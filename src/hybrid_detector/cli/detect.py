#!/usr/bin/env python3
"""Detect the Hybrid Carrier in one or more calibrated images.

Single camera, a self-contained camera file::

    python -m hybrid_detector.cli.detect \
        --model assets/models/hybrid_carrier_v1_20260904.json \
        --image cam_06=frame.png --camera cam_06=camera.json \
        --overlay outputs/hybrid/overlay --out outputs/hybrid/pose.json

Several cameras of the calibrated bench, solving one rig pose from all of them::

    python -m hybrid_detector.cli.detect \
        --model assets/models/hybrid_carrier_v1_20260904.json \
        --extrinsics <extrinsics summary> --intrinsics-dir <intrinsics dir> \
        --image cam_06=a.png --image cam_08=b.png --out pose.json

The report is written whether or not the solve succeeds, and it says which
stage ran out of evidence: no anchor, no hypothesis, too few facets, too few
edge samples.  A frame that simply could not be measured and a frame that was
measured badly need different fixes, and a bare "failed" hides which one it was.
"""

from __future__ import annotations

import argparse
import contextlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from hybrid_detector.calibration import CameraCalibration, load_rig, normalize_camera_model
from hybrid_detector.detector import (
    CarrierDetection,
    CarrierView,
    HybridCarrierModel,
    classify_colours,
    colour_reference_from_detections,
    detect_carrier,
    measure_edges,
    reject_mislocked_edges,
    solve_carrier_pose,
)

OVERLAY_COLOURS = {
    "red": (60, 60, 235),
    "blue": (235, 90, 60),
    "green": (70, 190, 70),
    "black": (200, 200, 200),
    "grey": (170, 170, 170),
    "white": (0, 220, 220),
}


def parse_pairs(values: list[str], what: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for value in values or []:
        if "=" not in value:
            raise SystemExit(f"--{what} wants NAME=PATH, got {value!r}")
        name, path = value.split("=", 1)
        out[name.strip()] = path.strip()
    return out


def camera_from_json(name: str, path: Path, width: int, height: int) -> CameraCalibration:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    return CameraCalibration(
        name=name,
        serial=str(cfg.get("serial", cfg.get("camera_serial", name))),
        width=int(cfg.get("width", cfg.get("image_width", width))),
        height=int(cfg.get("height", cfg.get("image_height", height))),
        K=np.asarray(cfg.get("K", cfg.get("camera_matrix")), dtype=np.float64).reshape(3, 3),
        D=np.asarray(
            cfg.get("dist", cfg.get("D", cfg.get("dist_coeffs", []))), dtype=np.float64
        ).reshape(-1),
        T_base_cam=np.asarray(cfg.get("T_base_cam", np.eye(4)), dtype=np.float64).reshape(4, 4),
        model=normalize_camera_model(str(cfg.get("model", "rational"))),
    )


def draw_overlay(
    image: np.ndarray,
    model: HybridCarrierModel,
    camera: CameraCalibration,
    detection: CarrierDetection,
    T_cam_rig: np.ndarray,
) -> np.ndarray:
    out = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for view in detection.facet_views:
        if len(view.polygon_uv) < 3:
            continue
        colour = OVERLAY_COLOURS.get(view.facet.colour, (255, 255, 255))
        thickness = 2 if view.usable else 1
        polygon = np.round(view.polygon_uv).astype(np.int32)
        cv2.polylines(out, [polygon], True, colour, thickness, cv2.LINE_AA)
        label = view.facet.name if view.usable else f"{view.facet.name}: {view.reason}"
        cv2.putText(
            out,
            label,
            tuple(np.round(view.polygon_uv.mean(axis=0)).astype(int)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            colour,
            1,
            cv2.LINE_AA,
        )
    # Every accepted edge sample, at the pixel it was actually found.
    for measurement in detection.measurements:
        cv2.circle(out, tuple(np.round(measurement.uv).astype(int)), 1, (0, 255, 255), -1)
    for anchor_detection in detection.anchors:
        quad = np.round(anchor_detection.corners_uv).astype(np.int32)
        cv2.polylines(out, [quad], True, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(
            out,
            f"id {anchor_detection.marker_id} q{anchor_detection.quadrant}",
            tuple(quad[0]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    if T_cam_rig is not None:
        rvec, _ = cv2.Rodrigues(np.ascontiguousarray(T_cam_rig[:3, :3]))
        with contextlib.suppress(cv2.error):
            cv2.drawFrameAxes(out, camera.K, camera.D, rvec, T_cam_rig[:3, 3], 0.04, 2)
    return out


def detection_report(
    detection: CarrierDetection, model: HybridCarrierModel, camera_name: str
) -> dict[str, Any]:
    band_widths = [m.band_width_px for m in detection.measurements if np.isfinite(m.band_width_px)]
    return {
        "camera": camera_name,
        "success": detection.success,
        "message": detection.message,
        "anchors": [
            {"marker_id": d.marker_id, "paste_quadrant": d.quadrant} for d in detection.anchors
        ],
        "hypothesis_margin": (
            None
            if not np.isfinite(detection.hypothesis_margin)
            else round(detection.hypothesis_margin, 3)
        ),
        "colour_reference": {
            "source": detection.reference.source,
            "white_bgr": np.round(detection.reference.white, 1).tolist(),
            "black_bgr": np.round(detection.reference.black, 1).tolist(),
        },
        "facets": [
            {
                "name": view.facet.name,
                "colour": view.facet.colour,
                "usable": view.usable,
                "reason": view.reason,
                "projected_area_px2": round(view.projected_area_px2, 1),
                "incidence_deg": round(view.incidence_deg, 1),
                "visible_fraction": round(view.visible_fraction, 3),
                "colour_coverage": (
                    None
                    if view.facet.name not in detection.coverage
                    else round(detection.coverage[view.facet.name], 3)
                ),
            }
            for view in detection.facet_views
        ],
        "edge_samples": len(detection.measurements),
        "edge_samples_by_profile": {
            kind: sum(1 for m in detection.measurements if m.profile == kind)
            for kind in sorted({m.profile for m in detection.measurements})
        },
        "measured_stroke_band_px": (
            {
                "median": round(float(np.median(band_widths)), 2),
                "p10": round(float(np.percentile(band_widths, 10)), 2),
                "p90": round(float(np.percentile(band_widths, 90)), 2),
                "note": (
                    "half-maximum width of the printed black band; compare against "
                    f"print.border.width_mm = {model.border_width_m * 1000:.2f} mm to check the print"
                ),
            }
            if band_widths
            else None
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True, type=Path, help="hybrid carrier descriptor JSON")
    parser.add_argument("--image", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--camera", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--extrinsics", type=Path, default=None)
    parser.add_argument("--intrinsics-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True, help="pose + diagnostics JSON")
    parser.add_argument("--overlay", type=Path, default=None, help="directory for overlay images")
    parser.add_argument("--anchor-sigma-px", type=float, default=0.3)
    parser.add_argument("--huber-px", type=float, default=2.0)
    parser.add_argument(
        "--unbiased-edges-only",
        action="store_true",
        help=(
            "use only edges whose ink band is self-centring. Costs samples but removes every "
            "landmark whose offset depends on the printed stroke width being what the model says"
        ),
    )
    parser.add_argument(
        "--allow-anchor-free",
        action="store_true",
        help=(
            "let a camera that sees no anchor be read from the paint alone. Off by default: it is "
            "gated harder than the anchored path and refuses rather than guesses, but on this rig "
            "the views without an anchor are also the views with the least measurable geometry, so "
            "expect most of them to come back as refusals with a reason attached"
        ),
    )
    parser.add_argument(
        "--min-anchors",
        type=int,
        default=1,
        help=(
            "refuse a camera that sees fewer anchors than this. On the seven-camera bench, "
            "2 is free -- every frame of the 2026-09-09 capture had at least two cameras "
            "holding two anchors -- and it removes every gross single-camera outlier"
        ),
    )
    parser.add_argument("--min-visible-fraction", type=float, default=0.6)
    parser.add_argument("--min-area-px2", type=float, default=150.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model = HybridCarrierModel.from_json(args.model)
    images = parse_pairs(args.image, "image")
    camera_files = parse_pairs(args.camera, "camera")

    loaded = {name: cv2.imread(str(path), cv2.IMREAD_COLOR) for name, path in images.items()}
    missing = [name for name, image in loaded.items() if image is None]
    if missing:
        raise SystemExit(f"could not read images for {missing}")

    cameras: dict[str, CameraCalibration] = {}
    provenance: dict[str, Any] = {}
    if args.extrinsics is not None and args.intrinsics_dir is not None:
        rig, rig_provenance = load_rig(args.extrinsics, args.intrinsics_dir, cameras=sorted(images))
        cameras = {camera.name: camera for camera in rig}
        provenance = rig_provenance.to_dict()
    for name, path in camera_files.items():
        height, width = loaded[name].shape[:2]
        cameras[name] = camera_from_json(name, Path(path), width, height)
    unknown = sorted(set(images) - set(cameras))
    if unknown:
        raise SystemExit(
            f"no calibration for {unknown}; pass --camera or --extrinsics/--intrinsics-dir"
        )

    detections: dict[str, CarrierDetection] = {}
    for name, image in loaded.items():
        detections[name] = detect_carrier(
            image,
            cameras[name],
            model,
            include_biased_edges=not args.unbiased_edges_only,
            min_visible_fraction=args.min_visible_fraction,
            min_area_px2=args.min_area_px2,
            huber_px=args.huber_px,
            anchor_sigma_px=args.anchor_sigma_px,
            min_anchors=int(args.min_anchors),
            allow_anchor_free=bool(args.allow_anchor_free),
        )

    solved = [name for name, detection in detections.items() if detection.success]
    joint: dict[str, Any] | None = None
    T_base_rig: np.ndarray | None = None
    if solved:
        seed = detections[solved[0]].T_base_rig
        views: list[CarrierView] = []
        for name in solved:
            camera = cameras[name]
            detection = detections[name]
            anchors = model.anchors_by_id
            object_points = np.vstack(
                [anchors[d.marker_id].object_points(d.quadrant) for d in detection.anchors]
            )
            image_points = np.vstack([d.corners_uv for d in detection.anchors])
            T_cam_rig = camera.T_cam_base @ seed
            reference = colour_reference_from_detections(loaded[name], detection.anchors)
            measurements = measure_edges(
                loaded[name],
                model,
                camera,
                T_cam_rig,
                reference=reference,
                views=detection.facet_views,
                include_biased=not args.unbiased_edges_only,
                masks=classify_colours(loaded[name], reference),
            )
            measurements = reject_mislocked_edges(measurements, camera, T_cam_rig)
            views.append(
                CarrierView(
                    camera=camera,
                    anchor_points_rig=object_points,
                    anchor_uv=image_points,
                    anchor_sigma_px=args.anchor_sigma_px,
                    edges=measurements,
                )
            )
        pose = solve_carrier_pose(views, seed, huber_px=args.huber_px)
        T_base_rig = pose.T_base_rig
        joint = {
            "cameras": solved,
            "T_base_rig": np.round(pose.T_base_rig, 9).tolist(),
            "anchor_rmse_px": None
            if not np.isfinite(pose.anchor_rmse_px)
            else round(pose.anchor_rmse_px, 4),
            "edge_rmse_px": None
            if not np.isfinite(pose.edge_rmse_px)
            else round(pose.edge_rmse_px, 4),
            "num_anchor_points": pose.num_anchor_points,
            "num_edge_samples": pose.num_edge_samples,
            "sigma_translation_m": pose.sigma_translation_m,
            "sigma_rotation_deg": pose.sigma_rotation_deg,
            "message": pose.message,
        }

    report = {
        "schema": "hybrid_carrier_detection/v1",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": {"path": str(args.model.resolve()), "carrier_id": model.carrier_id},
        "frame_convention": (
            "T_base_rig maps carrier coordinates into the calibration's world frame; with a "
            "single self-contained camera file that world frame is the camera itself"
        ),
        "calibration_provenance": provenance,
        "unbiased_edges_only": bool(args.unbiased_edges_only),
        "per_camera": [
            detection_report(detections[name], model, name) for name in sorted(detections)
        ],
        "joint": joint,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[INFO] wrote {args.out}")

    if args.overlay is not None:
        args.overlay.mkdir(parents=True, exist_ok=True)
        for name, detection in detections.items():
            camera = cameras[name]
            T_cam_rig = (
                camera.T_cam_base @ (T_base_rig if T_base_rig is not None else detection.T_base_rig)
                if (T_base_rig is not None or detection.T_base_rig is not None)
                else None
            )
            path = args.overlay / f"{name}_hybrid_carrier.png"
            cv2.imwrite(str(path), draw_overlay(loaded[name], model, camera, detection, T_cam_rig))
            print(f"[INFO] wrote {path}")

    for name in sorted(detections):
        detection = detections[name]
        usable = sum(1 for view in detection.facet_views if view.usable)
        print(
            f"[INFO] {name}: {'ok' if detection.success else 'FAILED'} "
            f"anchors={[d.marker_id for d in detection.anchors]} facets={usable} "
            f"edge_samples={len(detection.measurements)} {detection.message}"
        )
    if joint is not None:
        print(
            "[INFO] joint pose from {n} camera(s): anchor rmse {a} px, edge rmse {e} px, "
            "{s} edge samples".format(
                n=len(joint["cameras"]),
                a=joint["anchor_rmse_px"],
                e=joint["edge_rmse_px"],
                s=joint["num_edge_samples"],
            )
        )
    else:
        print("[WARN] no camera produced a pose")


if __name__ == "__main__":
    main()
