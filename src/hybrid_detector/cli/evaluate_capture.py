"""Freeze and evaluate paired, same-image capture ablations.

The freeze command validates a manifest and hashes every input before detection.
The run command checks those hashes, then compares anchors / colour / full on
identical decoded corners and images, with no temporal or reference pose seed.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import platform
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import scipy

from hybrid_detector.calibration import sha256_file
from hybrid_detector.cli.detect import camera_from_json, detection_report
from hybrid_detector.detector import (
    HybridCarrierModel,
    build_aruco_detector,
    detect_carrier,
    project_rig,
)
from hybrid_detector.evaluation import METHODS, make_summary, pose_difference, validate_pose

SCHEMA = "hybrid_detector/capture_evaluation/v1"
DETECTOR_PARAMETERS = (
    "min_anchors",
    "max_anchor_rms_ratio",
    "refine_iterations",
    "min_visible_fraction",
    "min_area_px2",
    "include_biased_edges",
    "huber_px",
    "anchor_sigma_px",
)
EVALUATION_DEFAULTS = {
    "translation_limit_mm": 10.0,
    "rotation_limit_deg": 2.0,
    "bootstrap_seed": 7,
    "bootstrap_repetitions": 2000,
}


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean_json(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(clean_json(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def environment() -> dict:
    package = Path(__file__).resolve().parents[1]
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "opencv": cv2.__version__,
        "platform": platform.platform(),
        "opencv_threads": 1,
        "source_sha256": {
            str(p.relative_to(package)): sha256_file(p) for p in sorted(package.rglob("*.py"))
        },
    }


def resolve(root: Path, value: str) -> Path:
    return (root / Path(value).expanduser()).resolve()


def freeze(manifest: Path, out: Path) -> dict:
    manifest, out = manifest.resolve(), out.resolve()
    doc = json.loads(manifest.read_text())
    if doc.get("schema") != SCHEMA:
        raise ValueError(f"manifest schema must be {SCHEMA}")
    for field in ("dataset_id", "world_frame", "purpose"):
        if not isinstance(doc.get(field), str) or not doc[field].strip():
            raise ValueError(f"manifest requires {field}")
    if not doc.get("cameras") or not doc.get("frames"):
        raise ValueError("manifest needs cameras and frames")
    detector = {
        key: inspect.signature(detect_carrier).parameters[key].default
        for key in DETECTOR_PARAMETERS
    }
    unknown = set(doc.get("detector", {})) - set(detector)
    if unknown:
        raise ValueError(f"unsupported detector parameters: {sorted(unknown)}")
    detector.update(doc.get("detector", {}))
    for key, value in detector.items():
        if key == "include_biased_edges":
            if not isinstance(value, bool):
                raise ValueError(f"{key} must be boolean")
        elif (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not np.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{key} must be finite and positive")
    for key in ("min_anchors", "refine_iterations"):
        if not isinstance(detector[key], int):
            raise ValueError(f"{key} must be an integer")
    if detector["min_visible_fraction"] > 1 or detector["max_anchor_rms_ratio"] < 1:
        raise ValueError("visibility must be <= 1 and anchor RMS ratio >= 1")
    evaluation = dict(EVALUATION_DEFAULTS)
    if set(doc.get("evaluation", {})) - set(evaluation):
        raise ValueError("unsupported evaluation parameter")
    evaluation.update(doc.get("evaluation", {}))
    for key in ("translation_limit_mm", "rotation_limit_deg"):
        value = evaluation[key]
        if not isinstance(value, (float, int)) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be finite and positive")
    for key in ("bootstrap_seed", "bootstrap_repetitions"):
        if type(evaluation[key]) is not int or evaluation[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    inputs = {}

    def bind(value: str) -> str:
        path = resolve(manifest.parent, value)
        key = os.path.relpath(path, out.parent)
        if key not in inputs:
            inputs[key] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        return key

    model_path = bind(doc["model"])
    model = HybridCarrierModel.from_json(resolve(out.parent, model_path))
    if not model.anchors or any(a.paste_quadrant is None for a in model.anchors):
        raise ValueError("capture evaluation requires pinned paste quadrants for every anchor")
    if len({a.dictionary for a in model.anchors}) != 1:
        raise ValueError("all anchors must use the same dictionary")
    cameras = {}
    for name, path in doc["cameras"].items():
        cameras[name] = bind(path)
        config = json.loads(resolve(out.parent, cameras[name]).read_text())
        for field in ("model", "T_base_cam"):
            if field not in config:
                raise ValueError(f"camera {name} must explicitly declare {field}")
        validate_pose(config["T_base_cam"], f"camera {name}")
        camera = camera_from_json(name, resolve(out.parent, cameras[name]), 0, 0)
        if (
            camera.width <= 0
            or camera.height <= 0
            or not np.isfinite(camera.K).all()
            or not np.isfinite(camera.D).all()
            or camera.K[0, 0] <= 0
            or camera.K[1, 1] <= 0
            or (camera.model == "fisheye" and camera.D.size != 4)
        ):
            raise ValueError(f"invalid camera calibration: {name}")
    reference = doc.get("reference")
    if reference is not None:
        if reference.get("kind") not in {"independent", "proxy", "synthetic"}:
            raise ValueError("reference.kind must be independent, proxy, or synthetic")
        for field in ("source", "registration", "synchronization", "uncertainty"):
            if not reference.get(field):
                raise ValueError(f"reference must document {field}")
    seen = set()
    frames = []
    for frame in doc["frames"]:
        for field in ("id", "group", "condition"):
            if not isinstance(frame.get(field), str) or not frame[field].strip():
                raise ValueError(f"every frame requires nonempty {field}")
        if frame["id"] in seen:
            raise ValueError(f"duplicate frame id: {frame['id']}")
        seen.add(frame["id"])
        if not frame.get("views") or set(frame["views"]) - set(cameras):
            raise ValueError(f"frame {frame['id']} has missing or unknown cameras")
        if len(frame["views"]) > 1 and not doc.get("synchronization"):
            raise ValueError("multi-camera frames require a synchronization description")
        item = {key: frame[key] for key in ("id", "group", "condition")}
        item["views"] = {}
        for camera_name, source in frame["views"].items():
            if set(source) == {"image"}:
                item["views"][camera_name] = {"image": bind(source["image"])}
            elif set(source) == {"video", "frame_index"}:
                if type(source["frame_index"]) is not int or source["frame_index"] < 0:
                    raise ValueError("video frame_index must be a nonnegative integer")
                item["views"][camera_name] = {
                    "video": bind(source["video"]),
                    "frame_index": source["frame_index"],
                }
            else:
                raise ValueError("view must specify image or video + frame_index")
        if frame.get("T_base_target_reference") is not None:
            if reference is None:
                raise ValueError("reference poses require reference provenance")
            item["T_base_target_reference"] = validate_pose(
                frame["T_base_target_reference"], frame["id"]
            ).tolist()
        frames.append(item)
    frozen = {
        "schema": SCHEMA + "/frozen",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": sha256_file(manifest),
        "dataset_id": doc["dataset_id"],
        "purpose": doc["purpose"],
        "world_frame": doc["world_frame"],
        "synchronization": doc.get("synchronization"),
        "reference": reference,
        "model": model_path,
        "cameras": cameras,
        "frames": frames,
        "inputs": inputs,
        "detector": detector,
        "evaluation": evaluation,
        "environment": environment(),
        "protocol": {
            "methods": list(METHODS),
            "allow_anchor_free": False,
            "temporal_seed": False,
            "shared_decoded_corners": True,
            "same_input_image": True,
            "colour": "CAD colour hypothesis scoring, with edge support and refinement disabled",
            "heldout_metric": "Predict decoded anchor corners in other synchronized cameras; never ground truth",
        },
        "inventory": {
            "frames": len(frames),
            "camera_frame_opportunities": sum(len(f["views"]) for f in frames),
            "groups": sorted({f["group"] for f in frames}),
            "reference_frames": sum("T_base_target_reference" in f for f in frames),
            "unique_input_files": len(inputs),
            "unique_input_bytes": sum(v["bytes"] for v in inputs.values()),
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, frozen)
    return frozen


class CaptureReader:
    def __init__(self, root: Path):
        self.root = root
        self.videos: dict[str, cv2.VideoCapture] = {}

    def read(self, source: dict) -> np.ndarray | None:
        if "image" in source:
            return cv2.imread(str(resolve(self.root, source["image"])), cv2.IMREAD_COLOR)
        key = source["video"]
        if key not in self.videos:
            self.videos[key] = cv2.VideoCapture(str(resolve(self.root, key)))
        capture = self.videos[key]
        index = source["frame_index"]
        if round(capture.get(cv2.CAP_PROP_POS_FRAMES)) != index:
            if not capture.set(cv2.CAP_PROP_POS_FRAMES, index):
                return None
        ok, image = capture.read()
        if not ok or abs(capture.get(cv2.CAP_PROP_POS_FRAMES) - (index + 1)) > 0.5:
            return None
        return image

    def close(self) -> None:
        for capture in self.videos.values():
            capture.release()


class FixedCorners:
    """Reuse only observed image corners, never another method's pose/quadrants."""

    def __init__(self, raw: tuple):
        self.raw = raw

    def detectMarkers(self, _image: np.ndarray) -> tuple:
        corners, ids, rejected = self.raw
        return ([c.copy() for c in corners], None if ids is None else ids.copy(), rejected)


def heldout_rmse(
    pose: np.ndarray, source_camera: str, raw: dict, cameras: dict, model: HybridCarrierModel
) -> tuple:
    per_camera = {}
    for name, (corners, ids, _) in raw.items():
        if name == source_camera or ids is None:
            continue
        errors = []
        for quad, marker_id in zip(corners, ids.ravel(), strict=True):
            anchor = model.anchors_by_id.get(int(marker_id))
            if anchor is None:
                continue
            points = anchor.object_points(anchor.paste_quadrant)
            local = cameras[name].T_cam_base @ pose
            if np.any((points @ local[:3, :3].T + local[:3, 3])[:, 2] <= 0):
                return None, {}, "predicted anchor behind held-out camera"
            predicted = project_rig(cameras[name], local, points)
            if not np.isfinite(predicted).all():
                return None, {}, "nonfinite held-out projection"
            errors.extend(
                np.sum((predicted - np.asarray(quad).reshape(4, 2)) ** 2, axis=1).tolist()
            )
        if errors:
            per_camera[name] = {"corners": len(errors), "sum_squared_px": sum(errors)}
    count = sum(v["corners"] for v in per_camera.values())
    return (
        float(np.sqrt(sum(v["sum_squared_px"] for v in per_camera.values()) / count))
        if count
        else None,
        per_camera,
        "ok" if count else "no decoded anchors in other cameras",
    )


def evaluate_frame(
    frame: dict, images: dict, cameras: dict, model: HybridCarrierModel, parameters: dict
) -> list[dict]:
    aruco = build_aruco_detector(model.anchors[0].dictionary)
    raw, invalid, decode_seconds = {}, {}, {}
    for name, image in images.items():
        camera = cameras[name]
        if image is None:
            invalid[name] = "input_decode_failed"
        elif image.shape[:2] != (camera.height, camera.width):
            invalid[name] = "image_calibration_size_mismatch"
        else:
            started = time.perf_counter()
            raw[name] = aruco.detectMarkers(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
            decode_seconds[name] = time.perf_counter() - started
    models = {
        "anchors": replace(model, facets=[], edges=[]),
        "colour": replace(model, edges=[]),
        "full": model,
    }
    rows = []
    for name, image in images.items():
        known = []
        if name in raw and raw[name][1] is not None:
            known = [int(i) for i in raw[name][1].ravel() if int(i) in model.anchors_by_id]
        row = {
            "frame": frame["id"],
            "group": frame["group"],
            "condition": frame["condition"],
            "camera": name,
            "decoded_anchor_count": len(known),
            "decoded_anchor_ids": known,
            "has_reference": "T_base_target_reference" in frame,
            "decoded_image_sha256": hashlib.sha256(image.tobytes()).hexdigest()
            if image is not None
            else None,
            "shared_corner_decode_seconds": decode_seconds.get(name),
            "methods": {},
        }
        # Rotate method order deterministically to reduce systematic warm-cache timing bias.
        offset = int(hashlib.sha256(f"{frame['id']}:{name}".encode()).hexdigest()[:8], 16) % len(
            METHODS
        )
        row["method_order"] = list(METHODS[offset:] + METHODS[:offset])
        for method in row["method_order"]:
            if name in invalid:
                row["methods"][method] = {
                    "success": False,
                    "message": invalid[name],
                    "seconds": None,
                }
                continue
            started = time.perf_counter()
            detection = detect_carrier(
                image,
                cameras[name],
                models[method],
                detector=FixedCorners(raw[name]),
                allow_anchor_free=False,
                **parameters,
            )
            out = detection_report(detection, models[method], name)
            out["seconds"] = time.perf_counter() - started
            out["T_base_target"] = detection.T_base_rig
            if detection.success:
                validate_pose(detection.T_base_rig, f"{frame['id']}:{name}:{method}")
                if row["has_reference"]:
                    out.update(
                        pose_difference(frame["T_base_target_reference"], detection.T_base_rig)
                    )
                error, evidence, status = heldout_rmse(
                    detection.T_base_rig, name, raw, cameras, model
                )
                out.update(
                    heldout_corner_rmse_px=error, heldout_evidence=evidence, heldout_status=status
                )
            if detection.pose is not None:
                pose = detection.pose
                out["pose_diagnostics"] = {
                    "anchor_rmse_px": pose.anchor_rmse_px,
                    "edge_rmse_px": pose.edge_rmse_px,
                    "variance_factor": pose.variance_factor,
                    "covariance": pose.covariance,
                    "sigma_translation_m": pose.sigma_translation_m,
                    "sigma_rotation_deg": pose.sigma_rotation_deg,
                    "solver_message": pose.message,
                }
            out["anchor_observations"] = [
                {"marker_id": a.marker_id, "quadrant": a.quadrant, "corners_uv": a.corners_uv}
                for a in detection.anchors
            ]
            out["edge_observations"] = [
                {
                    "edge_index": e.edge_index,
                    "point_rig": e.point_rig,
                    "uv": e.uv,
                    "normal_uv": e.normal_uv,
                    "sigma_px": e.sigma_px,
                    "profile": e.profile,
                }
                for e in detection.measurements
            ]
            row["methods"][method] = clean_json(out)
        row["difference_from_anchors"] = {
            method: pose_difference(
                row["methods"]["anchors"]["T_base_target"], row["methods"][method]["T_base_target"]
            )
            for method in METHODS[1:]
            if row["methods"]["anchors"]["success"] and row["methods"][method]["success"]
        }
        rows.append(row)
    return rows


def run(frozen_path: Path, out: Path) -> dict:
    frozen_path = frozen_path.resolve()
    frozen = json.loads(frozen_path.read_text())
    if frozen.get("schema") != SCHEMA + "/frozen":
        raise ValueError("run requires a frozen manifest")
    for key, expected in frozen["inputs"].items():
        path = resolve(frozen_path.parent, key)
        if path.stat().st_size != expected["bytes"] or sha256_file(path) != expected["sha256"]:
            raise ValueError(f"frozen input changed: {key}; make a new freeze")
    current = environment()
    for key in ("python", "numpy", "scipy", "opencv", "source_sha256"):
        if current[key] != frozen["environment"][key]:
            raise ValueError(f"frozen environment changed: {key}; make a new freeze")
    cv2.setNumThreads(1)
    model = HybridCarrierModel.from_json(resolve(frozen_path.parent, frozen["model"]))
    cameras = {
        name: camera_from_json(name, resolve(frozen_path.parent, path), 0, 0)
        for name, path in frozen["cameras"].items()
    }
    out.mkdir(parents=True, exist_ok=False)
    write_json(
        out / "run.json",
        {
            "frozen_manifest_sha256": sha256_file(frozen_path),
            "dataset_id": frozen["dataset_id"],
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "environment": current,
            "detector": frozen["detector"],
            "evaluation": frozen["evaluation"],
            "inventory": frozen["inventory"],
            "protocol": frozen["protocol"],
        },
    )
    reader = CaptureReader(frozen_path.parent)
    rows = []
    try:
        with (out / "observations.jsonl").open("x", encoding="utf-8") as stream:
            for index, frame in enumerate(frozen["frames"]):
                images = {name: reader.read(source) for name, source in frame["views"].items()}
                batch = evaluate_frame(frame, images, cameras, model, frozen["detector"])
                for row in batch:
                    stream.write(
                        json.dumps(clean_json(row), ensure_ascii=False, allow_nan=False) + "\n"
                    )
                stream.flush()
                # Full pixel evidence stays in JSONL; keep compact statistics in memory.
                for row in batch:
                    compact = {
                        key: row[key]
                        for key in (
                            "frame",
                            "group",
                            "condition",
                            "camera",
                            "decoded_anchor_count",
                            "has_reference",
                            "difference_from_anchors",
                        )
                    }
                    compact["methods"] = {
                        method: {
                            key: value
                            for key, value in result.items()
                            if key
                            in {
                                "success",
                                "message",
                                "translation_mm",
                                "rotation_deg",
                                "heldout_corner_rmse_px",
                                "heldout_status",
                                "seconds",
                            }
                        }
                        for method, result in row["methods"].items()
                    }
                    rows.append(compact)
                print(
                    f"[{index + 1}/{len(frozen['frames'])}] {frame['id']}: {len(batch)} views",
                    flush=True,
                )
    finally:
        reader.close()
    summary = make_summary(rows, frozen["evaluation"], frozen["reference"])
    summary.update(
        dataset_id=frozen["dataset_id"],
        purpose=frozen["purpose"],
        completed_utc=datetime.now(timezone.utc).isoformat(),
    )
    write_json(out / "summary.json", summary)
    write_report(out / "report.md", summary)
    return summary


def write_report(path: Path, summary: dict) -> None:
    def p90(entry, metric):
        value = entry.get(metric)
        return f"{value['p90']:.3f}" if value else "N/A"

    lines = [
        f"# Capture pilot: {summary['dataset_id']}",
        "",
        summary["purpose"],
        "",
        f"Camera-frame opportunities: {summary['opportunities']}; with reference: {summary['reference_opportunities']}.",
        "",
        summary["interpretation"],
        "",
        "| Method | Accepted | Failed | Translation P90 (mm) | Rotation P90 (deg) | Held-out P90 (px) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method, entry in summary["methods"].items():
        lines.append(
            f"| {method} | {entry['accepted']} | {entry['failed']} | "
            f"{p90(entry, 'translation_mm')} | {p90(entry, 'rotation_deg')} | "
            f"{p90(entry, 'heldout_corner_rmse_px')} |"
        )
    lines.extend(
        [
            "",
            "These error percentiles are conditional on accepted, evaluable results. "
            "See summary.json for matched pairs, failures, reference-limit coverage and group bootstrap intervals.",
            "",
            "## Difference from the anchor-only answer",
            "",
            "| Method | Translation P90 (mm) | Rotation P90 (deg) |",
            "|---|---:|---:|",
        ]
    )
    for method, entry in summary["difference_from_anchors"].items():
        lines.append(
            f"| {method} | {p90(entry, 'translation_mm')} | {p90(entry, 'rotation_deg')} |"
        )
    lines.extend(
        [
            "",
            "A different answer is not evidence of an improvement. "
            "Without independent reference poses this run cannot establish absolute accuracy or true pose gain.",
            "",
            "Shared corner decoding is timed separately in observations.jsonl. "
            "Per-method seconds exclude that shared work, image/video decoding and reporting; "
            "these are cold per-view ablations, not production multi-camera throughput.",
            "",
        ]
    )
    with path.open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, argument in (("freeze", "manifest"), ("run", "frozen")):
        child = subparsers.add_parser(command)
        child.add_argument(argument, type=Path)
        child.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "freeze":
            result = freeze(args.manifest, args.out)
            print(json.dumps(result["inventory"], indent=2))
        else:
            result = run(args.frozen, args.out)
            print(
                json.dumps(
                    {
                        "opportunities": result["opportunities"],
                        "reference_opportunities": result["reference_opportunities"],
                    }
                )
            )
    except (ValueError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
