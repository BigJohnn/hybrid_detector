#!/usr/bin/env python3
"""Score two fiducial targets against each other through one render -> detect loop.

"Is the painted carrier actually better than the ArUco rig we already have?" is
not answerable from each target's own report: the carrier is scored on rendered
poses, the rig on per-camera dispersion in real footage, and the two numbers are
different quantities that happen to share a unit.  This CLI puts every candidate
through the *same* harness -- same poses, same camera, same blur and noise, same
solver -- so the only thing that differs is what the body carries.

Two things it can do that a single-model smoke test cannot:

**Score a target against a model of itself that is wrong.**  A target spec is
``render[:detect]``: the image is rendered from the first model and solved
against the second.  That is how a geometry error gets priced -- a print scale
error, or an as-built rig read against nominal CAD -- instead of being defined
away by using one model for both halves.

**Take away the paint.**  ``--strip-paint`` drops the facets and edges from a
carrier and leaves its ArUco anchors, so "the same body, printed the plain way"
is a row in the same table rather than an argument.

**Put something in the way.**  ``--obstruction-sweep`` covers a fraction of the
target's silhouette with a slab the detector is never told about, which is the
honest form of the question the rig rows raise: the wrapper renders five
floating pads with no frame, no gripper and no hand, so it says what the target
could do if nothing were ever in front of it.  What the sweep answers is how
each target degrades as that stops being true -- and the two should not degrade
alike, because information spread over 31 edges and information carried by five
markers run out in different ways.

**Move the target while the shutter is open.**  ``--motion-sweep-mm`` scores
every target again at each smear length, on the same poses and the same motion
directions, so the rows are paired.  Note what this does and does not settle
about the budget's timing row.  That row is ``v * dt``: a correct pose carrying
a wrong time tag, which displaces the answer by the same vector whatever the
target is made of, and no target change touches it.  What the sweep prices is
the other half of the same velocity -- the target moved *during* the exposure,
so the picture the solver gets is worse -- and that half is target-dependent,
because a smeared ArUco corner and a smeared painted edge do not degrade alike.

A plain ArUco marker layout (``marker_layout/measured_v1``, what
``bootstrap_marker_layout`` writes) is admitted with ``--layout`` and wrapped as
an anchors-only carrier: each marker becomes one pad, and the pad is the only
occluding surface.  That wrapper is an idealisation in the target's favour --
there is no frame, no gripper and no hand in the render, so every marker whose
normal faces the camera is seen.  Real footage is much less generous: in the
2026-08-18 rig capture cam_06 and cam_08 decoded 1.08 and 1.25 markers per frame
out of five.  Read a rig row as an upper bound on that target, not a forecast.

What the numbers are not: this is a pinhole camera with exact intrinsics, no
distortion, Gaussian blur and Gaussian noise.  Absolute millimetres here are not
production error -- production carries fisheye calibration error, real lighting
and real motion.  The ratio between two rows measured on the same poses is the
result; the absolute value of one row is not.

Usage::

    python -m hybrid_detector.cli.compare_targets \
        --target carrier=fixtures/cad/hybrid_carrier_v1_20260907.json \
        --target carrier_plain=fixtures/cad/hybrid_carrier_v1_20260907.json \
        --strip-paint carrier_plain \
        --layout rig_measured=outputs/.../marker_layout_measured.json \
        --layout rig_no_ba=outputs/.../layout_cadprior.json:outputs/.../layout_cad.json \
        --layout-pad-size-mm 69.6 --poses 12 --out outputs/target_ab.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from hybrid_detector.calibration import CameraCalibration
from hybrid_detector.detector import HybridCarrierModel, detect_carrier, project_rig, transform
from hybrid_detector.render import render_carrier_exposure

# marker_layout/measured_v1 lists corners clockwise about the outward normal --
# the ArUco order -- so the pad, which the descriptor wants counter-clockwise,
# is the same ring reversed.
LAYOUT_UNIT_TO_MM = {"m": 1000.0, "mm": 1.0, "cm": 10.0}


def marker_layout_as_carrier(
    layout: dict[str, Any],
    *,
    carrier_id: str,
    pad_size_mm: float,
    dictionary: str = "DICT_6X6_50",
) -> dict[str, Any]:
    """Wrap a plain marker layout as an anchors-only carrier descriptor."""
    scale = LAYOUT_UNIT_TO_MM[str(layout.get("units", "m")).lower()]
    anchors: list[dict[str, Any]] = []
    triangles: list[list[list[float]]] = []
    face_ids: list[int] = []
    for index, marker in enumerate(layout["markers"]):
        corners = np.asarray(marker["corners_rig"], dtype=np.float64) * scale
        if corners.shape != (4, 3):
            raise SystemExit(f"marker {marker.get('id')}: expected 4 corners, got {corners.shape}")
        centre = corners.mean(axis=0)
        # The layout may or may not carry the derived fields; recompute rather
        # than trust one of two spellings.
        outward = np.cross(corners[1] - corners[0], corners[2] - corners[0])
        outward = -outward / np.linalg.norm(outward)  # corners run CW about it
        edges = [float(np.linalg.norm(corners[(i + 1) % 4] - corners[i])) for i in range(4)]
        edge = float(np.median(edges))
        if pad_size_mm < edge:
            raise SystemExit(
                f"--layout-pad-size-mm {pad_size_mm} is smaller than marker {edge:.2f} mm"
            )
        pad = centre + (corners - centre) * (pad_size_mm / edge)
        pad_ccw = pad[::-1]
        face_id = 1000 + index
        anchors.append(
            {
                "name": f"plate_{marker['id']}",
                "marker_id": int(marker["id"]),
                "dictionary": dictionary,
                "face_entity_id": face_id,
                "pad_corners": pad_ccw.tolist(),
                "marker_corners": corners.tolist(),
                "normal": outward.tolist(),
                "centre": centre.tolist(),
                "pad_size": float(pad_size_mm),
                "marker_size": edge,
                "quiet_zone": (float(pad_size_mm) - edge) / 2.0,
                "paste_quadrant": 0,
            }
        )
        triangles += [
            [pad_ccw[0].tolist(), pad_ccw[1].tolist(), pad_ccw[2].tolist()],
            [pad_ccw[0].tolist(), pad_ccw[2].tolist(), pad_ccw[3].tolist()],
        ]
        face_ids += [face_id, face_id]
    return {
        "schema": "hybrid_carrier/v1",
        "carrier_id": carrier_id,
        "units": "mm",
        "source": {
            "wrapped_layout_id": layout.get("layout_id"),
            "wrapper": "marker_layout_as_carrier",
        },
        "anchors": anchors,
        "facets": [],
        "edges": [],
        "occluders": {"triangles": triangles, "face_entity_ids": face_ids},
        "print": {"border": {"width_mm": 0.0, "alignment": "inside"}},
    }


def strip_paint(doc: dict[str, Any]) -> dict[str, Any]:
    """The same body with only its ArUco anchors printed.

    The occluders stay: the body still hides what is behind it, and each anchor
    still needs its own face in the depth buffer for the sticker to land on.
    """
    stripped = dict(doc)
    stripped["carrier_id"] = f"{doc.get('carrier_id', 'carrier')}_anchors_only"
    stripped["facets"] = []
    stripped["edges"] = []
    return stripped


def synthetic_camera(width: int, height: int, focal_px: float) -> CameraCalibration:
    return CameraCalibration(
        name="synthetic",
        serial="render-detect",
        width=int(width),
        height=int(height),
        K=np.array([[focal_px, 0.0, width / 2.0], [0.0, focal_px, height / 2.0], [0.0, 0.0, 1.0]]),
        D=np.zeros(5),
        T_base_cam=np.eye(4),
        model="rational",
    )


def pose_set(
    count: int, *, seed: int, tilt_deg: float, distance_m: tuple[float, float], offset_m: float
) -> list[np.ndarray]:
    """Poses that face the camera, tilted and rolled, at a working distance.

    The 180 deg flip about x puts the target's +z towards the camera; the tilt
    is bounded because a target seen edge-on is a visibility question, not an
    accuracy one, and mixing the two makes neither readable.
    """
    rng = np.random.default_rng(int(seed))
    flip = Rotation.from_euler("xyz", [180.0, 0.0, 0.0], degrees=True).as_matrix()
    poses = []
    for _ in range(int(count)):
        euler = rng.uniform(-1.0, 1.0, 3) * np.array([tilt_deg, tilt_deg, 180.0])
        T = np.eye(4)
        T[:3, :3] = flip @ Rotation.from_euler("xyz", euler, degrees=True).as_matrix()
        T[:3, 3] = [
            rng.uniform(-offset_m, offset_m),
            rng.uniform(-offset_m, offset_m),
            rng.uniform(*distance_m),
        ]
        poses.append(T)
    return poses


def motion_directions(count: int, *, seed: int) -> list[np.ndarray]:
    """One lateral direction per pose, shared by every target.

    Lateral, because motion along the line of sight barely smears: including it
    would leave some rungs of the ladder meaning much less smear than others and
    make the sweep unreadable.  Shared, because the sweep is only worth reading
    as a paired comparison -- same poses, same directions, one thing different.
    """
    rng = np.random.default_rng(int(seed) + 977)
    phi = rng.uniform(0.0, 2.0 * np.pi, int(count))
    return [np.array([np.cos(a), np.sin(a), 0.0]) for a in phi]


def exposure_poses(
    T_cam_rig: np.ndarray,
    *,
    smear_mm: float,
    lever_mm: float,
    direction: np.ndarray,
    focal_px: float,
    max_samples: int = 25,
) -> list[np.ndarray]:
    """Sample one pose across the exposure it was seen through.

    A hand does not translate cleanly, it swings, so the movement that smears
    the body sideways also turns it.  Both come off one number -- how far the
    target travelled while the shutter was open -- with the turn fixed at
    ``smear / lever`` radians about the axis a swing on that lever turns about.
    A rig of coplanar plates hardly notices the turn.  A body carrying facets at
    many angles might, and that asymmetry is a reason to sweep both rather than
    argue about which target motion favours.

    Samples are spaced about a pixel of smear apart, so what the exposure
    integrates is a smear and not a comb of separated copies.  ``max_samples``
    caps the cost; past it the comb returns, and the row says how many were used.
    """
    if smear_mm <= 0.0:
        return [np.asarray(T_cam_rig, dtype=np.float64)]
    depth = max(float(T_cam_rig[2, 3]), 1e-6)
    smear_px = smear_mm * 1e-3 * float(focal_px) / depth
    samples = int(min(int(max_samples), max(2, math.ceil(smear_px) + 1)))
    unit = np.asarray(direction, dtype=np.float64)
    unit = unit / max(float(np.linalg.norm(unit)), 1e-12)
    axis = np.cross(unit, np.array([0.0, 0.0, 1.0]))
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    turn = float(smear_mm) / max(float(lever_mm), 1e-6)
    out = []
    for fraction in np.linspace(-0.5, 0.5, samples):
        T = np.eye(4)
        T[:3, :3] = Rotation.from_rotvec(axis * turn * fraction).as_matrix() @ T_cam_rig[:3, :3]
        T[:3, 3] = T_cam_rig[:3, 3] + unit * (smear_mm * 1e-3 * fraction)
        out.append(T)
    return out


def obstruction_angles(count: int, *, seed: int) -> list[float]:
    """One cut orientation per pose, shared by every target, as with the smear."""
    rng = np.random.default_rng(int(seed) + 5501)
    return [float(a) for a in rng.uniform(0.0, 2.0 * np.pi, int(count))]


def _clip_half_plane(
    polygon: list[np.ndarray], normal: np.ndarray, offset: float
) -> list[np.ndarray]:
    """Sutherland-Hodgman against one half plane, keeping ``normal . x >= offset``."""
    out: list[np.ndarray] = []
    for index, start in enumerate(polygon):
        end = polygon[(index + 1) % len(polygon)]
        d_start = float(np.dot(normal, start)) - offset
        d_end = float(np.dot(normal, end)) - offset
        if d_start >= 0.0:
            out.append(start)
        if (d_start >= 0.0) != (d_end >= 0.0):
            out.append(start + (end - start) * (d_start / (d_start - d_end)))
    return out


def obstruction_triangles(
    model: HybridCarrierModel,
    camera: CameraCalibration,
    T_cam_rig: np.ndarray,
    *,
    fraction: float,
    angle: float,
    clearance_mm: float = 6.0,
    raster_px: int = 160,
) -> np.ndarray:
    """A slab in front of the target covering ``fraction`` of its silhouette.

    The cut is placed by area, not by geometry: the target's own skin is
    rasterised once, and the offset along the cut normal is the quantile that
    leaves the asked-for share of those pixels behind the slab.  So a rung of
    the sweep means the same thing for a five-plate rig and for a painted body,
    which two hand-shaped occluders sized in millimetres would not.

    The cut orientation is random per pose rather than fixed to the body, which
    is a deliberate softening: a real hand grips the same place every time and
    would always take the same features away, so this prices *a* share of the
    target being hidden and not the worst share.  Returned in the rig frame, so
    the slab travels with the body through an exposure -- it is the hand holding
    the target, not an arm crossing in front of it.
    """
    triangles = np.asarray(model.occluder_triangles, dtype=np.float64).reshape(-1, 3, 3)
    empty = np.zeros((0, 3, 3), dtype=np.float64)
    if fraction <= 0.0 or len(triangles) == 0:
        return empty
    cam = transform(T_cam_rig, triangles.reshape(-1, 3)).reshape(-1, 3, 3)
    keep = np.all(cam[:, :, 2] > 1e-6, axis=1)
    # Cull exactly as the renderer does, or the quantile is taken over a
    # silhouette wider than the one that gets painted and the rung lies about
    # how much it covered.
    facing = np.einsum(
        "ij,ij->i",
        np.cross(cam[:, 1] - cam[:, 0], cam[:, 2] - cam[:, 0]),
        cam.mean(axis=1),
    )
    keep &= facing < 0.0
    if not np.any(keep):
        return empty
    cam = cam[keep]
    uv = project_rig(camera, T_cam_rig, triangles[keep].reshape(-1, 3)).reshape(-1, 3, 2)
    keep = np.all(np.isfinite(uv).reshape(len(uv), -1), axis=1)
    if not np.any(keep):
        return empty
    cam, uv = cam[keep], uv[keep]

    flat = uv.reshape(-1, 2)
    lower, upper = flat.min(axis=0), flat.max(axis=0)
    span = float(np.max(upper - lower))
    if not np.isfinite(span) or span <= 0.0:
        return empty
    scale = float(raster_px) / span
    shape = (
        int(np.ceil((upper[1] - lower[1]) * scale)) + 2,
        int(np.ceil((upper[0] - lower[0]) * scale)) + 2,
    )
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.round((t - lower) * scale).astype(np.int32) for t in uv], 255)
    rows, columns = np.nonzero(mask)
    if not len(rows):
        return empty
    normal = np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
    along = (columns / scale + lower[0]) * normal[0] + (rows / scale + lower[1]) * normal[1]
    offset = float(np.quantile(along, 1.0 - float(fraction)))

    width, height = float(camera.width), float(camera.height)
    rectangle = [
        np.array([0.0, 0.0]),
        np.array([width, 0.0]),
        np.array([width, height]),
        np.array([0.0, height]),
    ]
    covered = _clip_half_plane(rectangle, normal, offset)
    if len(covered) < 3:
        return empty

    # The harness renders through an ideal pinhole, so the inverse of K is the
    # exact back projection; a distorted camera would want undistortPoints here.
    depth = max(float(cam[:, :, 2].min()) - float(clearance_mm) * 1e-3, 1e-3)
    inverse_k = np.linalg.inv(np.asarray(camera.K, dtype=np.float64))
    points_cam = []
    for point in covered:
        ray = inverse_k @ np.array([point[0], point[1], 1.0])
        points_cam.append(ray * (depth / max(float(ray[2]), 1e-12)))
    fan = np.array(
        [[points_cam[0], points_cam[i], points_cam[i + 1]] for i in range(1, len(points_cam) - 1)]
    )
    first = fan[0]
    if float(np.dot(np.cross(first[1] - first[0], first[2] - first[0]), first.mean(axis=0))) >= 0.0:
        fan = fan[:, ::-1, :]  # the renderer culls loops not wound CCW about the outward normal
    inverse_pose = np.linalg.inv(np.asarray(T_cam_rig, dtype=np.float64))
    return transform(inverse_pose, fan.reshape(-1, 3)).reshape(-1, 3, 3)


@dataclass(frozen=True)
class Target:
    label: str
    render: HybridCarrierModel
    detect: HybridCarrierModel
    render_source: str
    detect_source: str

    @property
    def same_model(self) -> bool:
        return self.render_source == self.detect_source


def score(
    target: Target,
    camera: CameraCalibration,
    poses: list[np.ndarray],
    *,
    blur_sigma: float,
    noise_sigma: float,
    motion_mm: float = 0.0,
    motion_lever_mm: float = 300.0,
    motion_directions_: list[np.ndarray] | None = None,
    motion_max_samples: int = 25,
    obstruction_fraction: float = 0.0,
    obstruction_angles_: list[float] | None = None,
    allow_anchor_free: bool = False,
) -> dict[str, Any]:
    started = time.time()
    translations, rotations, anchors, measurements = [], [], [], []
    samples: list[int] = []
    failures: list[dict[str, Any]] = []
    for index, T_true in enumerate(poses):
        during = exposure_poses(
            T_true,
            smear_mm=motion_mm,
            lever_mm=motion_lever_mm,
            direction=(motion_directions_ or [np.array([1.0, 0.0, 0.0])] * len(poses))[index],
            focal_px=float(camera.K[0, 0]),
            max_samples=motion_max_samples,
        )
        samples.append(len(during))
        # Built from the render geometry and handed only to the renderer: the
        # detector is not told where the obstruction is, because in the real
        # frame nothing tells it.
        slab = obstruction_triangles(
            target.render,
            camera,
            T_true,
            fraction=obstruction_fraction,
            angle=(obstruction_angles_ or [0.0] * len(poses))[index],
        )
        image = render_carrier_exposure(
            target.render,
            camera,
            during,
            obstructions=slab,
            blur_sigma=blur_sigma,
            noise_sigma=noise_sigma,
            seed=index,
        )
        detection = detect_carrier(
            image, camera, target.detect, allow_anchor_free=allow_anchor_free
        )
        if not detection.success:
            failures.append({"pose": index, "message": detection.message})
            continue
        delta = np.linalg.inv(T_true) @ detection.T_base_rig
        translations.append(float(np.linalg.norm(delta[:3, 3]) * 1000.0))
        rotations.append(
            float(np.degrees(np.linalg.norm(Rotation.from_matrix(delta[:3, :3]).as_rotvec())))
        )
        anchors.append(len(detection.anchors))
        measurements.append(len(detection.measurements))
    result: dict[str, Any] = {
        "label": target.label,
        "render_model": target.render_source,
        "detect_model": target.detect_source,
        "geometry_matches": target.same_model,
        "poses": len(poses),
        "motion_mm": float(motion_mm),
        "motion_lever_mm": float(motion_lever_mm),
        "obstruction_fraction": float(obstruction_fraction),
        "allow_anchor_free": bool(allow_anchor_free),
        "exposure_samples_p50": float(np.median(samples)) if samples else 0.0,
        "solved": len(translations),
        "failures": failures,
        "seconds": round(time.time() - started, 1),
        "translation_error_mm": translations,
        "rotation_error_deg": rotations,
        "anchors_per_pose": anchors,
        "edge_measurements_per_pose": measurements,
    }
    if translations:
        result["summary"] = {
            "translation_p50_mm": float(np.median(translations)),
            "translation_p90_mm": float(np.percentile(translations, 90)),
            "translation_max_mm": float(np.max(translations)),
            "rotation_p50_deg": float(np.median(rotations)),
            "anchors_p50": float(np.median(anchors)),
            "edge_measurements_p50": float(np.median(measurements)),
        }
    return result


def _split_spec(spec: str, flag: str) -> tuple[str, str, str]:
    label, sep, paths = spec.partition("=")
    if not sep or not label.strip():
        raise SystemExit(f"{flag} wants LABEL=PATH[:PATH], got {spec!r}")
    render, _, detect = paths.partition(":")
    if not render:
        raise SystemExit(f"{flag} wants LABEL=PATH[:PATH], got {spec!r}")
    return label.strip(), render, (detect or render)


def build_targets(args: argparse.Namespace) -> list[Target]:
    target_specs = [_split_spec(spec, "--target") for spec in args.target or []]
    layout_specs = [_split_spec(spec, "--layout") for spec in args.layout or []]
    if not target_specs and not layout_specs:
        raise SystemExit("nothing to compare: pass at least one --target or --layout")
    strip = set(args.strip_paint or [])
    unknown = strip - {label for label, _, _ in target_specs}
    if unknown:
        raise SystemExit(f"--strip-paint names no --target: {sorted(unknown)}")
    docs: dict[str, dict[str, Any]] = {}

    def carrier_doc(path: str) -> dict[str, Any]:
        if path not in docs:
            docs[path] = json.loads(Path(path).read_text())
        return docs[path]

    def layout_doc(path: str, label: str) -> dict[str, Any]:
        key = f"layout::{path}"
        if key not in docs:
            docs[key] = marker_layout_as_carrier(
                json.loads(Path(path).read_text()),
                carrier_id=f"{label}_anchors_only",
                pad_size_mm=args.layout_pad_size_mm,
                dictionary=args.layout_dictionary,
            )
        return docs[key]

    targets: list[Target] = []
    for label, render, detect in target_specs:
        pair = [carrier_doc(render), carrier_doc(detect)]
        if label in strip:
            pair = [strip_paint(doc) for doc in pair]
        targets.append(
            Target(
                label,
                HybridCarrierModel.from_dict(pair[0]),
                HybridCarrierModel.from_dict(pair[1]),
                render,
                detect,
            )
        )
    for label, render, detect in layout_specs:
        targets.append(
            Target(
                label,
                HybridCarrierModel.from_dict(layout_doc(render, label)),
                HybridCarrierModel.from_dict(layout_doc(detect, label)),
                render,
                detect,
            )
        )
    return targets


def format_table(results: list[dict[str, Any]]) -> str:
    """The comparison as a table.  A smear column appears only if anything moved."""
    moving = any(row.get("motion_mm", 0.0) > 0.0 for row in results)
    hidden = any(row.get("obstruction_fraction", 0.0) > 0.0 for row in results)
    smear_head = f" {'smear':>6s}" if moving else ""
    hidden_head = f" {'hid':>5s}" if hidden else ""
    header = (
        f"{'target':26s}{smear_head}{hidden_head} {'n':>3s} {'fail':>4s} {'p50':>7s} {'p90':>7s} "
        f"{'max':>7s} {'drot':>6s} {'anc':>4s} {'edge':>5s}"
    )
    lines = [header, "-" * len(header)]
    for row in results:
        smear = f" {row.get('motion_mm', 0.0):6.1f}" if moving else ""
        cover = f" {row.get('obstruction_fraction', 0.0):5.2f}" if hidden else ""
        s = row.get("summary")
        if not s:
            lines.append(
                f"{row['label']:26s}{smear}{cover} {0:3d} {len(row['failures']):4d}   (no pose solved)"
            )
            continue
        lines.append(
            f"{row['label']:26s}{smear}{cover} {row['solved']:3d} {len(row['failures']):4d} "
            f"{s['translation_p50_mm']:7.2f} {s['translation_p90_mm']:7.2f} "
            f"{s['translation_max_mm']:7.2f} {s['rotation_p50_deg']:6.2f} "
            f"{s['anchors_p50']:4.0f} {s['edge_measurements_p50']:5.0f}"
        )
    lines.append("")
    lines.append("p50/p90/max are |dt| in mm, drot is the rotation error in deg, anc and edge are")
    lines.append("the median anchors and edge measurements a pose was solved from.")
    if moving:
        lines.append("smear is how far the target travelled in mm while the shutter was open.")
    if hidden:
        lines.append("hid is the share of the target's silhouette a slab covered.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--target",
        action="append",
        metavar="LABEL=RENDER[:DETECT]",
        help="a hybrid carrier descriptor; give two paths to solve one geometry against another",
    )
    parser.add_argument(
        "--layout",
        action="append",
        metavar="LABEL=RENDER[:DETECT]",
        help="a marker_layout json, wrapped as an anchors-only carrier",
    )
    parser.add_argument(
        "--strip-paint",
        action="append",
        metavar="LABEL",
        help="drop the facets and edges of this --target, leaving its ArUco anchors",
    )
    parser.add_argument("--layout-pad-size-mm", type=float, default=69.6)
    parser.add_argument("--layout-dictionary", default="DICT_6X6_50")
    parser.add_argument("--poses", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--tilt-deg", type=float, default=26.0)
    parser.add_argument("--distance-m", type=float, nargs=2, default=(0.55, 0.85))
    parser.add_argument("--lateral-offset-m", type=float, default=0.05)
    parser.add_argument(
        "--motion-sweep-mm",
        type=float,
        nargs="+",
        default=[0.0],
        metavar="MM",
        help="score every target once per smear length: how far it moved during the exposure",
    )
    parser.add_argument(
        "--motion-lever-mm",
        type=float,
        default=300.0,
        help="the lever the hand swings on; the turn during the exposure is smear/lever radians",
    )
    parser.add_argument("--motion-max-samples", type=int, default=25)
    parser.add_argument(
        "--allow-anchor-free",
        action="store_true",
        help=(
            "let a frame that shows no anchor be seeded from the paint instead; only a "
            "painted target has anything to seed from, so this is the carrier's own answer "
            "to being covered and it is off by default"
        ),
    )
    parser.add_argument(
        "--obstruction-sweep",
        type=float,
        nargs="+",
        default=[0.0],
        metavar="FRACTION",
        help="score every target once per share of its silhouette a slab covers",
    )
    parser.add_argument("--blur-sigma", type=float, default=0.8)
    parser.add_argument("--noise-sigma", type=float, default=2.5)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--focal-px", type=float, default=1000.0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    targets = build_targets(args)
    camera = synthetic_camera(args.width, args.height, args.focal_px)
    poses = pose_set(
        args.poses,
        seed=args.seed,
        tilt_deg=args.tilt_deg,
        distance_m=tuple(args.distance_m),
        offset_m=args.lateral_offset_m,
    )

    directions = motion_directions(len(poses), seed=args.seed)
    angles = obstruction_angles(len(poses), seed=args.seed)
    levels = list(args.motion_sweep_mm) or [0.0]
    covers = list(args.obstruction_sweep) or [0.0]
    sweeping_motion = len(levels) > 1 or any(level > 0.0 for level in levels)
    sweeping_cover = len(covers) > 1 or any(cover > 0.0 for cover in covers)

    results = []
    for target in targets:
        for level in levels:
            for cover in covers:
                label = target.label
                if sweeping_motion:
                    label += f"@{level:g}mm"
                if sweeping_cover:
                    label += f"+{cover * 100:.0f}%hid"
                row = score(
                    target,
                    camera,
                    poses,
                    blur_sigma=args.blur_sigma,
                    noise_sigma=args.noise_sigma,
                    motion_mm=level,
                    motion_lever_mm=args.motion_lever_mm,
                    motion_directions_=directions,
                    motion_max_samples=args.motion_max_samples,
                    obstruction_fraction=cover,
                    obstruction_angles_=angles,
                    allow_anchor_free=args.allow_anchor_free,
                )
                row["label"] = label
                row["target"] = target.label
                results.append(row)
                s = row.get("summary")
                print(
                    f"{label:26s} n={row['solved']:2d} fail={len(row['failures'])}  "
                    + (
                        f"|dt| p50={s['translation_p50_mm']:5.2f} p90={s['translation_p90_mm']:5.2f} mm"
                        if s
                        else "(no pose solved)"
                    ),
                    flush=True,
                )

    print()
    print(format_table(results))

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "schema": "hybrid_detector/fiducial_target_ab/v1",
                    "camera": {
                        "width": args.width,
                        "height": args.height,
                        "focal_px": args.focal_px,
                        "model": "ideal pinhole, no distortion",
                    },
                    "render": {
                        "blur_sigma": args.blur_sigma,
                        "noise_sigma": args.noise_sigma,
                        "motion_sweep_mm": levels,
                        "motion_lever_mm": args.motion_lever_mm,
                        "motion_max_samples": args.motion_max_samples,
                        "obstruction_sweep": covers,
                        "allow_anchor_free": args.allow_anchor_free,
                    },
                    "poses": {
                        "count": args.poses,
                        "seed": args.seed,
                        "tilt_deg": args.tilt_deg,
                        "distance_m": list(args.distance_m),
                        "lateral_offset_m": args.lateral_offset_m,
                    },
                    "targets": results,
                },
                indent=1,
            )
        )
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
