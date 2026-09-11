"""Render what a camera should see of a Hybrid Carrier at a given pose.

Two jobs.  It is the ground truth the detector is tested against -- a renderer
built from the descriptor alone, so a pose that comes back wrong means the
descriptor and the detector disagree about the same object.  And it answers the
roadmap's H2 question offline: replay a historical trajectory through the real
camera rig and count, per frame, which facets were big enough, square-on enough
and unoccluded enough to measure, without printing anything.

The paint model is deliberately literal: bare body, then each facet's fill,
then its black stroke laid on the CAD boundary the way ``print.border`` says.
If the detector can only find edges in a render that assumes the stroke is
centred, that is a fact about the assumption, not a hidden success.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from hybrid_detector.calibration import CameraCalibration
from hybrid_detector.detector import (
    HybridCarrierModel,
    _inverse_depth_plane,
    project_rig,
    transform,
)

#: Face id the obstruction slabs are drawn under.  Nothing in a descriptor
#: reaches it, so a slab can never be mistaken for a facet or an anchor pad.
OBSTRUCTION_FACE_ID = 1_000_000

__all__ = [
    "OBSTRUCTION_FACE_ID",
    "PaintPalette",
    "render_carrier",
    "render_carrier_exposure",
]


@dataclass(frozen=True)
class PaintPalette:
    """BGR paint, as a camera would see it under even light."""

    body: tuple[int, int, int] = (196, 200, 205)
    background: tuple[int, int, int] = (70, 72, 78)
    border: tuple[int, int, int] = (18, 18, 18)
    red: tuple[int, int, int] = (40, 40, 200)
    blue: tuple[int, int, int] = (200, 70, 40)
    green: tuple[int, int, int] = (60, 170, 60)
    yellow: tuple[int, int, int] = (45, 205, 215)
    magenta: tuple[int, int, int] = (190, 55, 190)
    cyan: tuple[int, int, int] = (200, 190, 45)
    black: tuple[int, int, int] = (22, 22, 24)
    grey: tuple[int, int, int] = (128, 128, 128)
    white: tuple[int, int, int] = (240, 240, 240)
    obstruction: tuple[int, int, int] = (110, 112, 118)

    def of(self, colour: str) -> tuple[int, int, int]:
        return getattr(self, colour, self.body)


def _stroke_px(
    camera: CameraCalibration, T_cam_rig: np.ndarray, points_rig: np.ndarray, width_m: float
) -> int:
    """The printed stroke width in pixels, on this face, at this viewing angle.

    Focal length over depth is the scale of a surface facing the camera.  These
    facets rarely do -- several sit at 45 deg or worse -- and using the on-axis
    scale would draw a stroke two or three times too wide on exactly the faces
    the detector leans on.  The areal scale carries the foreshortening.
    """
    polygon = np.asarray(points_rig, dtype=np.float64).reshape(-1, 3)
    uv = project_rig(camera, T_cam_rig, polygon)
    if not np.isfinite(uv).all():
        return 1
    projected_area = abs(float(cv2.contourArea(uv.astype(np.float32))))
    normal = np.zeros(3)
    for i in range(len(polygon)):
        normal += np.cross(polygon[i], polygon[(i + 1) % len(polygon)])
    real_area = float(np.linalg.norm(normal)) / 2.0
    if real_area <= 1e-12 or projected_area <= 0.0:
        return 1
    return int(max(1, round(width_m * np.sqrt(projected_area / real_area))))


def render_carrier(
    model: HybridCarrierModel,
    camera: CameraCalibration,
    T_cam_rig: np.ndarray,
    *,
    palette: PaintPalette | None = None,
    marker_pixels: int = 512,
    paste_quadrants: dict[int, int] | None = None,
    obstructions: np.ndarray | None = None,
    noise_sigma: float = 0.0,
    blur_sigma: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """Paint the carrier into a fresh image through a depth buffer.

    A painter's-algorithm render would be shorter, but not usable as an oracle:
    at a shared edge the two faces are coincident, so whichever fill is drawn
    last erases half of the black band there and shifts its centre by a quarter
    of a stroke width.  A detector measured against that would look biased when
    the bias was in the picture.  So faces are resolved per pixel by depth, and a
    stroke is laid only where its own face is the visible surface.
    """
    palette = palette or PaintPalette()
    height, width = int(camera.height), int(camera.width)
    triangles = model.occluder_triangles
    face_ids = model.occluder_face_ids
    if obstructions is not None and len(obstructions) > 0:
        # Straight into the same depth buffer, so a slab hides a facet's fill,
        # its stroke and an anchor's sticker by the one rule that already
        # decides which of the body's own faces a pixel belongs to.
        extra = np.asarray(obstructions, dtype=np.float64).reshape(-1, 3, 3)
        triangles = (
            np.concatenate([triangles.reshape(-1, 3, 3), extra]) if len(triangles) else extra
        )
        face_ids = np.concatenate(
            [np.asarray(face_ids, dtype=np.int64), np.full(len(extra), OBSTRUCTION_FACE_ID)]
        )

    inverse_depth = np.full((height, width), -np.inf, dtype=np.float64)
    face_index = np.full((height, width), -1, dtype=np.int64)
    grid_v, grid_u = np.mgrid[0:height, 0:width].astype(np.float64)

    for index, tri in enumerate(triangles):
        cam = transform(T_cam_rig, tri)
        if np.any(cam[:, 2] <= 1e-6):
            continue
        if float(np.dot(np.cross(cam[1] - cam[0], cam[2] - cam[0]), cam.mean(axis=0))) >= 0.0:
            continue  # loops are wound CCW about the outward normal
        uv = project_rig(camera, T_cam_rig, tri)
        if not np.isfinite(uv).all():
            continue
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [np.round(uv).astype(np.int32)], 255)
        if not np.any(mask):
            continue
        plane = _inverse_depth_plane(uv, cam[:, 2])
        if plane is None:
            continue
        candidate = plane[0] * grid_u + plane[1] * grid_v + plane[2]
        nearer = (mask > 0) & (candidate > inverse_depth)
        inverse_depth[nearer] = candidate[nearer]
        face_index[nearer] = int(face_ids[index]) if index < len(face_ids) else -1

    colour_of: dict[int, tuple[int, int, int]] = {}
    for facet in model.facets:
        colour_of[int(facet.face_entity_id)] = palette.of(facet.colour)
    for anchor in model.anchors:
        colour_of[int(anchor.face_entity_id)] = palette.white
    colour_of[OBSTRUCTION_FACE_ID] = palette.obstruction

    image = np.full((height, width, 3), palette.background, dtype=np.uint8)
    body = face_index >= 0
    image[body] = palette.body
    for entity_id, colour in colour_of.items():
        image[face_index == entity_id] = colour

    # Strokes, clipped to where their own face is what the camera sees. At a
    # shared edge both faces are equidistant, so both strokes land and merge.
    for facet in model.facets:
        cam = transform(T_cam_rig, facet.polygon_m)
        if np.any(cam[:, 2] <= 1e-6):
            continue
        if float(np.dot(np.cross(cam[1] - cam[0], cam[2] - cam[0]), cam.mean(axis=0))) >= 0.0:
            continue
        uv = project_rig(camera, T_cam_rig, facet.polygon_m)
        if not np.isfinite(uv).all():
            continue
        plane = _inverse_depth_plane(uv, cam[:, 2])
        if plane is None:
            continue
        thickness = _stroke_px(camera, T_cam_rig, facet.polygon_m, float(model.border_width_m))
        outline = uv
        if model.border_alignment != "centred":
            sign = -1.0 if model.border_alignment == "inside" else 1.0
            centre = uv.mean(axis=0)
            radial = uv - centre
            scale = 1.0 + sign * 0.5 * thickness / np.maximum(
                np.linalg.norm(radial, axis=1, keepdims=True), 1.0
            )
            outline = centre + radial * scale
        stroke = np.zeros((height, width), dtype=np.uint8)
        cv2.polylines(
            stroke, [np.round(outline).astype(np.int32)], True, 255, thickness, lineType=cv2.LINE_AA
        )
        own = plane[0] * grid_u + plane[1] * grid_v + plane[2]
        visible = (stroke > 0) & (own >= inverse_depth * (1.0 - 2e-3))
        alpha = (stroke[visible].astype(np.float64) / 255.0)[:, None]
        image[visible] = np.round(
            image[visible] * (1.0 - alpha) + np.asarray(palette.border, dtype=np.float64) * alpha
        ).astype(np.uint8)

    # A descriptor that pins where its sticker was pasted is stating a physical
    # fact about the printed part, and the detector reads it as one: it does not
    # search the quarter turns for a pinned anchor.  So an oracle render that
    # ignored the pin would paste a sticker the detector is not allowed to find,
    # and every pose measured against it would be wrong by a quarter turn of the
    # pad.  A caller may still override, which is what the paste-rotation test does.
    quadrants = (
        dict(paste_quadrants)
        if paste_quadrants is not None
        else {
            int(anchor.marker_id): int(anchor.paste_quadrant)
            for anchor in model.anchors
            if anchor.paste_quadrant is not None
        }
    )
    for anchor in model.anchors:
        cam = transform(T_cam_rig, anchor.marker_corners_m)
        if np.any(cam[:, 2] <= 1e-6):
            continue
        if float(np.dot(np.cross(cam[1] - cam[0], cam[2] - cam[0]), cam.mean(axis=0))) <= 0.0:
            continue  # marker corners run clockwise, so the sign flips
        uv = project_rig(camera, T_cam_rig, anchor.marker_corners_m)
        if not np.isfinite(uv).all():
            continue
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, anchor.dictionary))
        sticker = cv2.cvtColor(
            cv2.aruco.generateImageMarker(dictionary, int(anchor.marker_id), int(marker_pixels)),
            cv2.COLOR_GRAY2BGR,
        )
        side = float(marker_pixels - 1)
        source = np.array([[0, 0], [side, 0], [side, side], [0, side]], dtype=np.float32)
        # A sticker pasted a quarter turn round still shows the same square; the
        # rotation moves which physical corner carries the marker's own corner 0.
        target = np.roll(
            uv.astype(np.float32), -int(quadrants.get(int(anchor.marker_id), 0)) % 4, axis=0
        )
        warped = cv2.warpPerspective(
            sticker,
            cv2.getPerspectiveTransform(source, target),
            (width, height),
            flags=cv2.INTER_LINEAR,
        )
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [np.round(uv).astype(np.int32)], 255)
        paste = (mask > 0) & (face_index == int(anchor.face_entity_id))
        image[paste] = warped[paste]

    if blur_sigma > 0.0:
        image = cv2.GaussianBlur(image, (0, 0), float(blur_sigma))
    if noise_sigma > 0.0:
        rng = np.random.default_rng(int(seed))
        image = np.clip(
            image.astype(np.float64) + rng.normal(0.0, float(noise_sigma), image.shape), 0, 255
        ).astype(np.uint8)
    return image


def render_carrier_exposure(
    model: HybridCarrierModel,
    camera: CameraCalibration,
    poses: Sequence[np.ndarray],
    *,
    palette: PaintPalette | None = None,
    marker_pixels: int = 512,
    paste_quadrants: dict[int, int] | None = None,
    obstructions: np.ndarray | None = None,
    noise_sigma: float = 0.0,
    blur_sigma: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """One frame, integrated over an exposure the target moved through.

    The cheap way to fake motion blur is to convolve a still with a line kernel,
    and it is wrong twice over for this job.  The kernel drags the background
    across the silhouette, so the outline -- which is exactly what the painted
    edges are measured on -- ends up carrying light that never fell there, and
    the measurement would be scored against a smear the optics cannot produce.
    And a line kernel has no way to express rotation, which a handheld target
    has as much of as it has translation.  Averaging renders taken along the
    exposure has neither problem, at the price of one render per sample.

    Blur and noise are applied once, to the integrated frame: a Gaussian PSF
    commutes with the average, and the sensor is read out once however far the
    target travelled.  Given a single pose this is exactly ``render_carrier``.
    """
    if len(poses) == 0:
        raise ValueError("render_carrier_exposure wants at least one pose")
    accumulator: np.ndarray | None = None
    for pose in poses:
        frame = render_carrier(
            model,
            camera,
            pose,
            palette=palette,
            marker_pixels=marker_pixels,
            paste_quadrants=paste_quadrants,
            obstructions=obstructions,
        ).astype(np.float64)
        accumulator = frame if accumulator is None else accumulator + frame
    image = np.round(accumulator / float(len(poses))).astype(np.uint8)
    if blur_sigma > 0.0:
        image = cv2.GaussianBlur(image, (0, 0), float(blur_sigma))
    if noise_sigma > 0.0:
        rng = np.random.default_rng(int(seed))
        image = np.clip(
            image.astype(np.float64) + rng.normal(0.0, float(noise_sigma), image.shape), 0, 255
        ).astype(np.uint8)
    return image
