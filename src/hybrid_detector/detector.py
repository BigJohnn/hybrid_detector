"""Hybrid Carrier V1: the CAD descriptor and the detector that reads it.

The V1 roadmap splits the target into two jobs that a single fiducial does
badly at once.  A couple of large ArUco anchors answer *which* rig this is and
roughly where it points; the painted polygon facets, whose 3D vertices and
edges CAD knows exactly, carry the millimetre-level pose.  Colour is only a
label -- it says which facet you are looking at, never where its boundary is.

Two properties of a *painted* facet drive everything in this module.

Edges are dark bands, not steps
    Every facet is printed with a black outline.  Where two painted facets
    meet, the two outlines merge into one dark band whose *centre* is the CAD
    edge, whichever way each stroke is aligned -- so those edges are
    self-centring and unbiased.  Where a painted facet meets bare body, only
    one stroke exists, and the dark band's centre sits half a stroke width off
    the CAD edge.  Snapping to the nearest Canny pixel finds an edge of the
    band, not its centre, and biases the pose by up to a full stroke width; a
    valley-centre search along the edge normal does not.

Position along an edge is not observable
    Only the component of an edge measurement perpendicular to the edge
    carries information.  Residuals are therefore point-to-line, one scalar
    per sample, never a two-vector to some nearest edge pixel.

Colour is read against references the image itself supplies: the white quiet
zone printed around each ArUco anchor and the anchor's own black modules pin
the white and black points of the current exposure, so facet classification
does not depend on a hand-tuned HSV box surviving the next lighting change.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.optimize import least_squares

from hybrid_detector.calibration import CameraCalibration
from hybrid_detector.marker_rig import (
    CornerObservation,
    _matrix_to_rodrigues,
    _rodrigues_to_matrix,
    project_camera_points,
)

__all__ = [
    "Anchor",
    "AnchorDetection",
    "CarrierDetection",
    "CarrierEdge",
    "ColourReference",
    "EdgeMeasurement",
    "Facet",
    "FacetView",
    "HybridCarrierModel",
    "corner_observations",
    "detect_carrier",
    "solve_carrier_pose",
]

UNIT_SCALE_TO_M = {"m": 1.0, "mm": 1e-3, "cm": 1e-2}

# Sticker rotation is a physical property of assembly; until it is measured and
# frozen into the model, the detector enumerates the four possibilities.
PASTE_QUADRANTS = (0, 1, 2, 3)


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Anchor:
    """One ArUco pad: a square face left unpainted so a sticker can go on it."""

    name: str
    marker_id: int
    dictionary: str
    face_entity_id: int
    pad_corners_m: np.ndarray  # (4, 3), CCW about the outward normal
    marker_corners_m: np.ndarray  # (4, 3), CW about the outward normal (ArUco order)
    normal: np.ndarray
    centre_m: np.ndarray
    pad_size_m: float
    marker_size_m: float
    quiet_zone_m: float
    paste_quadrant: int | None = None

    def object_points(self, quadrant: int = 0) -> np.ndarray:
        """Marker corners rolled by an unknown 90 deg paste rotation."""
        return np.roll(self.marker_corners_m, -int(quadrant) % 4, axis=0)


@dataclass(frozen=True)
class Facet:
    """One painted polygon face."""

    name: str
    colour: str
    face_entity_id: int
    polygon_m: np.ndarray  # (N, 3), CCW about the outward normal
    normal: np.ndarray
    centroid_m: np.ndarray
    area_m2: float
    use_for_pose: bool = True
    note: str = ""

    @property
    def num_vertices(self) -> int:
        return int(len(self.polygon_m))


@dataclass(frozen=True)
class CarrierEdge:
    """A CAD edge between two known faces, and the dark band printing puts on it.

    ``dark_lo_m``/``dark_hi_m`` bound the ink, signed along the edge normal that
    points from face A into face B, with ``None`` meaning "dark all the way out
    that side" (which is what a black *fill* does, as opposed to a stroke).  The
    detector reads the pair rather than a single bias because which landmark is
    measurable depends on the view: a band with two finite sides is measured at
    its centre, a one-sided one at its only transition.
    """

    face_a: str
    face_b: str
    colour_a: str
    colour_b: str
    p0_m: np.ndarray
    p1_m: np.ndarray
    ref_b_m: np.ndarray  # a point on the B side, to orient the image normal
    normal_b: np.ndarray  # face B's outward normal, to tell a silhouette from a fold
    length_m: float
    dihedral_deg: float
    has_ink: bool
    dark_lo_m: float | None
    dark_hi_m: float | None
    convex: bool
    use_for_pose: bool = True

    @property
    def direction(self) -> np.ndarray:
        d = self.p1_m - self.p0_m
        return d / max(float(np.linalg.norm(d)), 1e-12)

    @property
    def measurable(self) -> bool:
        """False when nothing is printed here, or when ink hides both landmarks."""
        return (
            self.use_for_pose
            and self.has_ink
            and not (self.dark_lo_m is None and self.dark_hi_m is None)
        )

    @property
    def bias_m(self) -> float:
        """Where the measurable landmark sits relative to the CAD edge."""
        if self.dark_lo_m is not None and self.dark_hi_m is not None:
            return 0.5 * (self.dark_lo_m + self.dark_hi_m)
        if self.dark_hi_m is None:
            return float(self.dark_lo_m or 0.0)
        return float(self.dark_hi_m or 0.0)

    @property
    def unbiased(self) -> bool:
        return self.measurable and abs(self.bias_m) < 1e-5

    @property
    def profile(self) -> str:
        if not self.measurable:
            return "unmeasurable"
        if self.dark_lo_m is not None and self.dark_hi_m is not None:
            return "band_centre"
        return "step_dark_positive" if self.dark_hi_m is None else "step_dark_negative"


@dataclass
class HybridCarrierModel:
    carrier_id: str
    anchors: list[Anchor]
    facets: list[Facet]
    edges: list[CarrierEdge]
    occluder_triangles: np.ndarray  # (T, 3, 3) in metres
    occluder_face_ids: np.ndarray  # (T,) the STEP face each triangle came from
    border_width_m: float
    border_alignment: str
    metadata: dict[str, Any] = field(default_factory=dict)

    # -- lookup ---------------------------------------------------------
    @property
    def anchors_by_id(self) -> dict[int, Anchor]:
        return {int(a.marker_id): a for a in self.anchors}

    @property
    def facets_by_name(self) -> dict[str, Facet]:
        return {f.name: f for f in self.facets}

    def edges_of(self, facet_name: str) -> list[CarrierEdge]:
        return [e for e in self.edges if facet_name in (e.face_a, e.face_b)]

    @property
    def side_resolving_facets(self) -> tuple[str, ...]:
        """Facets whose colour says which side of ``y = 0`` the rig is showing.

        The carrier is mirror-symmetric in shape, so a facet is only informative
        about the side if its mirror image is *not* painted the same colour.  A
        facet sitting on the mirror plane is its own twin and so never
        qualifies.  This is what an anchor-free pose stands on, and it is a
        property of how the thing was painted, not of the detector.
        """
        usable = [facet for facet in self.facets if facet.use_for_pose]
        out: list[str] = []
        for facet in usable:
            mirrored = facet.centroid_m.copy()
            mirrored[1] *= -1.0
            twins = [g for g in usable if float(np.linalg.norm(g.centroid_m - mirrored)) < 5e-4]
            if any(g.colour == facet.colour for g in twins):
                continue
            out.append(facet.name)
        return tuple(out)

    # -- io -------------------------------------------------------------
    @classmethod
    def from_json(cls, path: str | Path) -> HybridCarrierModel:
        path = Path(path)
        doc = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(doc, source=path)

    @classmethod
    def from_dict(cls, doc: dict[str, Any], *, source: Path | None = None) -> HybridCarrierModel:
        scale = UNIT_SCALE_TO_M[str(doc.get("units", "mm")).lower()]
        anchors = []
        for entry in doc["anchors"]:
            pad = np.asarray(entry["pad_corners"], dtype=np.float64) * scale
            marker = np.asarray(entry["marker_corners"], dtype=np.float64) * scale
            anchors.append(
                Anchor(
                    name=str(entry["name"]),
                    marker_id=int(entry["marker_id"]),
                    dictionary=str(entry.get("dictionary", "DICT_6X6_50")),
                    face_entity_id=int(entry.get("face_entity_id", -1)),
                    pad_corners_m=pad,
                    marker_corners_m=marker,
                    normal=np.asarray(entry["normal"], dtype=np.float64),
                    centre_m=np.asarray(entry["centre"], dtype=np.float64) * scale,
                    pad_size_m=float(entry["pad_size"]) * scale,
                    marker_size_m=float(entry["marker_size"]) * scale,
                    quiet_zone_m=float(entry["quiet_zone"]) * scale,
                    paste_quadrant=(
                        None
                        if entry.get("paste_quadrant") is None
                        else int(entry["paste_quadrant"])
                    ),
                )
            )
        facets = []
        for entry in doc["facets"]:
            polygon = np.asarray(entry["polygon"], dtype=np.float64) * scale
            facets.append(
                Facet(
                    name=str(entry["name"]),
                    colour=str(entry["colour"]),
                    face_entity_id=int(entry.get("face_entity_id", -1)),
                    polygon_m=polygon,
                    normal=np.asarray(entry["normal"], dtype=np.float64),
                    centroid_m=np.asarray(entry["centroid"], dtype=np.float64) * scale,
                    area_m2=float(entry["area"]) * scale * scale,
                    use_for_pose=bool(entry.get("use_for_pose", True)),
                    note=str(entry.get("note", "")),
                )
            )
        edges = []
        for entry in doc.get("edges", []):
            p0 = np.asarray(entry["p0"], dtype=np.float64) * scale
            p1 = np.asarray(entry["p1"], dtype=np.float64) * scale
            dark_lo = entry.get("dark_lo")
            dark_hi = entry.get("dark_hi")
            edges.append(
                CarrierEdge(
                    face_a=str(entry["face_a"]),
                    face_b=str(entry["face_b"]),
                    colour_a=str(entry["colour_a"]),
                    colour_b=str(entry["colour_b"]),
                    p0_m=p0,
                    p1_m=p1,
                    ref_b_m=np.asarray(entry["ref_b"], dtype=np.float64) * scale,
                    normal_b=np.asarray(entry["normal_b"], dtype=np.float64),
                    length_m=float(entry["length"]) * scale,
                    dihedral_deg=float(entry.get("dihedral_deg", 180.0)),
                    has_ink=bool(entry.get("has_ink", True)),
                    dark_lo_m=None if dark_lo is None else float(dark_lo) * scale,
                    dark_hi_m=None if dark_hi is None else float(dark_hi) * scale,
                    convex=bool(entry.get("convex", True)),
                    use_for_pose=bool(entry.get("use_for_pose", True)),
                )
            )
        occluders = doc.get("occluders", {})
        triangles = np.asarray(occluders.get("triangles", []), dtype=np.float64)
        if triangles.size:
            triangles = triangles.reshape(-1, 3, 3) * scale
        else:
            triangles = np.zeros((0, 3, 3), dtype=np.float64)
        triangle_faces = np.asarray(
            occluders.get("face_entity_ids", [-1] * len(triangles)), dtype=np.int64
        )
        border = doc.get("print", {}).get("border", {})
        metadata = {
            k: v for k, v in doc.items() if k not in {"anchors", "facets", "edges", "occluders"}
        }
        if source is not None:
            metadata["source"] = str(Path(source).resolve())
        return cls(
            carrier_id=str(doc.get("carrier_id", "hybrid_carrier")),
            anchors=anchors,
            facets=facets,
            edges=edges,
            occluder_triangles=triangles,
            occluder_face_ids=triangle_faces,
            border_width_m=float(border.get("width_mm", 1.0)) * 1e-3,
            border_alignment=str(border.get("alignment", "inside")),
            metadata=metadata,
        )


# ---------------------------------------------------------------------------
# projection helpers
# ---------------------------------------------------------------------------


def _as_T(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = _rodrigues_to_matrix(rvec)
    T[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return T


def _to_params(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    return np.concatenate([_matrix_to_rodrigues(T[:3, :3]), T[:3, 3]])


def _from_params(params: np.ndarray) -> np.ndarray:
    params = np.asarray(params, dtype=np.float64).reshape(6)
    return _as_T(params[:3], params[3:])


def transform(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return (
        points @ np.asarray(T, dtype=np.float64)[:3, :3].T + np.asarray(T, dtype=np.float64)[:3, 3]
    )


def project_rig(
    camera: CameraCalibration, T_cam_rig: np.ndarray, points_rig: np.ndarray
) -> np.ndarray:
    return project_camera_points(camera, transform(T_cam_rig, points_rig))


# ---------------------------------------------------------------------------
# visibility
# ---------------------------------------------------------------------------


@dataclass
class FacetView:
    facet: Facet
    polygon_uv: np.ndarray
    projected_area_px2: float
    incidence_deg: float
    visible_fraction: float
    reason: str = ""

    @property
    def usable(self) -> bool:
        return not self.reason


def _polygon_area_px(polygon_uv: np.ndarray) -> float:
    if len(polygon_uv) < 3:
        return 0.0
    return abs(float(cv2.contourArea(np.asarray(polygon_uv, dtype=np.float32))))


def _inverse_depth_plane(uv: np.ndarray, depths: np.ndarray) -> np.ndarray | None:
    """Least-squares ``1/z = a*u + b*v + c`` for one planar patch.

    For a plane, inverse depth really is affine in the *undistorted* image
    coordinates; fitting it to the projected vertices absorbs mild distortion
    and costs three unknowns instead of a per-pixel ray intersection.
    """
    if len(uv) < 3:
        return None
    design = np.column_stack([uv[:, 0], uv[:, 1], np.ones(len(uv))])
    try:
        coefficients, *_ = np.linalg.lstsq(design, 1.0 / depths, rcond=None)
    except np.linalg.LinAlgError:
        return None
    return coefficients


@dataclass(frozen=True)
class ProjectedOccluders:
    """The CAD skin projected once, for every facet of one pose to share."""

    uv: np.ndarray  # (N, 3, 2)
    planes: np.ndarray  # (N, 3), inverse-depth plane per triangle
    face_ids: np.ndarray  # (N,)
    bounds: np.ndarray  # (N, 4): umin, vmin, umax, vmax


def _project_occluders(
    model: HybridCarrierModel, camera: CameraCalibration, T_cam_rig: np.ndarray
) -> ProjectedOccluders | None:
    """Project the whole occluder skin in one pass.

    Every facet of a given pose sees the same skin from the same place, so
    projecting it per facet did the same work thirteen times over -- and it was
    the detector's largest single cost once the per-pixel fields were cached.
    """
    triangles = np.asarray(model.occluder_triangles, dtype=np.float64)
    if triangles.size == 0:
        return None
    flat = transform(T_cam_rig, triangles.reshape(-1, 3))
    depth = flat[:, 2].reshape(-1, 3)
    uv = project_camera_points(camera, flat).reshape(-1, 3, 2)
    keep = (depth > 1e-6).all(axis=1) & np.isfinite(uv).all(axis=(1, 2))
    if not keep.any():
        return None
    uv, depth = uv[keep], depth[keep]
    face_ids = np.asarray(model.occluder_face_ids, dtype=np.int64)
    face_ids = (
        face_ids[keep] if len(face_ids) == len(keep) else np.full(len(uv), -1, dtype=np.int64)
    )
    # 1/z is affine in image coordinates for a plane, and a triangle has exactly
    # three vertices, so the least-squares fit is an exact 3x3 solve.
    design = np.concatenate([uv, np.ones((len(uv), 3, 1))], axis=2)
    determinant = np.linalg.det(design)
    solvable = np.abs(determinant) > 1e-9
    if not solvable.any():
        return None
    uv, depth, face_ids, design = (
        uv[solvable],
        depth[solvable],
        face_ids[solvable],
        design[solvable],
    )
    # (N, 3) on the right is a stack of matrices to numpy 2, not a stack of vectors.
    planes = np.linalg.solve(design, (1.0 / depth)[..., None])[..., 0]
    return ProjectedOccluders(
        uv=uv,
        planes=planes,
        face_ids=face_ids,
        bounds=np.concatenate([uv.min(axis=1), uv.max(axis=1)], axis=1),
    )


def _occlusion_fraction(
    model: HybridCarrierModel,
    camera: CameraCalibration,
    T_cam_rig: np.ndarray,
    facet: Facet,
    polygon_uv: np.ndarray,
    *,
    raster_px: int = 96,
    depth_tolerance: float = 2e-3,
    occluders: ProjectedOccluders | None = None,
) -> float:
    """Fraction of the facet's projected area that nothing nearer covers.

    Back-face culling alone is not enough on this rig: the upper plates sit in
    front of the lower ones over a wide arc, and a facet that is half hidden
    still passes the normal test while its colour statistics and half its edges
    are meaningless.  A small inverse-depth buffer over the CAD skin answers it
    directly -- comparing per pixel, not per polygon, because two faces that
    meet at an edge have interleaved depths and a mean-depth test would have
    each of them shadow the other.
    """
    if occluders is None:
        occluders = _project_occluders(model, camera, T_cam_rig)
    if occluders is None or len(polygon_uv) < 3:
        return 1.0
    umin, vmin = polygon_uv.min(axis=0)
    umax, vmax = polygon_uv.max(axis=0)
    if not np.isfinite([umin, vmin, umax, vmax]).all() or umax <= umin or vmax <= vmin:
        return 0.0
    scale = float(raster_px) / max(umax - umin, vmax - vmin)
    origin = np.array([umin, vmin], dtype=np.float64)
    shape = (
        int(np.ceil((vmax - vmin) * scale)) + 2,
        int(np.ceil((umax - umin) * scale)) + 2,
    )

    def to_raster(uv: np.ndarray) -> np.ndarray:
        return (uv - origin) * scale

    facet_mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(facet_mask, [np.round(to_raster(polygon_uv)).astype(np.int32)], 255)
    total = int(np.count_nonzero(facet_mask))
    if total == 0:
        return 0.0

    plane = _inverse_depth_plane(polygon_uv, transform(T_cam_rig, facet.polygon_m)[:, 2])
    if plane is None:
        return 1.0
    grid_v, grid_u = np.mgrid[0 : shape[0], 0 : shape[1]].astype(np.float64)
    image_u = grid_u / scale + origin[0]
    image_v = grid_v / scale + origin[1]
    facet_inverse_depth = plane[0] * image_u + plane[1] * image_v + plane[2]

    hidden = np.zeros(shape, dtype=bool)
    # A triangle whose image bounding box misses the facet's cannot hide any of
    # it, and on this rig most of the skin is somewhere else in the frame.
    overlaps = (
        (occluders.bounds[:, 2] >= umin)
        & (occluders.bounds[:, 0] <= umax)
        & (occluders.bounds[:, 3] >= vmin)
        & (occluders.bounds[:, 1] <= vmax)
        & (occluders.face_ids != int(facet.face_entity_id))
    )
    for index in np.flatnonzero(overlaps):
        raster = to_raster(occluders.uv[index])
        lo = np.maximum(np.floor(raster.min(axis=0)).astype(np.int64), 0)
        hi = np.minimum(np.ceil(raster.max(axis=0)).astype(np.int64) + 1, [shape[1], shape[0]])
        if hi[0] <= lo[0] or hi[1] <= lo[1]:
            continue
        window = (slice(int(lo[1]), int(hi[1])), slice(int(lo[0]), int(hi[0])))
        tri_mask = np.zeros(shape, dtype=np.uint8)
        cv2.fillPoly(tri_mask, [np.round(raster).astype(np.int32)], 255)
        patch = tri_mask[window] > 0
        if not patch.any():
            continue
        tri_plane = occluders.planes[index]
        tri_inverse_depth = (
            tri_plane[0] * image_u[window] + tri_plane[1] * image_v[window] + tri_plane[2]
        )
        nearer = tri_inverse_depth > facet_inverse_depth[window] * (1.0 + float(depth_tolerance))
        hidden[window] |= patch & nearer
    covered = int(np.count_nonzero((facet_mask > 0) & hidden))
    return float(total - covered) / float(total)


def facet_views(
    model: HybridCarrierModel,
    camera: CameraCalibration,
    T_cam_rig: np.ndarray,
    *,
    min_area_px2: float = 150.0,
    max_incidence_deg: float = 78.0,
    min_visible_fraction: float = 0.6,
    check_occlusion: bool = True,
) -> list[FacetView]:
    """Which facets this camera can actually measure at this pose, and why not."""
    height, width = int(camera.height), int(camera.width)
    occluders = _project_occluders(model, camera, T_cam_rig) if check_occlusion else None
    out: list[FacetView] = []
    for facet in model.facets:
        cam_points = transform(T_cam_rig, facet.polygon_m)
        centre_cam = transform(T_cam_rig, facet.centroid_m.reshape(1, 3))[0]
        normal_cam = np.asarray(T_cam_rig, dtype=np.float64)[:3, :3] @ facet.normal
        view_dir = centre_cam / max(float(np.linalg.norm(centre_cam)), 1e-12)
        cos_incidence = float(-np.dot(normal_cam, view_dir))
        incidence = float(np.degrees(np.arccos(np.clip(cos_incidence, -1.0, 1.0))))
        polygon_uv = np.zeros((0, 2), dtype=np.float64)
        area = 0.0
        reason = ""
        if np.any(cam_points[:, 2] <= 1e-6):
            reason = "behind camera"
        elif cos_incidence <= 0.0:
            reason = "back facing"
        else:
            polygon_uv = project_camera_points(camera, cam_points)
            if not np.isfinite(polygon_uv).all():
                reason = "projection failed"
            else:
                area = _polygon_area_px(polygon_uv)
                inside = (
                    (polygon_uv[:, 0] > -width)
                    & (polygon_uv[:, 0] < 2 * width)
                    & (polygon_uv[:, 1] > -height)
                    & (polygon_uv[:, 1] < 2 * height)
                )
                if not np.any(inside):
                    reason = "outside image"
                elif area < float(min_area_px2):
                    reason = f"too small ({area:.0f} px2)"
                elif incidence > float(max_incidence_deg):
                    reason = f"grazing ({incidence:.0f} deg)"
        visible_fraction = 0.0
        if not reason:
            visible_fraction = (
                _occlusion_fraction(
                    model, camera, T_cam_rig, facet, polygon_uv, occluders=occluders
                )
                if check_occlusion
                else 1.0
            )
            if visible_fraction < float(min_visible_fraction):
                reason = f"occluded ({visible_fraction:.2f} visible)"
        out.append(
            FacetView(
                facet=facet,
                polygon_uv=polygon_uv,
                projected_area_px2=area,
                incidence_deg=incidence,
                visible_fraction=visible_fraction,
                reason=reason,
            )
        )
    return out


# ---------------------------------------------------------------------------
# colour, referenced to the image itself
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ColourReference:
    """The current exposure's white and black points, in BGR."""

    white: np.ndarray
    black: np.ndarray
    source: str

    def normalise(self, bgr: np.ndarray) -> np.ndarray:
        span = np.maximum(self.white - self.black, 1e-3)
        return (np.asarray(bgr, dtype=np.float64) - self.black) / span


DEFAULT_REFERENCE = ColourReference(
    white=np.array([235.0, 235.0, 235.0]),
    black=np.array([30.0, 30.0, 30.0]),
    source="fallback (no anchor sampled)",
)


def colour_reference_from_anchors(
    image: np.ndarray,
    model: HybridCarrierModel,
    camera: CameraCalibration,
    T_cam_rig: np.ndarray,
    anchor_ids: Sequence[int],
) -> ColourReference:
    """White from each anchor's printed quiet zone, black from its own modules.

    The quiet zone is a known-white ring of a known size sitting in the same
    plane, at the same distance, under the same light as the facets, so it
    tracks exposure and colour temperature for free.  A fixed HSV box does not.
    """
    anchors = model.anchors_by_id
    white_samples: list[np.ndarray] = []
    black_samples: list[np.ndarray] = []
    height, width = image.shape[:2]
    for marker_id in anchor_ids:
        anchor = anchors.get(int(marker_id))
        if anchor is None:
            continue
        pad_uv = project_rig(camera, T_cam_rig, anchor.pad_corners_m)
        marker_uv = project_rig(camera, T_cam_rig, anchor.marker_corners_m)
        if not (np.isfinite(pad_uv).all() and np.isfinite(marker_uv).all()):
            continue
        ring = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(ring, [np.round(pad_uv).astype(np.int32)], 255)
        cv2.fillPoly(ring, [np.round(marker_uv).astype(np.int32)], 0)
        ring = cv2.erode(ring, np.ones((3, 3), np.uint8))
        if np.count_nonzero(ring):
            pixels = image[ring > 0].reshape(-1, 3).astype(np.float64)
            white_samples.append(np.percentile(pixels, 70, axis=0))
        inner = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(inner, [np.round(marker_uv).astype(np.int32)], 255)
        inner = cv2.erode(inner, np.ones((5, 5), np.uint8))
        if np.count_nonzero(inner):
            pixels = image[inner > 0].reshape(-1, 3).astype(np.float64)
            black_samples.append(np.percentile(pixels, 15, axis=0))
    if not white_samples or not black_samples:
        return DEFAULT_REFERENCE
    return ColourReference(
        white=np.mean(white_samples, axis=0),
        black=np.mean(black_samples, axis=0),
        source=f"anchor quiet zone + modules ({len(white_samples)} anchors)",
    )


def clip_roi(roi: Sequence[float] | None, shape: Sequence[int]) -> tuple[int, int, int, int] | None:
    """``(x0, y0, x1, y1)`` clipped to the image, or None for the whole frame."""
    if roi is None:
        return None
    height, width = int(shape[0]), int(shape[1])
    x0 = int(max(0, np.floor(float(roi[0]))))
    y0 = int(max(0, np.floor(float(roi[1]))))
    x1 = int(min(width, np.ceil(float(roi[2]))))
    y1 = int(min(height, np.ceil(float(roi[3]))))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    if (x1 - x0) * (y1 - y0) >= width * height:
        return None
    return x0, y0, x1, y1


def rig_roi(
    camera: CameraCalibration,
    model: HybridCarrierModel,
    T_base_rig: np.ndarray,
    shape: Sequence[int],
    *,
    margin_px: float = 48.0,
) -> tuple[int, int, int, int] | None:
    """Where a rig at this pose can possibly appear, plus a margin.

    Every per-pixel field this module builds -- the ink map, the colour masks --
    costs the whole frame while only a couple of percent of it can contain the
    rig.  A pose from the previous frame says which couple of percent, and the
    margin has to cover what that pose does not know: how far the rig moved
    since, plus the edge search band, plus a pixel for the bilinear tap.  Too
    small a margin does not corrupt a measurement -- the fields read zero
    outside, so samples there are simply not found -- but it does silently cost
    edges, so the default is deliberately loose.
    """
    T_cam_rig = np.asarray(camera.T_cam_base, dtype=np.float64) @ np.asarray(
        T_base_rig, dtype=np.float64
    )
    points = [facet.polygon_m for facet in model.facets]
    points.extend(anchor.pad_corners_m for anchor in model.anchors)
    if not points:
        return None
    uv = project_rig(camera, T_cam_rig, np.vstack(points))
    uv = uv[np.isfinite(uv).all(axis=1)]
    if not len(uv):
        return None
    margin = float(margin_px)
    return clip_roi(
        (
            uv[:, 0].min() - margin,
            uv[:, 1].min() - margin,
            uv[:, 0].max() + margin,
            uv[:, 1].max() + margin,
        ),
        shape,
    )


def classify_colours(
    image: np.ndarray,
    reference: ColourReference,
    *,
    black_level: float = 0.22,
    chroma_margin: float = 0.14,
    roi: Sequence[float] | None = None,
) -> dict[str, np.ndarray]:
    """Per-pixel paint label under the current exposure.

    Works on the white-balanced, black-subtracted image, so the decision is
    about the paint and not about how bright the room is.

    ``roi`` restricts the work to a window; pixels outside it come back
    unlabelled in every mask, which is the safe direction -- a sample out there
    is dropped rather than matched against a colour nobody looked at.
    """
    window = clip_roi(roi, image.shape[:2])
    if window is not None:
        x0, y0, x1, y1 = window
        inner = classify_colours(
            image[y0:y1, x0:x1],
            reference,
            black_level=black_level,
            chroma_margin=chroma_margin,
        )
        out: dict[str, np.ndarray] = {}
        for name, mask in inner.items():
            full = np.zeros(image.shape[:2], dtype=np.uint8)
            full[y0:y1, x0:x1] = mask
            out[name] = full
        return out
    normalised = reference.normalise(image.astype(np.float64))
    blue, green, red = normalised[..., 0], normalised[..., 1], normalised[..., 2]
    luma = 0.114 * blue + 0.587 * green + 0.299 * red
    denominator = np.maximum(np.abs(blue) + np.abs(green) + np.abs(red), 1e-3)
    # A primary is one channel above the other two; a secondary is two channels
    # above the third.  Scoring them the same way keeps the two families on one
    # scale, and they do not collide: pure red scores ~0 as yellow (its green
    # and blue sit together) and pure yellow scores ~0 as red.
    red_score = (red - np.maximum(green, blue)) / denominator
    blue_score = (blue - np.maximum(green, red)) / denominator
    green_score = (green - np.maximum(red, blue)) / denominator
    yellow_score = (np.minimum(red, green) - blue) / denominator
    magenta_score = (np.minimum(red, blue) - green) / denominator
    cyan_score = (np.minimum(green, blue) - red) / denominator

    is_black = luma < float(black_level)
    is_red = (~is_black) & (red_score > float(chroma_margin))
    is_blue = (~is_black) & (blue_score > float(chroma_margin))
    is_green = (~is_black) & (green_score > float(chroma_margin))
    is_yellow = (~is_black) & (yellow_score > float(chroma_margin))
    is_magenta = (~is_black) & (magenta_score > float(chroma_margin))
    is_cyan = (~is_black) & (cyan_score > float(chroma_margin))
    is_light = ~(is_black | is_red | is_blue | is_green | is_yellow | is_magenta | is_cyan)
    return {
        "black": is_black.astype(np.uint8) * 255,
        "red": is_red.astype(np.uint8) * 255,
        "blue": is_blue.astype(np.uint8) * 255,
        "green": is_green.astype(np.uint8) * 255,
        "yellow": is_yellow.astype(np.uint8) * 255,
        "magenta": is_magenta.astype(np.uint8) * 255,
        "cyan": is_cyan.astype(np.uint8) * 255,
        "white": is_light.astype(np.uint8) * 255,
        "grey": is_light.astype(np.uint8) * 255,
        "body": is_light.astype(np.uint8) * 255,
    }


# A chromatic match is strong evidence: nothing else in a workcell is that red
# under a white-balanced exposure.  Black and white matches are weak -- shadow,
# a dark background, and any pale surface all produce them -- so a hypothesis
# must not be allowed to win on those alone.
COLOUR_EVIDENCE_WEIGHT = {
    "red": 1.0,
    "blue": 1.0,
    "green": 1.0,
    "yellow": 1.0,
    "magenta": 1.0,
    "cyan": 1.0,
    "black": 0.2,
    "white": 0.15,
    "grey": 0.15,
}


def facet_colour_coverage(
    masks: dict[str, np.ndarray],
    view: FacetView,
    *,
    border_px: float = 3.0,
) -> float | None:
    """How much of a facet's interior carries the colour the model predicts."""
    mask = masks.get(view.facet.colour)
    if mask is None or view.reason or len(view.polygon_uv) < 3:
        return None
    height, width = mask.shape[:2]
    filled = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(filled, [np.round(view.polygon_uv).astype(np.int32)], 255)
    # Stay clear of the printed black outline: it belongs to the edge stage,
    # and counting it here would penalise every non-black facet.
    radius = max(1, int(round(float(border_px))))
    filled = cv2.erode(filled, np.ones((2 * radius + 1, 2 * radius + 1), np.uint8))
    total = int(np.count_nonzero(filled))
    if total < 25:
        return None
    return float(np.count_nonzero((filled > 0) & (mask > 0))) / float(total)


# ---------------------------------------------------------------------------
# edges: where the millimetres come from
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EdgeMeasurement:
    """One subpixel crossing of a printed facet boundary."""

    edge_index: int
    point_rig: np.ndarray  # (3,) the CAD point this sample belongs to
    uv: np.ndarray  # (2,) measured position of the boundary
    normal_uv: np.ndarray  # (2,) unit, perpendicular to the edge in the image
    sigma_px: float
    contrast: float
    band_width_px: float
    residual_px: float  # how far the landmark sat from where the pose predicted it
    profile: str


def _bilinear(image: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Bilinear sample of a single-channel float image; NaN outside."""
    height, width = image.shape[:2]
    x = np.asarray(xy, dtype=np.float64)[..., 0]
    y = np.asarray(xy, dtype=np.float64)[..., 1]
    inside = (x >= 0) & (x <= width - 1.001) & (y >= 0) & (y <= height - 1.001)
    x0 = np.clip(np.floor(x), 0, width - 2).astype(np.int64)
    y0 = np.clip(np.floor(y), 0, height - 2).astype(np.int64)
    fx = x - x0
    fy = y - y0
    top = image[y0, x0] * (1 - fx) + image[y0, x0 + 1] * fx
    bottom = image[y0 + 1, x0] * (1 - fx) + image[y0 + 1, x0 + 1] * fx
    out = top * (1 - fy) + bottom * fy
    return np.where(inside, out, np.nan)


def ink_image(
    image: np.ndarray, reference: ColourReference, *, roi: Sequence[float] | None = None
) -> np.ndarray:
    """Where the printed black stroke is, as a scalar field in [0, 1].

    Grayscale is the wrong channel for this rig.  A saturated red or blue paint
    has roughly a third of the body's luminance, so a black outline drawn on it
    is barely a step in luma -- while the same outline is unmistakable in
    saturation.  Ink is the one thing on the part that is dark *and* neutral, so
    that conjunction is the detector, and it separates the stroke from red, blue
    and bare body by a wide margin instead of a marginal one.

    ``roi`` restricts the work to a window and leaves the rest of the field at
    zero: no ink anywhere the rig cannot be.  On a 1920x1080 frame the rig
    occupies a couple of percent of the pixels, and this map is the single most
    expensive thing the detector builds.
    """
    window = clip_roi(roi, image.shape[:2])
    if window is not None:
        x0, y0, x1, y1 = window
        full = np.zeros(image.shape[:2], dtype=np.float32)
        full[y0:y1, x0:x1] = ink_image(image[y0:y1, x0:x1], reference)
        return full
    if image.ndim == 2:
        stack = np.repeat(image[..., None].astype(np.float64), 3, axis=2)
    else:
        stack = image.astype(np.float64)
    normalised = reference.normalise(stack)
    luma = 0.114 * normalised[..., 0] + 0.587 * normalised[..., 1] + 0.299 * normalised[..., 2]
    peak = np.max(normalised, axis=2)
    chroma = (peak - np.min(normalised, axis=2)) / np.maximum(peak, 1e-3)
    darkness = np.clip(1.0 - luma, 0.0, 1.0)
    neutrality = np.clip(1.0 - chroma, 0.0, 1.0)
    return (darkness * neutrality).astype(np.float32)


def _crossings(taps: np.ndarray, profile: np.ndarray, level: float) -> list[tuple[float, bool]]:
    """Sub-tap positions where the profile crosses ``level``, with direction."""
    above = profile > level
    out: list[tuple[float, bool]] = []
    for i in range(len(profile) - 1):
        if above[i] == above[i + 1]:
            continue
        a, b = float(profile[i]), float(profile[i + 1])
        alpha = 0.0 if abs(b - a) < 1e-12 else (level - a) / (b - a)
        out.append((float(taps[i] + alpha * (taps[i + 1] - taps[i])), bool(above[i + 1])))
    return out


def _measure_profile(
    profile: np.ndarray,
    taps: np.ndarray,
    *,
    profile_kind: str,
    min_contrast: float,
) -> tuple[float, float, float] | None:
    """Locate one printed landmark in a 1-D ink profile.

    ``band_centre`` -- ink is bounded on both sides, so the band's midpoint is
    the landmark.  Two painted facets that meet put their two strokes here, and
    the midpoint lands on the CAD edge whichever way each stroke is aligned.

    ``step_dark_positive`` / ``step_dark_negative`` -- ink runs off one side of
    the window (a black *fill*, or unknown background behind a silhouette), so
    only the transition on the lit side is a landmark.  The peak is allowed to
    sit against the window edge here, which is exactly the case a
    peak-must-be-interior test would throw away.

    The dark level is the profile's *peak*, not a high percentile: a 3 px band
    inside a 30 px window occupies a tenth of the taps, so any percentile above
    about the 80th still sits in the light paint, which would drag the half level
    down and widen the measured band asymmetrically.
    """
    if np.isnan(profile).any():
        return None
    smoothed = np.convolve(profile, np.ones(3) / 3.0, mode="same")
    smoothed[0], smoothed[-1] = profile[0], profile[-1]
    low = float(np.percentile(profile, 20))
    high = float(smoothed.max())
    contrast = high - low
    if contrast < float(min_contrast):
        return None
    crossings = _crossings(taps, smoothed, 0.5 * (low + high))
    if not crossings:
        return None
    if profile_kind == "band_centre":
        pairs = [
            (rise, fall)
            for (rise, up), (fall, down) in zip(crossings, crossings[1:], strict=False)
            if up and not down
        ]
        if not pairs:
            return None
        rise, fall = min(pairs, key=lambda pair: abs(0.5 * (pair[0] + pair[1])))
        # Half-max crossings find the band; its ink-weighted centroid locates it
        # more precisely, because the centroid averages the two blurred flanks
        # instead of relying on where each one happens to cross a level.
        inside = (taps >= rise - 1.0) & (taps <= fall + 1.0)
        weight = np.clip(profile[inside] - low, 0.0, None)
        total = float(weight.sum())
        centre = (
            float(np.sum(weight * taps[inside]) / total) if total > 1e-6 else 0.5 * (rise + fall)
        )
        return centre, float(fall - rise), contrast
    wanted_rising = profile_kind == "step_dark_positive"
    candidates = [position for position, rising in crossings if rising == wanted_rising]
    if not candidates:
        return None
    position = min(candidates, key=abs)
    band = [p for p, rising in crossings if rising != wanted_rising]
    width = float(abs(min(band, key=abs) - position)) if band else float("nan")
    return position, width, contrast


# Faces that are not painted all land in the detector's one "light" class.
_LIGHT_CLASSES = {"body", "white", "grey"}


def _sides_agree(
    masks: dict[str, np.ndarray],
    uv: np.ndarray,
    normal: np.ndarray,
    offset_px: float,
    *,
    half_band: float,
    colour_a: str,
    colour_b: str,
    check_far_side: bool,
    gap_px: float = 2.0,
) -> bool:
    """Is the paint on each side of this landmark what the model says it is?

    The strongest thing the model knows about an edge is not where it is but
    what flanks it.  Checking that turns a landmark search into a verification:
    a sample that locked onto a shadow, a highlight or the next boundary along
    almost never has the right two paints beside it.
    """
    reach = float(half_band) + float(gap_px)
    for colour, side in ((colour_a, -1.0), (colour_b, +1.0)):
        if side > 0 and not check_far_side:
            continue
        key = "white" if colour in _LIGHT_CLASSES else colour
        mask = masks.get(key)
        if mask is None:
            continue
        point = uv + (offset_px + side * reach) * normal
        column, row = int(round(point[0])), int(round(point[1]))
        if not (0 <= row < mask.shape[0] and 0 <= column < mask.shape[1]):
            return False
        if not mask[row, column]:
            return False
    return True


def _front_facing(edge: CarrierEdge, T_cam_rig: np.ndarray) -> bool:
    """Is the face on the far side of this edge turned toward the camera?

    If it is not, whatever lies beyond the edge is background rather than the
    modelled paint, and only the near side of the ink band can be trusted.
    """
    normal_cam = np.asarray(T_cam_rig, dtype=np.float64)[:3, :3] @ edge.normal_b
    point_cam = transform(T_cam_rig, edge.ref_b_m.reshape(1, 3))[0]
    return float(np.dot(normal_cam, point_cam)) < 0.0


def _edge_profile_for_view(edge: CarrierEdge, face_b_front_facing: bool) -> tuple[str, float]:
    """Landmark kind and its offset from the CAD edge, for this viewpoint.

    A convex edge whose far face has turned away is a silhouette: what lies
    beyond it is background, not the modelled paint, so the far side of the
    band is not trustworthy and only the near transition is used.
    """
    if not edge.measurable:
        return "unmeasurable", 0.0
    if edge.dark_lo_m is not None and edge.dark_hi_m is not None:
        if edge.convex and not face_b_front_facing:
            return "step_dark_positive", float(edge.dark_lo_m)
        return "band_centre", edge.bias_m
    if edge.dark_hi_m is None:
        return "step_dark_positive", float(edge.dark_lo_m or 0.0)
    return "step_dark_negative", float(edge.dark_hi_m or 0.0)


def measure_edges(
    image: np.ndarray,
    model: HybridCarrierModel,
    camera: CameraCalibration,
    T_cam_rig: np.ndarray,
    *,
    reference: ColourReference,
    views: Sequence[FacetView] | None = None,
    sample_step_m: float = 3e-3,
    end_margin_m: float = 5e-3,
    search_px: float = 8.0,
    tap_px: float = 0.25,
    min_contrast: float = 0.12,
    include_biased: bool = True,
    masks: dict[str, np.ndarray] | None = None,
    stroke_width_uncertainty: float = 0.5,
    ink: np.ndarray | None = None,
) -> list[EdgeMeasurement]:
    """Find every printed facet boundary this pose predicts, to subpixel.

    Each sample searches only along the edge normal, because position along an
    edge is unobservable; the returned measurement is consumed as a
    point-to-line residual, one scalar, never as a two-vector to a snapped
    Canny pixel -- that snap would add tangential noise and, worse, lock onto
    whichever side of the black band happened to be nearest.

    ``ink`` accepts a map built earlier.  It depends only on the image and the
    exposure reference, never on the pose, so a caller running several search
    bands over one frame should build it once and hand it in; recomputing it per
    band was half the detector's runtime.
    """
    usable_names = None
    if views is not None:
        usable_names = {v.facet.name for v in views if v.usable and v.facet.use_for_pose}
    if ink is None:
        ink = ink_image(image, reference)

    taps = np.arange(-float(search_px), float(search_px) + 1e-9, float(tap_px))
    out: list[EdgeMeasurement] = []
    for edge_index, edge in enumerate(model.edges):
        if not edge.measurable:
            continue
        if (
            usable_names is not None
            and edge.face_a not in usable_names
            and edge.face_b not in usable_names
        ):
            continue
        profile, bias_m = _edge_profile_for_view(edge, _front_facing(edge, T_cam_rig))
        if profile == "unmeasurable":
            continue
        if abs(bias_m) > 1e-9 and not include_biased:
            continue
        length = float(edge.length_m)
        usable_length = length - 2.0 * float(end_margin_m)
        if usable_length <= 0.0:
            continue
        count = max(2, int(np.floor(usable_length / float(sample_step_m))) + 1)
        t = np.linspace(end_margin_m, length - end_margin_m, count)
        points_rig = edge.p0_m + np.outer(t, edge.direction)
        cam_points = transform(T_cam_rig, points_rig)
        if np.any(cam_points[:, 2] <= 1e-6):
            continue
        uv = project_camera_points(camera, cam_points)
        if not np.isfinite(uv).all():
            continue
        span = uv[-1] - uv[0]
        norm = float(np.linalg.norm(span))
        if norm < 4.0:  # shorter than a few pixels carries no geometry
            continue
        tangent = span / norm
        normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)
        # Orient the image normal face A -> face B, which is the sign convention
        # the model's dark interval is written in.
        ref_uv = project_rig(camera, T_cam_rig, edge.ref_b_m.reshape(1, 3))[0]
        if np.isfinite(ref_uv).all() and float(np.dot(normal, ref_uv - uv.mean(axis=0))) < 0.0:
            normal = -normal
        scale_px_per_m = norm / max(usable_length, 1e-9)
        bias_px = float(bias_m) * scale_px_per_m
        # A landmark that had to be un-biased is only as good as the stroke width
        # the model was told about.  Rather than trusting those edges fully or
        # throwing them away, carry the uncertainty of the correction into their
        # weight, so they inform the pose in proportion to what they are worth.
        bias_sigma_px = abs(bias_px) * float(stroke_width_uncertainty)
        expected_band_px = (
            float(edge.dark_hi_m - edge.dark_lo_m) * scale_px_per_m
            if (edge.dark_lo_m is not None and edge.dark_hi_m is not None)
            else float("nan")
        )
        far_side_known = profile == "band_centre"
        samples = uv[:, None, :] + taps[None, :, None] * normal[None, None, :]
        profiles = _bilinear(ink, samples)
        for k in range(count):
            measured = _measure_profile(
                profiles[k], taps, profile_kind=profile, min_contrast=min_contrast
            )
            if measured is None:
                continue
            offset_px, band_px, contrast = measured
            # The stroke has one width on the part, so its width in the image is
            # predictable; a landmark whose band is nothing like that width is a
            # lock onto something else -- a shadow, a highlight, the next edge.
            if np.isfinite(expected_band_px) and np.isfinite(band_px):
                floor = max(expected_band_px, 1.0)
                if not (0.35 * floor - 1.5 <= band_px <= 2.5 * floor + 3.0):
                    continue
            if masks is not None and not _sides_agree(
                masks,
                uv[k],
                normal,
                offset_px,
                half_band=0.5 * (band_px if np.isfinite(band_px) else 2.0),
                colour_a=edge.colour_a,
                colour_b=edge.colour_b,
                check_far_side=far_side_known,
            ):
                continue
            offset_px -= bias_px
            out.append(
                EdgeMeasurement(
                    edge_index=edge_index,
                    point_rig=points_rig[k],
                    uv=uv[k] + offset_px * normal,
                    normal_uv=normal,
                    sigma_px=float(
                        min(4.0, np.hypot(max(0.15, 0.35 / max(contrast, 1e-3)), bias_sigma_px))
                    ),
                    contrast=contrast,
                    band_width_px=band_px,
                    residual_px=float(offset_px),
                    profile=profile,
                )
            )
    return out


def edge_support(
    image: np.ndarray,
    model: HybridCarrierModel,
    camera: CameraCalibration,
    T_cam_rig: np.ndarray,
    *,
    reference: ColourReference,
    views: Sequence[FacetView] | None = None,
    masks: dict[str, np.ndarray] | None = None,
    search_px: float = 6.0,
    sample_step_m: float = 8e-3,
    ink: np.ndarray | None = None,
) -> float:
    """How well the CAD boundaries land on printed boundaries, in [0, 1].

    A soft inlier count: each accepted sample scores 1 when the landmark was
    found exactly where the pose predicted it and falls off to 0 at the edge of
    the search band.  Cheap enough to run once per pose hypothesis, and unlike a
    colour score it cannot be satisfied by a coincidence of paint.
    """
    predicted = measure_edges(
        image,
        model,
        camera,
        T_cam_rig,
        reference=reference,
        views=views,
        sample_step_m=sample_step_m,
        search_px=search_px,
        masks=masks,
        ink=ink,
    )
    # Normalise by every measurable edge in the *model*, not by the ones this
    # pose happens to predict.  A per-pose denominator would let a hypothesis
    # that predicts one visible facet and finds its edge score as highly as one
    # that predicts six and finds all of them.
    attempted = 0
    for edge in model.edges:
        if not edge.measurable:
            continue
        span = float(edge.length_m) - 2.0 * 5e-3
        if span > 0:
            attempted += max(2, int(np.floor(span / sample_step_m)) + 1)
    if attempted == 0:
        return 0.0
    scored = sum(max(0.0, 1.0 - abs(m.residual_px) / float(search_px)) for m in predicted)
    return float(scored / attempted)


def reject_mislocked_edges(
    measurements: Sequence[EdgeMeasurement],
    camera: CameraCalibration,
    T_cam_rig: np.ndarray,
    *,
    max_median_px: float = 1.5,
    min_samples: int = 3,
) -> list[EdgeMeasurement]:
    """Drop whole edges whose samples agree with each other but not with the pose.

    A robust loss handles scattered bad samples.  It does not handle the failure
    that actually happens here: a search band wide enough to reach the *next*
    boundary locks every sample of one edge onto it, producing a tight, confident,
    wrong line.  That signature is a large *median* residual on one edge while the
    rest of the rig fits, so it has to be judged per edge, not per sample.
    """
    if not measurements:
        return []
    by_edge: dict[int, list[EdgeMeasurement]] = {}
    for measurement in measurements:
        by_edge.setdefault(measurement.edge_index, []).append(measurement)
    kept: list[EdgeMeasurement] = []
    for group in by_edge.values():
        points = np.asarray([m.point_rig for m in group], dtype=np.float64)
        measured = np.asarray([m.uv for m in group], dtype=np.float64)
        normals = np.asarray([m.normal_uv for m in group], dtype=np.float64)
        predicted = project_rig(camera, T_cam_rig, points)
        residuals = np.abs(np.sum((predicted - measured) * normals, axis=1))
        if len(group) >= int(min_samples) and float(np.median(residuals)) > float(max_median_px):
            continue
        kept.extend(group)
    return kept


# ---------------------------------------------------------------------------
# pose
# ---------------------------------------------------------------------------


@dataclass
class CarrierView:
    """One camera's contribution to a single rig pose."""

    camera: CameraCalibration
    anchor_points_rig: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    anchor_uv: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    anchor_sigma_px: float = 0.3
    edges: list[EdgeMeasurement] = field(default_factory=list)


def _view_residuals(view: CarrierView, T_base_rig: np.ndarray) -> np.ndarray:
    camera = view.camera
    T_cam_rig = camera.T_cam_base @ T_base_rig
    blocks: list[np.ndarray] = []
    if len(view.anchor_points_rig):
        predicted = project_rig(camera, T_cam_rig, view.anchor_points_rig)
        blocks.append(((predicted - view.anchor_uv) / max(view.anchor_sigma_px, 1e-6)).ravel())
    if view.edges:
        points = np.asarray([m.point_rig for m in view.edges], dtype=np.float64)
        measured = np.asarray([m.uv for m in view.edges], dtype=np.float64)
        normals = np.asarray([m.normal_uv for m in view.edges], dtype=np.float64)
        sigma = np.asarray([m.sigma_px for m in view.edges], dtype=np.float64)
        predicted = project_rig(camera, T_cam_rig, points)
        blocks.append(np.sum((predicted - measured) * normals, axis=1) / np.maximum(sigma, 1e-6))
    if not blocks:
        return np.zeros(0, dtype=np.float64)
    return np.concatenate(blocks)


@dataclass
class PoseSolution:
    T_base_rig: np.ndarray
    anchor_rmse_px: float
    edge_rmse_px: float
    num_anchor_points: int
    num_edge_samples: int
    covariance: np.ndarray | None
    sigma_translation_m: float | None
    sigma_rotation_deg: float | None
    message: str = ""


def solve_carrier_pose(
    views: Sequence[CarrierView],
    initial_T_base_rig: np.ndarray,
    *,
    huber_px: float = 2.0,
    max_nfev: int = 60,
) -> PoseSolution:
    """One rigid pose from anchor corners plus point-to-line edge samples."""

    def residual(params: np.ndarray) -> np.ndarray:
        T = _from_params(params)
        blocks = [_view_residuals(view, T) for view in views]
        blocks = [b for b in blocks if b.size]
        return np.concatenate(blocks) if blocks else np.zeros(1, dtype=np.float64)

    x0 = _to_params(initial_T_base_rig)
    result = least_squares(
        residual, x0, loss="huber", f_scale=float(huber_px), max_nfev=int(max_nfev)
    )
    T = _from_params(result.x)

    anchor_errors: list[float] = []
    edge_errors: list[float] = []
    for view in views:
        T_cam_rig = view.camera.T_cam_base @ T
        if len(view.anchor_points_rig):
            predicted = project_rig(view.camera, T_cam_rig, view.anchor_points_rig)
            anchor_errors.extend(np.linalg.norm(predicted - view.anchor_uv, axis=1).tolist())
        if view.edges:
            points = np.asarray([m.point_rig for m in view.edges], dtype=np.float64)
            measured = np.asarray([m.uv for m in view.edges], dtype=np.float64)
            normals = np.asarray([m.normal_uv for m in view.edges], dtype=np.float64)
            predicted = project_rig(view.camera, T_cam_rig, points)
            edge_errors.extend(np.abs(np.sum((predicted - measured) * normals, axis=1)).tolist())

    covariance = None
    sigma_t = None
    sigma_r = None
    jacobian = result.jac
    if jacobian is not None and jacobian.size and jacobian.shape[0] >= 6:
        try:
            covariance = np.linalg.inv(jacobian.T @ jacobian)
            sigma_r = float(np.degrees(np.sqrt(max(np.trace(covariance[:3, :3]), 0.0))))
            sigma_t = float(np.sqrt(max(np.trace(covariance[3:, 3:]), 0.0)))
        except np.linalg.LinAlgError:
            covariance = None
    return PoseSolution(
        T_base_rig=T,
        anchor_rmse_px=float(np.sqrt(np.mean(np.square(anchor_errors))))
        if anchor_errors
        else float("nan"),
        edge_rmse_px=float(np.sqrt(np.mean(np.square(edge_errors))))
        if edge_errors
        else float("nan"),
        num_anchor_points=int(sum(len(v.anchor_points_rig) for v in views)),
        num_edge_samples=int(sum(len(v.edges) for v in views)),
        covariance=covariance,
        sigma_translation_m=sigma_t,
        sigma_rotation_deg=sigma_r,
        message="" if result.success else str(result.message),
    )


@dataclass
class BuildScale:
    """How big the part came out, against the CAD it was built from.

    Two scales, not one, because two different things can be the wrong size and
    they have opposite consequences.  ``body_ppm`` scales the distances *between*
    anchors, which is rig geometry: if the printed part shrank, every facet and
    anchor moved inward while the origin, being a point, did not, so fitting a
    CAD-sized model to it lands the origin off the socket -- that is TCP error.
    ``marker_ppm`` scales each sticker about its own centre, which is the decal
    coming out of the printer at other than 1:1.  A small sticker biases the
    range of every single-anchor view and nothing else; it does not move the
    origin at all.

    Measured on the 2026-09-09 bench capture the two are nothing alike: the
    printed anchors came out 1.6% small and the body 0.4%.  Correcting only the
    sticker halved the per-camera dispersion at every anchor count.

    Fitting one scale for both reports their average and blames the wrong thing.
    """

    body_ppm: float
    marker_ppm: float
    body_sigma_ppm: float
    marker_sigma_ppm: float
    socket_shift_mm: float
    lever_arm_mm: float
    anchor_rmse_px: float
    anchor_rmse_px_fixed_scale: float
    num_anchor_points: int
    num_anchors: int
    num_cameras: int
    message: str = ""


def estimate_build_scale(
    views: Sequence[CarrierView],
    initial_T_base_rig: np.ndarray,
    *,
    huber_px: float = 2.0,
    max_nfev: int = 80,
) -> BuildScale | None:
    """Fit the built part's body scale and sticker scale alongside the pose.

    This is where the sigma on the carrier -> TCP constant comes from.  The
    socket centre's *CAD* position is exact -- it is the origin -- so the
    constant is zero with no fit error in it.  What is not exact is the built
    part, and only its body scale moves the origin.

    Anchors only, deliberately: their corners are decoded correspondences, so a
    scale fitted on them is a ruler check.  Edge samples slide along their own
    boundaries, where a scale error can hide as a pose change.

    Returns ``None`` when the evidence cannot separate a scale from a range --
    one camera always explains a smaller part by a nearer one, so this needs two;
    and one anchor carries no body scale at all, so it needs two of those too.
    """
    usable: list[tuple[CarrierView, np.ndarray, np.ndarray]] = []
    markers: set[tuple[float, float, float]] = set()
    for view in views:
        points = np.asarray(view.anchor_points_rig, dtype=np.float64).reshape(-1, 3)
        if len(points) < 4 or len(points) % 4:
            continue
        # Four consecutive corners are one marker; its own centre is their mean,
        # which is what separates "the sticker is small" from "the rig is small".
        # Nothing here is carrier-specific -- any rig whose corner table is
        # grouped by marker fits, the production cube included.
        centres = points.reshape(-1, 4, 3).mean(axis=1)
        markers.update(tuple(np.round(centre, 6)) for centre in centres)
        usable.append((view, points, np.repeat(centres, 4, axis=0)))
    if len({view.camera.name for view, _, _ in usable}) < 2 or len(markers) < 2:
        return None

    lever_arm_mm = float(np.mean([np.linalg.norm(p, axis=1).mean() for _, p, _ in usable])) * 1000.0

    def scaled_views(body: float, marker: float) -> list[CarrierView]:
        return [
            CarrierView(
                camera=view.camera,
                anchor_points_rig=body * centres + marker * (points - centres),
                anchor_uv=view.anchor_uv,
                anchor_sigma_px=view.anchor_sigma_px,
            )
            for view, points, centres in usable
        ]

    def residual(params: np.ndarray) -> np.ndarray:
        T = _from_params(params[:6])
        blocks = [_view_residuals(v, T) for v in scaled_views(float(params[6]), float(params[7]))]
        return np.concatenate(blocks)

    x0 = np.concatenate([_to_params(initial_T_base_rig), [1.0, 1.0]])
    result = least_squares(
        residual, x0, loss="huber", f_scale=float(huber_px), max_nfev=int(max_nfev)
    )
    body, marker = float(result.x[6]), float(result.x[7])

    free = solve_carrier_pose(
        scaled_views(body, marker), _from_params(result.x[:6]), huber_px=huber_px
    )
    fixed = solve_carrier_pose(scaled_views(1.0, 1.0), initial_T_base_rig, huber_px=huber_px)

    body_sigma = marker_sigma = float("nan")
    jacobian = result.jac
    if jacobian is not None and jacobian.size and jacobian.shape[0] > 8:
        try:
            covariance = np.linalg.inv(jacobian.T @ jacobian)
            body_sigma = float(np.sqrt(max(covariance[6, 6], 0.0))) * 1e6
            marker_sigma = float(np.sqrt(max(covariance[7, 7], 0.0))) * 1e6
        except np.linalg.LinAlgError:
            pass

    return BuildScale(
        body_ppm=(body - 1.0) * 1e6,
        marker_ppm=(marker - 1.0) * 1e6,
        body_sigma_ppm=body_sigma,
        marker_sigma_ppm=marker_sigma,
        socket_shift_mm=abs(body - 1.0) * lever_arm_mm,
        lever_arm_mm=lever_arm_mm,
        anchor_rmse_px=float(free.anchor_rmse_px),
        anchor_rmse_px_fixed_scale=float(fixed.anchor_rmse_px),
        num_anchor_points=int(sum(len(p) for _, p, _ in usable)),
        num_anchors=len(markers),
        num_cameras=len({view.camera.name for view, _, _ in usable}),
        message="" if result.success else str(result.message),
    )


# ---------------------------------------------------------------------------
# detection
# ---------------------------------------------------------------------------


@dataclass
class AnchorDetection:
    marker_id: int
    corners_uv: np.ndarray  # (4, 2) in ArUco order
    quadrant: int = 0


@dataclass
class CarrierDetection:
    success: bool
    T_base_rig: np.ndarray | None
    anchors: list[AnchorDetection]
    facet_views: list[FacetView]
    coverage: dict[str, float]
    pose: PoseSolution | None
    reference: ColourReference
    hypothesis_margin: float
    measurements: list[EdgeMeasurement]
    message: str = ""
    # How this frame was read, for a caller that wants to know whether it is
    # paying for a re-acquisition or riding a track.
    seeded: bool = False
    reacquired: bool = False
    roi: tuple[int, int, int, int] | None = None

    @property
    def T_cam_rig(self) -> np.ndarray | None:
        return self.T_base_rig

    @property
    def measured_border_width_px(self) -> float | None:
        widths = [m.band_width_px for m in self.measurements if m.profile == "band_centre"]
        return float(np.median(widths)) if widths else None


def build_aruco_detector(dictionary_name: str, *, refine: bool = True) -> Any:
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
    params = cv2.aruco.DetectorParameters()
    if refine:
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        params.cornerRefinementWinSize = 5
        params.cornerRefinementMaxIterations = 50
        params.cornerRefinementMinAccuracy = 0.01
    return cv2.aruco.ArucoDetector(dictionary, params)


def colour_reference_from_detections(
    image: np.ndarray, detections: Sequence[AnchorDetection], *, quiet_zone_ratio: float = 1.25
) -> ColourReference:
    """White and black points straight off the anchor stickers, before any pose.

    An ArUco marker is a printed grey card that happens to encode a number: its
    white modules and quiet zone are the white patch, its black modules the
    black patch, both at the rig's own distance and under the rig's own light.
    """
    if image.ndim != 3 or not detections:
        return DEFAULT_REFERENCE
    height, width = image.shape[:2]
    whites: list[np.ndarray] = []
    blacks: list[np.ndarray] = []
    for detection in detections:
        quad = np.asarray(detection.corners_uv, dtype=np.float64)
        centre = quad.mean(axis=0)
        outer = centre + float(quiet_zone_ratio) * (quad - centre)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [np.round(outer).astype(np.int32)], 255)
        if np.count_nonzero(mask) < 64:
            continue
        pixels = image[mask > 0].reshape(-1, 3).astype(np.float64)
        whites.append(np.percentile(pixels, 85, axis=0))
        blacks.append(np.percentile(pixels, 10, axis=0))
    if not whites:
        return DEFAULT_REFERENCE
    return ColourReference(
        white=np.mean(whites, axis=0),
        black=np.mean(blacks, axis=0),
        source=f"anchor sticker patches ({len(whites)} markers)",
    )


def chromaticity_scores(image: np.ndarray) -> dict[str, np.ndarray]:
    """Hue evidence that needs no white point, because it is a ratio.

    Each score is a channel difference over the channel sum, so scaling the
    exposure or the whole illuminant divides out.  It is weaker than the
    white-balanced classification -- it cannot tell paint from a coloured cast,
    and it says nothing about black -- but it is enough to find where the paint
    is, which is the only thing needed before a white point exists.
    """
    bgr = np.asarray(image, dtype=np.float64)
    blue, green, red = bgr[..., 0], bgr[..., 1], bgr[..., 2]
    denominator = np.maximum(blue + green + red, 1e-3)
    return {
        "red": (red - np.maximum(green, blue)) / denominator,
        "green": (green - np.maximum(red, blue)) / denominator,
        "blue": (blue - np.maximum(green, red)) / denominator,
        "yellow": (np.minimum(red, green) - blue) / denominator,
        "magenta": (np.minimum(red, blue) - green) / denominator,
        "cyan": (np.minimum(green, blue) - red) / denominator,
    }


def colour_reference_from_image(
    image: np.ndarray,
    *,
    chroma_margin: float = 0.14,
    dilate_px: int = 15,
    min_paint_px: int = 400,
) -> ColourReference:
    """White and black points with no pose and no marker to read them off.

    The naive version of this -- percentiles over the whole frame -- fails
    exactly when it matters.  A rig at a metre covers a few percent of a 1080p
    frame, so both percentiles land in the background and the "white point"
    comes back as the tablecloth, after which every normalised pixel looks
    saturated and every colour mask fires everywhere.

    So find the paint first, by chromaticity, which needs no white point at all;
    then read the white and black points from a band around it, where the bare
    body and the printed borders are.  That is the same idea as the anchor
    version -- measure the reference on the rig, under the rig's own light --
    with the paint standing in for the sticker.
    """
    if image.ndim != 3:
        return DEFAULT_REFERENCE
    scores = chromaticity_scores(image)
    paint = np.zeros(image.shape[:2], dtype=np.uint8)
    for score in scores.values():
        paint |= (score > float(chroma_margin)).astype(np.uint8)
    if int(np.count_nonzero(paint)) < int(min_paint_px):
        return DEFAULT_REFERENCE
    radius = max(1, int(dilate_px))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    neighbourhood = cv2.dilate(paint, kernel) > 0
    pixels = image[neighbourhood].reshape(-1, 3).astype(np.float64)
    # The bare body is broad and the printed border is one or two pixels wide,
    # so the two points are not symmetric: 92% lands on the body, while any
    # percentile generous enough to be robust for black lands on dark paint
    # instead of on ink and leaves the borders looking like low contrast.
    white = np.percentile(pixels, 92.0, axis=0)
    black = np.percentile(pixels, 0.5, axis=0)
    if float(np.min(white - black)) < 25.0:  # no border and no bare body in view: nothing to key on
        return DEFAULT_REFERENCE
    return ColourReference(
        white=white, black=black, source="painted-region neighbourhood (no anchor)"
    )


@dataclass
class ColourRegion:
    """A blob of one paint in the image, reduced to a polygon."""

    colour: str
    polygon_uv: np.ndarray  # (N, 2), in image order
    area_px2: float
    centroid_uv: np.ndarray


def colour_regions(
    masks: dict[str, np.ndarray],
    *,
    min_area_px2: float = 400.0,
    epsilon_ratio: float = 0.02,
    max_per_colour: int = 3,
) -> list[ColourRegion]:
    """Painted blobs, largest first, as polygons.

    Only the chromatic masks are used.  White, grey and black are the colours a
    workcell is full of, so a region of them is not evidence of anything; the
    hypothesis has to be seeded by a paint that nothing else in the room shares.
    """
    out: list[ColourRegion] = []
    for colour in ("red", "green", "blue", "yellow", "magenta", "cyan"):
        mask = masks.get(colour)
        if mask is None or not np.any(mask):
            continue
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        found: list[ColourRegion] = []
        for contour in contours:
            area = abs(float(cv2.contourArea(contour)))
            if area < float(min_area_px2):
                continue
            perimeter = float(cv2.arcLength(contour, True))
            polygon = cv2.approxPolyDP(contour, float(epsilon_ratio) * perimeter, True)
            polygon = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
            if len(polygon) < 3:
                continue
            found.append(
                ColourRegion(
                    colour=colour,
                    polygon_uv=polygon,
                    area_px2=area,
                    centroid_uv=polygon.mean(axis=0),
                )
            )
        found.sort(key=lambda region: -region.area_px2)
        out.extend(found[: int(max_per_colour)])
    out.sort(key=lambda region: -region.area_px2)
    return out


def bootstrap_seeds(
    camera: CameraCalibration,
    model: HybridCarrierModel,
    regions: Sequence[ColourRegion],
    *,
    max_seeds: int = 96,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, str]]:
    """Coarse poses from the paint alone: no anchor, no id, no prior.

    A painted facet is a planar polygon whose 3D vertices CAD knows exactly, so
    one region matched to one facet is already a full planar PnP -- and a planar
    PnP has two solutions, both of which are returned for the same reason the
    anchor path returns both.  The correspondence is the unknown: the contour
    can start at any of the facet's vertices, so every cyclic shift is tried,
    and both traversal directions, because a mirrored match is cheaper to score
    and reject than to reason about.

    This is only affordable because the paint is chiral.  When mirror twins
    share a colour, every region has twice the candidate facets and the two
    hypotheses are indistinguishable by construction; with the twins painted
    apart, a region of a given colour has at most two candidates on the whole
    rig.  Returns ``(T_cam_rig, object_points, image_points, seed_name)``.
    """
    by_colour: dict[str, list[Facet]] = {}
    for facet in model.facets:
        if facet.use_for_pose:
            by_colour.setdefault(facet.colour, []).append(facet)

    seeds: list[tuple[np.ndarray, np.ndarray, np.ndarray, str]] = []
    for region in regions:
        for facet in by_colour.get(region.colour, []):
            if facet.num_vertices != len(region.polygon_uv):
                continue  # a different vertex count is a different face, or a bad contour
            count = facet.num_vertices
            for reverse in (False, True):
                observed = region.polygon_uv[::-1] if reverse else region.polygon_uv
                for shift in range(count):
                    image_points = np.roll(observed, shift, axis=0)
                    for T_cam_rig in _pnp_candidates(camera, facet.polygon_m, image_points):
                        seeds.append(
                            (
                                T_cam_rig,
                                facet.polygon_m,
                                image_points,
                                f"{facet.name}+{shift}{'r' if reverse else ''}",
                            )
                        )
                        if len(seeds) >= int(max_seeds):
                            return seeds
    return seeds


def polish_seed(
    camera: CameraCalibration,
    model: HybridCarrierModel,
    T_cam_rig: np.ndarray,
    regions: Sequence[ColourRegion],
    *,
    min_area_px2: float = 150.0,
    min_visible_fraction: float = 0.6,
    max_vertex_px: float = 25.0,
    passes: int = 2,
) -> tuple[np.ndarray, int]:
    """Re-fit a one-facet seed against every painted region it can now explain.

    A pose from a single polygon is planar: weakly conditioned in depth, and
    two-fold ambiguous by construction.  But it is good enough to say which
    observed blob is which facet, and once two non-coplanar facets are in the
    fit both problems disappear.  So the seed's real job is correspondence, not
    pose, and this is where the pose comes from.

    Returns the polished pose and how many facets ended up in it.
    """
    T = np.asarray(T_cam_rig, dtype=np.float64).copy()
    used = 0
    for _ in range(max(1, int(passes))):
        object_points: list[np.ndarray] = []
        image_points: list[np.ndarray] = []
        matched = 0
        views = facet_views(
            model, camera, T, min_area_px2=min_area_px2, min_visible_fraction=min_visible_fraction
        )
        for view in views:
            if not view.usable:
                continue
            same = [
                region
                for region in regions
                if region.colour == view.facet.colour
                and len(region.polygon_uv) == view.facet.num_vertices
            ]
            if not same:
                continue
            predicted = np.asarray(view.polygon_uv, dtype=np.float64)
            centre = predicted.mean(axis=0)
            region = min(same, key=lambda r: float(np.linalg.norm(r.centroid_uv - centre)))
            # Each model vertex takes the nearest unclaimed region vertex, and
            # the match is dropped whole unless every vertex finds one close by:
            # a partial polygon match is how a neighbouring facet of the same
            # colour gets silently substituted.
            free = list(range(len(region.polygon_uv)))
            pairs: list[tuple[int, int]] = []
            for index, point in enumerate(predicted):
                if not free:
                    break
                distances = [float(np.linalg.norm(region.polygon_uv[j] - point)) for j in free]
                best = int(np.argmin(distances))
                if distances[best] > float(max_vertex_px):
                    break
                pairs.append((index, free.pop(best)))
            if len(pairs) != len(predicted):
                continue
            object_points.append(np.asarray([view.facet.polygon_m[i] for i, _ in pairs]))
            image_points.append(np.asarray([region.polygon_uv[j] for _, j in pairs]))
            matched += 1
        if matched == 0 or sum(len(block) for block in object_points) < 4:
            break
        candidates = _pnp_candidates(camera, np.vstack(object_points), np.vstack(image_points))
        if not candidates:
            break
        # With more than one facet the points are no longer coplanar and there is
        # only one solution; with one facet keep the one nearest where we were.
        T = min(candidates, key=lambda C: float(np.linalg.norm((np.linalg.inv(T) @ C)[:3, 3])))
        used = matched
    return T, used


def signed_distance_fields(masks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Distance to each colour's boundary, negative inside, in pixels."""
    fields: dict[str, np.ndarray] = {}
    for colour, mask in masks.items():
        binary = (np.asarray(mask) > 0).astype(np.uint8)
        if not np.any(binary) or np.all(binary):
            continue
        inside = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
        outside = cv2.distanceTransform(1 - binary, cv2.DIST_L2, 3)
        fields[colour] = (outside - inside).astype(np.float64)
    return fields


def align_to_masks(
    camera: CameraCalibration,
    model: HybridCarrierModel,
    T_cam_rig: np.ndarray,
    fields: dict[str, np.ndarray],
    *,
    min_area_px2: float = 150.0,
    min_visible_fraction: float = 0.6,
    samples_per_edge: int = 12,
    passes: int = 3,
    trust_radius_m: float = 0.06,
) -> tuple[np.ndarray, int]:
    """Slide the pose until every facet outline sits on its own colour's boundary.

    The step between a polygon-matched seed and subpixel edge measurement.  The
    edge stage searches a few pixels along each normal and finds nothing at all
    when it starts further out than that, so it needs a pose that is already
    close; polygon vertices from ``approxPolyDP`` are not, being quantised and
    pulled inward by the printed border the mask does not include.

    A distance transform gives what the edge search cannot: a smooth cost with a
    basin tens of pixels wide, evaluated on every facet at once.  It aligns to
    the *mask* boundary, which sits inside the CAD edge by about half a stroke,
    so it is deliberately not the final answer -- it is what puts the edge stage
    inside its capture range, after which the band centre carries the pose.
    """
    T = np.asarray(T_cam_rig, dtype=np.float64).copy()
    used = 0
    for _ in range(max(1, int(passes))):
        points: list[np.ndarray] = []
        colours: list[str] = []
        views = facet_views(
            model, camera, T, min_area_px2=min_area_px2, min_visible_fraction=min_visible_fraction
        )
        for view in views:
            if not view.usable or view.facet.colour not in fields:
                continue
            polygon = view.facet.polygon_m
            count = len(polygon)
            for i in range(count):
                a, b = polygon[i], polygon[(i + 1) % count]
                t = np.linspace(0.1, 0.9, int(samples_per_edge))
                points.append(a + t[:, None] * (b - a))
                colours.extend([view.facet.colour] * len(t))
        if not points:
            break
        stacked = np.vstack(points)
        labels = np.asarray(colours)
        groups = [(colour, np.flatnonzero(labels == colour)) for colour in sorted(set(colours))]
        used = len(groups)

        def residual(
            params: np.ndarray, stacked: np.ndarray = stacked, groups: list = groups
        ) -> np.ndarray:
            uv = project_rig(camera, _from_params(params), stacked)
            out = np.zeros(len(uv), dtype=np.float64)
            for colour, index in groups:
                # Off the image the field says nothing, so charge a fixed cost
                # rather than a NaN: the fit should be pushed back into view,
                # not handed an undefined gradient.
                out[index] = np.nan_to_num(_bilinear(fields[colour], uv[index]), nan=50.0)
            return out

        result = least_squares(residual, _to_params(T), loss="huber", f_scale=4.0, max_nfev=40)
        moved = _from_params(result.x)
        # The field has local minima wherever an unexplained blob of the right
        # colour sits -- the reverse of an anchor pad, a misclassified
        # highlight -- and a pass that walks the rig across the scene has found
        # one of those, not the rig.  Keep the pose that came in.
        if float(np.linalg.norm((np.linalg.inv(T) @ moved)[:3, 3])) > float(trust_radius_m):
            break
        T = moved
    return T, used


def _pnp_candidates(
    camera: CameraCalibration, object_points: np.ndarray, image_points: np.ndarray
) -> list[np.ndarray]:
    """Every pose consistent with these correspondences, not just the first.

    One square marker is a planar target, and a planar target admits two poses
    that reproject identically -- the well-known flip about the plane.  Returning
    only one of them makes the detector right half the time and confidently wrong
    the other half; returning both lets the painted facets, which are not
    coplanar with the sticker, decide.  ``SOLVEPNP_IPPE_SQUARE`` is deliberately
    not used: it assumes the object points *are* the canonical centred square,
    which rig-frame coordinates are not, and it silently returns a pose for a
    different object when they are not.
    """
    object_points = np.ascontiguousarray(np.asarray(object_points, dtype=np.float64).reshape(-1, 3))
    image_points = np.ascontiguousarray(np.asarray(image_points, dtype=np.float64).reshape(-1, 2))
    if len(object_points) < 4:
        return []
    if camera.model == "fisheye":
        image_points = cv2.fisheye.undistortPoints(
            image_points.reshape(-1, 1, 2),
            camera.K,
            np.asarray(camera.D, dtype=np.float64).reshape(4, 1),
        ).reshape(-1, 2)
        K, D = np.eye(3), np.zeros(5)
    else:
        K, D = camera.K, camera.D

    centred = object_points - object_points.mean(axis=0)
    planar = bool(np.linalg.svd(centred, compute_uv=False)[2] < 1e-4)
    poses: list[np.ndarray] = []
    if planar:
        count, rvecs, tvecs, _ = cv2.solvePnPGeneric(
            object_points, image_points, K, D, flags=cv2.SOLVEPNP_IPPE
        )
        candidates = [(rvecs[i], tvecs[i]) for i in range(int(count))]
    else:
        ok, rvec, tvec = cv2.solvePnP(object_points, image_points, K, D, flags=cv2.SOLVEPNP_SQPNP)
        candidates = [(rvec, tvec)] if ok else []
    for rvec, tvec in candidates:
        rvec, tvec = cv2.solvePnPRefineLM(object_points, image_points, K, D, rvec, tvec)
        T_cam_rig = _as_T(rvec, tvec)
        if np.mean(transform(T_cam_rig, object_points)[:, 2]) > 0:
            poses.append(T_cam_rig)
    return poses


def _hypothesis_quadrants(
    model: HybridCarrierModel, detections: Sequence[AnchorDetection]
) -> list[tuple[int, ...]]:
    choices: list[Sequence[int]] = []
    anchors = model.anchors_by_id
    for detection in detections:
        anchor = anchors[int(detection.marker_id)]
        choices.append(
            (anchor.paste_quadrant,) if anchor.paste_quadrant is not None else PASTE_QUADRANTS
        )
    out: list[tuple[int, ...]] = [()]
    for options in choices:
        out = [prefix + (int(option),) for prefix in out for option in options]
    return out


def _quadrants_at_pose(
    model: HybridCarrierModel,
    camera: CameraCalibration,
    T_base_rig: np.ndarray,
    detections: Sequence[AnchorDetection],
) -> tuple[int, ...]:
    """Pick each sticker's rotation by reprojecting at a pose already known.

    The from-scratch path cannot do this -- it has no pose until a rotation is
    assumed, which is why it enumerates and then scores every combination
    against the image.  A tracked frame does have one, and then the choice is a
    reprojection comparison costing no image work at all.
    """
    T_cam_rig = camera.T_cam_base @ np.asarray(T_base_rig, dtype=np.float64)
    anchors = model.anchors_by_id
    out: list[int] = []
    for detection in detections:
        anchor = anchors[int(detection.marker_id)]
        if anchor.paste_quadrant is not None:
            out.append(int(anchor.paste_quadrant))
            continue
        errors = []
        for quadrant in PASTE_QUADRANTS:
            predicted = project_rig(camera, T_cam_rig, anchor.object_points(quadrant))
            errors.append(
                (
                    float(np.linalg.norm(predicted - detection.corners_uv, axis=1).mean()),
                    int(quadrant),
                )
            )
        out.append(min(errors)[1])
    return tuple(out)


def _score_hypothesis(
    image: np.ndarray,
    model: HybridCarrierModel,
    camera: CameraCalibration,
    T_cam_rig: np.ndarray,
    *,
    quadrants: tuple[int, ...],
    object_points: np.ndarray,
    image_points: np.ndarray,
    reference: ColourReference,
    masks: dict[str, np.ndarray],
    min_area_px2: float,
    min_visible_fraction: float,
    ink: np.ndarray | None = None,
) -> tuple[float, tuple[float, np.ndarray, tuple[int, ...], list[FacetView], dict[str, float]]]:
    """Rank one (sticker rotation, planar flip) hypothesis against the image.

    Three terms, and each answers something the others cannot.  Reprojection
    checks the anchors are self-consistent, but a wrong hypothesis can fit them
    perfectly.  Colour says which facet is which, weighted by how much a match of
    that colour is worth -- red is worth a lot, black next to nothing, since any
    shadow or dark background produces black.  Edge support asks whether the CAD
    boundaries actually land on printed boundaries, which is the term a wrong
    pose cannot fake.
    """
    views = facet_views(
        model,
        camera,
        T_cam_rig,
        min_area_px2=min_area_px2,
        min_visible_fraction=min_visible_fraction,
    )
    coverage: dict[str, float] = {}
    for view in views:
        value = facet_colour_coverage(masks, view) if masks else None
        if value is not None:
            coverage[view.facet.name] = value
    reprojection = float(
        np.sqrt(np.mean(np.square(project_rig(camera, T_cam_rig, object_points) - image_points)))
    )
    # Deliberately not a mean.  Each facet contributes on its own, in [-1, 1],
    # scaled by how much a match of its colour is worth and by whether it is
    # big enough to trust, so a hypothesis that explains six facets outscores one
    # that predicts a single facet and gets it right.
    agreement = 0.0
    for view in views:
        value = coverage.get(view.facet.name)
        if value is None:
            continue
        area_weight = min(1.0, view.projected_area_px2 / 2000.0)
        agreement += (
            COLOUR_EVIDENCE_WEIGHT.get(view.facet.colour, 0.5) * area_weight * (2.0 * value - 1.0)
        )
    support = edge_support(
        image, model, camera, T_cam_rig, reference=reference, views=views, masks=masks, ink=ink
    )
    score = 10.0 * agreement + 100.0 * support - 4.0 * reprojection
    T_base_rig = camera.T_base_cam @ T_cam_rig
    return score, (score, T_base_rig, quadrants, views, coverage)


def edge_constraints_span_the_pose(
    measurements: Sequence[EdgeMeasurement],
    model: HybridCarrierModel,
    *,
    min_distinct_edges: int = 3,
    min_normal_spread_deg: float = 25.0,
    min_facet_normals: int = 2,
) -> str:
    """Whether these point-to-line samples can determine six numbers.

    Each sample constrains one direction in the image, so a set of samples that
    all lie on parallel boundaries of one facet constrains far fewer than six
    degrees of freedom no matter how many samples there are.  A solver handed
    that finds an exact fit hundreds of millimetres away and reports a
    sub-pixel residual, which is why this is checked before the fit and not
    after it.  Returns "" when the evidence spans the pose, else why not.

    An anchored pose does not need the test -- decoded corners pin translation
    and scale on their own -- so it is the anchor-free path that calls it.
    """
    if not measurements:
        return "no edge samples"
    indices = {int(m.edge_index) for m in measurements}
    if len(indices) < int(min_distinct_edges):
        return f"{len(indices)} distinct edges (need {int(min_distinct_edges)})"
    angles = np.array([np.arctan2(m.normal_uv[1], m.normal_uv[0]) for m in measurements])
    # Boundaries and their opposites constrain the same direction, so fold the
    # normals onto a half turn before asking how much of it they cover.
    folded = np.mod(angles, np.pi)
    spread = float(np.degrees(np.max(folded) - np.min(folded)))
    spread = min(spread, 180.0 - spread) if spread > 90.0 else spread
    if spread < float(min_normal_spread_deg):
        return f"boundary normals span {spread:.0f} deg (need {float(min_normal_spread_deg):.0f})"
    planes = set()
    for measurement in measurements:
        edge = model.edges[int(measurement.edge_index)]
        planes.add(tuple(np.round(edge.normal_b, 3)))
    if len(planes) < int(min_facet_normals):
        return f"{len(planes)} facet plane(s) (need {int(min_facet_normals)}): the fit is planar"
    return ""


def _hypothesis_margin(
    score: float,
    T_base_rig: np.ndarray,
    scored: Sequence[tuple[float, np.ndarray]],
    *,
    same_pose_mm: float = 5.0,
    same_pose_deg: float = 3.0,
) -> float:
    """How far clear the winner is of the best *different* answer.

    Several seeds routinely converge on the same pose -- two cyclic shifts of a
    near-square outline, say -- and scoring the winner against one of its own
    duplicates reports a margin of nothing when the hypothesis is in fact
    unopposed.  Only a materially different pose counts as a rival.
    """
    best_rival = -np.inf
    for other_score, other_T in scored:
        delta = np.linalg.inv(T_base_rig) @ other_T
        translation_mm = float(np.linalg.norm(delta[:3, 3])) * 1000.0
        rotation_deg = float(np.degrees(np.linalg.norm(_matrix_to_rodrigues(delta[:3, :3]))))
        if translation_mm < float(same_pose_mm) and rotation_deg < float(same_pose_deg):
            continue
        best_rival = max(best_rival, float(other_score))
    return float(score - best_rival) if np.isfinite(best_rival) else float("inf")


def _refine_carrier_pose(
    image: np.ndarray,
    camera: CameraCalibration,
    model: HybridCarrierModel,
    T_base_rig: np.ndarray,
    *,
    reference: ColourReference,
    masks: dict[str, np.ndarray],
    anchor_object: np.ndarray,
    anchor_image: np.ndarray,
    anchor_sigma_px: float,
    huber_px: float,
    include_biased_edges: bool,
    min_area_px2: float,
    min_visible_fraction: float,
    refine_iterations: int,
    anchor_free: bool,
    trust_radius_m: float,
    max_sigma_m: float,
    ink: np.ndarray | None = None,
    start_px: float | None = None,
) -> tuple[np.ndarray, list[EdgeMeasurement], PoseSolution | None, str]:
    """Walk a coarse pose in against the printed boundaries.

    Split out of ``detect_carrier`` because the anchor-free path has to run it
    on several rival hypotheses: a coarse colour score is a weak discriminator,
    and which hypothesis actually admits a well-conditioned edge fit is the
    strong one.  Returns the pose, the measurements behind it, the solution, and
    why it stopped if it produced nothing.
    """
    T_base_rig = np.asarray(T_base_rig, dtype=np.float64).copy()
    measurements: list[EdgeMeasurement] = []
    pose: PoseSolution | None = None
    rank_failure = ""
    if ink is None:
        ink = ink_image(image, reference)
    # Coarse to fine. The first pass has to reach far enough to cover the anchor
    # pose's error; later passes narrow the band so a sample can no longer reach
    # a neighbouring boundary once the pose is close.
    iterations = max(1, int(refine_iterations))
    # More passes, not a wider window, for a pose seeded from paint.  Widening
    # the search is the intuitive fix and the wrong one: once the window spans
    # more than one printed boundary the profile stops having a single band to
    # centre on, and the measurement count collapses to zero rather than
    # degrading.  Walking in on the same narrow band is what converges.
    if anchor_free:
        iterations = max(iterations, 10)
    # A pose carried in from the previous frame starts far closer than one
    # solved from a single sticker, so the caller may start the walk narrower.
    if start_px is None:
        start_px = 5.0 if anchor_free else 8.0
    bands = np.geomspace(float(start_px), 2.5, iterations) if iterations > 1 else np.array([6.0])
    for band in bands:
        T_cam_rig = camera.T_cam_base @ T_base_rig
        views = facet_views(
            model,
            camera,
            T_cam_rig,
            min_area_px2=min_area_px2,
            min_visible_fraction=min_visible_fraction,
        )
        found = measure_edges(
            image,
            model,
            camera,
            T_cam_rig,
            reference=reference,
            views=views,
            search_px=float(band),
            include_biased=include_biased_edges,
            # The side-colour veto is only as good as the classification behind
            # it.  With an anchor, that comes off a printed grey card and is
            # worth vetoing on; without one it comes off frame statistics, and
            # measured against ground truth it throws away two of every five
            # good boundaries.  The band-width check, the mislock rejection and
            # the rank and trust gates downstream all still apply, and they do
            # not depend on the segmentation.
            masks=(masks or None) if not anchor_free else None,
            ink=ink,
        )
        found = reject_mislocked_edges(
            found, camera, T_cam_rig, max_median_px=max(1.5, float(band) * 0.5)
        )
        # A pass that finds nothing means this band straddled the boundary, not
        # that the boundary is gone: keep the last pass that did see something
        # so the frame is not reported as edgeless because its final pass was.
        if found:
            measurements = found
        view = CarrierView(
            camera=camera,
            anchor_points_rig=anchor_object,
            anchor_uv=anchor_image,
            anchor_sigma_px=float(anchor_sigma_px),
            edges=found,
        )
        if not len(view.edges) and not len(view.anchor_points_rig):
            # Nothing to fit at this band.  Running the solver on an empty
            # residual does not return the pose it was given, it wanders -- and
            # a narrower band later in the schedule may still find the boundary
            # this one straddled, so skip the pass rather than end the walk.
            continue
        if anchor_free:
            weakness = edge_constraints_span_the_pose(found, model)
            if weakness:
                rank_failure = weakness
                continue
        candidate = solve_carrier_pose([view], T_base_rig, huber_px=huber_px)
        if anchor_free:
            # The seed already explained two facets, so it is good to a couple of
            # centimetres.  A "refinement" that moves further than that has found
            # a different object, not a better fit for this one.
            step_m = float(
                np.linalg.norm((np.linalg.inv(T_base_rig) @ candidate.T_base_rig)[:3, 3])
            )
            if step_m > float(trust_radius_m):
                rank_failure = f"refinement stepped {step_m * 1000:.0f} mm, past the trust radius"
                continue
            # The rank test above is on the constraints; this is on the answer.
            # A fit that cannot say where the rig is to a few millimetres was
            # under-determined whatever the sample count said.
            sigma = candidate.sigma_translation_m
            if sigma is None or not np.isfinite(sigma) or sigma > float(max_sigma_m):
                rank_failure = (
                    "edge-only fit is under-determined"
                    if sigma is None
                    else f"edge-only fit is uncertain to {sigma * 1000:.1f} mm"
                )
                continue
        pose = candidate
        if np.allclose(candidate.T_base_rig, T_base_rig, atol=1e-9):
            break
        T_base_rig = candidate.T_base_rig

    return T_base_rig, measurements, pose, rank_failure


def detect_carrier(
    image: np.ndarray,
    camera: CameraCalibration,
    model: HybridCarrierModel,
    *,
    detector: Any | None = None,
    refine_iterations: int = 3,
    min_visible_fraction: float = 0.6,
    min_area_px2: float = 150.0,
    include_biased_edges: bool = True,
    huber_px: float = 2.0,
    anchor_sigma_px: float = 0.3,
    min_anchors: int = 1,
    allow_anchor_free: bool = False,
    anchor_free_min_margin: float = 6.0,
    anchor_free_min_facets: int = 2,
    anchor_free_trust_radius_m: float = 0.03,
    anchor_free_max_sigma_m: float = 0.004,
    seed: np.ndarray | None = None,
    seed_band_px: float = 5.0,
    seed_trust_radius_m: float = 0.05,
    seed_trust_deg: float = 20.0,
    roi: Sequence[float] | None = None,
    roi_margin_px: float = 48.0,
) -> CarrierDetection:
    """Anchors -> coarse pose -> facet ROIs -> subpixel edges -> object-level pose.

    ``seed`` is a ``T_base_rig`` this frame is expected to be near -- the
    previous frame's answer, in a video.  Video is continuous and this detector
    was throwing that away: at 30 fps the rig moves under two millimetres
    between frames, so the previous pose is a better start than anything the
    from-scratch path derives, and it also says *where in the image* to look,
    which is what the per-pixel fields cost.  A seeded frame therefore skips the
    sticker-rotation enumeration and the hypothesis scoring, builds its ink and
    colour fields over a window instead of the whole frame, and starts the edge
    walk narrow.  If the answer then lands further from the seed than
    ``seed_trust_radius_m`` / ``seed_trust_deg``, the seed is treated as stale
    and the frame is re-read from scratch, so a lost track costs one slow frame
    rather than a wrong pose.

    With ``allow_anchor_free`` a frame that shows no anchor is seeded from the
    paint instead (see ``bootstrap_seeds``).  It is off by default and gated
    harder than the anchored path -- a wider margin over the runner-up, more
    than one facet agreeing, and a refusal outright if the model's mirror twins
    share colours -- because an anchor id is a decoded number and a colour match
    is not.

    Returns whatever it got even when it fails, so a caller can see which stage
    ran out of evidence instead of only that the frame was dropped.
    """
    empty = CarrierDetection(
        success=False,
        T_base_rig=None,
        anchors=[],
        facet_views=[],
        coverage={},
        pose=None,
        reference=DEFAULT_REFERENCE,
        hypothesis_margin=0.0,
        measurements=[],
    )
    if not model.anchors:
        empty.message = "model has no anchors"
        return empty
    if detector is None:
        detector = build_aruco_detector(model.anchors[0].dictionary)

    seed_T = None if seed is None else np.asarray(seed, dtype=np.float64).reshape(4, 4)
    window = clip_roi(roi, image.shape[:2])
    if window is None and seed_T is not None:
        window = rig_roi(camera, model, seed_T, image.shape[:2], margin_px=float(roi_margin_px))
    empty.seeded = seed_T is not None
    empty.roi = window

    # Anchor detection stays on the whole frame even when a window is known.  It
    # costs a few milliseconds against the hundreds the per-pixel fields cost,
    # and paying it is what lets a rig that jumped out of the window be found
    # again instead of quietly going missing for the rest of the run.
    grey = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(grey)
    known = model.anchors_by_id
    detections = [
        AnchorDetection(
            marker_id=int(marker_id), corners_uv=np.asarray(quad, dtype=np.float64).reshape(4, 2)
        )
        for quad, marker_id in zip(
            corners or [], (ids.ravel() if ids is not None else []), strict=False
        )
        if int(marker_id) in known
    ]
    anchor_free = not detections
    if anchor_free and not allow_anchor_free:
        empty.message = "no rig anchor detected"
        return empty
    if detections and len(detections) < int(min_anchors):
        # A view holding one 48 mm square barely constrains rotation, and this
        # detector does not fail on that -- it returns a confident pose. On the
        # 2026-09-09 seven-camera bench capture (82 frames, 562 solved camera
        # views) every single view more than 30 mm or 5 deg from the fused pose
        # came from a one-anchor camera: 40 of 218, against 0 of 344 for views
        # with two or more. Costing nothing in coverage there -- every frame had
        # at least two cameras holding two anchors -- but a single-camera setup
        # has no such luxury, which is why this is a parameter and not a rule.
        empty.anchors = detections
        empty.reference = colour_reference_from_detections(image, detections)
        empty.message = (
            f"{len(detections)} anchor(s) detected, fewer than the {int(min_anchors)} required"
        )
        return empty
    if anchor_free and not model.side_resolving_facets:
        empty.message = (
            "no rig anchor detected, and the paint cannot stand in for one: every mirror twin "
            "shares a colour, so a facet-only pose would be a coin flip on which side of y = 0 "
            "the rig is facing"
        )
        return empty

    if anchor_free:
        # The paint-only seeds are found by scanning every colour region in the
        # frame, so this path cannot run inside a window.
        window = None
        empty.roi = None
        reference = colour_reference_from_image(image)
    else:
        reference = colour_reference_from_detections(image, detections)
    masks = classify_colours(image, reference, roi=window) if image.ndim == 3 else {}
    ink = ink_image(image, reference, roi=window)

    seeds: list[tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, ...]]] = []
    if anchor_free:
        regions = colour_regions(masks, min_area_px2=min_area_px2)
        fields = signed_distance_fields(masks)
        raw = bootstrap_seeds(camera, model, regions)
        if not raw:
            empty.reference = reference
            empty.message = "no anchor, and no painted region matched a facet outline"
            return empty
        seen: list[np.ndarray] = []
        for T_seed, obj, uv, _ in raw:
            polished, matched = polish_seed(
                camera,
                model,
                T_seed,
                regions,
                min_area_px2=min_area_px2,
                min_visible_fraction=min_visible_fraction,
            )
            if matched < 2:
                continue  # one facet is a correspondence, not a pose
            aligned, _ = align_to_masks(
                camera,
                model,
                polished,
                fields,
                min_area_px2=min_area_px2,
                min_visible_fraction=min_visible_fraction,
            )
            # Score both.  Mask alignment is what gets the pose inside the edge
            # search's capture range, but it is a fit to a noisy segmentation
            # and can be the worse of the two; letting the scorer decide costs
            # one more evaluation and removes a way to lose the frame outright.
            for candidate_T in (aligned, polished):
                if any(
                    float(np.linalg.norm((np.linalg.inv(prior) @ candidate_T)[:3, 3])) < 2e-3
                    for prior in seen
                ):
                    continue  # several seeds walk to the same answer; score it once
                seen.append(candidate_T)
                seeds.append((candidate_T, obj, uv, ()))
        if not seeds:
            empty.reference = reference
            empty.message = "no anchor, and no seed explained two facets at once"
            return empty
    elif seed_T is not None:
        image_points = np.vstack([d.corners_uv for d in detections])
        quadrants = _quadrants_at_pose(model, camera, seed_T, detections)
        object_points = np.vstack(
            [
                known[d.marker_id].object_points(q)
                for d, q in zip(detections, quadrants, strict=True)
            ]
        )
        seeds.append((camera.T_cam_base @ seed_T, object_points, image_points, quadrants))
    else:
        image_points = np.vstack([d.corners_uv for d in detections])
        for quadrants in _hypothesis_quadrants(model, detections):
            object_points = np.vstack(
                [
                    known[d.marker_id].object_points(q)
                    for d, q in zip(detections, quadrants, strict=True)
                ]
            )
            seeds.extend(
                (T, object_points, image_points, quadrants)
                for T in _pnp_candidates(camera, object_points, image_points)
            )

    entries: list[tuple[float, np.ndarray, tuple[int, ...], list[FacetView], dict[str, float]]] = []
    scored: list[tuple[float, np.ndarray]] = []
    if seed_T is not None and not anchor_free:
        # One hypothesis, and it did not come from this frame's pixels, so there
        # is nothing for the scorer to choose between. Skipping it also skips the
        # `edge_support` pass inside it, which is a full edge sweep of its own.
        T_cam_rig, object_points, image_points, quadrants = seeds[0]
        entries.append(
            (
                0.0,
                seed_T,
                quadrants,
                facet_views(
                    model,
                    camera,
                    T_cam_rig,
                    min_area_px2=min_area_px2,
                    min_visible_fraction=min_visible_fraction,
                ),
                {},
            )
        )
    else:
        for T_cam_rig, object_points, image_points, quadrants in seeds:
            score, entry = _score_hypothesis(
                image,
                model,
                camera,
                T_cam_rig,
                quadrants=quadrants,
                object_points=object_points,
                image_points=image_points,
                reference=reference,
                masks=masks,
                min_area_px2=min_area_px2,
                min_visible_fraction=min_visible_fraction,
                ink=ink,
            )
            scored.append((score, entry[1]))
            entries.append(entry)
        entries.sort(key=lambda item: -item[0])
    best = entries[0] if entries else None
    if best is None:
        empty.anchors = detections
        empty.reference = reference
        empty.message = (
            "no painted-region hypothesis produced a valid PnP pose"
            if anchor_free
            else "no anchor hypothesis produced a valid PnP pose"
        )
        return empty

    score, T_base_rig, quadrants, views, coverage = best
    # What the margin is asking is "could this frame be read as a different
    # object pose", so a rival has to be a different *reading*, not a coarser
    # start on the same one.  Anchor-free seeds are deliberately several
    # centimetres apart before refinement, so the window that counts as "the
    # same answer" is correspondingly wider there.
    margin = _hypothesis_margin(
        score,
        T_base_rig,
        scored,
        same_pose_mm=30.0 if anchor_free else 5.0,
        same_pose_deg=10.0 if anchor_free else 3.0,
    )

    if anchor_free:
        # An anchor id is decoded, a colour match is inferred.  Two independent
        # facets must agree and the winner must be clear of the runner-up, or
        # this frame goes back as a miss rather than as a guess.
        agreeing = sorted(name for name, value in coverage.items() if value >= 0.5)
        carriers = [name for name in agreeing if name in model.side_resolving_facets]
        if (
            len(agreeing) < int(anchor_free_min_facets)
            or margin < float(anchor_free_min_margin)
            or not carriers
        ):
            empty.reference = reference
            empty.facet_views = views
            empty.coverage = coverage
            empty.hypothesis_margin = margin
            empty.message = (
                f"anchor-free hypothesis rejected: {len(agreeing)} facets agreed "
                f"(need {int(anchor_free_min_facets)}), margin {margin:.1f} "
                f"(need {float(anchor_free_min_margin):.1f}), "
                f"{len(carriers)} of them resolve the side of y = 0 (need 1)"
            )
            return empty
        anchor_object = np.zeros((0, 3), dtype=np.float64)
        anchor_image = np.zeros((0, 2), dtype=np.float64)
    else:
        for detection, quadrant in zip(detections, quadrants, strict=True):
            detection.quadrant = int(quadrant)
        anchor_object = np.vstack(
            [known[d.marker_id].object_points(d.quadrant) for d in detections]
        )
        anchor_image = np.vstack([d.corners_uv for d in detections])

    shortlist = entries[:4] if anchor_free else entries[:1]
    attempts: list[tuple[float, np.ndarray, list[EdgeMeasurement], PoseSolution]] = []
    rank_failure = ""
    for entry_score, entry_T, _, _, _ in shortlist:
        refined, found, candidate, failure = _refine_carrier_pose(
            image,
            camera,
            model,
            entry_T,
            reference=reference,
            masks=masks,
            anchor_object=anchor_object,
            anchor_image=anchor_image,
            anchor_sigma_px=float(anchor_sigma_px),
            huber_px=float(huber_px),
            include_biased_edges=bool(include_biased_edges),
            min_area_px2=float(min_area_px2),
            min_visible_fraction=float(min_visible_fraction),
            refine_iterations=int(refine_iterations),
            anchor_free=anchor_free,
            trust_radius_m=float(anchor_free_trust_radius_m),
            max_sigma_m=float(anchor_free_max_sigma_m),
            ink=ink,
            start_px=float(seed_band_px) if seed_T is not None else None,
        )
        rank_failure = rank_failure or failure
        if candidate is None:
            continue
        if not anchor_free:
            attempts.append((entry_score, refined, found, candidate))
            break
        # Re-score at the refined pose.  The seed that survived refinement with
        # the most boundaries behind it and the smallest residual is the reading
        # of this frame; the coarse colour score only decided who got to try.
        final, _ = _score_hypothesis(
            image,
            model,
            camera,
            camera.T_cam_base @ refined,
            quadrants=(),
            object_points=np.asarray([m.point_rig for m in found]),
            image_points=np.asarray([m.uv for m in found]),
            reference=reference,
            masks=masks,
            min_area_px2=min_area_px2,
            min_visible_fraction=min_visible_fraction,
            ink=ink,
        )
        attempts.append((final, refined, found, candidate))
    if attempts:
        attempts.sort(key=lambda item: -item[0])
        _, T_base_rig, measurements, pose = attempts[0]
    else:
        measurements, pose = [], None

    if anchor_free and pose is None:
        empty.reference = reference
        empty.facet_views = views
        empty.coverage = coverage
        empty.hypothesis_margin = margin
        empty.measurements = measurements
        empty.message = "the paint identified the rig but did not measure it: " + (
            rank_failure or "no printed boundary was found to refine against"
        )
        return empty

    if seed_T is not None and not anchor_free:
        # Everything above assumed the seed. If the answer has walked out of the
        # window that assumption is good for, the seed was stale -- a dropped
        # frame, an occlusion, the rig picked up and moved -- and the honest
        # thing is to pay for one from-scratch read rather than return a pose
        # that was steered by a stale prior.
        stale = ""
        if pose is None:
            stale = "the seeded refinement produced no fit"
        else:
            delta = np.linalg.inv(seed_T) @ T_base_rig
            step_mm = float(np.linalg.norm(delta[:3, 3])) * 1000.0
            step_deg = float(np.degrees(np.linalg.norm(_matrix_to_rodrigues(delta[:3, :3]))))
            if step_mm > float(seed_trust_radius_m) * 1000.0 or step_deg > float(seed_trust_deg):
                stale = f"the seeded read moved {step_mm:.0f} mm / {step_deg:.0f} deg off the seed"
        if stale:
            again = detect_carrier(
                image,
                camera,
                model,
                detector=detector,
                refine_iterations=refine_iterations,
                min_visible_fraction=min_visible_fraction,
                min_area_px2=min_area_px2,
                include_biased_edges=include_biased_edges,
                huber_px=huber_px,
                anchor_sigma_px=anchor_sigma_px,
                min_anchors=min_anchors,
                allow_anchor_free=allow_anchor_free,
                anchor_free_min_margin=anchor_free_min_margin,
                anchor_free_min_facets=anchor_free_min_facets,
                anchor_free_trust_radius_m=anchor_free_trust_radius_m,
                anchor_free_max_sigma_m=anchor_free_max_sigma_m,
                seed=None,
            )
            again.reacquired = True
            again.message = f"{stale}; re-acquired from scratch: {again.message}"
            return again

    T_cam_rig = camera.T_cam_base @ T_base_rig
    views = facet_views(
        model,
        camera,
        T_cam_rig,
        min_area_px2=min_area_px2,
        min_visible_fraction=min_visible_fraction,
    )
    coverage = {}
    for item in views:
        value = facet_colour_coverage(masks, item) if masks else None
        if value is not None:
            coverage[item.facet.name] = value
    return CarrierDetection(
        success=True,
        T_base_rig=T_base_rig,
        anchors=detections,
        facet_views=views,
        coverage=coverage,
        pose=pose,
        reference=reference,
        hypothesis_margin=margin,
        measurements=measurements,
        seeded=seed_T is not None,
        roi=window,
        message=(
            "seeded from paint alone; no anchor in this frame"
            if anchor_free
            else (
                f"tracked from the previous pose, {len(detections)} anchor(s)"
                if seed_T is not None
                else f"seeded from {len(detections)} anchor(s)"
            )
        ),
    )


def corner_observations(
    detection: CarrierDetection, model: HybridCarrierModel, camera_name: str
) -> list[CornerObservation]:
    """Anchor corners as plain rig-corner observations.

    Lets the existing multi-camera machinery -- ``estimate_frame_pose``, the
    uncertainty maps, the pivot fit -- consume a Hybrid Carrier frame without
    knowing anything about facets.
    """
    anchors = model.anchors_by_id
    out: list[CornerObservation] = []
    for anchor_detection in detection.anchors:
        anchor = anchors.get(int(anchor_detection.marker_id))
        if anchor is None:
            continue
        object_points = anchor.object_points(anchor_detection.quadrant)
        for index in range(4):
            out.append(
                CornerObservation(
                    camera_name=str(camera_name),
                    marker_id=int(anchor_detection.marker_id),
                    corner_index=index,
                    point_rig=object_points[index],
                    uv=np.asarray(anchor_detection.corners_uv[index], dtype=np.float64),
                )
            )
    return out
