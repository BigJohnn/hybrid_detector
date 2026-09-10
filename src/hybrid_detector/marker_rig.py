"""Corner-level bundle adjustment for a rigid multi-marker target.

This is the Level 2/3 counterpart to :mod:`metrology.uncertainty`: instead of
triangulating one point, it estimates one rigid marker-rig pose from every
observed marker corner in every camera, then computes the local CRLB around that
solution.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from hybrid_detector.calibration import CameraCalibration, sha256_file

DEFAULT_POSE_JACOBIAN_STEP = 1e-6
DEFAULT_MIN_CORNERS = 8
DEFAULT_MIN_CAMERAS = 2
CONDITION_FLOOR = 1e-10


def _skew(v: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(v, dtype=np.float64).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def _rodrigues_to_matrix(rvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    return R


def _matrix_to_rodrigues(R: np.ndarray) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(np.ascontiguousarray(np.asarray(R, dtype=np.float64).reshape(3, 3)))
    return rvec.reshape(3)


def _as_size_m(value: Any, units: str = "m") -> float:
    scale = {"m": 1.0, "meter": 1.0, "meters": 1.0, "mm": 1e-3, "cm": 1e-2}[units]
    return float(value) * scale


def _as_xyz_m(value: Any, units: str = "m") -> np.ndarray:
    scale = {"m": 1.0, "meter": 1.0, "meters": 1.0, "mm": 1e-3, "cm": 1e-2}[units]
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size == 1:
        arr = np.repeat(float(arr[0]), 3)
    if arr.size != 3:
        raise ValueError("Expected a scalar or 3-vector size")
    return arr * scale


def _local_marker_corners(size_m: float) -> np.ndarray:
    h = float(size_m) / 2.0
    return np.array(
        [
            [-h, h, 0.0],
            [h, h, 0.0],
            [h, -h, 0.0],
            [-h, -h, 0.0],
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class MarkerLayout:
    """Known 3D corner coordinates for every marker in the rig frame."""

    layout_id: str
    marker_corners: dict[int, np.ndarray]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized = {}
        for marker_id, corners in self.marker_corners.items():
            arr = np.asarray(corners, dtype=np.float64)
            if arr.shape != (4, 3):
                raise ValueError(f"marker {marker_id} corners must have shape (4, 3)")
            normalized[int(marker_id)] = arr
        object.__setattr__(self, "marker_corners", normalized)

    @property
    def marker_ids(self) -> set[int]:
        return set(self.marker_corners)

    def corners_for(self, marker_id: int) -> np.ndarray:
        return self.marker_corners[int(marker_id)]

    @classmethod
    def from_json(cls, path: Path) -> MarkerLayout:
        path = Path(path)
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        units = str(data.get("units", "m")).strip().lower()
        markers = data.get("markers", [])
        if not isinstance(markers, list) or not markers:
            raise ValueError(f"{path}: markers must be a non-empty list")
        corners_by_id: dict[int, np.ndarray] = {}
        for entry in markers:
            marker_id = int(entry["id"])
            if "corners_rig" in entry:
                corners = np.asarray(entry["corners_rig"], dtype=np.float64)
                if units != "m":
                    corners = corners * {"mm": 1e-3, "cm": 1e-2}[units]
            else:
                size_m = _as_size_m(
                    entry.get("size", entry.get("size_m")), units if "size_m" not in entry else "m"
                )
                T = np.asarray(entry["T_rig_marker"], dtype=np.float64).reshape(4, 4)
                local = _local_marker_corners(size_m)
                corners = local @ T[:3, :3].T + T[:3, 3]
            corners_by_id[marker_id] = corners
        metadata = {"source": str(path.resolve()), "sha256": sha256_file(path)}
        metadata.update({k: v for k, v in data.items() if k not in {"markers"}})
        return cls(
            layout_id=str(data.get("layout_id", path.stem)),
            marker_corners=corners_by_id,
            metadata=metadata,
        )


def cube_marker_layout(
    *,
    marker_ids: Sequence[Any],
    cube_size_m: float | Sequence[float],
    marker_size_m: float,
    handedness: str = "right_hand",
    layout_id: str = "cube",
) -> MarkerLayout:
    """Build a six-face box-marker layout in metres.

    ``marker_ids`` follows the order ``[+Z, -Z, +Y, -Y, +X, -X]``.
    Corners follow OpenCV's clockwise marker convention.
    """
    ids = list(marker_ids)
    if len(ids) != 6:
        raise ValueError("cube marker_ids must have length 6: [+Z, -Z, +Y, -Y, +X, -X]")
    cx, cy, cz = _as_xyz_m(cube_size_m, "m") / 2.0
    m = float(marker_size_m) / 2.0
    right = str(handedness).strip().lower() == "right_hand"

    def corners(orientation: str) -> np.ndarray:
        if right:
            table = {
                "+Z": [[-m, m, cz], [m, m, cz], [m, -m, cz], [-m, -m, cz]],
                "-Z": [[-m, m, -cz], [-m, -m, -cz], [m, -m, -cz], [m, m, -cz]],
                "+Y": [[m, cy, -m], [m, cy, m], [-m, cy, m], [-m, cy, -m]],
                "-Y": [[-m, -cy, -m], [-m, -cy, m], [m, -cy, m], [m, -cy, -m]],
                "+X": [[cx, m, -m], [cx, -m, -m], [cx, -m, m], [cx, m, m]],
                "-X": [[-cx, m, -m], [-cx, m, m], [-cx, -m, m], [-cx, -m, -m]],
            }
        else:
            table = {
                "+Z": [[m, m, cz], [-m, m, cz], [-m, -m, cz], [m, -m, cz]],
                "-Z": [[m, m, -cz], [-m, m, -cz], [-m, -m, -cz], [m, -m, -cz]],
                "+Y": [[m, cy, m], [-m, cy, m], [-m, cy, -m], [m, cy, -m]],
                "-Y": [[-m, -cy, -m], [-m, -cy, m], [m, -cy, m], [m, -cy, -m]],
                "+X": [[cx, -m, -m], [cx, -m, m], [cx, m, m], [cx, m, -m]],
                "-X": [[-cx, m, -m], [-cx, m, m], [-cx, -m, m], [-cx, -m, -m]],
            }
        return np.asarray(table[orientation], dtype=np.float64)

    out: dict[int, np.ndarray] = {}
    for marker_id, orientation in zip(ids, ["+Z", "-Z", "+Y", "-Y", "+X", "-X"], strict=True):
        if marker_id is None or str(marker_id).strip().lower() in {"", "none", "null"}:
            continue
        out[int(marker_id)] = corners(orientation)
    return MarkerLayout(
        layout_id=layout_id,
        marker_corners=out,
        metadata={
            "schema": "marker_layout/cube_v1",
            "marker_ids_order": ["+Z", "-Z", "+Y", "-Y", "+X", "-X"],
            "cube_size_m": _as_xyz_m(cube_size_m, "m").tolist(),
            "marker_size_m": float(marker_size_m),
            "handedness": str(handedness),
        },
    )


@dataclass(frozen=True)
class CornerObservation:
    camera_name: str
    marker_id: int
    corner_index: int
    point_rig: np.ndarray
    uv: np.ndarray


@dataclass
class FrameEstimate:
    success: bool
    T_base_rig: np.ndarray | None
    reprojection_rmse_px: float | None
    num_observations: int
    num_cameras: int
    visible_marker_ids: list[int]
    sigma_rig: np.ndarray | None = None
    sigma_tcp: np.ndarray | None = None
    sigma_tcp_max_m: float | None = None
    fim_condition: float | None = None
    per_camera_rmse_px: dict[str, float] = field(default_factory=dict)
    per_camera_pose_error_m: dict[str, float] = field(default_factory=dict)
    per_camera_pose_error_deg: dict[str, float] = field(default_factory=dict)
    # Each camera's own single-view PnP pose as [x, y, z, qx, qy, qz, qw] in the
    # same base frame as ``T_base_rig``. The BA already solves these to report
    # ``per_camera_pose_error_m``; keeping the pose rather than only its
    # magnitude is what lets a caller tell a fixed extrinsics offset (the
    # per-camera difference is explained by one SE(3) error that reproduces
    # across episodes) from detection noise (it does not).
    per_camera_pnp_pose: dict[str, list[float]] = field(default_factory=dict)
    # The marker ids *this* camera contributed corners for, which is not
    # ``visible_marker_ids`` -- that is the union over all cameras. A rig whose
    # modelled marker layout is wrong biases each camera's PnP by an amount that
    # depends on which faces that camera sees, so the per-camera set is what
    # separates a layout error (bias tracks the marker set) from a camera error
    # (bias tracks the camera).
    per_camera_marker_ids: dict[str, list[int]] = field(default_factory=dict)
    excluded_camera_names: list[str] = field(default_factory=list)
    message: str = ""


def observations_from_detections(
    *,
    camera_name: str,
    raw_detections: Iterable[dict[str, Any]],
    layout: MarkerLayout,
) -> list[CornerObservation]:
    observations: list[CornerObservation] = []
    for det in raw_detections:
        marker_id = int(det["marker_id"])
        if marker_id not in layout.marker_ids:
            continue
        uv = np.asarray(det["points_2d"], dtype=np.float64).reshape(4, 2)
        corners = layout.corners_for(marker_id)
        for corner_index in range(4):
            observations.append(
                CornerObservation(
                    camera_name=str(camera_name),
                    marker_id=marker_id,
                    corner_index=corner_index,
                    point_rig=corners[corner_index],
                    uv=uv[corner_index],
                )
            )
    return observations


def project_camera_points(camera: CameraCalibration, points_cam: np.ndarray) -> np.ndarray:
    points_cam = np.asarray(points_cam, dtype=np.float64).reshape(-1, 3)
    zeros = np.zeros((3, 1), dtype=np.float64)
    if camera.model == "fisheye":
        uv, _ = cv2.fisheye.projectPoints(
            points_cam.reshape(1, -1, 3),
            zeros,
            zeros,
            camera.K,
            np.asarray(camera.D, dtype=np.float64).reshape(4, 1),
        )
    else:
        uv, _ = cv2.projectPoints(points_cam.reshape(-1, 1, 3), zeros, zeros, camera.K, camera.D)
    return uv.reshape(-1, 2)


def project_rig_points(
    camera: CameraCalibration, T_base_rig: np.ndarray, points_rig: np.ndarray
) -> np.ndarray:
    points_rig = np.asarray(points_rig, dtype=np.float64).reshape(-1, 3)
    T_base_rig = np.asarray(T_base_rig, dtype=np.float64).reshape(4, 4)
    points_base = points_rig @ T_base_rig[:3, :3].T + T_base_rig[:3, 3]
    T_cam_base = camera.T_cam_base
    points_cam = points_base @ T_cam_base[:3, :3].T + T_cam_base[:3, 3]
    return project_camera_points(camera, points_cam)


def _solve_pnp_for_camera(
    camera: CameraCalibration, observations: Sequence[CornerObservation]
) -> np.ndarray | None:
    object_points = np.asarray([obs.point_rig for obs in observations], dtype=np.float64).reshape(
        -1, 3
    )
    image_points = np.asarray([obs.uv for obs in observations], dtype=np.float64).reshape(-1, 2)
    if object_points.shape[0] < 4:
        return None
    if camera.model == "fisheye":
        undistorted = cv2.fisheye.undistortPoints(
            image_points.reshape(-1, 1, 2),
            camera.K,
            np.asarray(camera.D, dtype=np.float64).reshape(4, 1),
        ).reshape(-1, 2)
        ok, rvec, tvec = cv2.solvePnP(
            object_points, undistorted, np.eye(3), np.zeros(5), flags=cv2.SOLVEPNP_ITERATIVE
        )
    else:
        ok, rvec, tvec = cv2.solvePnP(
            object_points, image_points, camera.K, camera.D, flags=cv2.SOLVEPNP_ITERATIVE
        )
    if not ok:
        return None
    T_cam_rig = np.eye(4, dtype=np.float64)
    T_cam_rig[:3, :3] = _rodrigues_to_matrix(rvec)
    T_cam_rig[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return camera.T_base_cam @ T_cam_rig


def _params_to_T(params: np.ndarray) -> np.ndarray:
    params = np.asarray(params, dtype=np.float64).reshape(6)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = _rodrigues_to_matrix(params[3:])
    T[:3, 3] = params[:3]
    return T


def _T_to_params(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    return np.concatenate([T[:3, 3], _matrix_to_rodrigues(T[:3, :3])])


def _left_perturb(T: np.ndarray, delta: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    delta = np.asarray(delta, dtype=np.float64).reshape(6)
    out = T.copy()
    out[:3, :3] = _rodrigues_to_matrix(delta[3:]) @ T[:3, :3]
    out[:3, 3] = T[:3, 3] + delta[:3]
    return out


def group_observations(
    observations: Sequence[CornerObservation],
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Pack observations per camera once, as arrays.

    The pose Jacobian evaluates the residual thirteen times at the same set of
    observations, so rebuilding these arrays inside the residual made the numeric
    differentiation mostly array construction. Callers that evaluate repeatedly
    (the Jacobian, and any map that sweeps poses) group once and pass it in.
    """
    grouped: dict[str, list[CornerObservation]] = defaultdict(list)
    for obs in observations:
        grouped[obs.camera_name].append(obs)
    return [
        (
            camera_name,
            np.asarray([obs.point_rig for obs in grouped[camera_name]], dtype=np.float64),
            np.asarray([obs.uv for obs in grouped[camera_name]], dtype=np.float64),
        )
        for camera_name in sorted(grouped)
    ]


def _residuals(
    cameras: dict[str, CameraCalibration],
    observations: Sequence[CornerObservation],
    T_base_rig: np.ndarray,
    *,
    weighted: bool,
    grouped: Sequence[tuple[str, np.ndarray, np.ndarray]] | None = None,
) -> np.ndarray:
    rows = []
    if grouped is None:
        grouped = group_observations(observations)
    for camera_name, points, measured in grouped:
        camera = cameras[camera_name]
        projected = project_rig_points(camera, T_base_rig, points)
        residual = projected - measured
        if weighted:
            residual = residual / float(camera.sigma_pixel)
        rows.append(residual.reshape(-1))
    return np.concatenate(rows) if rows else np.zeros(0, dtype=np.float64)


def _pose_jacobian(
    cameras: dict[str, CameraCalibration],
    observations: Sequence[CornerObservation],
    T_base_rig: np.ndarray,
    *,
    step: float = DEFAULT_POSE_JACOBIAN_STEP,
) -> np.ndarray:
    grouped = group_observations(observations)
    base = _residuals(cameras, observations, T_base_rig, weighted=True, grouped=grouped)
    J = np.empty((base.size, 6), dtype=np.float64)
    delta = np.zeros(6, dtype=np.float64)
    for axis in range(6):
        delta[:] = 0.0
        delta[axis] = float(step)
        plus = _residuals(
            cameras, observations, _left_perturb(T_base_rig, delta), weighted=True, grouped=grouped
        )
        delta[axis] = -float(step)
        minus = _residuals(
            cameras, observations, _left_perturb(T_base_rig, delta), weighted=True, grouped=grouped
        )
        J[:, axis] = (plus - minus) / (2.0 * float(step))
    return J


def covariance_from_jacobian(J: np.ndarray) -> tuple[np.ndarray | None, float | None]:
    J = np.asarray(J, dtype=np.float64)
    if J.ndim != 2 or J.shape[1] != 6 or J.shape[0] < 6:
        return None, None
    information = J.T @ J
    eigenvalues = np.linalg.eigvalsh(information)
    if eigenvalues[0] <= max(eigenvalues[-1], 1e-30) * CONDITION_FLOOR:
        return None, float("inf")
    return np.linalg.inv(information), float(eigenvalues[-1] / eigenvalues[0])


def propagate_tcp_covariance(
    sigma_rig: np.ndarray,
    T_base_rig: np.ndarray,
    tcp_in_rig_m: Sequence[float],
) -> np.ndarray:
    """Propagate ``[dt_base, dtheta_base]`` rig covariance to TCP position."""
    sigma_rig = np.asarray(sigma_rig, dtype=np.float64).reshape(6, 6)
    T_base_rig = np.asarray(T_base_rig, dtype=np.float64).reshape(4, 4)
    r_base = T_base_rig[:3, :3] @ np.asarray(tcp_in_rig_m, dtype=np.float64).reshape(3)
    A = np.concatenate([np.eye(3), -_skew(r_base)], axis=1)
    return A @ sigma_rig @ A.T


def estimate_frame_pose(
    cameras: Sequence[CameraCalibration],
    observations: Sequence[CornerObservation],
    *,
    initial_T_base_rig: np.ndarray | None = None,
    tcp_in_rig_m: Sequence[float] | None = None,
    min_cameras: int = DEFAULT_MIN_CAMERAS,
    min_corners: int = DEFAULT_MIN_CORNERS,
    huber_px: float = 2.0,
    max_nfev: int = 40,
    max_camera_rmse_px: float | None = 8.0,
) -> FrameEstimate:
    cameras_by_name = {camera.name: camera for camera in cameras}
    usable = [obs for obs in observations if obs.camera_name in cameras_by_name]
    camera_names = sorted({obs.camera_name for obs in usable})
    marker_ids = sorted({int(obs.marker_id) for obs in usable})
    if len(camera_names) < int(min_cameras) or len(usable) < int(min_corners):
        return FrameEstimate(
            success=False,
            T_base_rig=None,
            reprojection_rmse_px=None,
            num_observations=len(usable),
            num_cameras=len(camera_names),
            visible_marker_ids=marker_ids,
            message="not enough cameras or corners",
        )

    initial = initial_T_base_rig
    if initial is None:
        candidates = []
        by_camera: dict[str, list[CornerObservation]] = defaultdict(list)
        for obs in usable:
            by_camera[obs.camera_name].append(obs)
        for camera_name, obs_group in by_camera.items():
            candidate = _solve_pnp_for_camera(cameras_by_name[camera_name], obs_group)
            if candidate is None:
                continue
            residual = _residuals(cameras_by_name, usable, candidate, weighted=False)
            rmse = (
                float(np.sqrt(np.mean(residual.reshape(-1, 2) ** 2)))
                if residual.size
                else float("inf")
            )
            candidates.append((rmse, candidate))
        if not candidates:
            return FrameEstimate(
                success=False,
                T_base_rig=None,
                reprojection_rmse_px=None,
                num_observations=len(usable),
                num_cameras=len(camera_names),
                visible_marker_ids=marker_ids,
                message="no PnP initializer",
            )
        initial = min(candidates, key=lambda item: item[0])[1]

    def objective(params: np.ndarray) -> np.ndarray:
        return _residuals(cameras_by_name, usable, _params_to_T(params), weighted=True)

    solution = least_squares(
        objective,
        _T_to_params(initial),
        method="trf",
        loss="huber",
        f_scale=float(huber_px),
        max_nfev=int(max_nfev),
        x_scale="jac",
    )
    T = _params_to_T(solution.x)
    unweighted = _residuals(cameras_by_name, usable, T, weighted=False).reshape(-1, 2)
    rmse_px = (
        float(np.sqrt(np.mean(np.sum(unweighted**2, axis=1)))) if unweighted.size else float("nan")
    )

    per_camera_rmse: dict[str, float] = {}
    by_camera = defaultdict(list)
    for obs in usable:
        by_camera[obs.camera_name].append(obs)
    per_camera_pose_error_m: dict[str, float] = {}
    per_camera_pose_error_deg: dict[str, float] = {}
    per_camera_pnp_pose: dict[str, list[float]] = {}
    per_camera_marker_ids: dict[str, list[int]] = {}
    for camera_name, obs_group in by_camera.items():
        per_camera_marker_ids[camera_name] = sorted({int(obs.marker_id) for obs in obs_group})
        r = _residuals(cameras_by_name, obs_group, T, weighted=False).reshape(-1, 2)
        per_camera_rmse[camera_name] = (
            float(np.sqrt(np.mean(np.sum(r**2, axis=1)))) if r.size else float("nan")
        )
        pnp_T = _solve_pnp_for_camera(cameras_by_name[camera_name], obs_group)
        if pnp_T is not None:
            per_camera_pose_error_m[camera_name] = float(np.linalg.norm(pnp_T[:3, 3] - T[:3, 3]))
            dR = pnp_T[:3, :3].T @ T[:3, :3]
            per_camera_pose_error_deg[camera_name] = float(
                Rotation.from_matrix(dR).magnitude() * 180.0 / np.pi
            )
            per_camera_pnp_pose[camera_name] = [
                *(float(v) for v in pnp_T[:3, 3]),
                *(float(v) for v in Rotation.from_matrix(pnp_T[:3, :3]).as_quat()),
            ]

    if max_camera_rmse_px is not None:
        bad_cameras = sorted(
            name
            for name, value in per_camera_rmse.items()
            if np.isfinite(value) and value > float(max_camera_rmse_px)
        )
        if bad_cameras:
            filtered = [obs for obs in usable if obs.camera_name not in set(bad_cameras)]
            remaining_cameras = {obs.camera_name for obs in filtered}
            if len(remaining_cameras) >= int(min_cameras) and len(filtered) >= int(min_corners):
                refined = estimate_frame_pose(
                    cameras,
                    filtered,
                    initial_T_base_rig=T,
                    tcp_in_rig_m=tcp_in_rig_m,
                    min_cameras=min_cameras,
                    min_corners=min_corners,
                    huber_px=huber_px,
                    max_nfev=max_nfev,
                    max_camera_rmse_px=None,
                )
                refined.excluded_camera_names = bad_cameras
                suffix = f"excluded cameras by reprojection RMSE: {','.join(bad_cameras)}"
                refined.message = f"{refined.message}; {suffix}" if refined.message else suffix
                return refined
            return FrameEstimate(
                success=False,
                T_base_rig=T,
                reprojection_rmse_px=rmse_px,
                num_observations=len(usable),
                num_cameras=len(camera_names),
                visible_marker_ids=marker_ids,
                per_camera_rmse_px=per_camera_rmse,
                per_camera_pose_error_m=per_camera_pose_error_m,
                per_camera_pose_error_deg=per_camera_pose_error_deg,
                per_camera_pnp_pose=per_camera_pnp_pose,
                per_camera_marker_ids=per_camera_marker_ids,
                excluded_camera_names=bad_cameras,
                message=(
                    f"camera reprojection RMSE gate rejected {','.join(bad_cameras)} "
                    "but not enough cameras/corners remain"
                ),
            )

    J = _pose_jacobian(cameras_by_name, usable, T)
    sigma_rig, condition = covariance_from_jacobian(J)
    sigma_tcp = None
    sigma_tcp_max = None
    if sigma_rig is not None and tcp_in_rig_m is not None:
        sigma_tcp = propagate_tcp_covariance(sigma_rig, T, tcp_in_rig_m)
        sigma_tcp_max = float(np.sqrt(np.max(np.linalg.eigvalsh(sigma_tcp))))

    return FrameEstimate(
        success=bool(solution.success),
        T_base_rig=T,
        reprojection_rmse_px=rmse_px,
        num_observations=len(usable),
        num_cameras=len(camera_names),
        visible_marker_ids=marker_ids,
        sigma_rig=sigma_rig,
        sigma_tcp=sigma_tcp,
        sigma_tcp_max_m=sigma_tcp_max,
        fim_condition=condition,
        per_camera_rmse_px=per_camera_rmse,
        per_camera_pose_error_m=per_camera_pose_error_m,
        per_camera_pose_error_deg=per_camera_pose_error_deg,
        per_camera_pnp_pose=per_camera_pnp_pose,
        per_camera_marker_ids=per_camera_marker_ids,
        message=solution.message,
    )
