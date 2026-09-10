"""Load the calibration products the production tracking pipeline consumes.

The calibration contract is explicit so uncertainty follows the measured rig:

* ``joint_solution.cameras.<name>.base_to_camera.matrix_4x4`` is ``T_base_cam``,
  i.e. it maps camera-frame coordinates into the robot base frame. The joint
  solve is preferred whenever it reports ``status: ok``; the per-camera solve is
  the fallback, exactly as ``load_camera_poses_in_base()`` does.
* Intrinsics are the vendor "producer" JSON: a 3x3 ``camera_matrix`` plus an
  8-coefficient OpenCV rational ``dist_coeffs`` (k1 k2 p1 p2 k3 k4 k5 k6).

Every loaded file is hashed into the provenance record. Tracking summaries
currently reference calibrations *by run name only*, so editing a calibration
directory without renaming it goes unnoticed; anything this package emits
carries content hashes instead.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# The world frame of this rig is the FR3 robot base. Named here once so callers
# and plots do not have to re-guess it.
WORLD_FRAME = "robot_base"

DEFAULT_SIGMA_PIXEL = 0.2

# The names a producer JSON may carry for the same two distortion models.
_MODEL_ALIASES = {
    "": "rational",
    "pinhole": "rational",
    "opencv_rational": "rational",
    "equidistant": "fisheye",
    "opencv_fisheye": "fisheye",
}


def normalize_camera_model(model: str) -> str:
    """Map a producer JSON's model name onto ``rational`` or ``fisheye``.

    Shared so that every caller reading an intrinsics file agrees on which of
    the two distortion models the four or eight coefficients belong to.
    Guessing it instead -- assuming ``rational`` because that is the common case
    -- silently reinterprets a fisheye D as the first four rational
    coefficients, which is not a worse fit but a different lens.
    """
    normalized = _MODEL_ALIASES.get(str(model).strip().lower(), str(model).strip().lower())
    if normalized not in {"rational", "fisheye"}:
        raise ValueError(f"Unsupported camera model: {model!r}")
    return normalized


@dataclass
class CameraCalibration:
    """One fixed camera: intrinsics, distortion, and pose in the world frame."""

    name: str
    serial: str
    width: int
    height: int
    K: np.ndarray  # (3, 3)
    D: np.ndarray  # (N,) OpenCV distortion coefficients
    T_base_cam: np.ndarray  # (4, 4), maps camera coords -> base coords
    model: str = "rational"
    sigma_pixel: float = DEFAULT_SIGMA_PIXEL

    def __post_init__(self) -> None:
        self.model = normalize_camera_model(self.model)

    @property
    def T_cam_base(self) -> np.ndarray:
        """Inverse pose: maps base coords -> camera coords."""
        R = self.T_base_cam[:3, :3]
        t = self.T_base_cam[:3, 3]
        out = np.eye(4, dtype=np.float64)
        out[:3, :3] = R.T
        out[:3, 3] = -R.T @ t
        return out

    @property
    def rvec(self) -> np.ndarray:
        """Rodrigues vector of ``T_cam_base``, as ``cv2.projectPoints`` wants."""
        rvec, _ = cv2.Rodrigues(np.ascontiguousarray(self.T_cam_base[:3, :3]))
        return rvec.astype(np.float64).reshape(3, 1)

    @property
    def tvec(self) -> np.ndarray:
        return self.T_cam_base[:3, 3].astype(np.float64).reshape(3, 1)

    @property
    def position_in_base(self) -> np.ndarray:
        return self.T_base_cam[:3, 3].copy()

    @property
    def max_normalized_radius(self) -> float:
        """Largest undistorted normalized radius still inside the image.

        Used as a field-of-view guard. A rational distortion model can fold
        far-off-axis points back into the image, so an "is the projection inside
        the image rectangle" test alone would report such points as visible.

        Two traps, both hit by the GMSL2 vendor intrinsics:

        * ``cv2.undistortPoints`` inverts the model with a *fixed five*
          Gauss-Newton steps, which does not converge on 120-degree lenses --
          on these cameras it lands 35-55 px off at the image corners. Hence
          ``undistortPointsIter`` with an explicit termination criterion.
        * some fits are not monotonic across the frame: ``cam_08`` peaks at
          51.7 degrees while its corner sits at 70.4, so pixels beyond radius
          845 (18% of the frame) have no unique ray at all. Inverting there is
          meaningless, so the forward model's own monotonic limit caps the
          result.
        """
        forward_limit = self._monotonic_radius_limit()

        corners = np.array(
            [
                [0.0, 0.0],
                [self.width - 1.0, 0.0],
                [0.0, self.height - 1.0],
                [self.width - 1.0, self.height - 1.0],
            ],
            dtype=np.float64,
        ).reshape(-1, 1, 2)
        criteria = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 100, 1e-12)
        try:
            if self.model == "fisheye":
                normalized = cv2.fisheye.undistortPoints(
                    corners, self.K, np.asarray(self.D, dtype=np.float64).reshape(4, 1)
                )
            else:
                normalized = cv2.undistortPointsIter(corners, self.K, self.D, None, None, criteria)
            radius = float(np.max(np.linalg.norm(normalized.reshape(-1, 2), axis=1)))
            if np.isfinite(radius) and radius > 0.0:
                return min(radius, forward_limit)
        except cv2.error:  # pragma: no cover - defensive, model failed to invert
            pass
        # Fall back to an undistorted pinhole estimate of the corner radius.
        fx, fy = float(self.K[0, 0]), float(self.K[1, 1])
        cx, cy = float(self.K[0, 2]), float(self.K[1, 2])
        dx = max(cx, self.width - 1.0 - cx) / max(fx, 1e-9)
        dy = max(cy, self.height - 1.0 - cy) / max(fy, 1e-9)
        return min(float(np.hypot(dx, dy)), forward_limit)

    def _monotonic_radius_limit(self, max_angle_deg: float = 85.0, samples: int = 20000) -> float:
        """Normalized radius at which the radial map stops increasing.

        Purely forward, so it needs no inversion and cannot itself be fooled by
        a non-convergent solver. Beyond this radius the model is multi-valued
        and any point mapping there must be treated as unobservable.
        """
        angles_all = np.radians(np.linspace(1e-3, max_angle_deg, samples))
        if self.model == "fisheye":
            # Equidistant: r = theta + k1 t^3 + k2 t^5 + k3 t^7 + k4 t^9. The
            # radius that matters is still tan(theta), so convert at the end.
            k = np.asarray(self.D, dtype=np.float64).reshape(-1)[:4]
            mapped = (
                angles_all
                + k[0] * angles_all**3
                + k[1] * angles_all**5
                + k[2] * angles_all**7
                + k[3] * angles_all**9
            )
            decreasing_fe = np.flatnonzero(np.diff(mapped) < 0.0)
            if decreasing_fe.size == 0:
                return float("inf")
            return float(np.tan(angles_all[decreasing_fe[0]]))

        coefficients = np.asarray(self.D, dtype=np.float64).reshape(-1)
        padded = np.zeros(8, dtype=np.float64)
        padded[: min(8, coefficients.size)] = coefficients[: min(8, coefficients.size)]
        k1, k2, _p1, _p2, k3, k4, k5, k6 = padded

        angles = np.radians(np.linspace(1e-3, max_angle_deg, samples))
        radius = np.tan(angles)
        r2 = radius**2
        numerator = 1.0 + k1 * r2 + k2 * r2**2 + k3 * r2**3
        denominator = 1.0 + k4 * r2 + k5 * r2**2 + k6 * r2**3
        with np.errstate(divide="ignore", invalid="ignore"):
            distorted = radius * numerator / denominator

        finite = np.isfinite(distorted)
        if not finite.all():
            first_bad = int(np.argmin(finite))
            if first_bad == 0:
                return float("inf")
            radius, distorted = radius[:first_bad], distorted[:first_bad]

        decreasing = np.flatnonzero(np.diff(distorted) < 0.0)
        if decreasing.size == 0:
            return float("inf")
        return float(radius[decreasing[0]])


@dataclass
class RigProvenance:
    """What was loaded, and a content hash of each file."""

    world_frame: str = WORLD_FRAME
    extrinsics_summary: str = ""
    extrinsics_source: str = ""  # "joint_solution" or "per_camera"
    intrinsics_dir: str = ""
    files: dict[str, str] = field(default_factory=dict)  # path -> sha256
    cameras: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "world_frame": self.world_frame,
            "extrinsics_summary": self.extrinsics_summary,
            "extrinsics_source": self.extrinsics_source,
            "intrinsics_dir": self.intrinsics_dir,
            "cameras": list(self.cameras),
            "file_sha256": dict(self.files),
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def resolve_extrinsics_summary(path: Path) -> Path:
    """Accept either the summary file or the directory that contains it."""
    path = Path(path).expanduser()
    if path.is_dir():
        candidate = path / "summary.json"
        if not candidate.exists():
            raise FileNotFoundError(f"No summary.json in extrinsics directory: {path}")
        return candidate.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Extrinsics summary not found: {path}")
    return path.resolve()


def load_camera_poses_in_base(
    summary_path: Path, use_joint_solution: bool = True
) -> tuple[dict[str, np.ndarray], str]:
    """Return ``{camera_name: T_base_cam}`` plus which solve it came from."""
    summary = _load_json(summary_path)

    if use_joint_solution:
        joint = summary.get("joint_solution", {}) or {}
        if str(joint.get("status", "")).strip().lower() == "ok":
            poses = _poses_from_camera_map(joint.get("cameras", {}) or {}, require_status_ok=False)
            if poses:
                return poses, "joint_solution"

    poses = _poses_from_camera_map(summary.get("cameras", {}) or {}, require_status_ok=True)
    if not poses:
        raise RuntimeError(f"No usable camera poses in: {summary_path}")
    return poses, "per_camera"


def _poses_from_camera_map(cams: dict[str, Any], require_status_ok: bool) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for camera_name, info in (cams or {}).items():
        info = info or {}
        if require_status_ok and str(info.get("status", "")).strip().lower() != "ok":
            continue
        matrix = np.asarray(
            (info.get("base_to_camera", {}) or {}).get("matrix_4x4", []), dtype=np.float64
        )
        if matrix.shape != (4, 4):
            continue
        out[str(camera_name)] = matrix
    return out


def find_intrinsics_json(intrinsics_dir: Path, camera_name: str) -> Path | None:
    """Locate ``converted/<camera_name>_<serial>/intrinsics_producer.json``."""
    converted = intrinsics_dir / "converted"
    search_roots = [converted, intrinsics_dir]
    for root in search_roots:
        if not root.is_dir():
            continue
        matches = sorted(root.glob(f"{camera_name}_*/intrinsics_producer.json"))
        if matches:
            return matches[0].resolve()
        direct = root / camera_name / "intrinsics_producer.json"
        if direct.exists():
            return direct.resolve()
    return None


def load_intrinsics_json(path: Path) -> dict[str, Any]:
    data = _load_json(path)
    K = np.asarray(data.get("camera_matrix", []), dtype=np.float64)
    if K.shape != (3, 3):
        raise ValueError(f"camera_matrix is not 3x3 in: {path}")
    D = np.asarray(data.get("dist_coeffs", []), dtype=np.float64).reshape(-1)
    if D.size not in (4, 5, 8, 12, 14):
        raise ValueError(f"Unsupported dist_coeffs length {D.size} in: {path}")
    width = int(data.get("image_width", 0))
    height = int(data.get("image_height", 0))
    if width <= 0 or height <= 0:
        raise ValueError(f"Missing image size in: {path}")
    return {
        "K": K,
        "D": D,
        "width": width,
        "height": height,
        "serial": str(data.get("camera_serial", "")).strip(),
        # The logical name, which is what every other stage keys on. Directory
        # names carry it too but glued to the serial, so parsing them back is a
        # guess about serial formats that this field makes unnecessary.
        "name": str(data.get("camera_name", "")).strip(),
        "model": str(data.get("model", "")).strip() or "opencv_rational",
    }


def load_rig(
    extrinsics: Path,
    intrinsics_dir: Path,
    *,
    use_joint_solution: bool = True,
    cameras: Sequence[str] | None = None,
    sigma_pixel: float = DEFAULT_SIGMA_PIXEL,
    sigma_pixel_by_camera: dict[str, float] | None = None,
) -> tuple[list[CameraCalibration], RigProvenance]:
    """Load every fixed camera as a :class:`CameraCalibration`.

    ``sigma_pixel`` is the assumed per-corner measurement noise. It is a stated
    assumption, not a measurement -- see the roadmap note about not leaving it
    hard-coded at 0.2 px forever.
    """
    summary_path = resolve_extrinsics_summary(Path(extrinsics))
    intrinsics_dir = Path(intrinsics_dir).expanduser().resolve()

    poses, source = load_camera_poses_in_base(summary_path, use_joint_solution=use_joint_solution)

    wanted = list(cameras) if cameras else sorted(poses)
    missing_pose = [name for name in wanted if name not in poses]
    if missing_pose:
        raise KeyError(f"No extrinsics for requested cameras: {missing_pose}")

    provenance = RigProvenance(
        extrinsics_summary=str(summary_path),
        extrinsics_source=source,
        intrinsics_dir=str(intrinsics_dir),
    )
    provenance.files[str(summary_path)] = sha256_file(summary_path)

    per_camera_sigma = dict(sigma_pixel_by_camera or {})
    rig: list[CameraCalibration] = []
    for name in wanted:
        intr_path = find_intrinsics_json(intrinsics_dir, name)
        if intr_path is None:
            raise FileNotFoundError(
                f"No intrinsics_producer.json for {name} under {intrinsics_dir}"
            )
        intr = load_intrinsics_json(intr_path)
        provenance.files[str(intr_path)] = sha256_file(intr_path)
        rig.append(
            CameraCalibration(
                name=name,
                serial=intr["serial"],
                width=intr["width"],
                height=intr["height"],
                K=intr["K"],
                D=intr["D"],
                T_base_cam=poses[name],
                model=intr["model"],
                sigma_pixel=float(per_camera_sigma.get(name, sigma_pixel)),
            )
        )

    provenance.cameras = [cam.name for cam in rig]
    return rig, provenance


def load_rig_from_reports(
    intrinsics_report: Path,
    extrinsics_report: Path,
    *,
    model: str = "rational",
    sigma_pixel: float = DEFAULT_SIGMA_PIXEL,
    sigma_pixel_by_camera: dict[str, float] | None = None,
) -> tuple[list[CameraCalibration], RigProvenance]:
    """Load a rig from this package's own self-calibration output.

    The world frame here is whatever the bundle adjustment used as its gauge --
    currently one of the cameras, not the robot base. That makes the axes far
    less interpretable than the old base-frame calibration, which is exactly the
    problem the anchor-plate world frame is meant to solve.
    """
    intrinsics_report = Path(intrinsics_report).expanduser().resolve()
    extrinsics_report = Path(extrinsics_report).expanduser().resolve()
    intrinsics = _load_json(intrinsics_report)
    extrinsics = _load_json(extrinsics_report)

    reference = str(extrinsics.get("reference", ""))
    poses = {k: np.asarray(v, dtype=np.float64) for k, v in extrinsics["T_ref_cam"].items()}

    provenance = RigProvenance(
        world_frame=f"bundle gauge = {reference}",
        extrinsics_summary=str(extrinsics_report),
        extrinsics_source="bundle_adjustment",
        intrinsics_dir=str(intrinsics_report),
    )
    provenance.files[str(intrinsics_report)] = sha256_file(intrinsics_report)
    provenance.files[str(extrinsics_report)] = sha256_file(extrinsics_report)

    per_camera_sigma = dict(sigma_pixel_by_camera or {})
    rig: list[CameraCalibration] = []
    for name in sorted(poses):
        entry = intrinsics["cameras"].get(name)
        if entry is None:
            raise KeyError(f"No intrinsics for {name} in {intrinsics_report}")
        block = entry["models"].get(model)
        if not block or "K" not in block:
            raise KeyError(f"No {model} intrinsics for {name}")
        width, height = entry["image_size"]
        rig.append(
            CameraCalibration(
                name=name,
                serial=name,
                width=int(width),
                height=int(height),
                K=np.asarray(block["K"], dtype=np.float64),
                D=np.asarray(block["D"], dtype=np.float64).reshape(-1),
                T_base_cam=poses[name],
                model=model,
                sigma_pixel=float(per_camera_sigma.get(name, sigma_pixel)),
            )
        )

    provenance.cameras = [c.name for c in rig]
    return rig, provenance
