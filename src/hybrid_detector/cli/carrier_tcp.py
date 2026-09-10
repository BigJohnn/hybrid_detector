"""Carrier -> TCP constant, read off the CAD instead of fitted from a capture.

The marker cube needed a pivot capture for this number because nothing in its
CAD knows where the gripper's TCP is: the cube is bolted to the device, the TCP
is 247 mm away down a chain the cube frame cannot see, and the only way to tie
them together is to seat a datum ball and sweep.

This rig does not have that problem.  Its part studio puts the **seating socket
sphere centre on the part origin**, and that socket centre is the TCP -- it is
the one point on the rig whose position is fixed by a seated ball rather than
by the printed skin.  So the carrier -> TCP translation is not a measurement
waiting to be taken, it is zero by construction, and the honest work is to
verify the CAD really says that and to put a defensible sigma on it.

    python -m hybrid_detector.cli.carrier_tcp \
        assets/models/hybrid_carrier_v1_20260907.json \
        --out .../marker_to_tcp_calibration_hybrid_carrier_v1_20260909.json \
        --translation-sigma-mm 0.7 --rotation-sigma-deg 1.0

The emitted file is a ``marker_rig_to_tcp_calibration/v1`` bundle -- the same
schema and the same ``cubes`` key the production ``ee_from_cube`` loader reads,
so a carrier run consumes it through the code path that is already tested.

Rotation
--------
A point never determines a rotation, and no amount of CAD changes that on its
own; what CAD *does* give here is the physical role of each axis:

* the socket opens along ``-z_rig``, so a ball seats into the rig from ``-z``.
  On the gripper the equivalent direction -- body, through TCP_closed, out to
  the table -- is ``-z_tcp``.  That pins ``z_tcp = z_rig``.
* the remaining roll about that axis is the finger-closing direction on a
  gripper.  This rig has no fingers, so the only distinguished direction left
  in its xy-plane is the boom, ``+x_rig``, and ``x_tcp = x_rig`` pins it.

Both together are the identity, which is the default.  It is a *convention*
pinned to two physical facts, not a measurement, and ``--rotation-rig-tcp``
overrides it -- pass one deliberately before mixing carrier episodes into a
dataset whose other episodes were labelled through a cube bundle, because the
orientation convention has to be the same one on both.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "marker_rig_to_tcp_calibration/v1"
# Above this the socket is not on the origin and the whole premise is gone: the
# rig frame is then not the TCP frame, and emitting a zero translation would be
# writing a constant that CAD contradicts.
ORIGIN_TOLERANCE_MM = 0.05


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_rotation(text: str) -> np.ndarray:
    rows = [row for row in str(text).split(";") if row.strip()]
    matrix = np.array([[float(x) for x in row.split(",")] for row in rows], dtype=np.float64)
    if matrix.shape != (3, 3):
        raise SystemExit(f"--rotation-rig-tcp must be 3x3, got {matrix.shape}")
    if abs(float(np.linalg.det(matrix)) - 1.0) > 1e-6:
        raise SystemExit(
            f"--rotation-rig-tcp is not a proper rotation (det={float(np.linalg.det(matrix)):.6f})"
        )
    if not np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-6):
        raise SystemExit("--rotation-rig-tcp is not orthonormal")
    return matrix


def socket_centre_m(model: dict[str, Any], descriptor: Path) -> np.ndarray:
    """The seating socket centre, in metres, or a refusal saying why not."""
    pivot = model.get("pivot")
    if not isinstance(pivot, dict) or pivot.get("socket_sphere_centre_mm") is None:
        raise SystemExit(
            f"{descriptor}: no pivot socket in this descriptor. Rebuild it with a "
            "build_hybrid_carrier_model new enough to record one -- without a socket the rig has "
            "no CAD-known TCP and this tool has nothing to emit."
        )
    centre_mm = np.asarray(pivot["socket_sphere_centre_mm"], dtype=np.float64).reshape(3)
    offset_mm = float(np.linalg.norm(centre_mm))
    if offset_mm > ORIGIN_TOLERANCE_MM:
        raise SystemExit(
            f"{descriptor}: the socket sphere centre is {offset_mm:.3f} mm off the part origin, past "
            f"the {ORIGIN_TOLERANCE_MM} mm this tool accepts. The rig frame is then not the TCP frame, "
            "so the constant is that offset and not zero -- which is a CAD change to make deliberately, "
            "not something to absorb here."
        )
    return centre_mm / 1e3


def build_bundle(
    model: dict[str, Any],
    descriptor: Path,
    *,
    rig_name: str,
    R_rig_tcp: np.ndarray,
    translation_sigma_mm: float,
    rotation_sigma_deg: float,
    rotation_source: str,
    open_items: list[str],
    validated: bool,
) -> dict[str, Any]:
    centre = socket_centre_m(model, descriptor)
    T_rig_tcp = np.eye(4, dtype=np.float64)
    T_rig_tcp[:3, :3] = R_rig_tcp
    T_rig_tcp[:3, 3] = centre
    pivot = model["pivot"]
    carrier_id = str(model.get("carrier_id") or descriptor.stem)
    return {
        "schema": SCHEMA,
        "calibration_id": f"cad_{carrier_id}_socket_centre",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "validated": bool(validated),
        # `cubes` is the key the production ee_from_cube loader reads. This rig is
        # not a cube; keeping the key is what lets a carrier run go through the
        # loader that is already tested rather than a second copy of it.
        "cubes": {
            rig_name: {
                "device_id": carrier_id,
                "T_rig_tcp_is_T_cube_tcp": (
                    "same field, different rig: the tracked frame here is the carrier's CAD frame"
                ),
                "T_cube_tcp": T_rig_tcp.tolist(),
                "tcp_link": "carrier_socket_centre (the seated ball centre; no URDF chain involved)",
                "translation_sigma_mm": [float(translation_sigma_mm)] * 3,
                "rotation_sigma_deg": float(rotation_sigma_deg),
                "translation_source": (
                    "CAD: the seating socket sphere centre sits on the part-studio origin "
                    f"({pivot['socket_sphere_radius_mm']} mm radius, STEP face "
                    f"#{pivot.get('socket_face_entity_id')}), so the rig frame is the TCP frame and "
                    "the translation is zero by construction, not by fit"
                ),
                "rotation_source": rotation_source,
                "mount": {
                    "socket_sphere_radius_mm": pivot["socket_sphere_radius_mm"],
                    "socket_centre_offset_from_origin_mm": pivot.get("offset_from_origin_mm"),
                    "socket_centre_beyond_tcp_closed_mm": 0.0,
                    "note": (
                        "no gripper in this chain. The cube bundles carry this field because a "
                        "seated ball sits some distance beyond the closed fingers; here the seated "
                        "ball centre is the TCP itself, so the offset is zero by definition rather "
                        "than by a caliper."
                    ),
                },
                "source_descriptor": {
                    "path": str(descriptor),
                    "sha256": sha256_of(descriptor),
                    "carrier_id": carrier_id,
                    "step_file": (model.get("source") or {}).get("step_file"),
                    "step_sha256": (model.get("source") or {}).get("sha256"),
                },
                "open_items": open_items,
            }
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("descriptor", type=Path, help="a hybrid_carrier_cad/v1 JSON")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--rig-name",
        default=None,
        help="the name the tracking run uses for this rig (default: the descriptor's carrier_id)",
    )
    parser.add_argument(
        "--rotation-rig-tcp",
        default="1,0,0;0,1,0;0,0,1",
        help="R_rig_tcp as 'r00,r01,r02;r10,...'. Default identity -- see the module docstring.",
    )
    parser.add_argument(
        "--translation-sigma-mm",
        type=float,
        required=True,
        help=(
            "1-sigma on the socket centre in the tracked frame. This is NOT the socket's CAD "
            "position, which is exact; it is how far the built part's skin has moved relative to "
            "its socket -- print shrinkage over the ~137 mm from the socket to the painted head is "
            "the dominant term. Measure it with `carrier_scale_check`, do not guess it."
        ),
    )
    parser.add_argument("--rotation-sigma-deg", type=float, default=1.0)
    parser.add_argument(
        "--rotation-source",
        default=(
            "convention, not measurement: z_tcp = z_rig because the socket opens along -z_rig and "
            "a ball seats from that side, which is the gripper's body->TCP->table direction -z_tcp; "
            "x_tcp = x_rig because the boom is the only distinguished direction left in the plane. "
            "A rig with no fingers has no measured roll to inherit."
        ),
    )
    parser.add_argument(
        "--open-item",
        action="append",
        default=[],
        dest="open_items",
        metavar="TEXT",
        help="repeatable; recorded in the bundle and printed by the tracker on every run",
    )
    parser.add_argument(
        "--validated",
        action="store_true",
        help="only after the open items are closed; leaving it off makes the tracker warn every run",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model = json.loads(args.descriptor.read_text(encoding="utf-8"))
    if not str(model.get("schema", "")).startswith("hybrid_carrier_cad/"):
        raise SystemExit(f"{args.descriptor}: unexpected schema {model.get('schema')!r}")
    open_items = list(args.open_items) or [
        "the rotation is a convention pinned to the seating direction and the boom, not a measurement; "
        "pin it deliberately before mixing carrier episodes with cube-labelled ones",
    ]
    bundle = build_bundle(
        model,
        args.descriptor,
        rig_name=str(args.rig_name or model.get("carrier_id")),
        R_rig_tcp=parse_rotation(args.rotation_rig_tcp),
        translation_sigma_mm=float(args.translation_sigma_mm),
        rotation_sigma_deg=float(args.rotation_sigma_deg),
        rotation_source=str(args.rotation_source),
        open_items=open_items,
        validated=bool(args.validated),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
    name = next(iter(bundle["cubes"]))
    print(f"wrote {args.out}")
    print(f"  rig {name!r}: translation {np.zeros(3)} m (socket centre on the part origin)")
    print(
        f"  sigma {args.translation_sigma_mm} mm / {args.rotation_sigma_deg} deg, validated={bundle['validated']}"
    )


if __name__ == "__main__":
    main()
