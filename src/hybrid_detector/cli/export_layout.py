"""Carrier descriptor -> the ``marker_layout.json`` the production tracker reads.

``CubeSpec.marker_layout_path`` is the seam the production tracker documents for
a rig that is not a cube: set it and the analytic cube model is not used at all,
and the corner bundle adjustment solves against the corner table in that file
instead.  This writes that table for a Hybrid Carrier, so the rig swaps into the
tracker through a path that is already tested rather than a second estimator.

    python -m hybrid_detector.cli.export_layout \
        assets/models/hybrid_carrier_v1_20260907.json \
        --out .../hybrid_carrier_v1_20260907_marker_layout.json

Two things it refuses to leave implicit, because both are silent when wrong:

* **the sticker rotation.**  A layout is a table of corner coordinates, so the
  paste quadrant has to be baked into it.  An unmeasured quadrant is a 90 degree
  error that produces a confident pose, so an unpinned anchor is refused here
  rather than exported at quadrant 0.
* **the dictionary.**  The layout declares ``aruco_dictionary``, and the tracker
  refuses a config that claims a different one -- swapping the rig without
  swapping the dictionary decodes every id as something else.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from hybrid_detector.detector import HybridCarrierModel


def build_layout(model: HybridCarrierModel, *, layout_id: str, source: Path) -> dict:
    unpinned = [int(a.marker_id) for a in model.anchors if a.paste_quadrant is None]
    if unpinned:
        raise SystemExit(
            f"anchors {unpinned} have no measured paste_quadrant. A corner table has to state where "
            "the sticker's first corner is, and guessing it is a 90 degree pose error that the solve "
            "reports as a success. Measure the quadrants and rebuild the descriptor with "
            "--paste-quadrant, then export."
        )
    dictionaries = {a.dictionary for a in model.anchors}
    if len(dictionaries) != 1:
        raise SystemExit(
            f"anchors span several dictionaries {sorted(dictionaries)}; a layout carries one"
        )
    return {
        "layout_id": layout_id,
        "units": "m",
        "aruco_dictionary": dictionaries.pop(),
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": {
            "descriptor": str(source),
            "carrier_id": model.carrier_id,
            "frame": (
                "the carrier's CAD frame: origin on the seating socket sphere centre, which is "
                "also this rig's TCP"
            ),
        },
        "note": (
            "anchor corners only. The painted facets carry pose too, but as point-to-line samples "
            "on boundaries rather than as corners, and a marker layout has no way to express one -- "
            "they enter through the carrier solve, not through this table."
        ),
        "markers": [
            {
                "id": int(anchor.marker_id),
                "name": anchor.name,
                "paste_quadrant": int(anchor.paste_quadrant),
                "size_m": float(anchor.marker_size_m),
                "corners_rig": np.asarray(
                    anchor.object_points(int(anchor.paste_quadrant))
                ).tolist(),
            }
            for anchor in model.anchors
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("descriptor", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--layout-id", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model = HybridCarrierModel.from_json(args.descriptor)
    layout = build_layout(
        model, layout_id=str(args.layout_id or model.carrier_id), source=args.descriptor
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(layout, indent=2) + "\n", encoding="utf-8")
    ids = [m["id"] for m in layout["markers"]]
    print(f"wrote {args.out}: {len(ids)} markers {ids} in {layout['aruco_dictionary']}")


if __name__ == "__main__":
    main()
