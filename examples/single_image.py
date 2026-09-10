#!/usr/bin/env python3
"""Run the included real-image example without any external data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from hybrid_detector.cli.detect import camera_from_json, detection_report, draw_overlay
from hybrid_detector.detector import HybridCarrierModel, detect_carrier

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "assets" / "models" / "hybrid_carrier_v1_20260907.json"
IMAGE = ROOT / "examples" / "data" / "cam14_ep1_frame000000.jpg"
CAMERA = ROOT / "examples" / "data" / "cam14_intrinsics.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "example")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image = cv2.imread(str(IMAGE), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(IMAGE)

    model = HybridCarrierModel.from_json(MODEL)
    camera = camera_from_json("cam14", CAMERA, image.shape[1], image.shape[0])
    detection = detect_carrier(image, camera, model, min_anchors=2)

    report = detection_report(detection, model, camera.name)
    report["model"] = str(MODEL.relative_to(ROOT))
    report["image"] = str(IMAGE.relative_to(ROOT))
    report["camera_file"] = str(CAMERA.relative_to(ROOT))
    report["T_camera_target"] = (
        None
        if detection.T_base_rig is None
        else np.asarray(detection.T_base_rig, dtype=np.float64).round(12).tolist()
    )

    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    overlay = image
    if detection.T_base_rig is not None:
        overlay = draw_overlay(image, model, camera, detection, detection.T_base_rig)
    overlay_path = args.output / "overlay.jpg"
    if not cv2.imwrite(str(overlay_path), overlay):
        raise RuntimeError(f"failed to write {overlay_path}")

    print(
        json.dumps(
            {
                "success": detection.success,
                "message": detection.message,
                "anchors": [item.marker_id for item in detection.anchors],
                "edge_samples": len(detection.measurements),
                "report": str(report_path),
                "overlay": str(overlay_path),
            },
            indent=2,
        )
    )
    return 0 if detection.success else 2


if __name__ == "__main__":
    raise SystemExit(main())
