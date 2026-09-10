"""Public API for the standalone Hybrid Detector package."""

from hybrid_detector.calibration import CameraCalibration
from hybrid_detector.detector import (
    Anchor,
    AnchorDetection,
    CarrierDetection,
    CarrierEdge,
    CarrierView,
    ColourReference,
    EdgeMeasurement,
    Facet,
    HybridCarrierModel,
    detect_carrier,
    solve_carrier_pose,
)

__all__ = [
    "Anchor",
    "AnchorDetection",
    "CameraCalibration",
    "CarrierDetection",
    "CarrierEdge",
    "CarrierView",
    "ColourReference",
    "EdgeMeasurement",
    "Facet",
    "HybridCarrierModel",
    "detect_carrier",
    "solve_carrier_pose",
]
__version__ = "0.1.0"
