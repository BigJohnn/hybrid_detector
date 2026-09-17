"""Paired capture evaluation: keep failures, references and evidence distinct."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

METHODS = ("anchors", "colour", "full")


def validate_pose(value: Any, name: str) -> np.ndarray:
    pose = np.asarray(value, dtype=float)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError(f"{name}: expected a finite 4x4 transform")
    if not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-7):
        raise ValueError(f"{name}: invalid homogeneous row")
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-6
    ):
        raise ValueError(f"{name}: rotation must be in SO(3)")
    return pose


def pose_difference(expected: Any, actual: Any) -> dict[str, float]:
    a, b = np.asarray(expected), np.asarray(actual)
    return {
        "translation_mm": float(np.linalg.norm(b[:3, 3] - a[:3, 3]) * 1000),
        "rotation_deg": float(
            np.degrees(Rotation.from_matrix(a[:3, :3].T @ b[:3, :3]).magnitude())
        ),
    }


def distribution(values: Sequence[float]) -> dict[str, float | int] | None:
    if not len(values):
        return None
    values = np.asarray(values, dtype=float)
    return {
        "n": len(values),
        "p50": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def paired_metric(
    rows: list[dict], method: str, metric: str, *, repetitions: int, seed: int
) -> dict:
    """Resample entire acquisition groups, never individual camera/video rows."""
    pairs = []
    for row in rows:
        base, candidate = row["methods"]["anchors"], row["methods"][method]
        if base["success"] and candidate["success"]:
            left, right = base.get(metric), candidate.get(metric)
            if left is not None and right is not None:
                pairs.append((row["group"], float(left), float(right)))
    if not pairs:
        return {"paired_n": 0, "group_n": 0, "cluster_bootstrap_p90_delta_95pct": None}
    groups = sorted({p[0] for p in pairs})
    left, right = np.asarray([(p[1], p[2]) for p in pairs]).T
    p90_left, p90_right = float(np.percentile(left, 90)), float(np.percentile(right, 90))
    interval = None
    if len(groups) >= 2 and repetitions:
        clusters = [np.asarray([(p[1], p[2]) for p in pairs if p[0] == group]) for group in groups]
        rng = np.random.default_rng(seed)
        deltas = []
        for _ in range(repetitions):
            sample = np.concatenate(
                [clusters[i] for i in rng.integers(len(groups), size=len(groups))]
            )
            deltas.append(float(np.percentile(sample[:, 1], 90) - np.percentile(sample[:, 0], 90)))
        interval = np.percentile(deltas, [2.5, 97.5]).tolist()
    return {
        "paired_n": len(pairs),
        "group_n": len(groups),
        "anchors_p90": p90_left,
        "candidate_p90": p90_right,
        "p90_delta": p90_right - p90_left,
        "p90_reduction_fraction": 1 - p90_right / p90_left if p90_left > 0 else None,
        "candidate_better_fraction": float(np.mean(right < left)),
        "cluster_bootstrap_p90_delta_95pct": interval,
        "interval_note": "Descriptive only with few independent acquisition groups; not frame-wise replication.",
    }


def summarize(rows: list[dict], config: dict, *, bootstrap: bool = True) -> dict:
    summary: dict[str, Any] = {"opportunities": len(rows), "methods": {}, "paired": {}}
    reference_n = sum(row["has_reference"] for row in rows)
    summary["reference_opportunities"] = reference_n
    for method in METHODS:
        outputs = [row["methods"][method] for row in rows]
        accepted = [out for out in outputs if out["success"]]
        evaluated = [out for out in accepted if out.get("translation_mm") is not None]
        bad = sum(
            out["translation_mm"] > config["translation_limit_mm"]
            or out["rotation_deg"] > config["rotation_limit_deg"]
            for out in evaluated
        )
        entry = {
            "accepted": len(accepted),
            "failed": len(outputs) - len(accepted),
            "acceptance_rate": len(accepted) / len(outputs) if outputs else None,
            "failure_reasons": dict(
                Counter(out["message"] for out in outputs if not out["success"])
            ),
            "heldout_status_counts": dict(
                Counter(out.get("heldout_status", "not evaluated") for out in outputs)
            ),
            "reference_evaluated": len(evaluated),
            "accepted_over_reference_limits": bad,
            "over_limit_rate_among_reference_evaluated": bad / len(evaluated)
            if evaluated
            else None,
            "within_limit_coverage_of_reference_opportunities": (
                (len(evaluated) - bad) / reference_n if reference_n else None
            ),
        }
        for metric in ("translation_mm", "rotation_deg", "heldout_corner_rmse_px", "seconds"):
            entry[metric] = distribution(
                [
                    out[metric]
                    for out in (outputs if metric == "seconds" else accepted)
                    if out.get(metric) is not None
                ]
            )
        summary["methods"][method] = entry
    for method in METHODS[1:]:
        summary["paired"][method] = {
            metric: paired_metric(
                rows,
                method,
                metric,
                repetitions=config["bootstrap_repetitions"] if bootstrap else 0,
                seed=config["bootstrap_seed"],
            )
            for metric in ("translation_mm", "rotation_deg", "heldout_corner_rmse_px")
        }
    return summary


def make_summary(rows: list[dict], config: dict, reference: dict | None) -> dict:
    result = summarize(rows, config)
    result["reference"] = reference
    result["interpretation"] = (
        "Pose errors are relative to the declared reference; reference uncertainty is not subtracted. "
        "Held-out camera residuals measure cross-view consistency, not absolute accuracy. "
        "Acceptance does not imply correctness. Paired metrics use common successful rows only; "
        "read them with acceptance and within-limit coverage."
    )
    result["accuracy_claim_supported_by_reference_type"] = bool(
        reference and reference["kind"] == "independent" and result["reference_opportunities"]
    )
    result["difference_from_anchors"] = {
        method: {
            metric: distribution(
                [
                    r["difference_from_anchors"][method][metric]
                    for r in rows
                    if method in r.get("difference_from_anchors", {})
                ]
            )
            for metric in ("translation_mm", "rotation_deg")
        }
        for method in METHODS[1:]
    }
    result["by"] = {}
    for key in ("group", "camera", "condition", "decoded_anchor_count"):
        result["by"][key] = {
            str(value): summarize([r for r in rows if r[key] == value], config, bootstrap=False)
            for value in sorted({r[key] for r in rows}, key=str)
        }
    return result
