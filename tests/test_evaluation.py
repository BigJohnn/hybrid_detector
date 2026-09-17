"""Protect experimental conclusions against selection and provenance mistakes."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from hybrid_detector.calibration import CameraCalibration
from hybrid_detector.cli.evaluate_capture import (
    SCHEMA,
    FixedCorners,
    evaluate_frame,
    freeze,
    heldout_rmse,
    run,
)
from hybrid_detector.detector import HybridCarrierModel, project_rig
from hybrid_detector.evaluation import METHODS, make_summary, paired_metric, validate_pose

ROOT = Path(__file__).resolve().parents[1]
CONFIG = {
    "translation_limit_mm": 10.0,
    "rotation_limit_deg": 2.0,
    "bootstrap_seed": 7,
    "bootstrap_repetitions": 100,
}


def row(group="day1", reference=True, errors=(1.0, 2.0, 3.0)):
    return {
        "group": group,
        "camera": "cam",
        "condition": "test",
        "decoded_anchor_count": 1,
        "has_reference": reference,
        "methods": {
            method: {
                "success": error is not None,
                "message": "ok" if error is not None else "no anchor",
                "translation_mm": error if reference else None,
                "rotation_deg": 0.5 if reference and error is not None else None,
            }
            for method, error in zip(METHODS, errors, strict=True)
        },
    }


def test_rejections_remain_in_coverage_and_pairs_use_identical_successful_rows():
    rows = [row(errors=(1, 2, 3)), row(errors=(100, 100, None)), row(errors=(2, 2, 20))]
    report = make_summary(rows, CONFIG, {"kind": "independent"})
    full = report["methods"]["full"]
    assert full["accepted"] == 2 and full["failed"] == 1
    assert full["accepted_over_reference_limits"] == 1
    assert full["within_limit_coverage_of_reference_opportunities"] == pytest.approx(1 / 3)
    assert full["over_limit_rate_among_reference_evaluated"] == 0.5
    paired = report["paired"]["full"]["translation_mm"]
    assert paired["paired_n"] == 2
    assert paired["anchors_p90"] == pytest.approx(1.9)
    assert paired["p90_delta"] > 0


def test_absent_reference_does_not_become_zero_error_or_perfect_coverage():
    report = make_summary([row(reference=False)], CONFIG, None)
    assert report["reference_opportunities"] == 0
    assert not report["accuracy_claim_supported_by_reference_type"]
    assert report["methods"]["full"]["translation_mm"] is None
    assert report["methods"]["full"]["within_limit_coverage_of_reference_opportunities"] is None
    assert report["paired"]["full"]["translation_mm"]["paired_n"] == 0


def test_no_confidence_interval_from_one_video_even_with_many_frames():
    result = paired_metric([row()] * 100, "full", "translation_mm", repetitions=100, seed=7)
    assert result["group_n"] == 1
    assert result["cluster_bootstrap_p90_delta_95pct"] is None


def test_cluster_bootstrap_resamples_whole_groups():
    rows = [row("day1", errors=(10, 10, 1)), row("day2", errors=(10, 10, 100))]
    result = paired_metric(rows, "full", "translation_mm", repetitions=1000, seed=7)
    assert result["cluster_bootstrap_p90_delta_95pct"] == pytest.approx([-9, 90])
    assert result == paired_metric(rows, "full", "translation_mm", repetitions=1000, seed=7)


@pytest.fixture
def manifest(tmp_path):
    image = tmp_path / "frame.png"
    assert cv2.imwrite(str(image), np.zeros((48, 64, 3), np.uint8))
    camera = tmp_path / "camera.json"
    camera.write_text(
        json.dumps(
            {
                "width": 64,
                "height": 48,
                "model": "rational",
                "K": [[100, 0, 32], [0, 100, 24], [0, 0, 1]],
                "D": [0, 0, 0, 0, 0],
                "T_base_cam": np.eye(4).tolist(),
            }
        )
    )
    doc = {
        "schema": SCHEMA,
        "dataset_id": "test",
        "world_frame": "cam",
        "purpose": "test",
        "model": str(ROOT / "assets/models/hybrid_carrier_v1_20260907.json"),
        "cameras": {"cam": str(camera)},
        "evaluation": {"bootstrap_repetitions": 0},
        "frames": [
            {
                "id": "frame0",
                "group": "day1",
                "condition": "test",
                "views": {"cam": {"image": str(image)}},
            }
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(doc))
    return path


def test_frozen_run_keeps_blank_frame_and_has_no_accuracy_claim(manifest, tmp_path):
    lock = tmp_path / "frozen.json"
    frozen = freeze(manifest, lock)
    assert frozen["inventory"]["camera_frame_opportunities"] == 1
    report = run(lock, tmp_path / "results")
    for method in METHODS:
        assert report["methods"][method]["failed"] == 1
        assert report["methods"][method]["acceptance_rate"] == 0
    assert not report["accuracy_claim_supported_by_reference_type"]
    observations = [
        json.loads(line)
        for line in (tmp_path / "results/observations.jsonl").read_text().splitlines()
    ]
    assert len(observations) == 1
    with pytest.raises(FileExistsError):
        run(lock, tmp_path / "results")


def test_freeze_refuses_changed_inputs_before_creating_results(manifest, tmp_path):
    lock = tmp_path / "frozen.json"
    freeze(manifest, lock)
    (tmp_path / "frame.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="frozen input changed"):
        run(lock, tmp_path / "results")
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize(
    "fault", ["duplicate", "reference", "extrinsic", "synchronization", "parameter"]
)
def test_freeze_refuses_ambiguous_experiment_contract(manifest, tmp_path, fault):
    doc = json.loads(manifest.read_text())
    if fault == "duplicate":
        doc["frames"].append(doc["frames"][0])
    elif fault == "reference":
        doc["frames"][0]["T_base_target_reference"] = np.eye(4).tolist()
    elif fault == "extrinsic":
        config = json.loads((tmp_path / "camera.json").read_text())
        del config["T_base_cam"]
        (tmp_path / "camera.json").write_text(json.dumps(config))
    elif fault == "synchronization":
        doc["cameras"]["cam2"] = doc["cameras"]["cam"]
        doc["frames"][0]["views"]["cam2"] = doc["frames"][0]["views"]["cam"]
    else:
        doc["detector"] = {"seed": np.eye(4).tolist()}
    manifest.write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        freeze(manifest, tmp_path / "frozen.json")


def test_shared_corners_cannot_be_mutated_by_one_method():
    raw = ([np.zeros((1, 4, 2))], np.array([[30]]), [])
    detector = FixedCorners(raw)
    corners, ids, _ = detector.detectMarkers(None)
    corners[0][:] = 100
    ids[0, 0] = 49
    again, again_ids, _ = detector.detectMarkers(None)
    assert not again[0].any()
    assert again_ids[0, 0] == 30


def test_heldout_residual_uses_only_other_camera_pixels():
    model = HybridCarrierModel.from_json(ROOT / "assets/models/hybrid_carrier_v1_20260907.json")
    cameras = {
        name: CameraCalibration(
            name,
            name,
            640,
            480,
            np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1.0]]),
            np.zeros(5),
            np.eye(4),
        )
        for name in ("a", "b")
    }
    pose = np.eye(4)
    pose[2, 3] = 2
    anchor = model.anchors[0]
    uv = project_rig(cameras["b"], pose, anchor.object_points(anchor.paste_quadrant))
    raw = {
        "a": ([uv + 1000], np.array([[anchor.marker_id]]), []),
        "b": ([uv + np.array([3, 4])], np.array([[anchor.marker_id]]), []),
    }
    error, evidence, status = heldout_rmse(pose, "a", raw, cameras, model)
    assert error == pytest.approx(5)
    assert set(evidence) == {"b"} and status == "ok"


def test_unreadable_images_remain_in_the_denominator():
    model = HybridCarrierModel.from_json(ROOT / "assets/models/hybrid_carrier_v1_20260907.json")
    camera = CameraCalibration("a", "a", 640, 480, np.eye(3), np.zeros(5), np.eye(4))
    rows = evaluate_frame(
        {"id": "f", "group": "g", "condition": "c"}, {"a": None}, {"a": camera}, model, {}
    )
    assert len(rows) == 1
    assert all(rows[0]["methods"][m]["message"] == "input_decode_failed" for m in METHODS)


def test_transform_validation_rejects_reflection():
    reflection = np.eye(4)
    reflection[0, 0] = -1
    with pytest.raises(ValueError, match=r"SO\(3\)"):
        validate_pose(reflection, "reflection")


def test_reference_pose_is_used_only_for_scoring():
    from hybrid_detector.cli.detect import camera_from_json

    cv2.setNumThreads(1)
    model = HybridCarrierModel.from_json(ROOT / "assets/models/hybrid_carrier_v1_20260907.json")
    image = cv2.imread(str(ROOT / "examples/data/cam14_ep1_frame000000.jpg"))
    camera = camera_from_json("cam14", ROOT / "examples/data/cam14_intrinsics.json", 0, 0)
    frame = {"id": "example", "group": "capture", "condition": "example"}
    before = evaluate_frame(frame, {"cam14": image}, {"cam14": camera}, model, {})[0]
    wrong_reference = np.eye(4)
    wrong_reference[0, 3] = 100.0
    after = evaluate_frame(
        {**frame, "T_base_target_reference": wrong_reference.tolist()},
        {"cam14": image},
        {"cam14": camera},
        model,
        {},
    )[0]
    for method in METHODS:
        first, second = before["methods"][method], after["methods"][method]
        assert first["success"] and second["success"]
        np.testing.assert_allclose(first["T_base_target"], second["T_base_target"], atol=1e-12)
        assert "translation_mm" not in first
        assert second["translation_mm"] > 90000
        assert first["anchor_observations"] == before["methods"]["anchors"]["anchor_observations"]
    assert before["methods"]["full"]["edge_samples"] > 0
    assert before["methods"]["colour"]["edge_samples"] == 0
    assert before["methods"]["anchors"]["edge_samples"] == 0


def test_video_reader_uses_requested_zero_based_index(tmp_path):
    from hybrid_detector.cli.evaluate_capture import CaptureReader

    path = tmp_path / "video.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    if not writer.isOpened():
        pytest.skip("MJPG encoder unavailable")
    for value in (20, 100, 220):
        writer.write(np.full((48, 64, 3), value, np.uint8))
    writer.release()
    reader = CaptureReader(tmp_path)
    try:
        for index, value in ((2, 220), (0, 20), (1, 100)):
            decoded = reader.read({"video": path.name, "frame_index": index})
            assert decoded is not None
            assert float(decoded.mean()) == pytest.approx(value, abs=5)
        assert reader.read({"video": path.name, "frame_index": 100}) is None
    finally:
        reader.close()
