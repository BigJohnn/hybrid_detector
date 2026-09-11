"""End-to-end checks on the Hybrid Carrier descriptor and detector.

The pose tests run against a synthetic render of the descriptor rather than a
photograph, which buys an exact ground truth and costs realism -- lighting,
specular paint, motion blur and print tolerance are all absent.  So they are
written to check the things a render *can* settle: that the pipeline recovers
the pose it was given, that the painted boundaries add information rather than
noise, and that each stage refuses rather than guesses when its evidence is
gone.  Absolute accuracy in millimetres is for the bench, not for these.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from hybrid_detector.calibration import CameraCalibration
from hybrid_detector.cli.build_model import build_model, dark_interval, diagnostics, warnings_for
from hybrid_detector.detector import (
    CarrierView,
    ColourReference,
    EdgeMeasurement,
    HybridCarrierModel,
    build_aruco_detector,
    classify_colours,
    colour_reference_from_image,
    colour_regions,
    corner_observations,
    detect_carrier,
    edge_constraints_span_the_pose,
    estimate_build_scale,
    facet_views,
    ink_image,
    measure_edges,
    project_rig,
    rig_roi,
)
from hybrid_detector.render import PaintPalette, render_carrier
from hybrid_detector.step import read_step

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "assets" / "models" / "hybrid_carrier_v1_20260904.json"
STEP_PATH = ROOT / "assets" / "cad" / "finalrig0904.step"

# The print differs from the CAD on purpose, for two reasons.  Black paint
# absorbs its own black border, which turns every edge a black face touches
# into a one-sided step, so no face is printed black.  And the shape is mirror
# symmetric about y = 0, so a mirror pair painted the same colour cannot say
# which side of the rig it is: every twin is painted apart, which is what lets
# a frame with no anchor in it be read at all.
REPAINT = {
    "black_pz": "green",
    "black_ny": "blue",
    "black_py": "magenta",
    "red_px_yp": "yellow",
    "red_ny": "green",
    "red_py": "cyan",
    "blue_pz_yn": "yellow",
}


@pytest.fixture(scope="module")
def model() -> HybridCarrierModel:
    return HybridCarrierModel.from_json(MODEL_PATH)


@pytest.fixture(scope="module")
def camera() -> CameraCalibration:
    return CameraCalibration(
        name="cam_test",
        serial="synthetic",
        width=1920,
        height=1080,
        K=np.array([[1400.0, 0.0, 960.0], [0.0, 1400.0, 540.0], [0.0, 0.0, 1.0]]),
        D=np.zeros(5),
        T_base_cam=np.eye(4),
        model="rational",
    )


def pose_from(euler_deg, translation_m) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = (
        Rotation.from_euler("xyz", [180.0, 0.0, 0.0], degrees=True).as_matrix()
        @ Rotation.from_euler("xyz", euler_deg, degrees=True).as_matrix()
    )
    T[:3, 3] = translation_m
    return T


def pose_error(expected: np.ndarray, actual: np.ndarray) -> tuple[float, float]:
    delta = np.linalg.inv(expected) @ actual
    translation_mm = float(np.linalg.norm(delta[:3, 3]) * 1000.0)
    rotation_deg = float(
        np.degrees(np.linalg.norm(Rotation.from_matrix(delta[:3, :3]).as_rotvec()))
    )
    return translation_mm, rotation_deg


# ---------------------------------------------------------------------------
# the descriptor
# ---------------------------------------------------------------------------


def test_descriptor_is_reproducible_from_the_step():
    """The committed descriptor is what the builder produces from the committed CAD."""
    built = build_model(
        read_step(STEP_PATH),
        anchor_ids=[30, 31],
        dictionary="DICT_6X6_50",
        pad_size_mm=60.0,
        marker_size_mm=48.0,
        pad_tolerance_mm=0.35,
        border_width_mm=1.0,
        border_alignment="centred",
        sticker_covers_pad=False,
        carrier_id="hybrid_carrier_v1_20260904",
        repaint=REPAINT,
    )
    stored = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    for key in ("anchors", "facets", "edges", "print", "occluders"):
        assert built[key] == stored[key], key


def _build(**overrides):
    kwargs = dict(
        anchor_ids=[30, 31],
        dictionary="DICT_6X6_50",
        pad_size_mm=60.0,
        marker_size_mm=48.0,
        pad_tolerance_mm=0.35,
        border_width_mm=1.0,
        border_alignment="centred",
        sticker_covers_pad=False,
        carrier_id="hybrid_carrier_v1_20260904",
        repaint=REPAINT,
    )
    kwargs.update(overrides)
    return build_model(read_step(STEP_PATH), **kwargs)


def test_a_measured_sticker_rotation_survives_a_rebuild():
    """The 2026-09-09 bench capture measured these; a rebuild must not drop them back
    to None, which is what silently returns the detector to a 4^n search."""
    built = _build(paste_quadrant_by_id={30: 3, 31: 2})
    assert {a["marker_id"]: a["paste_quadrant"] for a in built["anchors"]} == {30: 3, 31: 2}
    assert not any(
        "sticker rotation frozen" in warning for warning in warnings_for(built, diagnostics(built))
    )


def test_an_unmeasured_sticker_rotation_is_still_announced():
    built = _build()
    assert all(a["paste_quadrant"] is None for a in built["anchors"])
    assert any(
        "sticker rotation frozen" in warning for warning in warnings_for(built, diagnostics(built))
    )


def test_a_sticker_rotation_for_an_id_no_pad_carries_is_refused():
    with pytest.raises(SystemExit, match=r"marker id\(s\) \[47\]"):
        _build(paste_quadrant_by_id={47: 1})


def test_a_sticker_rotation_outside_the_four_quarter_turns_is_refused():
    with pytest.raises(SystemExit, match="0, 1, 2 or 3"):
        _build(paste_quadrant_by_id={30: 4})


def test_anchor_marker_corners_are_aruco_ordered(model: HybridCarrierModel):
    """ArUco reports corners clockwise about the outward normal; so must we."""
    for anchor in model.anchors:
        corners = anchor.marker_corners_m
        edges = [float(np.linalg.norm(corners[(i + 1) % 4] - corners[i])) for i in range(4)]
        assert edges == pytest.approx([0.048] * 4, abs=1e-6)
        area_vector = np.zeros(3)
        for i in range(4):
            area_vector += np.cross(corners[i], corners[(i + 1) % 4])
        assert float(np.dot(area_vector, anchor.normal)) < 0.0
        assert np.allclose(corners.mean(axis=0), anchor.centre_m, atol=1e-9)


def test_two_painted_facets_meeting_give_an_unbiased_landmark():
    """The band centre is the CAD edge for any stroke alignment -- that is the point."""
    for alignment in ("inside", "centred", "outside"):
        has_ink, low, high = dark_interval("red", "blue", 1.0, alignment)
        assert has_ink and low is not None and high is not None
        assert 0.5 * (low + high) == pytest.approx(0.0, abs=1e-9)


def test_a_black_fill_swallows_its_own_border():
    """One side unbounded, so the landmark moves off the CAD edge by half a stroke."""
    has_ink, low, high = dark_interval("red", "black", 1.0, "centred")
    assert has_ink and high is None
    assert low == pytest.approx(-0.5)


def test_a_sticker_that_covers_the_pad_biases_its_edges():
    unbiased = dark_interval("white", "red", 1.0, "centred", carries_ink_a=True)
    covered = dark_interval("white", "red", 1.0, "centred", carries_ink_a=False)
    assert 0.5 * (unbiased[1] + unbiased[2]) == pytest.approx(0.0)
    assert 0.5 * (covered[1] + covered[2]) == pytest.approx(0.25)


def test_the_print_leaves_no_black_facet_and_so_no_biased_edge():
    """The whole point of repainting them: a black fill is what makes an edge one-sided."""
    stored = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    stats = diagnostics(stored)
    assert "black" not in stats["facet_colour_counts"]
    assert stats["num_unbiased_edges"] == stats["num_measurable_edges"]
    assert not any("cancel their own black border" in warning for warning in stored["warnings"])


def test_centring_the_stroke_is_what_makes_the_edges_self_centring():
    stats = diagnostics(json.loads(MODEL_PATH.read_text(encoding="utf-8")))
    by_alignment = stats["unbiased_edge_length_mm_by_alignment"]
    assert by_alignment["centred"] > 2.0 * by_alignment["inside"]
    assert by_alignment["inside"] == by_alignment["outside"]


def test_the_skin_is_fully_chiral(model: HybridCarrierModel):
    """No mirror twin shares a colour, so every off-plane facet names its side.

    This is the property an anchor-free reading stands on, and it is a fact
    about how the thing is painted rather than about the detector -- so it is
    asserted on the descriptor, not on a detection.
    """
    stored = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    stats = diagnostics(stored)
    assert stats["mirror_pairs_y"] == []  # no twin pair left sharing a colour
    assert not any("mirror-symmetric" in warning for warning in stored["warnings"])
    on_plane = {f.name for f in model.facets if f.use_for_pose and abs(f.centroid_m[1]) < 1e-3}
    off_plane = {f.name for f in model.facets if f.use_for_pose} - on_plane
    assert set(model.side_resolving_facets) == off_plane
    assert on_plane.isdisjoint(model.side_resolving_facets)


def test_adjacent_facets_never_share_a_colour(model: HybridCarrierModel):
    """A shared edge between two facets of one colour has no colour contrast."""
    for edge in model.edges:
        if edge.face_a in model.facets_by_name and edge.face_b in model.facets_by_name:
            assert edge.colour_a != edge.colour_b, f"{edge.face_a}|{edge.face_b}"


def test_the_reverse_side_of_an_anchor_pad_is_not_a_facet(model: HybridCarrierModel):
    excluded = [facet for facet in model.facets if not facet.use_for_pose]
    assert [facet.name for facet in excluded] == ["grey_nz"]
    assert "reverse side of ArUco pad" in excluded[0].note


# ---------------------------------------------------------------------------
# the detector
# ---------------------------------------------------------------------------


def test_ink_separates_the_stroke_from_saturated_paint():
    """Grayscale would not: red at full saturation is nearly as dark as black."""
    reference = ColourReference(
        white=np.array([251.0] * 3), black=np.array([0.0] * 3), source="test"
    )
    patches = np.array(
        [[[40, 40, 200], [200, 70, 40], [18, 18, 18], [196, 200, 205]]], dtype=np.uint8
    )
    red, blue, stroke, body = ink_image(patches, reference)[0]
    assert stroke > 0.8
    assert max(red, blue, body) < 0.35
    assert stroke - max(red, blue, body) > 0.45


def test_yellow_is_a_colour_of_its_own_and_not_a_pale_red():
    """The repaint leans on it, so it has to survive next to red and next to bare body.

    Yellow is two channels up rather than one, so the primary tests score it at
    about zero -- and red scores about zero as yellow, for the same reason.
    Without its own channel it would fall through to "light" and read as body.
    """
    reference = ColourReference(
        white=np.array([251.0] * 3), black=np.array([0.0] * 3), source="test"
    )
    palette = PaintPalette()
    patches = np.array([[palette.yellow, palette.red, palette.green, palette.body]], dtype=np.uint8)
    masks = classify_colours(patches, reference)
    yellow, red, green, body = (masks["yellow"][0] > 0).tolist()
    assert (yellow, red, green, body) == (True, False, False, False)
    assert (masks["red"][0] > 0).tolist() == [False, True, False, False]
    assert (masks["green"][0] > 0).tolist() == [False, False, True, False]
    assert (masks["body"][0] > 0).tolist() == [False, False, False, True]


@pytest.mark.parametrize(
    "euler,translation",
    [
        ((25.0, 20.0, 0.0), (-0.09, -0.03, 0.75)),
        ((-15.0, 30.0, 120.0), (0.02, 0.01, 0.9)),
        ((10.0, -25.0, -140.0), (0.0, -0.02, 0.7)),
    ],
)
def test_detector_recovers_the_rendered_pose(model, camera, euler, translation):
    T_true = pose_from(euler, translation)
    image = render_carrier(model, camera, T_true, blur_sigma=0.8, noise_sigma=2.5, seed=3)
    detection = detect_carrier(image, camera, model)
    assert detection.success, detection.message
    translation_mm, rotation_deg = pose_error(T_true, detection.T_base_rig)
    assert translation_mm < 6.0
    assert rotation_deg < 1.5
    assert detection.pose is not None and detection.pose.num_edge_samples > 40


def test_the_sticker_rotation_is_recovered_not_assumed(model, camera):
    """The paste quadrant is a physical unknown; a wrong guess is a 90 deg error."""
    T_true = pose_from((25.0, 20.0, 0.0), (-0.09, -0.03, 0.75))
    quadrants = {int(anchor.marker_id): 3 for anchor in model.anchors}
    image = render_carrier(model, camera, T_true, paste_quadrants=quadrants, blur_sigma=0.8)
    detection = detect_carrier(image, camera, model)
    assert detection.success
    assert {d.quadrant for d in detection.anchors} == {3}
    translation_mm, rotation_deg = pose_error(T_true, detection.T_base_rig)
    assert translation_mm < 6.0 and rotation_deg < 1.5


def test_the_render_pastes_the_quadrant_the_descriptor_pinned(camera):
    """A pinned paste rotation binds the oracle, not just the detector.

    The 0907 descriptor records where each sticker actually sits on the printed
    part, and the detector reads that as a measured fact: it does not search the
    quarter turns for a pinned anchor.  A render that pasted at quadrant 0
    anyway would put the marker somewhere the detector is not allowed to look,
    and every synthetic comparison run on this descriptor would be scored
    against a target that cannot be read -- silently, because such a pose still
    solves and still reports success.  It was wrong by 150-370 mm.
    """
    pinned = HybridCarrierModel.from_json(
        ROOT / "assets" / "models" / "hybrid_carrier_v1_20260907.json"
    )
    assert {int(a.marker_id): a.paste_quadrant for a in pinned.anchors} == {30: 3, 31: 3, 32: 2}
    T_true = pose_from((20.0, -15.0, 35.0), (0.01, -0.02, 0.72))
    image = render_carrier(pinned, camera, T_true, blur_sigma=0.8, noise_sigma=2.5, seed=3)
    detection = detect_carrier(image, camera, pinned)
    assert detection.success, detection.message
    translation_mm, rotation_deg = pose_error(T_true, detection.T_base_rig)
    assert translation_mm < 6.0
    assert rotation_deg < 1.5
    assert len(detection.measurements) > 40


def test_painted_edges_beat_the_anchors_on_their_own(model, camera):
    """The whole premise of V1: the facets have to add pose, not just identity."""
    rng = np.random.default_rng(11)
    anchor_only: list[float] = []
    with_edges: list[float] = []
    for seed in range(8):
        T_true = pose_from(
            rng.uniform(-1, 1, 3) * np.array([30.0, 30.0, 180.0]),
            [rng.uniform(-0.05, 0.05), rng.uniform(-0.04, 0.04), rng.uniform(0.65, 1.0)],
        )
        image = render_carrier(model, camera, T_true, blur_sigma=0.8, noise_sigma=2.5, seed=seed)
        detection = detect_carrier(image, camera, model)
        if not detection.success:
            continue
        from hybrid_detector.detector import _pnp_candidates

        anchors = model.anchors_by_id
        object_points = np.vstack(
            [anchors[d.marker_id].object_points(d.quadrant) for d in detection.anchors]
        )
        image_points = np.vstack([d.corners_uv for d in detection.anchors])
        candidates = _pnp_candidates(camera, object_points, image_points)
        coarse = min(candidates, key=lambda T: pose_error(T_true, T)[0])
        anchor_only.append(pose_error(T_true, coarse)[0])
        with_edges.append(pose_error(T_true, detection.T_base_rig)[0])
    assert len(with_edges) >= 6
    assert float(np.median(with_edges)) < float(np.median(anchor_only))


def test_facets_report_why_they_are_unusable(model, camera):
    T_true = pose_from((25.0, 20.0, 0.0), (-0.09, -0.03, 0.75))
    image = render_carrier(model, camera, T_true, blur_sigma=0.8)
    detection = detect_carrier(image, camera, model)
    reasons = {view.facet.name: view.reason for view in detection.facet_views}
    assert any("back facing" in reason for reason in reasons.values())
    assert all(view.usable or view.reason for view in detection.facet_views)


def test_a_single_anchor_view_can_be_refused(model, camera):
    """Every gross outlier of the 2026-09-09 bench capture was a one-anchor view."""
    T_cam_rig = pose_from((0.0, 0.0, 0.0), (0.0, 0.0, 0.8))
    image = render_carrier(model, camera, T_cam_rig)
    permissive = detect_carrier(image, camera, model, min_anchors=1)
    strict = detect_carrier(image, camera, model, min_anchors=len(permissive.anchors) + 1)
    assert permissive.success
    assert not strict.success
    assert strict.T_base_rig is None
    assert f"fewer than the {len(permissive.anchors) + 1} required" in strict.message
    # The refusal still says what it saw, so a caller can tell "not enough evidence"
    # from "nothing there at all".
    assert [d.marker_id for d in strict.anchors] == [d.marker_id for d in permissive.anchors]


def test_no_anchor_means_no_pose(model, camera):
    """Colour is mirror-symmetric on this rig, so facets alone must not be trusted."""
    import cv2

    from hybrid_detector.detector import project_rig

    T_true = pose_from((25.0, 20.0, 0.0), (-0.09, -0.03, 0.75))
    image = render_carrier(model, camera, T_true, blur_sigma=0.8)
    for anchor in model.anchors:  # a sticker peeled off, or lost to glare
        uv = project_rig(camera, T_true, anchor.pad_corners_m)
        if np.isfinite(uv).all():
            cv2.fillPoly(image, [np.round(uv).astype(np.int32)], (200, 200, 200))
    detection = detect_carrier(image, camera, model)
    assert not detection.success
    assert "anchor" in detection.message


def test_corner_observations_bridge_to_the_existing_rig_solver(model, camera):
    T_true = pose_from((25.0, 20.0, 0.0), (-0.09, -0.03, 0.75))
    image = render_carrier(model, camera, T_true, blur_sigma=0.8)
    detection = detect_carrier(image, camera, model)
    observations = corner_observations(detection, model, "cam_test")
    assert len(observations) == 4 * len(detection.anchors)
    assert {o.camera_name for o in observations} == {"cam_test"}
    assert all(o.point_rig.shape == (3,) and o.uv.shape == (2,) for o in observations)


# ---------------------------------------------------------------------------
# reading the rig with no anchor in the frame
# ---------------------------------------------------------------------------


def _pose(rvec, translation):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(rvec).as_matrix()
    T[:3, 3] = translation
    return T


# Anchors face up and back, so this view -- looking at the painted side from
# below -- sees six facets and neither sticker.  It is the case the bootstrap
# exists for.
ANCHOR_FREE_POSE = _pose([-2.429951, 0.429388, 0.881959], [-0.023271, 0.008519, 0.784925])


def test_the_colour_reference_is_taken_from_the_rig_not_from_the_frame(model, camera):
    """A frame percentile measures the tablecloth, and then everything is saturated.

    The rig covers a few percent of a 1080p frame, so a whole-frame high
    percentile lands in the background: white and black come back a dozen grey
    levels apart, normalising blows the difference up, and every colour mask
    fires everywhere at once.  The reference has to be read next to the paint.
    """
    image = render_carrier(model, camera, ANCHOR_FREE_POSE, blur_sigma=0.8, noise_sigma=2.5, seed=3)
    naive_white = np.percentile(image.reshape(-1, 3), 98.0, axis=0)
    naive_black = np.percentile(image.reshape(-1, 3), 2.0, axis=0)
    assert float(np.min(naive_white - naive_black)) < 30.0  # the failure this guards against

    reference = colour_reference_from_image(image)
    assert float(np.min(reference.white - reference.black)) > 120.0
    masks = classify_colours(image, reference)
    # Under the bug every hue claimed about an eighth of the frame each, the
    # rig included and the background with it.  The rig is a few percent of the
    # frame, so that is what all six together should come to.
    painted = np.zeros(image.shape[:2], dtype=bool)
    for colour in ("red", "green", "blue", "yellow", "magenta", "cyan"):
        painted |= masks[colour] > 0
    assert float(np.mean(painted)) < 0.05


def test_colour_regions_find_the_facets_that_are_showing(model, camera):
    image = render_carrier(model, camera, ANCHOR_FREE_POSE, blur_sigma=0.8, noise_sigma=2.5, seed=3)
    masks = classify_colours(image, colour_reference_from_image(image))
    regions = colour_regions(masks, min_area_px2=150.0)
    showing = {
        (view.facet.colour, view.facet.num_vertices)
        for view in facet_views(model, camera, ANCHOR_FREE_POSE)
        if view.usable and view.facet.use_for_pose
    }
    found = {(region.colour, len(region.polygon_uv)) for region in regions}
    assert showing <= found, f"missed {showing - found}"


def test_the_bootstrap_reads_a_frame_with_no_anchor_in_it(model, camera):
    """Paint alone: identity from the colours, pose from the printed boundaries."""
    image = render_carrier(model, camera, ANCHOR_FREE_POSE, blur_sigma=0.8, noise_sigma=2.5, seed=3)
    detector = build_aruco_detector(model.anchors[0].dictionary)
    _, ids, _ = detector.detectMarkers(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    assert ids is None or not any(int(i) in model.anchors_by_id for i in ids.ravel())

    assert not detect_carrier(image, camera, model).success  # refused by default

    detection = detect_carrier(image, camera, model, allow_anchor_free=True)
    assert detection.success, detection.message
    assert not detection.anchors
    delta = np.linalg.inv(ANCHOR_FREE_POSE) @ detection.T_base_rig
    assert float(np.linalg.norm(delta[:3, 3])) < 0.005
    assert float(np.degrees(np.linalg.norm(Rotation.from_matrix(delta[:3, :3]).as_rotvec()))) < 1.5


def test_an_edge_only_fit_is_refused_when_the_constraints_cannot_span_the_pose():
    """Samples along one boundary fit exactly at any depth; the count never says so."""
    model = HybridCarrierModel.from_json(MODEL_PATH)
    one_edge = [
        EdgeMeasurement(
            edge_index=0,
            point_rig=np.zeros(3),
            uv=np.zeros(2),
            normal_uv=np.array([1.0, 0.0]),
            sigma_px=0.3,
            contrast=0.5,
            band_width_px=2.0,
            residual_px=0.0,
            profile="band_centre",
        )
        for _ in range(200)
    ]
    assert "distinct edges" in edge_constraints_span_the_pose(one_edge, model)
    assert edge_constraints_span_the_pose([], model) == "no edge samples"


# ---------------------------------------------------------------------------
# tracking: the same answer, for a fraction of the work
# ---------------------------------------------------------------------------


def _camera_at(name: str, translation, euler_deg) -> CameraCalibration:
    T_base_cam = np.eye(4)
    T_base_cam[:3, :3] = Rotation.from_euler("xyz", euler_deg, degrees=True).as_matrix()
    T_base_cam[:3, 3] = translation
    return CameraCalibration(
        name=name,
        serial=f"synthetic_{name}",
        width=1920,
        height=1080,
        K=np.array([[1400.0, 0.0, 960.0], [0.0, 1400.0, 540.0], [0.0, 0.0, 1.0]]),
        D=np.zeros(5),
        T_base_cam=T_base_cam,
        model="rational",
    )


def test_a_window_gives_the_same_fields_inside_it_and_nothing_outside(model, camera):
    T_true = pose_from((25.0, 20.0, 0.0), (-0.09, -0.03, 0.75))
    image = render_carrier(model, camera, T_true, blur_sigma=0.8, seed=5)
    window = rig_roi(camera, model, T_true, image.shape[:2], margin_px=40.0)
    assert window is not None
    x0, y0, x1, y1 = window
    inside = (slice(y0, y1), slice(x0, x1))

    reference = colour_reference_from_image(image)
    assert np.array_equal(
        ink_image(image, reference)[inside], ink_image(image, reference, roi=window)[inside]
    )
    assert not np.any(np.delete(ink_image(image, reference, roi=window), np.s_[y0:y1], axis=0))
    full = classify_colours(image, reference)
    windowed = classify_colours(image, reference, roi=window)
    for colour, mask in full.items():
        assert np.array_equal(mask[inside], windowed[colour][inside]), colour
        assert not windowed[colour][:y0].any(), colour


def test_the_window_a_seed_implies_covers_the_whole_rig(model, camera):
    """The margin has to swallow the seed's own error, or edges go missing."""
    T_true = pose_from((-15.0, 30.0, 120.0), (0.02, 0.01, 0.9))
    window = rig_roi(camera, model, T_true, (1080, 1920), margin_px=40.0)
    assert window is not None
    x0, y0, x1, y1 = window
    T_cam_rig = camera.T_cam_base @ T_true
    points = np.vstack([facet.polygon_m for facet in model.facets])
    uv = project_rig(camera, T_cam_rig, points)
    assert uv[:, 0].min() >= x0 and uv[:, 0].max() <= x1
    assert uv[:, 1].min() >= y0 and uv[:, 1].max() <= y1
    # And it is worth having: a rig this size covers a few percent of a frame.
    assert (x1 - x0) * (y1 - y0) < 0.25 * 1920 * 1080


def test_a_prebuilt_ink_map_changes_no_measurement(model, camera):
    """The map depends on the frame, never on the pose, so reusing it is exact."""
    T_true = pose_from((25.0, 20.0, 0.0), (-0.09, -0.03, 0.75))
    image = render_carrier(model, camera, T_true, blur_sigma=0.8, seed=7)
    reference = colour_reference_from_image(image)
    T_cam_rig = camera.T_cam_base @ T_true
    fresh = measure_edges(image, model, camera, T_cam_rig, reference=reference)
    reused = measure_edges(
        image, model, camera, T_cam_rig, reference=reference, ink=ink_image(image, reference)
    )
    assert len(fresh) == len(reused) > 20
    assert np.allclose([m.uv for m in fresh], [m.uv for m in reused])


def test_a_seeded_frame_reads_the_same_pose_as_a_cold_one(model, camera):
    T_true = pose_from((25.0, 20.0, 0.0), (-0.09, -0.03, 0.75))
    image = render_carrier(model, camera, T_true, blur_sigma=0.8, noise_sigma=2.5, seed=3)
    cold = detect_carrier(image, camera, model)
    assert cold.success and not cold.seeded

    # A seed one frame old: a few millimetres and a degree away.
    seed = T_true.copy()
    seed[:3, 3] += np.array([0.003, -0.002, 0.004])
    seed[:3, :3] = Rotation.from_euler("z", 1.0, degrees=True).as_matrix() @ seed[:3, :3]
    warm = detect_carrier(image, camera, model, seed=seed)
    assert warm.success and warm.seeded and not warm.reacquired
    assert warm.roi is not None
    translation_mm, rotation_deg = pose_error(cold.T_base_rig, warm.T_base_rig)
    assert translation_mm < 2.0 and rotation_deg < 0.5


def test_a_stale_seed_is_thrown_away_rather_than_believed(model, camera):
    """A track that has lost the rig must cost a slow frame, not a wrong pose."""
    T_true = pose_from((25.0, 20.0, 0.0), (-0.09, -0.03, 0.75))
    image = render_carrier(model, camera, T_true, blur_sigma=0.8, noise_sigma=2.5, seed=3)
    stale = T_true.copy()
    stale[:3, 3] += np.array([0.25, 0.18, -0.12])
    detection = detect_carrier(image, camera, model, seed=stale)
    assert detection.success and detection.reacquired
    assert "re-acquired from scratch" in detection.message
    translation_mm, rotation_deg = pose_error(T_true, detection.T_base_rig)
    assert translation_mm < 6.0 and rotation_deg < 1.5


def test_a_seeded_frame_still_reads_the_sticker_rotation_off_the_image(model, camera):
    """Seeded or not, a wrong paste quadrant is a 90 deg error, never assumed."""
    unpinned = HybridCarrierModel.from_json(MODEL_PATH)
    assert all(anchor.paste_quadrant is None for anchor in unpinned.anchors)
    T_true = pose_from((25.0, 20.0, 0.0), (-0.09, -0.03, 0.75))
    quadrants = {int(anchor.marker_id): 2 for anchor in unpinned.anchors}
    image = render_carrier(unpinned, camera, T_true, paste_quadrants=quadrants, blur_sigma=0.8)
    detection = detect_carrier(image, camera, unpinned, seed=T_true)
    assert detection.success and detection.seeded
    assert {d.quadrant for d in detection.anchors} == {2}


# ---------------------------------------------------------------------------
# how big the part came out
# ---------------------------------------------------------------------------


def _scaled_anchor_views(model, cameras, T_base_rig, body: float, marker: float):
    views = []
    for camera in cameras:
        object_points, image_points = [], []
        for anchor in model.anchors:
            corners = anchor.object_points(0)
            centre = corners.mean(axis=0)
            built = body * centre + marker * (corners - centre)
            object_points.append(corners)
            image_points.append(project_rig(camera, camera.T_cam_base @ T_base_rig, built))
        views.append(
            CarrierView(
                camera=camera,
                anchor_points_rig=np.vstack(object_points),
                anchor_uv=np.vstack(image_points),
                anchor_sigma_px=0.3,
            )
        )
    return views


def test_build_scale_separates_a_small_sticker_from_a_small_part(model):
    """Two different faults, opposite consequences; one scale would average them."""
    cameras = [
        _camera_at("cam_a", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        _camera_at("cam_b", (0.6, 0.0, 0.1), (0.0, -35.0, 0.0)),
        _camera_at("cam_c", (-0.5, 0.2, 0.05), (0.0, 30.0, 0.0)),
    ]
    T_base_rig = pose_from((20.0, 15.0, 10.0), (0.0, 0.0, 1.1))
    views = _scaled_anchor_views(model, cameras, T_base_rig, body=0.996, marker=0.984)
    estimate = estimate_build_scale(views, T_base_rig)
    assert estimate is not None
    assert estimate.body_ppm == pytest.approx(-4000.0, abs=250.0)
    assert estimate.marker_ppm == pytest.approx(-16000.0, abs=250.0)
    # 0.4% on a ~135 mm lever arm is where the TCP sigma comes from.
    assert estimate.socket_shift_mm == pytest.approx(0.004 * estimate.lever_arm_mm, rel=0.1)


def test_build_scale_refuses_a_single_camera(model):
    """One camera explains a smaller part by a nearer one, exactly."""
    cameras = [_camera_at("cam_a", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))]
    T_base_rig = pose_from((20.0, 15.0, 10.0), (0.0, 0.0, 1.1))
    views = _scaled_anchor_views(model, cameras, T_base_rig, body=0.996, marker=0.984)
    assert estimate_build_scale(views, T_base_rig) is None
