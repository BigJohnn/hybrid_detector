# Validation status

## Current-code audit (2026-09-11)

The historical tables below have not been regenerated in the new capture evaluator.
Current `detect_carrier` already filters competing anchor hypotheses using
`max_anchor_rms_ratio=3.0`; statements below that this floor is unimplemented
are historical and no longer describe the complete current code. This does not
validate the threshold on unseen real data, or extend the edge-refinement
rank/trust/sigma gates to the anchored path. Follow the
[paired real-capture pilot protocol](real_capture_pilot.md) to freeze the actual
source and assess it without sharing pose estimates across methods.

## 2026-09-09 bench capture

Seven calibrated fisheye cameras observed the 0907 target. Two processable
episodes contributed 2403 frames and 16,821 camera-frame opportunities.

| Metric | Result |
|---|---:|
| Camera views with a detection | 10,249 / 16,821 (60.9%) |
| Joint poses solved | 2403 / 2403 |
| Median cameras per joint pose | 7 |
| Median corners per joint pose | 52 |
| Median joint reprojection RMSE | 1.261 px |
| P95 joint reprojection RMSE | 1.872 px |
| Median reported position sigma | 0.061 mm |
| P95 reported position sigma | 0.077 mm |
| Median reported rotation sigma | 0.0372 deg |
| P95 reported rotation sigma | 0.0486 deg |
| Median joint-vs-camera-fusion delta | 1.170 mm |
| P95 joint-vs-camera-fusion delta | 2.620 mm |

Per-camera detection rates ranged from 25.8% to 87.0%. This is compatible with a
100% joint-pose rate: the joint estimator needs enough total views per time
step, not every camera to succeed.

**The two reported-sigma rows above are superseded.** They were produced before
`solve_carrier_pose` scaled its covariance by the a posteriori variance factor,
so they report the assumed noise model rather than the observed one. The
assumption was a flat 0.3 px per anchor corner against a measured 1.261 px
median joint RMSE, which is a variance factor near 17 and a sigma roughly four
times larger than quoted: about 0.25 mm, not 0.061 mm. Recomputing them exactly
needs the 0909 capture, which is not in this repository. Two further corrections
are still owed on top of that one: the variance factor prices the noise scale
but not the correlation between samples taken along one physical boundary, which
on synthetic ground truth leaves the reported sigma about 2x optimistic
(\|dt\|/sigma_t of 1.98 near and 2.37 far, where a calibrated sigma would give
about 0.9); and none of it prices systematic camera-calibration error at all.
Quote no uncertainty from this capture until it has been recomputed.

![Observed carrier overlays](images/bench_overlay_cam13_cam14.jpg)

## Earlier controlled check

A 40-frame sample solved 40/40 frames with a median of seven cameras. Joint
reprojection RMSE was 1.590 px, with 142 edge samples per frame and 0.77 px edge
RMSE. Facet refinement moved the anchor bundle-adjustment answer by less than
0.05 mm in that sample.

## The synthetic oracle was not an oracle (2026-09-10)

Every synthetic comparison run against the shipped 0907 descriptor before this
date is void. `render_carrier` pasted each ArUco sticker at quadrant 0, while
the 0907 descriptor pins `paste_quadrant` to 3, 3, and 2 -- a measured fact
about the printed part, which the detector honours without searching the quarter
turns. So the renderer drew the marker one or two quarter turns away from the
only place the detector was allowed to look. The resulting poses were wrong by
150 to 370 mm and 48 to 142 deg, **and still reported `success`**, because a
consistent quarter-turn error on every anchor is a self-consistent solution.

The 0904 descriptor leaves `paste_quadrant` unset, so it never showed the fault,
and every test in the suite used 0904. Real bench captures are unaffected: there
the sticker is physically where the descriptor says it is.

`render_carrier` now defaults to the descriptor's pinned quadrants, and
`test_the_render_pastes_the_quadrant_the_descriptor_pinned` fails by 280 mm
without the fix. The general lesson for the paper's synthetic section: a
render/detect round trip agrees with itself for reasons that have nothing to do
with being right, so every synthetic harness needs at least one assertion
against an externally known pose.

## T0.1 gain sweep, synthetic (2026-09-10)

The question this answers is the one that decides whether there is a paper: do
the coloured facets beat the *same carrier with its paint stripped*? The
baseline has to be the same body, because comparing against a different target
compares geometry rather than paint. `--strip-paint` edits a named `--target` in
place rather than adding a row, so the same descriptor is passed twice under two
labels. `rig` is the measured marker layout wrapped as an anchors-only carrier,
carried as an independent third reference. One camera, 60 poses, `--seed 7`,
shared across every row.

### Far range and high obliquity (0.9-2.2 m, 50 deg tilt)

| target | n | fail | p50 | p90 | max | drot | anc | edge |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| painted | 60 | 0 | 6.07 | 19.10 | 56.10 | 0.90 | 2 | 178 |
| plain | 60 | 0 | 9.82 | 60.75 | 199.17 | 1.53 | 2 | 0 |
| rig | 60 | 0 | 9.86 | 46.95 | 204.90 | 1.55 | 2 | 0 |

p90 falls 68.6%, max 71.8%, rotation 41%, with no loss of solved frames. The
predeclared bar was a 30% p90 reduction at unchanged `n` in at least one
operating regime, so this regime clears it outright. `plain` and `rig` landing
together is the check that the stripped baseline is a fair one.

### Occlusion ladder (0.55-0.85 m, 26 deg tilt)

`hid` is the share of the silhouette a slab covers. p90, mm:

| hid | painted | plain | rig |
|---:|--:|--:|--:|
| 0.00 | 1.72 | 2.83 | 3.03 |
| 0.15 | 2.33 | 4.77 | 6.65 |
| 0.30 | 2.92 | 8.53 | 9.48 |
| 0.45 | 4.29 | 9.26 | 23.05 |
| 0.60 | **50.40** | 14.52 | 38.29 |

The median improves everywhere, 60% cover included (2.35 mm against 6.05 mm).
It is the tail that inverts: past roughly half the silhouette a minority of
frames go badly wrong, and the paint is what makes them wrong.

### Heavy cover with `--allow-anchor-free` (0.5 and 0.7 hidden)

| target | hid | n | fail | p50 | p90 | max | drot | edge |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| painted | 0.50 | 55 | 5 | 1.55 | 5.13 | 185.70 | 0.35 | 102 |
| plain | 0.50 | 53 | 7 | 5.48 | 14.78 | 311.15 | 0.74 | 0 |
| rig | 0.50 | 46 | 14 | 5.93 | 18.90 | 293.22 | 0.73 | 0 |
| painted | 0.70 | 30 | 30 | 6.16 | 86.11 | 145.67 | 1.13 | 36 |
| plain | 0.70 | 27 | 33 | 6.13 | 13.64 | 46.32 | 0.80 | 0 |
| rig | 0.70 | 22 | 38 | 6.84 | 11.24 | 14.60 | 0.80 | 0 |

At half cover the paint clears the bar on both counts: p90 down 65% and *more*
frames solved, not fewer. At 0.7 it loses the tail for the reason described
below, and the 185.70 mm outlier at 0.5 is the same failure appearing once.

This is the one sweep the covariance correction moves, because
`anchor_free_max_sigma_m` compares against a sigma that is now scaled. Solved
frames went 52 to 55 at half cover and 27 to 30 at 0.7: on synthetic imagery the
variance factor runs below 1, so the admission loosened. On the bench capture
the factor is nearer 17, so the same constant will *tighten* by about four
times. It was tuned against an uncalibrated sigma and has to be re-derived
against a calibrated one before the real pipeline is run again. The far-range
and occlusion-ladder tables above are unaffected -- the anchored path never
reads the covariance -- and re-running the far-range sweep after the correction
reproduced it exactly.

### What breaks at 60% cover

Seven of 37 solved frames exceed 15 mm. They are not a gradual degradation:

| | good (n=30) | bad (n=7) |
|---|---|---|
| \|dt\| mm | 2.15 [0.12, 7.97] | 67.86 [19.47, 145.67] |
| variance factor | 0.51 [0.16, 1.61] | 3.00 [0.73, 16.58] |
| sigma_t mm | 1.07 [0.52, 5.12] | 6.43 [2.98, 16.87] |
| edge samples | 84 [41, 126] | 16 [0, 33] |
| anchor points | 4 | 4 |

The obvious reading -- a starved edge set drags the pose off -- is wrong, and
the frame that disproves it is pose 26: the painted descriptor misses it by
67.86 mm having found **zero** edges, while the stripped descriptor reads the
same image to 2.38 mm. With no edge measurements both solves are anchors-only,
so nothing separates them except which PnP hypothesis won. The same holds for
the rest: pose 39 is 145.67 mm painted against 2.93 mm anchors-only, pose 42
123.19 against 6.53.

The winning painted hypothesis on pose 39 carries an anchor variance factor of
21.4 -- roughly 1.4 px RMS on corners assumed good to 0.3 px -- where the
anchors-only pipeline's winner carries 1.37. **Under heavy occlusion the facet
colour score, computed over a mostly hidden target, out-votes the anchor
geometry and selects the wrong branch of the single-marker pose ambiguity.**
The edge refinement then faithfully polishes the wrong answer.

Two things follow, neither of them fixed here:

1. Hypothesis ranking needs an anchor-evidence floor, so a colour score
   measured on a 40%-visible target cannot override corners that fit. This is a
   design decision about the evidence hierarchy, not a tuning constant.
2. `edge_constraints_span_the_pose`, the trust radius, and the sigma admission
   all sit inside `if anchor_free:` (`detector.py:2288-2310`). Decoding one
   sticker currently buys the edge fit unconditional acceptance. The comment at
   line 2258 claims these gates "all still apply"; they do not.

A relative admission rule -- keep the refinement only if it narrows the sigma
the anchors alone reported -- was implemented and reverted: measured against a
baseline computed at the *same* wrong hypothesis it never fires, because the
refinement genuinely is more precise than the bad pose it started from. It
priced well only in an offline comparison across two separate pipelines, which
is not a test the detector can run on itself.

## What the evidence establishes

- The measured model and dictionary produce detections on real images.
- Multi-camera anchor geometry is numerically stable on the captured trajectory.
- Facet boundaries reach the optimizer and remain close to the anchor solution.
- The system exposes per-view evidence and can reject weak measurements.

## What it does not establish

- Traceable absolute 6-DoF accuracy against an independent metrology system.
- Cross-day camera calibration stability.
- Generalization across lighting, paint batches, printers, and lenses.
- Remove/reseat repeatability of the carrier socket.
- A measured target-to-tool roll convention.
- Superiority over all modern fiducial or learned keypoint baselines.

The current CAD-derived target-to-TCP bundle is deliberately marked
`validated: false`. Its translation is zero by construction at the socket
centre; its rotation is a convention rather than a measurement.

## Required publication experiments

1. Register an independent ground-truth system and report translation and
   rotation error, not reprojection error alone.
2. Run repeated captures across days, cameras, lighting, ranges, and incidence
   angles with frozen parameters.
3. Compare against AprilTag/ArUco, ChArUco/board, and at least one strong
   correspondence or learned-pose baseline.
4. Report detection rate and error conditional on anchor count and occlusion.
5. Perform ablations: anchors only, facets only, anchors + colour, and full
   anchor + colour + sub-pixel edges.
6. Separate body-scale, sticker-scale, camera calibration, and mounting errors.
7. Publish failure cases and predefine exclusion thresholds.
8. Settle the evidence hierarchy under occlusion: give hypothesis ranking an
   anchor-evidence floor so a colour score measured on a mostly hidden target
   cannot override corners that fit, and extend the rank, trust-radius, and
   sigma admissions to the anchored path. Both are prerequisites for claiming
   that the estimator "refuses under-constrained solutions", which is one of
   only two surviving novelty candidates in
   [ip_and_publication.md](ip_and_publication.md).
9. Recompute every reported uncertainty on the 0909 capture after the
   covariance correction, and quantify the residual correlation between samples
   along one boundary rather than leaving the reported sigma 2x optimistic.
