# IP and publication assessment

Assessment date: 2026-09-10. Revised 2026-09-10 to add the model-based 3D edge
tracking family, which the first pass of this screen missed entirely. This is an
engineering prior-art screen, not a patentability opinion or freedom-to-operate
analysis.

## Executive assessment

Publishing a useful robotics-vision paper is feasible after stronger
ground-truth experiments. A broad patent claim over a coloured 3D fiducial,
marker-assisted pose initialization, or geometric boundary refinement is
unlikely to be defensible because each theme has substantial prior art.

A narrower invention case remains plausible around the complete measurement
contract:

1. decoded anchors establish identity, metric scale, and a coarse pose;
2. that pose and an exact CAD model predict visible physical facets;
3. colour is evaluated only inside predicted supports and is used as a
   correspondence gate;
4. sub-pixel samples constrain motion only along known boundary normals;
5. anchor and edge residuals are jointly optimized with robust loss;
6. observability checks retain the anchor-only solution when facet evidence is
   weak; and
7. the physical descriptor records paste rotation, measured print scale,
   occlusion geometry, and target-to-tool provenance.

The potentially differentiating unit is this gated, uncertainty-reporting
system and its co-designed physical carrier, not any one familiar component.
Patentability is therefore **possible but unproven**; the present confidence is
medium-low until a professional claim chart and jurisdiction-specific search
are complete.

That list needs one correction. Items 2, 3, 4, and 5 are, taken together, the
standard model-based 3D edge tracking recipe, published since 1990 and shipped
in an open-source library. The first version of this screen compared the method
only against planar fiducial markers and coloured-marker patents, and so read as
novel a sequence that a reader from the augmented-reality tracking community
would recognize on sight. See
[the family this screen missed](#the-family-this-screen-missed-model-based-3d-edge-tracking)
below. The invention case has to be argued somewhere other than the
predict-project-measure loop; the two candidates that survive are the two-scale
build calibration and the pre-fit observability admission, both of which are
about what the estimator refuses to do rather than how it iterates.

## Closest technical themes found

The following references should be treated as starting points, not an exhaustive
search:

| Reference | Relevant overlap | Remaining distinction to investigate |
|---|---|---|
| [ChromaTag (ICCV 2017)](https://openaccess.thecvf.com/content_iccv_2017/html/DeGol_ChromaTag_A_Colored_ICCV_2017_paper.html) | Colour improves detection while grayscale structure supports precise localization. | It is a planar encoded tag, not pose-gated sampling of CAD-defined 3D facet boundaries. |
| [STag](https://arxiv.org/abs/1707.06292) | A coarse outer-square estimate is refined using a more repeatable inner boundary. | The refinement geometry and target are planar; it does not disclose the same anchor/CAD-facet evidence contract. |
| [AR-HueCode](https://doi.org/10.1080/01691864.2026.2623883) | Multiple differently sized markers are overlaid in colour and fused for range and occlusion robustness. | Its colour encodes overlapping markers and pose graph fusion, rather than semantic 3D facet edges gated by a CAD pose. |
| [EP2157545A1](https://patents.google.com/patent/EP2157545A1/en) | An AR marker may have a primary recognition facet and non-parallel secondary facets. | The claims and family must be charted against the carrier geometry and use of secondary surfaces. |
| [US9526587B2](https://patents.google.com/patent/US9526587B2/en) | Known 3D marker geometry is projected after pose initialization and image evidence verifies candidate features. | The disclosed surgical marker patterns and feature verification differ, but the predict-then-test structure is important prior art. |
| [CN110197509B](https://patents.google.com/patent/CN110197509B/zh) | A coloured artificial target is detected and used for camera-pose solving. | Claims need a Chinese-language element-by-element review against colour classification and pose solving. |
| [US20250200792A1](https://patents.google.com/patent/US20250200792A1/en) | A colour-patterned spherical marker supports recognition and 6-DoF positioning. | It uses coloured spherical cells and learned recognition rather than planar CAD facets and normal-profile edges. |

Other baseline systems that belong in the paper comparison include
[AprilTag](https://april.eecs.umich.edu/software/apriltag.html),
[TopoTag](https://arxiv.org/abs/1908.01450), and
[RUNE-Tag](https://www.dsi.unive.it/~bergamasco/runetag/).

## The family this screen missed: model-based 3D edge tracking

Every reference in the table above is a *marker*. The method in this repository
is only half a marker method. Steps 3 through 6 of [method.md](method.md) --
project a known CAD model from a current pose estimate, decide which faces are
front-facing, search along each predicted boundary's normal for the image
evidence, and fold the resulting one-dimensional displacements into a robust
pose update -- are the defining loop of model-based tracking (MBT), a literature
that starts before ArUco existed and that no coloured-marker search will
surface.

| Reference | What it already discloses | Distance from this work |
|---|---|---|
| Harris and Stennett, *RAPiD -- a video rate object tracker*, BMVC 1990 | The loop itself: a known 3D model at a predicted pose, control points on its edges, a one-dimensional search along each edge normal, and a linearized pose update from the normal displacements. | This is the ancestor of step 4 and step 6. Nothing about the residual form here is new. |
| Drummond and Cipolla, *Real-time visual tracking of complex structures*, IEEE TPAMI 24(7), 2002 | Adds hidden-line removal from the predicted pose (step 3), robust M-estimation over the normal residuals (step 6), and a Lie-algebra pose parameterization. Handles articulated and self-occluding structures. | The visibility prediction and the robust joint solve are both here. Our Huber loss on `se(3)` increments is the textbook version. |
| Comport, Marchand, Pressigout and Chaumette, *Real-time markerless tracking for augmented reality: the virtual visual servoing framework*, IEEE TVCG 12(4), 2006 | The same contract as a virtual visual servoing problem, with Tukey weighting; shipped as ViSP's `vpMbEdgeTracker` / `vpMbGenericTracker`, which also accepts a fiducial marker for initialization. | **The closest single reference.** An open, maintained implementation that takes a CAD model, initializes from a marker, and refines on edge-normal residuals. Any claim over the loop has to distinguish itself from a library a reviewer can `apt install`. |
| Wuest, Vial and Stricker, *Adaptive line tracking with multiple hypotheses for augmented reality*, ISMAR 2005 | Multiple candidate responses per normal search, carried forward and disambiguated rather than committed to at first crossing. | Our `_measure_profile` commits to one landmark per profile and rejects on profile shape instead. That is a simplification of this, not an advance on it. |
| Petit, Marchand and Kanani, *Combining complementary edge, keypoint and colour features in model-based tracking for highly dynamic scenes*, ICRA 2014 | Colour used as a complementary cue *inside a model-based edge tracker*, fused with edge and keypoint residuals in one pose solve. | This is step 5 -- colour supporting or rejecting predictions inside projected regions -- with the same motivation. Our colour is a hard correspondence gate rather than a fused residual, which is a difference in role, not in kind. |
| Prisacariu and Reid, *PWP3D: real-time segmentation and tracking of 3D objects*, IJCV 98(3), 2012 | 6-DoF pose of a known 3D model driven purely by region colour statistics -- the pose that best separates foreground from background colour models. | Establishes colour-plus-known-3D-model pose estimation as a whole field. Our facets are painted and semantically labelled rather than statistically segmented, but "colour tells the CAD model where it is" is not new. |
| Tjaden, Schwanecke, Schoemer and Kraus, *A region-based Gauss-Newton approach to real-time monocular multiple object tracking*, IEEE TPAMI 41(8), 2019 | Per-region colour histograms attached to a known mesh, optimized to a pose by Gauss-Newton, robust to partial occlusion. | Same as above, with the optimizer we use. |

### Where that leaves each step

| method.md step | Prior art status |
|---|---|
| 1. decode anchors, establish identity and scale | ArUco/AprilTag. Not novel. |
| 2. PnP hypotheses, paste-quadrant search | The quadrant search is unusual but is a consequence of the manufacturing choice, not an estimator contribution. |
| 3. predict front-facing facets from the pose | Drummond and Cipolla 2002 (hidden-line removal). Not novel. |
| 4. sub-pixel measurement along boundary normals | Harris and Stennett 1990. Not novel. |
| 5. colour classification inside projected supports | Petit et al. 2014; PWP3D. Not novel as a cue; the *gate* framing is a narrow difference. |
| 6. robust joint solve on 2-D corner and 1-D normal residuals | Comport et al. 2006. Not novel. |
| 7. refuse under-constrained solutions | **Candidate.** A pre-fit geometric admission -- fold edge normals onto a half turn, measure angular spread, count distinct facet planes -- run *before* the solve. Degeneracy detection by Jacobian conditioning after the fact is common; deciding from the geometry that the measurement set cannot be taken is the part to chart. `detector.py:2115`. |
| (build) two-scale print calibration | **Candidate, strongest.** Estimating an inter-anchor body scale and a sticker-image scale as separate parameters and letting only the body scale move the tool origin. Photogrammetric scale self-calibration is old; treating the printed part and the printed sticker as two independently mis-scaled objects, with different consequences for the reported TCP, is what needs a search. `detector.py:1409`. |

### Consequences for the two go/no-go gates

- **Patent search gate.** Counsel's search must cover MBT, not only marker
  patents. Add CPC `G06T7/246` (tracking) and `G06T7/75`
  (model-based pose) alongside the marker classes already listed, and hand
  counsel the six references above as named art to design around.
- **Novelty gate for the paper.** A submission that presents steps 3 to 6 as
  the contribution will be desk-rejected by any reviewer from ISMAR, ICRA, or
  TVCG. The paper's claim has to be the carrier and the calibration contract,
  with ViSP's `vpMbGenericTracker` run as a baseline rather than ignored.

## Candidate claim families

These are drafting hypotheses for counsel, not claims:

- **Method:** initialize from one or more decoded fiducials; render visible
  semantic CAD facets; form signed one-dimensional residuals at their physical
  boundaries; jointly optimize pose; accept facet refinement only when an
  information/observability criterion is met.
- **Apparatus:** a rigid carrier whose decoded anchor planes and coloured
  boundary-bearing facets are arranged to preserve pose observability across a
  specified camera envelope, with measured geometry captured in a machine
  descriptor.
- **Calibration/manufacture:** bind each physical unit's measured sticker scale,
  paste quadrant, CAD revision, print manifest, and target-to-tool transform to
  detection outputs through content hashes.
- **Multi-camera:** select or down-weight camera evidence using per-view
  geometric consistency while estimating a single target pose and its
  covariance.

The first claim family appears technically strongest. The apparatus family is
more exposed to older multi-facet and instrument-marker patents. Provenance by
itself may be viewed as routine engineering unless tied to a concrete reduction
in pose error or prevention of a specific failure.

## Disclosure controls

Chinese patent law defines prior art globally and provides only specific
six-month exceptions, such as prescribed exhibitions or conferences and
unauthorized disclosure. It should not be treated as a general grace period;
see [CNIPA Patent Law Articles 22–25](https://english.cnipa.gov.cn/art/2022/10/13/art_3068_179273.html).
Article 25 also excludes certain two-dimensional printed designs whose main
purpose is indication, which makes protection of the print sheet alone
especially uncertain.

Before filing:

1. keep this repository private and do not add an open-source license;
2. inventory every prior disclosure, demo, email recipient, customer delivery,
   and conference submission with dates and confidentiality terms;
3. settle inventorship and employer/contract ownership separately from Git
   authorship;
4. preserve dated CAD, source, raw images, lab notes, print measurements, and
   benchmark configurations;
5. ask patent counsel to search CPC classes around `G06T7/73`, `G06T7/246`,
   `G06T7/75`, `G06V20/20`, `G06V10/44`, and optical marker subclasses,
   including CN/US/EP/PCT families, and to treat model-based tracking as an
   in-scope field rather than a neighbouring one;
6. draft an element-by-element claim chart against the seven marker references
   and the seven model-based tracking references above, with ViSP's
   `vpMbGenericTracker` charted as a working implementation, not just a paper;
   and
7. file before submitting a paper, posting video, distributing binaries/PDFs,
   or opening this repository.

If international protection matters, coordinate the first filing and
twelve-month priority/PCT schedule before disclosure. WIPO likewise recommends
filing before public disclosure and using confidentiality agreements where
pre-filing disclosure is unavoidable:
[WIPO patent FAQ](https://www.wipo.int/en/web/patents/faq_patents) and
[WIPO patent protection](https://www.wipo.int/en/web/patents/protection).

## Publication readiness

The method is publishable in principle, but the current evidence is an
engineering validation set rather than a paper-ready benchmark. Before making a
superiority claim, complete the experiments in [validation.md](validation.md)
and freeze:

- task definition and primary metric;
- independent 6-DoF ground truth and its uncertainty;
- camera/lighting/range/angle/occlusion matrix;
- anchor-only, edge-only, colour-gating, and full-method ablations;
- matched-area and matched-print-quality baselines;
- runtime distribution and failure taxonomy;
- held-out physical carriers and cross-day calibration;
- statistical intervals and all exclusion rules.

The paper's strongest honest contribution is likely a co-designed target and
evidence-aware estimator for wide-baseline robot pose measurement, supported by
real occlusion/obliquity experiments. Reprojection residual and self-reported
covariance cannot substitute for external accuracy ground truth.

## Go/no-go gates

- **Patent search gate:** counsel finds a claim scope that survives the closest
  predict-project-test and multi-facet marker references *and* the model-based
  edge tracking family, ViSP included.
- **Novelty gate:** no uncontrolled public disclosure predates the intended
  priority filing.
- **Measurement gate:** external ground truth shows a significant improvement
  over strong marker baselines in at least one predeclared operating regime.
- **Reproducibility gate:** a clean environment reproduces the included example
  and a frozen benchmark from documented inputs.
- **Release gate:** ownership, inventorship, license, redaction, and export/data
  rights are signed off.

Until all five pass, keep the status as private research and avoid public
novelty or accuracy claims.
