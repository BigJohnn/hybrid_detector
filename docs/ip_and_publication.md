# IP and publication assessment

Assessment date: 2026-09-10. This is an engineering prior-art screen, not a
patentability opinion or freedom-to-operate analysis.

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
5. ask patent counsel to search CPC classes around `G06T7/73`, `G06V20/20`,
   `G06V10/44`, and optical marker subclasses, including CN/US/EP/PCT families;
6. draft an element-by-element claim chart against at least the seven references
   above; and
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
  predict-project-test and multi-facet marker references.
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
