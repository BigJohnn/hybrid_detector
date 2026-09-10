# Method

## Problem statement

Given one or more calibrated images, estimate the rigid transform
`T_world_target` of a known hybrid target. A valid answer must state which
measurements support it; a plausible pose without correspondence evidence is
not a measurement.

## Coordinate contract

- Descriptor points are metres in the target CAD frame.
- `CameraCalibration.T_base_cam` maps camera-frame points into a caller-defined
  world frame.
- Single-camera use normally sets `T_base_cam = I`, so the returned
  `T_base_rig` is `T_camera_target`.
- Anchor corner order follows OpenCV ArUco corner order after applying the
  measured sticker paste quadrant.
- Both OpenCV rational and fisheye/equidistant camera models are explicit.

## Estimator

### 1. Decode anchors

Detect ArUco quads and keep only IDs declared by the descriptor. Marker family,
physical size, 3D corner coordinates, and paste rotation all belong to the
model. A dictionary mismatch is a hard configuration error.

### 2. Generate coarse hypotheses

Each anchor produces PnP candidates. With multiple anchors, candidates are
scored jointly. Unknown paste orientation may be enumerated during model
development; production descriptors should pin it.

### 3. Predict visibility

The coarse pose projects the CAD skin. A facet is eligible only if it is
front-facing, sufficiently large in the image, inside the calibrated field of
view, and not substantially self-occluded.

### 4. Classify colour in projected ROIs

Colour references are estimated from anchor black/white regions when possible.
Chromaticity scores are evaluated only where CAD predicts a facet, avoiding
global colour segmentation and its unconstrained correspondences.

### 5. Measure physical boundaries

For each eligible CAD edge, sample intensity/colour profiles along the projected
normal. The descriptor records whether ink is centred, one-sided, or absorbed
by a dark facet. The detector returns pixel location, normal, uncertainty,
contrast, band width, and residual for every accepted sample.

### 6. Robust pose refinement

Optimize a six-parameter pose against:

- two-dimensional anchor corner residuals; and
- one-dimensional edge-normal residuals.

Huber loss limits the influence of segmentation outliers. Edge constraints must
span enough pose directions; otherwise the anchor-only result is retained.

### 7. Multi-camera solve

Each camera contributes anchor corners and optional edge samples in its own
calibration model. The joint optimization estimates one target pose. Cameras
with excessive residual can be excluded explicitly rather than averaged into
the result.

## Failure semantics

The detector distinguishes:

- no declared anchor decoded;
- no geometrically consistent hypothesis;
- insufficient usable facets;
- insufficient edge samples;
- underconstrained edge geometry;
- successful anchor pose with facet refinement skipped;
- successful joint anchor-and-facet pose.

This distinction is essential for diagnosing optics, print quality, visibility,
calibration, and model errors separately.

## Physical model

The 0907 descriptor records:

- anchor IDs 30/31/32 from `DICT_6X6_50`;
- measured sticker side length 47.24 mm;
- paste quadrants 30 -> 3, 31 -> 3, 32 -> 2;
- CAD facet polygons, normals, adjacency, and semantic colours;
- printed-border geometry and occlusion triangles;
- a socket sphere whose centre defines the target origin.

The algorithm never infers the target-to-tool transform from an image. That is
a separate mechanical/calibration contract.
