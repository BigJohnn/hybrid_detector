# Development history

This file records the engineering lineage without depending on another
repository or dataset location. Dates use the Asia/Shanghai time zone.

## 2026-09-04: V1 concept

The first design combined two or more decoded planar anchors with four to six
large coloured CAD facets. The intended division of labour was already clear:
anchors provide identity and metric initialization; the larger physical
boundaries provide longer-baseline localization.

The design gates were:

1. generate exact 3D vertices, normals, and anchor-to-target transforms;
2. render calibrated viewpoints to inspect projected size and self-occlusion;
3. compare against a conventional marker carrier on real multi-camera data.

## 2026-09-07: measured prototype

The prototype was rebuilt from AP242 geometry with semantic surface colours.
Three `DICT_6X6_50` anchors were mounted and their paste orientations recorded.
The nominal 48 mm sticker size was replaced by the measured 47.24 mm value.

Deliverables preserved in this repository include:

- the original and colour-annotated STEP models;
- measured JSON descriptors and 2D marker layout;
- an exact-scale A4 print PDF and print manifest;
- an installation guide image;
- deterministic code for rebuilding geometry and visual aids.

## 2026-09-09: first real-image validation

Seven calibrated fisheye views were processed over 2,403 synchronized frames.
The joint estimator returned a pose for every processable frame. These results
establish pipeline operation and internal consistency, not traceable accuracy;
the precise scope and limitations are in [validation.md](validation.md).

## 2026-09-10: standalone research package

The detector, geometry/calibration contracts, CLIs, tests, real-image example,
manufacturing assets, and evidence documents were assembled as one independent
Python package. An automated test scans both imports and text files to prevent
accidental dependencies on external source trees.

The next milestone is a frozen preregistered benchmark with independent pose
ground truth, followed by intellectual-property review before public release.
