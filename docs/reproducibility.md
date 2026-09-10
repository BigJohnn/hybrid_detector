# Reproducibility guide

## Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
pytest
```

Record Python, OpenCV, NumPy, SciPy, operating system, and CPU architecture for
every benchmark. Do not compare wall-clock numbers across unrecorded machines.

## Real-image smoke test

```bash
python examples/single_image.py --output outputs/example
```

The example contains one real 1920x1080 image, a matching fisheye calibration,
and no external file references.

## Detector CLI

```bash
hybrid-detect \
  --model assets/models/hybrid_carrier_v1_20260907.json \
  --image cam14=examples/data/cam14_ep1_frame000000.jpg \
  --camera cam14=examples/data/cam14_intrinsics.json \
  --out outputs/example/report.json \
  --overlay outputs/example
```

## Rebuild a descriptor

```bash
hybrid-build-model assets/cad/rig0907.step \
  --out outputs/rebuilt_model.json \
  --anchor-ids 30 31 32 \
  --dictionary DICT_6X6_50 \
  --marker-size-mm 47.24
```

The exact production command also requires the recorded repaint and paste
quadrants. Read `hybrid-build-model --help` and
`assets/print/rig0907_patterns_RGB_exact1mm_manifest.json` before regenerating
the released model.

## Manufacturing assets

- Print `assets/print/rig0907_patterns_RGB_exact1mm_A4.pdf` at 100% scale with
  all printer scaling disabled.
- Verify dimensions using the manifest's scale marks.
- Apply facets using `assets/print/rig0907_RGB_installation_guide.png`.
- Record actual marker side length and paste quadrant after assembly.

## Provenance rules

For every experiment archive:

- descriptor and camera-calibration SHA-256 values;
- input image hashes or immutable dataset revision;
- command and full configuration;
- detector version/commit;
- per-frame raw evidence, not only fused poses;
- environment and runtime;
- exclusions and reasons.

Do not overwrite a released descriptor in place. Add a dated model and state
which physical build it describes.
