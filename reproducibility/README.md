# Reproducibility

This directory contains paper-aligned reproduction units for `whpc-kit`.

## Current Status

The repository currently provides:

1. smoke scripts for `FP1` through `FP5`;
2. native full dataset-backed entry points for FP1/M1 and FP2/M2;
3. companion-checkout full orchestration entry points for FP3/M3, FP4/M4, and FP5/M5;
4. JSON manifests describing scope, supported modes, expected datasets, and expected outputs.

For a compact paper-to-command map, see [`REPRODUCIBILITY_MATRIX.md`](REPRODUCIBILITY_MATRIX.md).

Current entry points:

```bash
python3 reproducibility/fp1_supervised_mm.py --mode smoke
python3 reproducibility/fp2_oneclass_adaptive.py --mode smoke
python3 reproducibility/fp3_online_drift.py --mode smoke
python3 reproducibility/fp4_openworld_uncertainty.py --mode smoke
python3 reproducibility/fp5_trustworthy.py --mode smoke
```

Current full entry points:

```bash
python3 reproducibility/fp1_supervised_mm.py --mode full --unsw-raw-dir <path> --nsl-raw-dir <path>
python3 reproducibility/fp2_oneclass_adaptive.py --mode full --unsw-raw-dir <path> --nsl-raw-dir <path> --cic-raw-dir <path>
python3 reproducibility/fp3_online_drift.py --mode full --research-root ../W-HPC
python3 reproducibility/fp4_openworld_uncertainty.py --mode full --research-root ../W-HPC
python3 reproducibility/fp5_trustworthy.py --mode full --research-root ../W-HPC
```

FP1 full mode now defaults to the frozen M1 protocol for MM-WHPC:

1. uniform local sample weights;
2. max aggregation;
3. symmetric representative-count search over `M in {1,2,3,4,5}`;
4. inner validation on the official training split;
5. final refit on the full official training split.

Use `--mm-selection-protocol custom` only when you intentionally want exploratory MM-WHPC selection outside the manuscript freeze.

FP2 full mode now defaults to the frozen M2 protocol and distinguishes between:

1. the current research freeze reconstructed from `../W-HPC/results/part3_final_thresholds/final_thresholds.json`;
2. the published M2 reference values stored in `reproducibility/reference/m2_published_reference.json`.

Each full FP2 dataset summary includes a `reference_alignment` block comparing the current rerun against the published M2 table values.

FP3-FP5 full mode currently use a different strategy:

1. the public `whpc-kit` repository keeps the smoke validation and reusable APIs local;
2. the exact frozen manuscript protocol for M3-M5 is orchestrated from a local companion checkout at `../W-HPC`;
3. each runner writes a compact `whpc-kit` summary JSON under `reproducibility/artifacts/` plus the generated companion artifacts it invoked.

Example with the local research checkout next to `whpc-kit`:

```bash
python3 -m pip install -e .[dev,repro]
python3 reproducibility/fp1_supervised_mm.py --mode full \
  --unsw-raw-dir ../W-HPC/data/raw \
  --nsl-raw-dir ../W-HPC/data/raw/archive
python3 reproducibility/fp2_oneclass_adaptive.py --mode full \
  --unsw-raw-dir ../W-HPC/data/raw \
  --nsl-raw-dir ../W-HPC/data/raw/archive \
  --cic-raw-dir ../W-HPC/data/raw/CIC-IDS2017
python3 reproducibility/fp3_online_drift.py --mode full \
  --research-root ../W-HPC
python3 reproducibility/fp4_openworld_uncertainty.py --mode full \
  --research-root ../W-HPC
python3 reproducibility/fp5_trustworthy.py --mode full \
  --research-root ../W-HPC
```

Optional explicit FP2 references:

```bash
python3 reproducibility/fp2_oneclass_adaptive.py --mode full \
  --unsw-raw-dir ../W-HPC/data/raw \
  --nsl-raw-dir ../W-HPC/data/raw/archive \
  --cic-raw-dir ../W-HPC/data/raw/CIC-IDS2017 \
  --m2-final-thresholds ../W-HPC/results/part3_final_thresholds/final_thresholds.json \
  --m2-published-reference reproducibility/reference/m2_published_reference.json
```

Expected full-mode artifacts:

1. `reproducibility/artifacts/fp1_full_summary.json`
2. `reproducibility/artifacts/fp1_full_validation.csv`
3. `reproducibility/artifacts/fp1_full_test.csv`
4. `reproducibility/artifacts/fp2_full_summary.json`
5. `reproducibility/artifacts/fp2_full_validation.csv`
6. `reproducibility/artifacts/fp2_full_test.csv`
7. `reproducibility/artifacts/fp3_full_summary.json`
8. `reproducibility/artifacts/fp4_full_summary.json`
9. `reproducibility/artifacts/fp5_full_summary.json`

Current alignment note:

1. FP2 matches the published M2 values for UNSW-NB15 up to rounding noise.
2. FP2 matches NSL-KDD closely, with a small drift on the fixed frozen Part 4 row.
3. CIC-IDS2017 is tracked as a published-freeze vs recomputed-checkout discrepancy, not as a silent reproduction success.

## Design Rules

Each paper-aligned unit should eventually define:

1. smoke mode;
2. full mode;
3. expected datasets;
4. output files;
5. output schemas;
6. paper tables/figures supported by the outputs.

Smoke mode must work without private datasets. Full mode may require public datasets placed locally according to the manifest.
For CIC-IDS2017 full-mode runs, install the `repro` extra so parquet readers are available.
For FP3-FP5 full-mode runs, the current public contract also expects a local companion checkout at `../W-HPC` containing the frozen manuscript experiment scripts and public datasets.
