# Contributing

Thanks for contributing to `whpc-kit`.

## Scope

This repository is the public package and reproducibility layer for the W-HPC research line.

Please keep contributions aligned with that scope:

1. reusable package code under `src/whpc/`,
2. public tests and examples,
3. reproducibility entry points and manifests,
4. user-facing documentation.

Do not add:

1. raw datasets,
2. manuscript drafts,
3. large generated result folders,
4. machine-specific paths or private notes.

## Development Setup

```bash
python3 -m pip install -e .[dev]
```

If your change touches dataset-backed reproduction utilities:

```bash
python3 -m pip install -e .[dev,repro]
```

## Before Opening A Pull Request

Run the checks that match your change.

Core test suite:

```bash
python3 -m pytest \
  tests/test_public_api_smoke.py \
  tests/test_scores.py \
  tests/test_representatives.py \
  tests/test_thresholds.py \
  tests/test_weights.py \
  tests/test_drift_monitoring.py \
  tests/test_open_world.py \
  tests/test_querying.py \
  tests/test_selection.py \
  tests/test_reports.py
```

Smoke example:

```bash
python3 examples/smoke_synthetic.py
```

Smoke reproducibility entry points:

```bash
python3 reproducibility/fp1_supervised_mm.py --mode smoke
python3 reproducibility/fp2_oneclass_adaptive.py --mode smoke
python3 reproducibility/fp3_online_drift.py --mode smoke
python3 reproducibility/fp4_openworld_uncertainty.py --mode smoke
python3 reproducibility/fp5_trustworthy.py --mode smoke
```

## Reproducibility Notes

`FP1` and `FP2` run full protocol logic natively in this repository.

`FP3`, `FP4`, and `FP5` currently use companion-checkout orchestration in `../W-HPC` for exact manuscript-aligned full runs. Please preserve that distinction unless you are intentionally refactoring the public reproducibility contract.

## Pull Request Style

1. Keep changes focused.
2. Update docs when the public behavior changes.
3. Prefer small, reviewable commits.
4. Explain scope, verification, and any known limitations in the PR description.
