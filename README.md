# whpc-kit

`whpc-kit` is the public software and reproducibility repository for the W-HPC research line.
It packages the core models behind a Python API and exposes a clean bridge to the manuscript results without shipping datasets, drafts, or large research artifacts.

## What It Provides

1. A Python package under `src/whpc/` with supervised, multimodal, one-class, adaptive, open-world, querying, explainability, and robustness utilities.
2. Small examples and tests suitable for public CI.
3. Reproducibility entry points for `FP1-FP5`, covering the manuscript line from `M1` to `M5`.
4. Documentation for installation, scope, API naming, and product direction.

## What It Does Not Include

1. Manuscript drafts.
2. Raw datasets.
3. Large generated result folders.
4. Private notes or machine-specific paths.
5. Exploratory research scripts that are not part of the public reproducibility contract.

## Install

Development install:

```bash
python3 -m pip install -e .[dev]
```

If you want dataset-backed reproduction support:

```bash
python3 -m pip install -e .[dev,repro]
```

If you want optional explainability baselines later:

```bash
python3 -m pip install -e .[dev,repro,explain]
```

## Quick Start

Run the synthetic smoke example:

```bash
python3 examples/smoke_synthetic.py
```

Run the smoke reproducibility entry points:

```bash
python3 reproducibility/fp1_supervised_mm.py --mode smoke
python3 reproducibility/fp2_oneclass_adaptive.py --mode smoke
python3 reproducibility/fp3_online_drift.py --mode smoke
python3 reproducibility/fp4_openworld_uncertainty.py --mode smoke
python3 reproducibility/fp5_trustworthy.py --mode smoke
```

## Reproducibility Model

The public reproducibility layer is intentionally split in two:

1. `FP1` and `FP2` run full protocol logic natively inside `whpc-kit`.
2. `FP3`, `FP4`, and `FP5` keep lightweight public runners here, but their `full` mode orchestrates the frozen experiment scripts from a companion `../W-HPC` checkout.

This keeps the public repository compact while preserving exact manuscript-aligned replay for the later stages.

Examples:

```bash
python3 reproducibility/fp1_supervised_mm.py --mode full \
  --unsw-raw-dir ../W-HPC/data/raw \
  --nsl-raw-dir ../W-HPC/data/raw/archive

python3 reproducibility/fp2_oneclass_adaptive.py --mode full \
  --unsw-raw-dir ../W-HPC/data/raw \
  --nsl-raw-dir ../W-HPC/data/raw/archive \
  --cic-raw-dir ../W-HPC/data/raw/CIC-IDS2017

python3 reproducibility/fp3_online_drift.py --mode full --research-root ../W-HPC
python3 reproducibility/fp4_openworld_uncertainty.py --mode full --research-root ../W-HPC
python3 reproducibility/fp5_trustworthy.py --mode full --research-root ../W-HPC
```

## Manuscript Mapping

| Manuscript | Parts | Focus |
| --- | --- | --- |
| M1 | Parts 1-2 | Supervised W-HPC and multimodal W-HPC |
| M2 | Parts 3-4 | One-class multimodal W-HPC and adaptive one-class W-HPC |
| M3 | Parts 5-6 | Online streaming W-HPC and drift-aware monitoring |
| M4 | Parts 7-8 | Open-world detection and uncertainty/active querying |
| M5 | Parts 9-10 | Explainable W-HPC and robustness analysis |

## Public API Notes

Top-level imports:

```python
from whpc import WHPCClassifier, MMWHPCClassifier, OCMMWHPCDetector
```

`WHPCSelector` is included as an advisory helper. It does not replace dataset-specific validation or manuscript-level protocol selection.

## Repository Layout

```text
src/whpc/            reusable package code
tests/               public tests and smoke coverage
examples/            small runnable examples
reproducibility/     manuscript entry points and manifests
docs/                user-facing project documentation
```

## Documentation

Start with:

```text
docs/GETTING_STARTED.md
docs/SCOPE.md
docs/REPRODUCIBILITY_NOTES.md
docs/PRODUCT_ROADMAP.md
docs/API_NAMING_PROPOSAL.md
reproducibility/README.md
```

## Roadmap

| Version | Scope |
| --- | --- |
| v0.1 | Core W-HPC, MM-WHPC, OC-WHPC, OC-MM-WHPC, Adaptive OC-MM-WHPC, tests, examples |
| v0.2 | Online and drift-aware W-HPC |
| v0.3 | Open-world, uncertainty-aware, and active-querying W-HPC |
| v0.4 | Explainable and robust W-HPC |
| v1.0 | Stable API, docs, reproducibility manifests, selector, reports, release-ready package |
| post-v1.0 | Software paper, optional CLI, community GUI, advanced product planning |

## License

Apache License 2.0. See [LICENSE](LICENSE).
