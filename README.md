# whpc-kit

`whpc-kit` is the public software artifact for the W-HPC research line. It is intended to provide a clean Python package, documented APIs, and reproducibility support for the results reported across the W-HPC manuscripts.

The research repository keeps manuscripts, exploratory experiments, local results, and draft assets. This repository should remain focused on reusable software and reproducibility entry points.

## Current Scope

1. Package the W-HPC model family as reusable Python software.
2. Provide a scikit-learn-like API for supervised, multimodal, one-class, adaptive, online, drift-aware, open-world, uncertainty/querying, explainability, and robustness workflows.
3. Provide reproducibility manifests, smoke commands, and manuscript-level full entry points for M1-M5 without embedding datasets in the public repository.
4. Keep examples and tests small enough for public CI.
5. Add model selection and report/model-card utilities before v1.0.
6. Use Apache License 2.0 for public release.

Current selector status:

1. `WHPCSelector` is advisory.
2. It provides scenario-based heuristic recommendations only.
3. It does not replace dataset-specific validation, ablation, or manuscript-level protocol selection.

See:

```text
docs/SCOPE.md
docs/PRODUCT_ROADMAP.md
docs/API_NAMING_PROPOSAL.md
docs/GETTING_STARTED.md
docs/RELEASE_CHECKLIST_v0.1.md
docs/RELEASE_CHECKLIST_v1.0.md
```

## Product Direction

The priority is the Python library/API. A CLI may be useful later, but it is not required for the first stable versions. GUI products such as WHPC Studio Community or WHPC Studio Pro should be considered only after the core package reaches v1.0.

Planned layers:

1. `whpc-kit`: open-source Python library and reproducibility artifact.
2. Optional WHPC CLI: terminal interface for training, evaluation, prediction, and reports.
3. WHPC Studio Community: local/free GUI for non-programming users.
4. WHPC Studio Pro: advanced or commercial product with dashboards, monitoring, connectors, alerts, and integrations.

## Manuscript Mapping

| Manuscript | Parts | Focus |
| --- | --- | --- |
| M1 | Parts 1-2 | Supervised W-HPC and multimodal W-HPC |
| M2 | Parts 3-4 | One-class multimodal W-HPC and adaptive one-class W-HPC |
| M3 | Parts 5-6 | Online streaming W-HPC and drift-aware monitoring |
| M4 | Parts 7-8 | Open-world detection and uncertainty/active querying |
| M5 | Parts 9-10 | Explainable W-HPC and robustness analysis |

## Repository Boundary

This repository should include:

1. `src/whpc/`: reusable implementation.
2. `tests/`: unit and smoke tests.
3. `examples/`: small runnable examples.
4. `reproducibility/`: manifests and lightweight reproduction entry points.
5. `docs/`: user-facing documentation.
6. optional `reports/` or report utilities inside the package.

This repository should not include:

1. manuscript drafts;
2. raw datasets;
3. large generated result folders;
4. exploratory scripts that are not part of the public artifact;
5. private notes or local-only paths.

## Initial Development Plan

1. Import the stable `whpc` package code from the research repository.
2. Normalize the public API around `WHPCClassifier`, `MMWHPCClassifier`, `OCWHPCDetector`, `OCMMWHPCDetector`, and `AdaptiveOCMMWHPCDetector`.
3. Port the tests needed to protect the public API.
4. Add examples that run without private datasets.
5. Define reproducibility manifests for M1-M5.
6. Add smoke reproduction commands that can run without private datasets.
7. Add full reproduction instructions for users who obtain the required datasets.
8. Add model selector and report/model-card utilities before v1.0.

## Version Roadmap

| Version | Scope |
| --- | --- |
| v0.1 | Core W-HPC, MM-WHPC, OC-WHPC, OC-MM-WHPC, Adaptive OC-MM-WHPC, tests, examples |
| v0.2 | Online and drift-aware W-HPC |
| v0.3 | Open-world, uncertainty-aware, and active-querying W-HPC |
| v0.4 | Explainable and robust W-HPC |
| v1.0 | Stable API, model selector, reports, docs, reproducibility manifests, release-ready package |
| post-v1.0 | Software paper, optional CLI, WHPC Studio Community, possible WHPC Studio Pro planning |

## Quick Start

Development install:

```bash
python3 -m pip install -e .[dev]
```

Full reproducibility install:

```bash
python3 -m pip install -e .[dev,repro]
```

Synthetic smoke example:

```bash
python3 examples/smoke_synthetic.py
```

Current smoke-level reproducibility commands:

```bash
python3 reproducibility/fp1_supervised_mm.py --mode smoke
python3 reproducibility/fp2_oneclass_adaptive.py --mode smoke
python3 reproducibility/fp3_online_drift.py --mode smoke
python3 reproducibility/fp4_openworld_uncertainty.py --mode smoke
python3 reproducibility/fp5_trustworthy.py --mode smoke
```

Dataset-backed reproduction currently available for FP1-FP5:

```bash
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

These commands write JSON summaries plus validation/test CSV files under `reproducibility/artifacts/`.

By default, `reproducibility/fp1_supervised_mm.py --mode full` reproduces the frozen M1 MM-WHPC protocol rather than an exploratory multimodal sweep.
By default, `reproducibility/fp2_oneclass_adaptive.py --mode full` reproduces the frozen M2 protocol and compares current reruns against the published M2 table values stored in `reproducibility/reference/m2_published_reference.json`.
By default, `reproducibility/fp3_online_drift.py`, `fp4_openworld_uncertainty.py`, and `fp5_trustworthy.py` orchestrate the frozen M3-M5 experiment scripts from a companion `../W-HPC` checkout to keep the manuscript-level protocol exact while the public package remains lightweight.

Current FP2 status:

1. UNSW-NB15 is aligned to the published M2 values up to rounding noise.
2. NSL-KDD is aligned with a small drift on the fixed frozen Part 4 row.
3. CIC-IDS2017 keeps an explicit distinction between published M2 values and current recomputed values from the research checkout.

Current FP3-FP5 status:

1. smoke runners are self-contained and validate the public package utilities directly;
2. full runners produce `whpc-kit` artifact summaries while delegating the exact frozen paper protocol to the companion `W-HPC` checkout;
3. this keeps the public repository small and honest while still exposing a reproducible bridge to M3-M5.

## Selector Scope

`WHPCSelector` should be interpreted as a lightweight recommendation helper in `v1.0.0`, not as a validated automatic model-selection layer. It is useful for orienting users toward a plausible W-HPC family member, but the final choice must still be confirmed with dataset-specific validation and the relevant reproduction protocol.

## License

Apache License 2.0. See `LICENSE`.
