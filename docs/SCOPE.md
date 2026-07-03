# whpc-kit Scope

`whpc-kit` is the software-first artifact for the W-HPC research line. Its purpose is to turn the W-HPC family into a reusable Python package, keep a reproducibility bridge to the manuscripts, and create the technical base for later user-facing products.

The immediate priority is the Python library and reproducibility layer. A software paper, graphical applications, and commercial products should come only after the core package is stable.

## Product Layers

### 1. Python Library/API

This is the main scope of `whpc-kit`.

The package should expose a scikit-learn-like API:

```python
from whpc import WHPCClassifier, MMWHPCClassifier

model = MMWHPCClassifier(M=5, aggregation="max")
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
scores = model.decision_function(X_test)
```

The distribution name is `whpc-kit`, but the import package should remain `whpc`:

```python
from whpc import WHPCClassifier
```

This avoids breaking the research code while allowing the public repository to have a clear product name.

### 2. Reproducibility Layer

The kit must provide a clean bridge from software to manuscripts. It should not contain manuscript drafts or large local result folders, but it must document and validate how the central results can be reproduced.

Expected manuscript-level entry points:

```bash
python reproducibility/fp1_supervised_mm.py
python reproducibility/fp2_oneclass_adaptive.py
python reproducibility/fp3_online_drift.py
python reproducibility/fp4_openworld_uncertainty.py
python reproducibility/fp5_trustworthy.py
```

Each entry point should eventually support:

1. a smoke mode that runs with synthetic or tiny public data;
2. a full mode for users who place the required public datasets under `data/`;
3. result schema validation;
4. explicit links to the manuscript tables and figures it supports.

Current status:

1. `FP1` and `FP2` already have native full modes inside `whpc-kit`.
2. `FP3`, `FP4`, and `FP5` expose full modes through companion-checkout orchestration of the frozen manuscript scripts in `../W-HPC`.
3. This split is intentional for v1.0: the public package stays compact, while exact manuscript replay remains available.

### 3. Optional CLI

A command-line interface can be useful, but it should not be a priority before the Python API is stable.

Potential future examples:

```bash
whpc train --model mm-whpc --data train.csv --target label --M 5
whpc evaluate --model saved_model.pkl --data test.csv
whpc predict --model saved_model.pkl --input new_data.csv --output predictions.csv
```

Current decision: keep CLI as optional post-API work. Do not block v0.1-v0.4 on CLI development.

### 4. WHPC Studio Community

After `whpc-kit` reaches a stable v1.0, a local/community GUI can be developed as a separate product or separate repository.

Target users: people who want to experiment with W-HPC without writing Python code.

Potential functions:

1. load CSV files;
2. choose W-HPC, MM-WHPC, or OC-MM-WHPC;
3. train and evaluate models;
4. show metrics and confusion matrices;
5. inspect scores, margins, and explanations;
6. export predictions and reports;
7. save trained models.

Possible technologies:

1. Streamlit;
2. Gradio;
3. PySide/PyQt;
4. local web app with Python backend.

This should not be mixed into the core library until the API is stable.

### 5. WHPC Studio Pro

This is a longer-term advanced or commercial product, separate from the open-source core.

Potential functions:

1. dashboards;
2. advanced reports;
3. log connectors;
4. anomaly analysis workflows;
5. decision explanations;
6. monitoring and alerts;
7. model comparison;
8. support for larger datasets;
9. professional export;
10. IDS/SIEM integration;
11. users, projects, and governance.

This layer should remain out of scope for the open-source core unless a specific reason appears later.

## Model Scope

The Python package should eventually expose the central W-HPC family:

| Public class | Model |
| --- | --- |
| `WHPCClassifier` | Supervised W-HPC |
| `MMWHPCClassifier` | Multimodal W-HPC |
| `OCWHPCDetector` | One-class W-HPC |
| `OCMMWHPCDetector` | One-class multimodal W-HPC |
| `AdaptiveOCMMWHPCDetector` | Adaptive representative/mode selection |
| `OnlineWHPCDetector` | Online or streaming W-HPC |
| `DriftAwareWHPCDetector` | Drift-aware W-HPC |
| `OpenWorldWHPCDetector` | Open-world/unknown detection |
| `UncertaintyWHPCDetector` | Uncertainty and rejection-aware W-HPC |
| `ActiveQueryWHPCDetector` | Query/feedback-aware W-HPC |
| `ExplainableWHPCMixin` or utilities | Scores, margins, representatives, local/global explanations |
| `RobustWHPCAnalyzer` or utilities | Perturbation and robustness analysis |

Names may be refined during implementation, but the public API should be consistent and documented.

## Architecture Target

The package is organized around a modular architecture. The practical target is:

```text
src/whpc/
  core/
    scores.py
    prototypes.py
    margins.py
    distances.py
  models/
    whpc.py
    mm_whpc.py
    oc_whpc.py
    oc_mm_whpc.py
    adaptive_oc_mm_whpc.py
    online_whpc.py
    drift_aware_whpc.py
    open_world_whpc.py
    uncertainty_whpc.py
  selection/
    model_selector.py
  explainability/
    explanations.py
  robustness/
    perturbation_tests.py
  evaluation/
    metrics.py
  reports/
    model_card.py
  datasets/
    loaders.py
```

Supporting directories:

```text
tests/
examples/
docs/
reproducibility/
```

## User Scope

The kit is intended for:

1. machine learning and cybersecurity researchers;
2. students reproducing W-HPC models;
3. intrusion detection laboratories;
4. institutions with network traffic data;
5. data teams comparing lightweight models;
6. organizations that need explainable and adaptable prototypes.

## Version Roadmap

| Version | Scope |
| --- | --- |
| v0.1 | W-HPC, MM-WHPC, OC-WHPC, OC-MM-WHPC, Adaptive OC-MM-WHPC, basic metrics, examples, tests |
| v0.2 | Online W-HPC and Drift-Aware W-HPC |
| v0.3 | Open-World W-HPC, Uncertainty-Aware W-HPC, and Active Querying |
| v0.4 | Explainable W-HPC and Robust W-HPC |
| v1.0 | Stable API, documentation, model selector, reports/model cards, reproducibility manifests, tests, release-ready package |
| post-v1.0 | Software paper, optional CLI, WHPC Studio Community, possible WHPC Studio Pro planning |

## Reproducibility Scope

`whpc-kit` should reproduce central results, not every exploratory experiment.

Each FP/manuscript reproduction unit should define:

1. required datasets;
2. expected data layout;
3. command for smoke mode;
4. command for full mode;
5. expected output files;
6. output schemas;
7. expected metrics or tolerances where appropriate;
8. manuscript tables/figures supported by the outputs.

## Public Repository Rules

For the public repository:

1. confirm Apache 2.0 is acceptable for the project;
2. review university/institutional intellectual-property rules;
3. confirm coauthor approval if required;
4. avoid committing raw datasets;
5. avoid committing private paths, private notes, or unpublished manuscript drafts;
6. document dataset acquisition instead of redistributing datasets;
7. include tests and maintenance/versioning policy.

## Non-Goals

1. Do not build WHPC Studio before the Python API is stable.
2. Do not prioritize CLI unless it becomes necessary for reproducibility.
3. Do not include manuscripts in this repository.
4. Do not include large generated result folders.
5. Do not port every exploratory experiment from the research workspace.
6. Do not present W-HPC as a universal IDS solution.

## Communication Principle

The project should be presented as:

> An open-source research, reproduction, and evaluation toolkit for the W-HPC family, aimed at supporting academic and institutional adoption in intrusion detection.

It should not be presented as a complete production IDS by itself.
