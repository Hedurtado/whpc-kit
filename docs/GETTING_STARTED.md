# Getting Started

## Install

For development:

```bash
python3 -m pip install -e .[dev]
```

If you want explainability baselines later:

```bash
python3 -m pip install -e .[dev,explain]
```

## Run The Synthetic Smoke Example

After installation:

```bash
python3 examples/smoke_synthetic.py
```

Expected behavior:

1. `WHPCClassifier` predicts the synthetic axis dataset correctly.
2. `MMWHPCClassifier` predicts the same synthetic dataset correctly.
3. `OCMMWHPCDetector` returns higher scores for normal-aligned points than for off-axis anomalous points.

If you prefer not to install the package yet, you can run:

```bash
PYTHONPATH=src python3 examples/smoke_synthetic.py
```

## Run The Current Core Test Set

```bash
python3 -m pytest \
  tests/test_public_api_smoke.py \
  tests/test_scores.py \
  tests/test_representatives.py \
  tests/test_thresholds.py \
  tests/test_weights.py
```

## Current Public API Layers

Top-level imports:

```python
from whpc import WHPCClassifier, MMWHPCClassifier, OCMMWHPCDetector
```

Modular imports:

```python
from whpc.models import WHPCClassifier, MMWHPCClassifier
from whpc.core import cosine_score, build_class_representative, fit_threshold
from whpc.evaluation import robustness_metrics
```

## Reproducibility Entry Points

Current smoke-level manuscript entry points:

```bash
python3 reproducibility/fp1_supervised_mm.py --mode smoke
python3 reproducibility/fp2_oneclass_adaptive.py --mode smoke
python3 reproducibility/fp3_online_drift.py --mode smoke
python3 reproducibility/fp4_openworld_uncertainty.py --mode smoke
python3 reproducibility/fp5_trustworthy.py --mode smoke
```

Full dataset-backed reproduction for M1 and M2 is documented in the manifests and remains a follow-up task.
