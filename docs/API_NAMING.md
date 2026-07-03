# API Naming Guide

This guide defines the target public names for `whpc-kit`. It does not require immediate renaming of the imported research code; aliases can be introduced gradually to preserve compatibility.

## Naming Principles

1. Use `Classifier` for supervised closed-set estimators.
2. Use `Detector` for anomaly, one-class, open-world, uncertainty, and streaming detection estimators.
3. Use short W-HPC family names in public imports.
4. Keep research-specific class names as backwards-compatible aliases when useful.
5. Avoid exposing manuscript or part numbers in public class names.

## Recommended Public Classes

| Recommended name | Current or source name | Scope |
| --- | --- | --- |
| `WHPCClassifier` | `WHPCClassifier` | Supervised W-HPC |
| `MMWHPCClassifier` | `MMWHPCClassifier` | Multimodal supervised W-HPC |
| `WHPCOpenSetClassifier` | `WHPCOpenSetClassifier` | Threshold-based open-set wrapper for supervised W-HPC |
| `OCWHPCDetector` | new or future alias | One-class W-HPC |
| `OCMMWHPCDetector` | `OCMMWHPCClassifier` | One-class multimodal W-HPC |
| `AdaptiveOCMMWHPCDetector` | `AdaptiveOCMMWHPCClassifier` | Adaptive one-class multimodal W-HPC |
| `OnlineWHPCDetector` | future class | Online/streaming W-HPC |
| `DriftAwareWHPCDetector` | current drift utilities plus future estimator | Drift-aware W-HPC |
| `OpenWorldWHPCDetector` | `OpenWorldOCMMWHPCClassifier` | Open-world/unknown detection |
| `UncertaintyWHPCDetector` | querying/uncertainty utilities plus future estimator | Rejection and uncertainty-aware W-HPC |
| `ActiveQueryWHPCDetector` | querying utilities plus future estimator | Active querying with feedback |
| `RobustWHPCAnalyzer` | robustness utilities | Robustness and perturbation analysis |
| `WHPCSelector` | future class | Advisory model recommendation by scenario |

## Compatibility Aliases

The current research names can remain available during transition:

| Current name | Proposed public alias |
| --- | --- |
| `OCMMWHPCClassifier` | `OCMMWHPCDetector` |
| `AdaptiveOCMMWHPCClassifier` | `AdaptiveOCMMWHPCDetector` |
| `OpenWorldOCMMWHPCClassifier` | `OpenWorldWHPCDetector` |

## Preferred Import Style

For common users:

```python
from whpc import WHPCClassifier, MMWHPCClassifier, OCMMWHPCDetector
```

For advanced users:

```python
from whpc.models import MMWHPCClassifier
from whpc.selection import WHPCSelector
from whpc.robustness import RobustWHPCAnalyzer
```

## Parameter Naming

Prefer clear public parameters:

| Preferred parameter | Notes |
| --- | --- |
| `M` | Short alias for number of representatives when consistent with manuscripts. |
| `n_representatives` | Explicit alias for users who prefer scikit-learn-style names. |
| `aggregation` | Public alias for prototype aggregation. |
| `threshold_metric` | Short public alias for threshold-selection metric. |
| `random_state` | Keep scikit-learn convention. |

Implementation can accept both manuscript-style aliases and explicit names, but documentation should choose one primary style per example.

## Recommendation

Use detector names for one-class, open-world, uncertainty, active-querying, and drift-aware estimators. This is clearer for intrusion-detection users because these models often output anomaly/unknown/reject decisions rather than ordinary multiclass labels.

`WHPCSelector` should remain explicitly advisory in `v1.0.0`: it can expose scenario-driven heuristics and explanatory reasons, but it should not be presented as a substitute for empirical model selection on a target dataset.
