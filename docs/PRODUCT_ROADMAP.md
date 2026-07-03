# Product Roadmap

This roadmap separates the core open-source package from later product layers.

## Stage 1: Core Python Library

Goal: make `whpc-kit` installable, tested, and usable from Python.

Deliverables:

1. `src/whpc/` package with stable imports.
2. Core models from M1-M2.
3. Examples that run with synthetic or tiny public datasets.
4. Unit tests for scores, representatives, prediction, thresholds, and metrics.
5. Public API documentation.

Target version: v0.1.

## Stage 2: Streaming And Drift

Goal: add the dynamic W-HPC family.

Deliverables:

1. online update API;
2. drift monitoring utilities;
3. streaming examples;
4. smoke reproducibility for M3.

Target version: v0.2.

## Stage 3: Open-World And Querying

Goal: support unknown detection, rejection, uncertainty, and feedback.

Deliverables:

1. open-world detector;
2. uncertainty/rejection utilities;
3. active-querying utilities;
4. smoke reproducibility for M4.

Target version: v0.3.

## Stage 4: Trustworthy W-HPC

Goal: expose explainability and robustness utilities.

Deliverables:

1. local explanation utilities;
2. representative summaries;
3. feature/group contribution summaries;
4. perturbation utilities;
5. robustness metrics;
6. smoke reproducibility for M5.

Target version: v0.4.

## Stage 5: v1.0 Release

Goal: make the package stable enough to support public use and software-paper consideration.

Deliverables:

1. stable public API;
2. model selector;
3. model cards or reports;
4. reproducibility manifests for the five paper groups;
5. complete documentation;
6. maintenance and versioning policy;
7. CI-ready tests;
8. release checklist.

Target version: v1.0.

## Stage 6: Post-v1.0 Products

These should be considered only after the core kit is stable.

### Optional CLI

The CLI is useful but not mandatory. It should be added only if it clearly improves reproducibility or user adoption.

Potential commands:

```bash
whpc train
whpc evaluate
whpc predict
whpc report
```

Recommendation: defer until after the Python API stabilizes.

### WHPC Studio Community

A free/local GUI for non-programming users.

Recommendation: separate repository or clearly separated app directory after v1.0.

### WHPC Studio Pro

Advanced/commercial layer with dashboards, connectors, monitoring, alerts, professional reporting, and IDS/SIEM integration.

Recommendation: separate product, not part of the open-source core.
