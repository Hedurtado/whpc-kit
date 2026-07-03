# Changelog

## [0.1.0] - 2026-07-02

### Added

- standalone `whpc-kit` repository structure with Apache 2.0 license and `src/` layout;
- imported reusable W-HPC package core from the research workspace;
- top-level and modular imports through `whpc`, `whpc.models`, `whpc.core`, and `whpc.evaluation`;
- detector aliases for one-class and open-world APIs;
- basic selector layer through `whpc.selection.WHPCSelector`;
- basic report/model-card layer through `whpc.reports`;
- synthetic smoke example for supervised, multimodal, and one-class flows;
- unit tests for public API smoke, scores, representatives, thresholds, weights, drift, open-world, querying, selection, and reports;
- manuscript-level smoke reproducibility scripts and manifests for `FP1-FP5`;
- dataset-backed full reproduction entry points for `FP1` and `FP2`;
- frozen M1 alignment for `FP1`;
- frozen M2 alignment for `FP2`, including published-reference tracking and explicit CIC drift reporting;
- GitHub Actions CI workflow for tests and smoke scripts;
- getting-started, roadmap, naming, scope, and FP1-FP2 alignment documentation.

### Changed

- querying utilities now tolerate missing `rejected_by_unknown_risk` support when used directly with raw open-world support dictionaries.

### Notes

- `WHPCSelector` is advisory and does not replace dataset-specific validation.
- `FP2` reproduces the published M2 values for UNSW-NB15 and closely for NSL-KDD.
- `CIC-IDS2017` remains documented as a published-freeze versus recomputed-checkout discrepancy rather than a silent exact reproduction claim.
