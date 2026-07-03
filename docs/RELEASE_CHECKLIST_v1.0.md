# v1.0 Release Checklist

## Current Status

Technical v1.0 status target as of 2026-07-02:

1. package API is public and stable enough for the first broad tag;
2. reproducibility manifests exist for `FP1-FP5`;
3. `FP1-FP2` have native full reproduction runners;
4. `FP3-FP5` have companion-checkout full orchestration runners aligned to the frozen M3-M5 paper protocols;
5. smoke runners remain self-contained for the public package layer.

## Package And Reproducibility Contract

1. Confirm `README.md`, `docs/GETTING_STARTED.md`, and `reproducibility/README.md` describe the same current behavior.
2. Confirm manifests for `FP1-FP5` list actual supported modes and commands.
3. Confirm companion-checkout expectations for `FP3-FP5` are explicit and honest.
4. Confirm `PYTHONPATH=src` or editable-install usage is documented where needed.

## Verification

1. Core tests should pass in the maintained environment.
2. Smoke runners `FP1-FP5` should pass in the maintained environment.
3. `FP3 full`, `FP4 full`, and `FP5 full` should be spot-checked on at least one local dataset configuration.
4. Exact large-scale manuscript reruns remain optional follow-up work when wall-clock time is acceptable.

## Release Hygiene

1. `pyproject.toml` version matches the tag `v1.0.0`.
2. Public docs do not claim fully native M3-M5 reproduction inside `whpc-kit`.
3. No raw datasets or large generated research folders are staged.
4. Apache 2.0 remains the declared license.

## Immediate Post-v1.0 Work

1. Decide whether to internalize more of the M3-M5 protocol into the public repo or keep the companion-checkout split.
2. Decide whether a software paper is needed or whether `whpc-kit` acts as the technical reproducibility companion for the manuscript line.
3. Revisit CLI only if it clearly improves reproducibility or user adoption.
