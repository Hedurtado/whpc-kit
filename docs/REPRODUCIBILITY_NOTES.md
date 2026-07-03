# Reproducibility Notes

This note summarizes the public reproducibility contract of `whpc-kit`.

## Current Split

`whpc-kit` uses two reproducibility layers:

1. `FP1` and `FP2` run their full protocol logic natively inside this repository.
2. `FP3`, `FP4`, and `FP5` expose public runners here, but their `full` mode orchestrates the frozen manuscript scripts from a companion `../W-HPC` checkout.

This split is intentional:

1. the public repository stays compact;
2. the reusable package code remains easy to inspect and install;
3. later manuscript stages can still be replayed with the exact frozen research protocol.

## Alignment Notes

For the first two manuscript groups:

1. `FP1` is aligned to the frozen `M1` protocol.
2. `FP2` is aligned to the frozen `M2` protocol.
3. `FP2` also keeps an explicit distinction between published reference values and values recomputed from the current research checkout when those differ.

For the later manuscript groups:

1. `FP3` corresponds to `M3` and covers online streaming plus drift-aware evaluation.
2. `FP4` corresponds to `M4` and covers open-world evaluation plus query-feedback and guarded retraining.
3. `FP5` corresponds to `M5` and covers explainability plus robustness diagnostics.

## Scope Boundary

The repository does not aim to mirror every exploratory run from the research workspace.
Its goal is to provide:

1. a stable public package,
2. small smoke validations for CI,
3. a documented bridge to the manuscript protocols,
4. a clean public surface without datasets, drafts, or large local outputs.
