# v0.1 Release Checklist

## Current Status

Technical v0.1 status as of 2026-07-02: ready to close.

Verified locally in this repository:

1. core test suite passed: `38 passed`;
2. synthetic smoke example passed;
3. smoke reproducibility scripts `FP1-FP5` passed;
4. FP1 is aligned to the frozen M1 protocol;
5. FP2 is aligned to the published M2 values for UNSW-NB15 and closely aligned for NSL-KDD, with the CIC-IDS2017 discrepancy documented explicitly as a published-freeze versus recomputed-checkout difference.

Not verified in this session:

1. a fresh `pip install -e .[dev,repro]` in a clean environment, because the local `.venv` currently does not expose `pip`;
2. live GitHub Actions status;
3. external coauthor, institutional, or release-timing approvals.

## Scope Freeze

1. Confirm v0.1 only promises the M1-M2 core plus package foundations.
2. Confirm FP3-FP5 smoke scripts are clearly labeled as smoke-only, not full reproductions.
3. Confirm public API names and aliases are acceptable for the first tag.

## Package Quality

1. `python -m pip install -e .[dev]` works on a clean environment.
   Status: not re-verified in this session.
2. Core test suite passes.
   Status: verified locally on 2026-07-02.
3. Synthetic smoke example passes.
   Status: verified locally on 2026-07-02.
4. Reproducibility smoke scripts `FP1-FP5` pass.
   Status: verified locally on 2026-07-02.
5. CI workflow is present and green.
   Status: workflow present; green status not verified here.

## Documentation

1. README reflects actual current scope.
2. Getting-started guide matches current commands.
3. Scope, roadmap, and API naming proposal are internally consistent.
4. Reproducibility manifests are present for `FP1-FP5`.
5. FP1-FP2 alignment note documents the current manuscript-level status.

## Release Hygiene

1. Version in `pyproject.toml` matches the intended tag.
2. `CHANGELOG.md` is updated.
3. `.gitignore` excludes generated artifacts and caches.
4. No raw datasets, local paths, manuscript drafts, or large result folders are included.

## Institutional And Collaboration Checks

1. Review Apache 2.0 decision.
2. Review coauthor approval requirements.
3. Review institutional IP constraints.
4. Confirm public release timing relative to manuscript/preprint strategy.

## Post-Release Immediate Follow-Up

1. Keep CIC-IDS2017 tracked as a manuscript-freeze versus recomputed-checkout discrepancy until the research repo resolves that drift.
2. Expand package architecture with dedicated public modules where needed.
3. Decide whether CLI remains deferred after observing early user needs.
