# Reproducibility Matrix

This matrix defines what `whpc-kit` should support for each W-HPC paper group. The public kit is expected to reproduce the central reported results, not every exploratory run from the research workspace.

| Paper group | Parts | Target command | Public artifact target |
| --- | --- | --- | --- |
| M1 | Parts 1-2 | `python reproducibility/fp1_supervised_mm.py` | Supervised and multimodal W-HPC APIs, examples, smoke validation, full reproduction instructions |
| M2 | Parts 3-4 | `python reproducibility/fp2_oneclass_adaptive.py` | One-class and adaptive W-HPC APIs, examples, smoke validation, full reproduction instructions |
| M3 | Parts 5-6 | `python reproducibility/fp3_online_drift.py` | Online streaming and drift-aware APIs, examples, smoke validation, full reproduction instructions |
| M4 | Parts 7-8 | `python reproducibility/fp4_openworld_uncertainty.py` | Open-world, uncertainty, and query-feedback APIs, examples, smoke validation, full reproduction instructions |
| M5 | Parts 9-10 | `python reproducibility/fp5_trustworthy.py` | Explainability and robustness APIs, examples, smoke validation, full reproduction instructions |

## Required Contract Per Paper Group

1. Required datasets and acquisition notes.
2. Minimal smoke command.
3. Full reproduction command or command group.
4. Expected output files.
5. Expected output schemas.
6. Link back to the corresponding paper tables and figures.

## Public Reproducibility Policy

Smoke commands must run without private or local datasets. Full commands may require users to place external datasets under `data/`, but the repository must document expected paths and fail with clear messages when data is missing.

Current implementation split:

1. `FP1` and `FP2` full mode run natively inside `whpc-kit`.
2. `FP3`, `FP4`, and `FP5` full mode currently orchestrate the frozen companion checkout at `../W-HPC` so the exact paper protocol can be replayed without copying large experimental scaffolding into the public repository.

## What Should Stay Out

1. Raw datasets.
2. Paper drafts.
3. Large generated result folders.
4. Private local paths.
5. Exploratory scripts that are not needed for central reproducibility.

## Relationship To Product Roadmap

The reproducibility layer supports the Python package. It should not force a CLI. If a CLI is added later, it can wrap these same reproduction commands, but the first stable interface should remain Python scripts and documented APIs.
