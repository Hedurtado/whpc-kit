from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from whpc import DriftAlarmConfig, compute_block_drift_signals, evaluate_drift_alarm


DEFAULT_RESEARCH_ROOT = Path(__file__).resolve().parents[2] / "W-HPC"
PART6_DATASETS = ("cic-ids2017", "unsw-nb15", "nsl-kdd")
PART5_POLICIES = (
    "static",
    "online_refit_conservative",
    "online_refit_gated",
    "online_refit_shadow_guarded",
)
PART6_POLICIES = (
    "online_refit_shadow_guarded",
    "drift_monitor_only",
    "drift_freeze_on_alarm",
    "drift_tighten_on_alarm",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproducibility entry point for FP3/M3.")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(PART6_DATASETS),
        choices=list(PART6_DATASETS),
    )
    parser.add_argument("--research-root", type=Path, default=DEFAULT_RESEARCH_ROOT)
    parser.add_argument("--cic-raw-dir", type=Path, default=None)
    parser.add_argument("--unsw-raw-dir", type=Path, default=None)
    parser.add_argument("--nsl-raw-dir", type=Path, default=None)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-rows-per-source", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output is None:
        args.output = (
            Path("reproducibility/artifacts/fp3_smoke_summary.json")
            if args.mode == "smoke"
            else Path("reproducibility/artifacts/fp3_full_summary.json")
        )
    if args.mode == "full":
        _run_full(args)
        return
    _run_smoke(args)


def _run_smoke(args: argparse.Namespace) -> None:
    signals = compute_block_drift_signals(
        normality_scores=np.array([0.80, 0.72, 0.50, 0.49, 0.20]),
        y_pred_anomalous=np.array([0, 0, 1, 1, 1]),
        accepted_update_mask=np.array([True, True, False, False, False]),
        threshold=0.50,
        reference_score_mean=0.70,
        borderline_margin=0.02,
    )
    alarm = evaluate_drift_alarm(
        signals,
        DriftAlarmConfig(
            anomaly_rate_threshold=0.50,
            score_drop_threshold=0.10,
            borderline_rate_threshold=0.20,
            accepted_update_floor=0.45,
            min_active_conditions=2,
        ),
    )
    summary = {
        "mode": "smoke",
        "mean_normality_score": signals.mean_normality_score,
        "score_drop": signals.score_drop,
        "predicted_anomaly_rate": signals.predicted_anomaly_rate,
        "borderline_rate": signals.borderline_rate,
        "accepted_update_rate": signals.accepted_update_rate,
        "is_alarm": alarm.is_alarm,
        "active_conditions": alarm.active_conditions,
    }
    _write_json(args.output, summary)
    print(json.dumps(summary, indent=2))


def _run_full(args: argparse.Namespace) -> None:
    research_root = args.research_root.resolve()
    _require_file(research_root / "experiments" / "run_part5_online_evaluation.py")
    _require_file(research_root / "experiments" / "run_part6_drift_aware_evaluation.py")

    cic_raw_dir = (research_root / "data" / "raw" / "CIC-IDS2017") if args.cic_raw_dir is None else args.cic_raw_dir
    unsw_raw_dir = (research_root / "data" / "raw") if args.unsw_raw_dir is None else args.unsw_raw_dir
    nsl_raw_dir = (research_root / "data" / "raw" / "archive") if args.nsl_raw_dir is None else args.nsl_raw_dir

    artifacts_root = (args.output.parent / "fp3_full").resolve()
    part5_results_dir = (artifacts_root / "part5_online_all_datasets").resolve()
    commands: list[dict[str, object]] = []

    part5_cmd = [
        sys.executable,
        str(research_root / "experiments" / "run_part5_online_evaluation.py"),
        "--datasets",
        *args.datasets,
        "--cic-raw-dir",
        str(cic_raw_dir),
        "--unsw-raw-dir",
        str(unsw_raw_dir),
        "--nsl-raw-dir",
        str(nsl_raw_dir),
        "--validation-size",
        str(args.validation_size),
        "--random-state",
        str(args.random_state),
        "--policies",
        *PART5_POLICIES,
        "--results-dir",
        str(part5_results_dir),
    ]
    if args.max_rows_per_source is not None:
        part5_cmd.extend(["--max-rows-per-source", str(args.max_rows_per_source)])
    _run_command(part5_cmd, cwd=research_root)
    commands.append({"step": "part5_online_evaluation", "command": part5_cmd, "results_dir": str(part5_results_dir)})

    part6_runs: dict[str, dict[str, object]] = {}
    for dataset_name in args.datasets:
        dataset_results_dir = (artifacts_root / f"part6_{dataset_name.replace('-', '_')}").resolve()
        part6_cmd = [
            sys.executable,
            str(research_root / "experiments" / "run_part6_drift_aware_evaluation.py"),
            "--dataset",
            dataset_name,
            "--cic-raw-dir",
            str(cic_raw_dir),
            "--unsw-raw-dir",
            str(unsw_raw_dir),
            "--nsl-raw-dir",
            str(nsl_raw_dir),
            "--validation-size",
            str(args.validation_size),
            "--random-state",
            str(args.random_state),
            "--policies",
            *PART6_POLICIES,
            "--drift-anomaly-rate-threshold",
            "0.90",
            "--drift-score-drop-threshold",
            "0.10",
            "--drift-borderline-rate-threshold",
            "0.10",
            "--drift-accepted-update-floor",
            "0.001",
            "--drift-min-active-conditions",
            "3",
            "--drift-tightened-update-margin",
            "0.08",
            "--drift-tightened-update-cap",
            "100",
            "--results-dir",
            str(dataset_results_dir),
        ]
        if dataset_name == "cic-ids2017" and args.max_rows_per_source is not None:
            part6_cmd.extend(["--max-rows-per-source", str(args.max_rows_per_source)])
        _run_command(part6_cmd, cwd=research_root)
        commands.append(
            {"step": f"part6_drift_aware_{dataset_name}", "command": part6_cmd, "results_dir": str(dataset_results_dir)}
        )
        part6_runs[dataset_name] = {
            "results_dir": str(dataset_results_dir),
            "policy_summary": _index_rows(
                dataset_results_dir / "part6_drift_policy_summary.csv",
                keys=("policy",),
            ),
        }

    part5_summary = _index_rows(
        part5_results_dir / "part5_online_policy_summary.csv",
        keys=("dataset", "policy"),
    )
    selected_part5 = {
        dataset_name: {
            policy: part5_summary.get(f"{dataset_name}::{policy}", {})
            for policy in PART5_POLICIES
        }
        for dataset_name in args.datasets
    }
    selected_part6 = {
        dataset_name: {
            policy: part6_runs[dataset_name]["policy_summary"].get(policy, {})
            for policy in PART6_POLICIES
        }
        for dataset_name in args.datasets
    }

    summary = {
        "mode": "full",
        "full_protocol": "m3_frozen_companion_checkout",
        "research_root": str(research_root),
        "datasets": args.datasets,
        "part5_selected_core": "online_refit_shadow_guarded",
        "part6_selected_alarm_rule": {
            "drift_anomaly_rate_threshold": 0.90,
            "drift_score_drop_threshold": 0.10,
            "drift_borderline_rate_threshold": 0.10,
            "drift_accepted_update_floor": 0.001,
            "drift_min_active_conditions": 3,
        },
        "part6_selected_response": {
            "policy": "drift_tighten_on_alarm",
            "drift_tightened_update_margin": 0.08,
            "drift_tightened_update_cap": 100,
        },
        "artifacts": {
            "part5_results_dir": str(part5_results_dir),
            "part6_results_dirs": {dataset_name: part6_runs[dataset_name]["results_dir"] for dataset_name in args.datasets},
        },
        "selected_results": {
            "part5": selected_part5,
            "part6": selected_part6,
        },
        "commands": commands,
        "notes": [
            "Full mode orchestrates the frozen M3 protocol from the companion W-HPC checkout.",
            "Part 5 keeps static, conservative, gated, and shadow-guarded runs for direct comparison.",
            "Part 6 applies the selected alarm rule and on-alarm response policies from the closed M3 results.",
        ],
    }
    _write_json(args.output, summary)
    print(json.dumps({"summary_json": str(args.output), "datasets": args.datasets}, indent=2))


def _run_command(command: list[str], *, cwd: Path) -> None:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if completed.returncode != 0:
        tail = "\n".join(completed.stdout.splitlines()[-40:])
        raise RuntimeError(f"Command failed ({completed.returncode}): {' '.join(command)}\n{tail}")


def _index_rows(path: Path, *, keys: tuple[str, ...]) -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = "::".join(str(row[name]) for name in keys)
            rows[key] = {name: _coerce(value) for name, value in row.items()}
    return rows


def _coerce(value: str | None) -> object:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return ""
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        integer = int(text)
    except ValueError:
        integer = None
    if integer is not None and text == str(integer):
        return integer
    try:
        return float(text)
    except ValueError:
        return text


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required companion-checkout file not found: {path}")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
