from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from whpc import (
    OCMMWHPCDetector,
    OpenWorldWHPCDetector,
    accepted_risk_candidate_mask,
    apply_oracle_feedback,
    compute_feedback_metrics,
    compute_open_world_metrics,
    compute_query_scores,
    query_candidate_mask,
    select_dual_top_k_queries,
)


DEFAULT_RESEARCH_ROOT = Path(__file__).resolve().parents[2] / "W-HPC"
PART7_DATASETS = ("cic-ids2017", "unsw-nb15", "nsl-kdd")
PART8_FEEDBACK_POLICIES = ("accepted_risk_topk", "dual_topk", "topk", "random")
PART8_RETRAINING_POLICIES = ("accepted_risk_topk", "dual_topk")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproducibility entry point for FP4/M4.")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(PART7_DATASETS),
        choices=list(PART7_DATASETS),
    )
    parser.add_argument("--research-root", type=Path, default=DEFAULT_RESEARCH_ROOT)
    parser.add_argument("--cic-raw-dir", type=Path, default=None)
    parser.add_argument("--unsw-raw-dir", type=Path, default=None)
    parser.add_argument("--nsl-raw-dir", type=Path, default=None)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-rows-per-source", type=int, default=50000)
    parser.add_argument("--query-budget-per-block", type=int, default=100)
    return parser.parse_args()


def make_detector() -> OpenWorldWHPCDetector:
    base = OCMMWHPCDetector(
        n_normal_representatives=1,
        representative_partition_strategy="kmeans",
        threshold=0.60,
        normalize=True,
        random_state=0,
    )
    detector = OpenWorldWHPCDetector(base_detector=base, min_decision_margin=0.05, reject_on_alarm=True)
    detector.fit(np.array([[1.0, 0.0], [0.95, 0.05], [0.90, 0.10]]))
    detector.unknown_risk_score_floor_ = -0.50
    detector.unknown_risk_gap_floor_ = 0.50
    detector.unknown_risk_margin_floor_ = 0.60
    return detector


def main() -> None:
    args = parse_args()
    if args.output is None:
        args.output = (
            Path("reproducibility/artifacts/fp4_smoke_summary.json")
            if args.mode == "smoke"
            else Path("reproducibility/artifacts/fp4_full_summary.json")
        )
    if args.mode == "full":
        _run_full(args)
        return
    _run_smoke(args)


def _run_smoke(args: argparse.Namespace) -> None:
    detector = make_detector()
    X = np.array([[1.0, 0.0], [-1.0, 0.0], [0.60, 0.80]])
    y_true = np.array([0, 1, 1])
    unknown_mask = np.array([False, False, True])
    alarm_mask = np.array([False, True, False])

    support = detector.decision_support(X, alarm_mask=alarm_mask)
    rejected_by_unknown_risk = support["base_is_anomalous"] & support["unknown_risk_mask"] & ~support["reject_mask"]
    support["rejected_by_unknown_risk"] = rejected_by_unknown_risk
    support["reject_mask"] = support["reject_mask"] | rejected_by_unknown_risk

    labels = detector.predict(X, alarm_mask=alarm_mask).astype(object)
    labels[rejected_by_unknown_risk] = detector.unknown_label

    query_scores = compute_query_scores(support, labels, alarm_active=True, unknown_label=detector.unknown_label)
    reject_candidates = query_candidate_mask(support, labels, alarm_active=True, unknown_label=detector.unknown_label)
    accepted_candidates = accepted_risk_candidate_mask(
        support,
        labels,
        alarm_active=True,
        unknown_label=detector.unknown_label,
    )
    queried = select_dual_top_k_queries(
        query_scores,
        reject_candidates,
        query_scores,
        accepted_candidates,
        budget=2,
        primary_budget_fraction=0.5,
    )
    corrected = apply_oracle_feedback(y_true, labels, queried)
    metrics = compute_feedback_metrics(y_true, labels, corrected, unknown_mask, queried)
    open_world_metrics = compute_open_world_metrics(y_true, labels, unknown_mask)

    summary = {
        "mode": "smoke",
        "labels": labels.tolist(),
        "query_scores": query_scores.tolist(),
        "queried": queried.tolist(),
        "coverage": open_world_metrics["coverage"],
        "unknown_rejection_recall": open_world_metrics["unknown_rejection_recall"],
        "harmful_accept_rate": open_world_metrics["harmful_accept_rate"],
        "query_rate": metrics["query_rate"],
        "accepted_accuracy_after_feedback": metrics["accepted_accuracy_after_feedback"],
    }
    _write_json(args.output, summary)
    print(json.dumps(summary, indent=2))


def _run_full(args: argparse.Namespace) -> None:
    research_root = args.research_root.resolve()
    _require_file(research_root / "experiments" / "run_part7_open_world_protocol.py")
    _require_file(research_root / "experiments" / "run_part8_query_feedback_protocol.py")
    _require_file(research_root / "experiments" / "run_part8_query_retraining_protocol.py")

    cic_raw_dir = (research_root / "data" / "raw" / "CIC-IDS2017") if args.cic_raw_dir is None else args.cic_raw_dir
    unsw_raw_dir = (research_root / "data" / "raw") if args.unsw_raw_dir is None else args.unsw_raw_dir
    nsl_raw_dir = (research_root / "data" / "raw" / "archive") if args.nsl_raw_dir is None else args.nsl_raw_dir

    artifacts_root = (args.output.parent / "fp4_full").resolve()
    commands: list[dict[str, object]] = []
    part7_runs: dict[str, dict[str, object]] = {}
    part8_feedback_runs: dict[str, dict[str, object]] = {}
    part8_retraining_runs: dict[str, dict[str, object]] = {}

    for dataset_name in args.datasets:
        common = [
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
            "--anomaly-support-quantile",
            "0.01",
            "--unknown-risk-quantile",
            "0.10",
            "--unknown-risk-min-conditions",
            "1",
            "--drift-min-active-conditions",
            "2",
        ]
        if args.max_rows_per_source is not None:
            common.extend(["--max-rows-per-source", str(args.max_rows_per_source)])

        part7_results_dir = (artifacts_root / f"part7_{dataset_name.replace('-', '_')}").resolve()
        part7_cmd = [
            sys.executable,
            str(research_root / "experiments" / "run_part7_open_world_protocol.py"),
            *common,
            "--results-dir",
            str(part7_results_dir),
        ]
        _run_command(part7_cmd, cwd=research_root)
        commands.append({"step": f"part7_open_world_{dataset_name}", "command": part7_cmd, "results_dir": str(part7_results_dir)})
        part7_runs[dataset_name] = {
            "results_dir": str(part7_results_dir),
            "policy_summary": _index_rows(part7_results_dir / "part7_open_world_policy_summary.csv", keys=("policy",)),
        }

        feedback_results_dir = (artifacts_root / f"part8_feedback_{dataset_name.replace('-', '_')}").resolve()
        feedback_cmd = [
            sys.executable,
            str(research_root / "experiments" / "run_part8_query_feedback_protocol.py"),
            *common,
            "--base-open-world-policy",
            "ow_m3_unknown_risk",
            "--query-policies",
            *PART8_FEEDBACK_POLICIES,
            "--query-budget-per-block",
            str(args.query_budget_per_block),
            "--results-dir",
            str(feedback_results_dir),
        ]
        _run_command(feedback_cmd, cwd=research_root)
        commands.append(
            {"step": f"part8_query_feedback_{dataset_name}", "command": feedback_cmd, "results_dir": str(feedback_results_dir)}
        )
        part8_feedback_runs[dataset_name] = {
            "results_dir": str(feedback_results_dir),
            "summary": _index_rows(feedback_results_dir / "part8_query_feedback_summary.csv", keys=("query_policy",)),
        }

        retraining_results_dir = (artifacts_root / f"part8_retraining_{dataset_name.replace('-', '_')}").resolve()
        retraining_cmd = [
            sys.executable,
            str(research_root / "experiments" / "run_part8_query_retraining_protocol.py"),
            *common,
            "--base-open-world-policy",
            "ow_m3_unknown_risk",
            "--query-policies",
            *PART8_RETRAINING_POLICIES,
            "--retraining-modes",
            "feedback_only",
            "queried_retrain_guarded",
            "--query-budget-per-block",
            str(args.query_budget_per_block),
            "--feedback-source",
            "oracle",
            "--results-dir",
            str(retraining_results_dir),
        ]
        _run_command(retraining_cmd, cwd=research_root)
        commands.append(
            {
                "step": f"part8_query_retraining_{dataset_name}",
                "command": retraining_cmd,
                "results_dir": str(retraining_results_dir),
            }
        )
        part8_retraining_runs[dataset_name] = {
            "results_dir": str(retraining_results_dir),
            "summary": _index_rows(
                retraining_results_dir / "part8_query_retraining_summary.csv",
                keys=("query_policy", "retraining_mode"),
            ),
        }

    summary = {
        "mode": "full",
        "full_protocol": "m4_frozen_companion_checkout",
        "research_root": str(research_root),
        "datasets": args.datasets,
        "part7_selected_open_world_point": "ow_m3_unknown_risk",
        "part7_high_rejection_reference": "ow_m3_alarm_aware",
        "part8_selected_query_policy": {
            "primary": "accepted_risk_topk",
            "balanced_variant": "dual_topk",
            "query_budget_per_block": args.query_budget_per_block,
        },
        "artifacts": {
            "part7_results_dirs": {dataset_name: part7_runs[dataset_name]["results_dir"] for dataset_name in args.datasets},
            "part8_feedback_results_dirs": {
                dataset_name: part8_feedback_runs[dataset_name]["results_dir"] for dataset_name in args.datasets
            },
            "part8_retraining_results_dirs": {
                dataset_name: part8_retraining_runs[dataset_name]["results_dir"] for dataset_name in args.datasets
            },
        },
        "selected_results": {
            "part7": {
                dataset_name: {
                    policy: part7_runs[dataset_name]["policy_summary"].get(policy, {})
                    for policy in ("ow_ocmmwhpc", "ow_m3_alarm_aware", "ow_m3_unknown_risk")
                }
                for dataset_name in args.datasets
            },
            "part8_feedback": {
                dataset_name: {
                    policy: part8_feedback_runs[dataset_name]["summary"].get(policy, {})
                    for policy in PART8_FEEDBACK_POLICIES
                }
                for dataset_name in args.datasets
            },
            "part8_retraining": {
                dataset_name: {
                    f"{query_policy}::{mode}": part8_retraining_runs[dataset_name]["summary"].get(
                        f"{query_policy}::{mode}",
                        {},
                    )
                    for query_policy in PART8_RETRAINING_POLICIES
                    for mode in ("feedback_only", "queried_retrain_guarded")
                }
                for dataset_name in args.datasets
            },
        },
        "commands": commands,
        "notes": [
            "Full mode orchestrates the frozen M4 protocol from the companion W-HPC checkout.",
            "Part 7 uses the selected support and alarm settings from the closed open-world tradeoff analysis.",
            "Part 8 runs the selected accepted-risk querying policy, the balanced dual-top-k variant, and guarded oracle retraining.",
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
