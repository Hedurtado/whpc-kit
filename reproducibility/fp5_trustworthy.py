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
    add_gaussian_noise,
    explain_ocmmwhpc_decisions,
    explanation_jaccard_summary,
    explanation_summary,
    parse_top_feature_indices,
    robustness_metrics,
)


DEFAULT_RESEARCH_ROOT = Path(__file__).resolve().parents[2] / "W-HPC"
PART9_10_DATASETS = ("cic-ids2017", "unsw-nb15", "nsl-kdd")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproducibility entry point for FP5/M5.")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(PART9_10_DATASETS),
        choices=list(PART9_10_DATASETS),
    )
    parser.add_argument("--research-root", type=Path, default=DEFAULT_RESEARCH_ROOT)
    parser.add_argument("--cic-raw-dir", type=Path, default=None)
    parser.add_argument("--unsw-raw-dir", type=Path, default=None)
    parser.add_argument("--nsl-raw-dir", type=Path, default=None)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-rows-per-source", type=int, default=50000)
    parser.add_argument("--run-external-robustness-baselines", action="store_true")
    return parser.parse_args()


def make_normal_only_bimodal_dataset() -> np.ndarray:
    return np.array(
        [
            [2.0, 0.0],
            [3.0, 0.0],
            [-2.0, 0.0],
            [-3.0, 0.0],
        ]
    )


def make_validation_dataset() -> tuple[np.ndarray, np.ndarray]:
    X_val = np.array(
        [
            [2.5, 0.0],
            [-2.5, 0.0],
            [0.0, 2.0],
            [0.0, -2.0],
        ]
    )
    y_val = np.array([0, 0, 1, 1])
    return X_val, y_val


def main() -> None:
    args = parse_args()
    if args.output is None:
        args.output = (
            Path("reproducibility/artifacts/fp5_smoke_summary.json")
            if args.mode == "smoke"
            else Path("reproducibility/artifacts/fp5_full_summary.json")
        )
    if args.mode == "full":
        _run_full(args)
        return
    _run_smoke(args)


def _run_smoke(args: argparse.Namespace) -> None:
    X_train_normal = make_normal_only_bimodal_dataset()
    X_eval, y_true = make_validation_dataset()
    feature_names = ["feature_0", "feature_1"]

    detector = OCMMWHPCDetector(n_normal_representatives=2, normalize=True, random_state=0)
    detector.fit(X_train_normal)
    detector.calibrate_threshold(X_eval, y_true)

    clean_scores = detector.score_samples(X_eval)
    clean_pred = (detector.predict(X_eval) == detector.anomaly_label).astype(int)
    clean_explanations = explain_ocmmwhpc_decisions(
        detector,
        X_eval,
        y_true=y_true,
        feature_names=feature_names,
        top_k_features=2,
    )
    explanation_diag = explanation_summary(clean_explanations)
    top_indices = parse_top_feature_indices(clean_explanations, feature_names, column="top_abs_features")

    rng = np.random.default_rng(0)
    X_perturbed = add_gaussian_noise(X_eval, std=0.05, rng=rng)
    perturbed_scores = detector.score_samples(X_perturbed)
    perturbed_pred = (detector.predict(X_perturbed) == detector.anomaly_label).astype(int)
    perturbed_explanations = explain_ocmmwhpc_decisions(
        detector,
        X_perturbed,
        y_true=y_true,
        feature_names=feature_names,
        top_k_features=2,
    )
    robustness_diag = robustness_metrics(
        y_true,
        clean_scores,
        clean_pred,
        perturbed_scores,
        perturbed_pred,
        threshold=float(detector.threshold_),
    )
    jaccard_diag = explanation_jaccard_summary(clean_explanations, perturbed_explanations)

    summary = {
        "mode": "smoke",
        "n_explained": explanation_diag["n_explained"],
        "mean_absolute_threshold_margin": explanation_diag["mean_absolute_threshold_margin"],
        "prediction_flip_rate": robustness_diag["prediction_flip_rate"],
        "mean_abs_score_shift": robustness_diag["mean_abs_score_shift"],
        "mean_topk_feature_jaccard": jaccard_diag["mean_topk_feature_jaccard"],
        "parsed_top_feature_lengths": [len(indices) for indices in top_indices],
    }
    _write_json(args.output, summary)
    print(json.dumps(summary, indent=2))


def _run_full(args: argparse.Namespace) -> None:
    research_root = args.research_root.resolve()
    _require_file(research_root / "experiments" / "run_part9_explainability_protocol.py")
    _require_file(research_root / "experiments" / "run_part10_robustness_protocol.py")

    cic_raw_dir = (research_root / "data" / "raw" / "CIC-IDS2017") if args.cic_raw_dir is None else args.cic_raw_dir
    unsw_raw_dir = (research_root / "data" / "raw") if args.unsw_raw_dir is None else args.unsw_raw_dir
    nsl_raw_dir = (research_root / "data" / "raw" / "archive") if args.nsl_raw_dir is None else args.nsl_raw_dir

    artifacts_root = (args.output.parent / "fp5_full").resolve()
    commands: list[dict[str, object]] = []
    part9_runs: dict[str, dict[str, object]] = {}
    part10_runs: dict[str, dict[str, object]] = {}

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
            "--max-rows-per-source",
            str(args.max_rows_per_source),
        ]

        part9_results_dir = (artifacts_root / f"part9_{dataset_name.replace('-', '_')}").resolve()
        part9_cmd = [
            sys.executable,
            str(research_root / "experiments" / "run_part9_explainability_protocol.py"),
            *common,
            "--results-dir",
            str(part9_results_dir),
        ]
        _run_command(part9_cmd, cwd=research_root)
        commands.append({"step": f"part9_explainability_{dataset_name}", "command": part9_cmd, "results_dir": str(part9_results_dir)})
        part9_runs[dataset_name] = {
            "results_dir": str(part9_results_dir),
            "summary": _load_json(part9_results_dir / "part9_explainability.json"),
        }

        part10_results_dir = (artifacts_root / f"part10_{dataset_name.replace('-', '_')}").resolve()
        part10_cmd = [
            sys.executable,
            str(research_root / "experiments" / "run_part10_robustness_protocol.py"),
            *common,
            "--explanation-max-samples",
            "5000",
            "--results-dir",
            str(part10_results_dir),
        ]
        _run_command(part10_cmd, cwd=research_root)
        commands.append({"step": f"part10_robustness_{dataset_name}", "command": part10_cmd, "results_dir": str(part10_results_dir)})
        part10_runs[dataset_name] = {
            "results_dir": str(part10_results_dir),
            "summary_rows": _index_rows(
                part10_results_dir / "part10_robustness_summary.csv",
                keys=("perturbation", "severity"),
            ),
        }

    external_baselines = {}
    if args.run_external_robustness_baselines:
        _require_file(research_root / "experiments" / "run_part10_external_robustness_baselines.py")
        external_results_dir = (artifacts_root / "part10_external_robustness_baselines").resolve()
        external_cmd = [
            sys.executable,
            str(research_root / "experiments" / "run_part10_external_robustness_baselines.py"),
            "--datasets",
            "unsw-nb15",
            "nsl-kdd",
            "--baselines",
            "isolation_forest",
            "sgd_oneclass_svm",
            "--results-dir",
            str(external_results_dir),
        ]
        _run_command(external_cmd, cwd=research_root)
        commands.append(
            {"step": "part10_external_robustness_baselines", "command": external_cmd, "results_dir": str(external_results_dir)}
        )
        external_baselines = {
            "results_dir": str(external_results_dir),
            "summary_rows": _read_rows(external_results_dir / "part10_external_robustness_summary.csv"),
        }

    summary = {
        "mode": "full",
        "full_protocol": "m5_frozen_companion_checkout",
        "research_root": str(research_root),
        "datasets": args.datasets,
        "part9_selected_scope": "native explanations, representative summaries, and seed stability",
        "part10_selected_scope": "controlled robustness diagnostics with clean, Gaussian, masking, and adversarial-style perturbations",
        "artifacts": {
            "part9_results_dirs": {dataset_name: part9_runs[dataset_name]["results_dir"] for dataset_name in args.datasets},
            "part10_results_dirs": {dataset_name: part10_runs[dataset_name]["results_dir"] for dataset_name in args.datasets},
            "external_robustness_baselines": external_baselines.get("results_dir", ""),
        },
        "selected_results": {
            "part9": {
                dataset_name: _part9_selected_metrics(part9_runs[dataset_name]["summary"])
                for dataset_name in args.datasets
            },
            "part10": {
                dataset_name: {
                    key: part10_runs[dataset_name]["summary_rows"].get(key, {})
                    for key in (
                        "clean::clean",
                        "gaussian_noise::high",
                        "random_mask::high",
                        "top_feature_mask::high",
                        "adversarial_lite::high",
                        "adversarial_score::high",
                    )
                }
                for dataset_name in args.datasets
            },
            "part10_external_baselines": external_baselines.get("summary_rows", []),
        },
        "commands": commands,
        "notes": [
            "Full mode orchestrates the frozen M5 protocol from the companion W-HPC checkout.",
            "Part 9 keeps the lightweight native explanation export, representative summaries, and seed-stability diagnostics.",
            "Part 10 keeps the bounded robustness-diagnostics interpretation rather than a certified-adversarial claim.",
            "Optional SHAP/LIME and other post-hoc controls remain companion-checkout extras outside the default public runner.",
        ],
    }
    _write_json(args.output, summary)
    print(json.dumps({"summary_json": str(args.output), "datasets": args.datasets}, indent=2))


def _part9_selected_metrics(payload: dict[str, object]) -> dict[str, object]:
    summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
    return {
        "dataset": summary.get("dataset", ""),
        "stream_mode": summary.get("stream_mode", ""),
        "f1_anomalous": summary.get("f1_anomalous", ""),
        "anomaly_recall": summary.get("anomaly_recall", ""),
        "balanced_accuracy": summary.get("balanced_accuracy", ""),
        "explanation_ms_per_sample": summary.get("explanation_ms_per_sample", ""),
        "mean_absolute_threshold_margin": summary.get("mean_absolute_threshold_margin", ""),
        "mean_representative_gap": summary.get("mean_representative_gap", ""),
        "mean_topk_abs_contribution_share": summary.get("mean_topk_abs_contribution_share", ""),
        "mean_top5_feature_jaccard": summary.get("mean_top5_feature_jaccard", ""),
        "mean_top5_group_jaccard": summary.get("mean_top5_group_jaccard", ""),
    }


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


def _read_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append({name: _coerce(value) for name, value in row.items()})
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


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required companion-checkout file not found: {path}")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
