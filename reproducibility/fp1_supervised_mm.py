from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from whpc import (
    MMWHPCClassifier,
    WHPCClassifier,
    load_nsl_kdd,
    load_unsw_nb15,
    make_nsl_kdd_preprocessor,
    make_unsw_preprocessor,
)
from whpc.preprocessing import PreprocessingSpec


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    loader: object
    preprocessor_factory: object


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke reproducibility entry point for FP1/M1.")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--unsw-raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--nsl-raw-dir", type=Path, default=Path("data/raw/archive"))
    parser.add_argument("--datasets", nargs="+", default=["unsw-nb15", "nsl-kdd"], choices=["unsw-nb15", "nsl-kdd"])
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--boundary-scales", type=float, nargs="+", default=[0.1, 0.5])
    parser.add_argument("--boundary-gammas", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0, 8.0])
    parser.add_argument(
        "--mm-selection-protocol",
        choices=["m1_frozen", "custom"],
        default="m1_frozen",
        help="M1 frozen protocol keeps uniform local weights, symmetric representative counts, and max aggregation.",
    )
    parser.add_argument("--sample-weight-strategies", nargs="+", default=["uniform"])
    parser.add_argument("--prototype-counts", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--prototype-aggregation", choices=["max", "softmax"], default="max")
    parser.add_argument("--prototype-softmax-alpha", type=float, default=10.0)
    parser.add_argument("--prototype-penalty", type=float, default=0.0)
    return parser.parse_args()


def make_axis_dataset() -> tuple[np.ndarray, np.ndarray]:
    X = np.array(
        [
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [0.0, 1.0],
            [0.0, 2.0],
            [0.0, 3.0],
        ]
    )
    y = np.array([0, 0, 0, 1, 1, 1])
    return X, y


def make_bimodal_dataset() -> tuple[np.ndarray, np.ndarray]:
    X = np.array(
        [
            [1.0, 0.0],
            [2.0, 0.0],
            [-1.0, 0.0],
            [-2.0, 0.0],
            [0.0, 1.0],
            [0.0, 2.0],
            [0.0, -1.0],
            [0.0, -2.0],
        ]
    )
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    return X, y


def main() -> None:
    args = parse_args()
    if args.output is None:
        args.output = (
            Path("reproducibility/artifacts/fp1_smoke_summary.json")
            if args.mode == "smoke"
            else Path("reproducibility/artifacts/fp1_full_summary.json")
        )
    if args.mode == "full":
        _run_full(args)
        return

    axis_X, axis_y = make_axis_dataset()
    bimodal_X, bimodal_y = make_bimodal_dataset()

    whpc = WHPCClassifier()
    whpc.fit(axis_X, axis_y)
    mm_whpc = MMWHPCClassifier(n_representatives_per_class=2, random_state=0)
    mm_whpc.fit(bimodal_X, bimodal_y)

    summary = {
        "mode": "smoke",
        "whpc_train_accuracy": float(np.mean(whpc.predict(axis_X) == axis_y)),
        "mm_whpc_train_accuracy": float(np.mean(mm_whpc.predict(bimodal_X) == bimodal_y)),
        "whpc_score_shape": list(whpc.predict_scores(axis_X[:2]).shape),
        "mm_whpc_score_shape": list(mm_whpc.predict_scores(bimodal_X[:2]).shape),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def _run_full(args: argparse.Namespace) -> None:
    _validate_full_args(args)
    dataset_configs = {
        "unsw-nb15": DatasetConfig(
            "unsw-nb15",
            lambda: load_unsw_nb15(raw_dir=args.unsw_raw_dir, target="binary"),
            make_unsw_preprocessor,
        ),
        "nsl-kdd": DatasetConfig(
            "nsl-kdd",
            lambda: load_nsl_kdd(raw_dir=args.nsl_raw_dir, target="binary"),
            make_nsl_kdd_preprocessor,
        ),
    }
    validation_rows: list[dict[str, object]] = []
    test_rows: list[dict[str, object]] = []
    payload: dict[str, object] = {
        "mode": "full",
        "protocol": {
            "datasets": args.datasets,
            "validation_size": args.validation_size,
            "random_state": args.random_state,
            "boundary_scales": args.boundary_scales,
            "boundary_gammas": args.boundary_gammas,
            "mm_selection_protocol": args.mm_selection_protocol,
            "sample_weight_strategies": args.sample_weight_strategies,
            "prototype_counts": args.prototype_counts,
            "prototype_aggregation": args.prototype_aggregation,
            "prototype_softmax_alpha": args.prototype_softmax_alpha,
            "prototype_penalty": args.prototype_penalty,
        },
        "datasets": {},
    }

    for dataset_name in args.datasets:
        dataset_payload, dataset_validation_rows, dataset_test_rows = _run_dataset(dataset_configs[dataset_name], args)
        payload["datasets"][dataset_name] = dataset_payload
        validation_rows.extend(dataset_validation_rows)
        test_rows.extend(dataset_test_rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_rows(args.output.with_name("fp1_full_validation.csv"), validation_rows)
    _write_rows(args.output.with_name("fp1_full_test.csv"), test_rows)
    print(json.dumps({"summary_json": str(args.output), "validation_rows": len(validation_rows), "test_rows": len(test_rows)}, indent=2))


def _run_dataset(
    dataset_config: DatasetConfig,
    args: argparse.Namespace,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    dataset = dataset_config.loader()

    X_train_inner, X_val_inner, y_train_inner, y_val_inner = train_test_split(
        dataset.X_train,
        dataset.y_train,
        test_size=args.validation_size,
        random_state=args.random_state,
        stratify=dataset.y_train,
    )
    inner_preprocessor = dataset_config.preprocessor_factory(X_train_inner)
    X_train_inner_t = inner_preprocessor.transformer.fit_transform(X_train_inner)
    X_val_inner_t = inner_preprocessor.transformer.transform(X_val_inner)
    y_train_inner_np = y_train_inner.to_numpy()
    y_val_inner_np = y_val_inner.to_numpy()

    center_validation_rows: list[dict[str, object]] = []
    for boundary_scale in args.boundary_scales:
        for boundary_gamma in args.boundary_gammas:
            model = WHPCClassifier(
                normalize=True,
                score_metric="cosine",
                feature_weight=None,
                sample_weight_strategy="center_boundary",
                sample_weight_temperature=args.temperature,
                sample_weight_boundary_gamma=boundary_gamma,
                sample_weight_boundary_scale=boundary_scale,
            )
            model.fit(X_train_inner_t, y_train_inner_np)
            y_val_pred = model.predict(X_val_inner_t)
            center_validation_rows.append(
                {
                    "dataset": dataset_config.name,
                    "phase": "fp1_whpc_validation",
                    "boundary_scale": boundary_scale,
                    "boundary_gamma": boundary_gamma,
                    **_classification_metrics(y_val_inner_np, y_val_pred),
                }
            )
    selected_center = max(
        center_validation_rows,
        key=lambda row: (float(row["f1_macro"]), float(row["balanced_accuracy"]), float(row["accuracy"])),
    )

    class_labels = sorted(set(y_train_inner_np.tolist()))
    normal_label, attack_label = class_labels[0], class_labels[1]
    mm_strategies = _resolve_mm_sample_weight_strategies(args)
    mm_validation_rows: list[dict[str, object]] = []
    for strategy in mm_strategies:
        for prototype_count in args.prototype_counts:
            prototype_count_by_class = {normal_label: prototype_count, attack_label: prototype_count}
            model = MMWHPCClassifier(
                n_representatives_per_class=prototype_count_by_class,
                representative_partition_strategy="kmeans",
                prototype_aggregation=args.prototype_aggregation,
                prototype_softmax_alpha=args.prototype_softmax_alpha,
                sample_weight_strategy=strategy,
                sample_weight_temperature=args.temperature,
                sample_weight_boundary_gamma=float(selected_center["boundary_gamma"]),
                sample_weight_boundary_scale=float(selected_center["boundary_scale"]),
                feature_weight=None,
                normalize=True,
                score_metric="cosine",
                random_state=args.random_state,
            )
            model.fit(X_train_inner_t, y_train_inner_np)
            y_val_pred = model.predict(X_val_inner_t)
            metrics = _classification_metrics(y_val_inner_np, y_val_pred)
            total_representatives = 2 * prototype_count
            mm_validation_rows.append(
                {
                    "dataset": dataset_config.name,
                    "phase": "fp1_mm_validation",
                    "sample_weight_strategy": strategy,
                    "n_representatives_normal": prototype_count,
                    "n_representatives_attack": prototype_count,
                    "total_representatives": total_representatives,
                    "selection_score": float(metrics["f1_macro"] - args.prototype_penalty * (total_representatives - 2)),
                    **metrics,
                }
            )
    selected_mm = max(
        mm_validation_rows,
        key=lambda row: (float(row["selection_score"]), float(row["f1_macro"]), float(row["balanced_accuracy"])),
    )

    full_preprocessor = dataset_config.preprocessor_factory(dataset.X_train)
    X_train_t = full_preprocessor.transformer.fit_transform(dataset.X_train)
    X_test_t = full_preprocessor.transformer.transform(dataset.X_test)
    y_train_np = dataset.y_train.to_numpy()
    y_test_np = dataset.y_test.to_numpy()

    whpc_variants = [
        ("uniform", {"sample_weight_strategy": "uniform"}),
        ("intra_class_core", {"sample_weight_strategy": "intra_class_core", "sample_weight_temperature": args.temperature}),
        (
            "center_boundary_validated",
            {
                "sample_weight_strategy": "center_boundary",
                "sample_weight_temperature": args.temperature,
                "sample_weight_boundary_gamma": float(selected_center["boundary_gamma"]),
                "sample_weight_boundary_scale": float(selected_center["boundary_scale"]),
            },
        ),
    ]
    test_rows: list[dict[str, object]] = []
    baseline_pred: np.ndarray | None = None
    baseline_metrics: dict[str, float] | None = None
    for variant_name, model_kwargs in whpc_variants:
        model = WHPCClassifier(normalize=True, score_metric="cosine", feature_weight=None, **model_kwargs)
        model.fit(X_train_t, y_train_np)
        y_pred = model.predict(X_test_t)
        metrics = _classification_metrics(y_test_np, y_pred)
        if baseline_pred is None:
            baseline_pred = y_pred
            baseline_metrics = metrics
        delta_f1 = float(metrics["f1_macro"] - baseline_metrics["f1_macro"]) if baseline_metrics is not None else 0.0
        test_rows.append(
            {
                "dataset": dataset_config.name,
                "phase": "fp1_whpc_test",
                "variant": variant_name,
                "prediction_change_rate_vs_uniform": float(np.mean(y_pred != baseline_pred)) if baseline_pred is not None else 0.0,
                "delta_f1_macro_vs_uniform": delta_f1,
                **metrics,
            }
        )

    mm_model = MMWHPCClassifier(
        n_representatives_per_class={
            normal_label: int(selected_mm["n_representatives_normal"]),
            attack_label: int(selected_mm["n_representatives_attack"]),
        },
        representative_partition_strategy="kmeans",
        prototype_aggregation=args.prototype_aggregation,
        prototype_softmax_alpha=args.prototype_softmax_alpha,
        sample_weight_strategy=str(selected_mm["sample_weight_strategy"]),
        sample_weight_temperature=args.temperature,
        sample_weight_boundary_gamma=float(selected_center["boundary_gamma"]),
        sample_weight_boundary_scale=float(selected_center["boundary_scale"]),
        feature_weight=None,
        normalize=True,
        score_metric="cosine",
        random_state=args.random_state,
    )
    mm_model.fit(X_train_t, y_train_np)
    mm_pred = mm_model.predict(X_test_t)
    mm_metrics = _classification_metrics(y_test_np, mm_pred)
    test_rows.append(
        {
            "dataset": dataset_config.name,
            "phase": "fp1_mm_test",
            "variant": "selected_multimodal",
            "sample_weight_strategy": selected_mm["sample_weight_strategy"],
            "n_representatives_normal": selected_mm["n_representatives_normal"],
            "n_representatives_attack": selected_mm["n_representatives_attack"],
            "prediction_change_rate_vs_whpc_uniform": float(np.mean(mm_pred != baseline_pred)) if baseline_pred is not None else 0.0,
            **mm_metrics,
            **{f"delta_{key}_vs_whpc_uniform": float(mm_metrics[key] - baseline_metrics[key]) for key in baseline_metrics},
        }
    )

    dataset_payload = {
        "metadata": dataset.metadata,
        "validated_center_boundary": {"selected": selected_center, "rows": center_validation_rows},
        "validated_multimodal": {"selected": selected_mm, "rows": mm_validation_rows},
        "test": {"rows": test_rows},
    }
    return dataset_payload, center_validation_rows + mm_validation_rows, test_rows


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def _resolve_mm_sample_weight_strategies(args: argparse.Namespace) -> list[str]:
    if args.mm_selection_protocol == "m1_frozen":
        return ["uniform"]
    return list(args.sample_weight_strategies)


def _validate_full_args(args: argparse.Namespace) -> None:
    if args.validation_size <= 0.0 or args.validation_size >= 1.0:
        raise ValueError("validation_size must be between 0 and 1.")
    if any(count < 1 for count in args.prototype_counts):
        raise ValueError("All prototype counts must be at least 1.")
    if any(scale <= 0.0 for scale in args.boundary_scales):
        raise ValueError("All boundary scales must be positive.")
    if any(gamma < 0.0 for gamma in args.boundary_gammas):
        raise ValueError("All boundary gammas must be non-negative.")
    if args.prototype_softmax_alpha <= 0.0:
        raise ValueError("prototype_softmax_alpha must be positive.")
    if args.prototype_penalty < 0.0:
        raise ValueError("prototype_penalty must be non-negative.")
    if args.mm_selection_protocol == "m1_frozen":
        if args.prototype_aggregation != "max":
            raise ValueError("M1 frozen MM-WHPC protocol requires prototype_aggregation='max'.")
        if args.sample_weight_strategies != ["uniform"]:
            raise ValueError(
                "M1 frozen MM-WHPC protocol requires sample_weight_strategies=['uniform']. "
                "Use --mm-selection-protocol custom to explore other local weighting rules."
            )


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)


if __name__ == "__main__":
    main()
