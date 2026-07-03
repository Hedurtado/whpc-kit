from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

from whpc import (
    AdaptiveOCMMWHPCDetector,
    OCMMWHPCDetector,
    compute_feature_weights,
    make_cic_ids2017_preprocessor,
    make_nsl_kdd_preprocessor,
    make_unsw_preprocessor,
)
from whpc.multimodal_aggregation import aggregate_prototype_scores
from whpc.one_class_protocols import (
    CIC_IDS2017_AUXILIARY_ATTACK_FAMILIES,
    CIC_IDS2017_SEEN_ATTACK_FAMILIES,
    CIC_IDS2017_UNSEEN_ATTACK_FAMILIES,
    UNSW_NB15_AUXILIARY_ATTACK_FAMILIES,
    UNSW_NB15_SEEN_ATTACK_FAMILIES,
    UNSW_NB15_UNSEEN_ATTACK_FAMILIES,
    OneClassDatasetSplit,
    build_cic_ids2017_one_class_split,
    build_nsl_kdd_one_class_split,
    build_unsw_nb15_one_class_split,
    compute_one_class_metrics,
    load_cic_ids2017_part3_frames,
    load_nsl_kdd_raw_frames,
    load_unsw_nb15_raw_frames,
    select_best_one_class_row,
)


UNSUPERVISED_BETA_STRATEGIES = {"identity", "variance", "mean_activation"}
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_M2_FINAL_THRESHOLDS_PATH = PROJECT_ROOT / "W-HPC/results/part3_final_thresholds/final_thresholds.json"
DEFAULT_M2_PUBLISHED_REFERENCE_PATH = Path(__file__).resolve().parent / "reference" / "m2_published_reference.json"


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    preprocessor_factory: object


@dataclass(frozen=True)
class Part3Variant:
    label: str
    sample_weight_strategy: str
    prototype_aggregation: str
    prototype_softmax_alpha: float | None
    n_normal_representatives: int
    boundary_scale: float | None
    boundary_gamma: float | None
    beta_strategy: str
    density_lambda: float


@dataclass(frozen=True)
class Part4Variant:
    label: str
    model_family: str
    sample_weight_strategy: str
    prototype_aggregation: str
    prototype_softmax_alpha: float | None
    n_normal_representatives: int
    candidate_n_normal_representatives: tuple[int, ...]
    boundary_scale: float | None
    boundary_gamma: float | None
    beta_strategy: str
    density_lambda: float
    structure_selection_strategy: str | None = None
    structure_selection_metric: str | None = None
    structure_selection_beta: float | None = None
    structure_complexity_penalty: float = 0.0


@dataclass(frozen=True)
class ModeDensityCache:
    neighbors_by_mode: dict[int, NearestNeighbors]
    reference_distances_by_mode: dict[int, np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke reproducibility entry point for FP2/M2.")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--datasets", nargs="+", default=["unsw-nb15", "nsl-kdd", "cic-ids2017"], choices=["unsw-nb15", "nsl-kdd", "cic-ids2017"])
    parser.add_argument("--unsw-raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--nsl-raw-dir", type=Path, default=Path("data/raw/archive"))
    parser.add_argument("--cic-raw-dir", type=Path, default=Path("data/raw/CIC-IDS2017"))
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--test-normal-size", type=float, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--full-protocol", choices=["m2_frozen", "custom"], default="m2_frozen")
    parser.add_argument("--beta-calibration-size", type=float, default=0.5)
    parser.add_argument("--beta-max-supervised-samples", type=int, default=50000)
    parser.add_argument("--density-k", type=int, default=15)
    parser.add_argument("--density-reference-cap", type=int, default=10000)
    parser.add_argument("--adaptive-random-states", type=int, nargs="+", default=[7, 42, 84])
    parser.add_argument("--adaptive-complexity-penalty", type=float, default=0.001)
    parser.add_argument("--m2-final-thresholds", type=Path, default=DEFAULT_M2_FINAL_THRESHOLDS_PATH)
    parser.add_argument("--m2-published-reference", type=Path, default=DEFAULT_M2_PUBLISHED_REFERENCE_PATH)
    parser.add_argument("--n-normal-representatives", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--prototype-aggregations", nargs="+", default=["max", "softmax"], choices=["max", "softmax"])
    parser.add_argument("--prototype-softmax-alphas", type=float, nargs="+", default=[5.0, 10.0, 20.0])
    parser.add_argument("--sample-weight-strategies", nargs="+", default=["uniform", "center_boundary"], choices=["uniform", "intra_class_core", "mixed_intra_class_core", "optimized_margin", "center_boundary"])
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--boundary-scale", type=float, default=0.5)
    parser.add_argument("--boundary-gamma", type=float, default=8.0)
    parser.add_argument("--candidate-m", type=int, nargs="+", default=[1, 2, 3, 4, 5])
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
            Path("reproducibility/artifacts/fp2_smoke_summary.json")
            if args.mode == "smoke"
            else Path("reproducibility/artifacts/fp2_full_summary.json")
        )
    if args.mode == "full":
        _run_full(args)
        return

    X_train_normal = make_normal_only_bimodal_dataset()
    X_val, y_val = make_validation_dataset()

    ocmm = OCMMWHPCDetector(n_normal_representatives=2, normalize=True, random_state=0)
    ocmm.fit(X_train_normal)
    ocmm_threshold = ocmm.calibrate_threshold(X_val, y_val)
    ocmm_predictions = ocmm.predict(X_val)

    adaptive = AdaptiveOCMMWHPCDetector(
        structure_selection_strategy="grid",
        candidate_n_normal_representatives=(1, 2),
        normalize=True,
        random_state=0,
    )
    adaptive.fit(X_train_normal, X_val=X_val, y_val=y_val)
    adaptive_predictions = adaptive.predict(X_val)

    summary = {
        "mode": "smoke",
        "ocmm_threshold": float(ocmm_threshold),
        "ocmm_predictions": ocmm_predictions.tolist(),
        "adaptive_threshold": float(adaptive.threshold_),
        "adaptive_predictions": adaptive_predictions.tolist(),
        "adaptive_selected_representatives": int(adaptive.n_normal_representatives_),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def _run_full(args: argparse.Namespace) -> None:
    if args.full_protocol == "m2_frozen":
        _run_full_m2_frozen(args)
        return
    _run_full_custom(args)


def _run_full_m2_frozen(args: argparse.Namespace) -> None:
    _validate_frozen_args(args)
    dataset_configs = _dataset_configs()
    validation_rows: list[dict[str, object]] = []
    test_rows: list[dict[str, object]] = []
    payload: dict[str, object] = {
        "mode": "full",
        "protocol": {
            "full_protocol": args.full_protocol,
            "datasets": args.datasets,
            "validation_size": args.validation_size,
            "test_normal_size": args.test_normal_size,
            "random_state": args.random_state,
            "beta_calibration_size": args.beta_calibration_size,
            "beta_max_supervised_samples": args.beta_max_supervised_samples,
            "density_k": args.density_k,
            "density_reference_cap": args.density_reference_cap,
            "adaptive_random_states": args.adaptive_random_states,
            "adaptive_complexity_penalty": args.adaptive_complexity_penalty,
            "candidate_m": args.candidate_m,
            "m2_final_thresholds": str(args.m2_final_thresholds),
            "m2_published_reference": str(args.m2_published_reference),
        },
        "datasets": {},
    }

    for dataset_name in args.datasets:
        dataset_payload, dataset_validation_rows, dataset_test_rows = _run_dataset_m2_frozen(dataset_configs[dataset_name], args)
        payload["datasets"][dataset_name] = dataset_payload
        validation_rows.extend(dataset_validation_rows)
        test_rows.extend(dataset_test_rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_rows(args.output.with_name("fp2_full_validation.csv"), validation_rows)
    _write_rows(args.output.with_name("fp2_full_test.csv"), test_rows)
    print(json.dumps({"summary_json": str(args.output), "validation_rows": len(validation_rows), "test_rows": len(test_rows)}, indent=2))


def _run_dataset_m2_frozen(
    dataset_config: DatasetConfig,
    args: argparse.Namespace,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    split = _load_split_for_dataset(dataset_config.name, args)
    part3_validation_rows, part3_test_rows, part3_payload = _run_part3_frozen(dataset_config, split, args)
    part4_validation_rows, part4_test_rows, part4_payload = _run_part4_frozen(dataset_config, args)
    dataset_payload = {
        "split_metadata": split.metadata,
        "part3": part3_payload,
        "part4": part4_payload,
        "reference_alignment": _build_m2_reference_alignment(dataset_config.name, part3_payload, part4_payload, args),
    }
    validation_rows = part3_validation_rows + part4_validation_rows
    test_rows = part3_test_rows + part4_test_rows
    return dataset_payload, validation_rows, test_rows


def _run_part3_frozen(
    dataset_config: DatasetConfig,
    split: OneClassDatasetSplit,
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    preprocessor = dataset_config.preprocessor_factory(split.X_train_normal)
    X_train_t = preprocessor.transformer.fit_transform(split.X_train_normal)
    X_val_t = preprocessor.transformer.transform(split.X_val)
    X_test_standard_t = preprocessor.transformer.transform(split.X_test_standard)
    X_test_unseen_t = preprocessor.transformer.transform(split.X_test_unseen)
    unseen_attack_families = tuple(split.metadata["unseen_attack_families"])

    X_beta_aux_t, X_beta_cal_t, y_beta_aux, y_beta_cal, _, families_beta_cal = _build_beta_split(
        X_val_t,
        np.asarray(split.y_val, dtype=int),
        split.val_attack_families.reset_index(drop=True),
        args,
    )

    variants = [
        _baseline_part3_variant(dataset_config.name),
        _frozen_part3_variant(dataset_config.name, args),
    ]

    validation_rows: list[dict[str, object]] = []
    test_rows: list[dict[str, object]] = []
    variant_payloads: list[dict[str, object]] = []

    for variant in variants:
        beta = _compute_variant_beta(variant, X_train_t, X_beta_aux_t, y_beta_aux, args.random_state)
        model = _make_ocmm_variant_model(variant, beta, args.random_state)
        model.fit(X_train_t)
        density_cache = _build_density_cache_for_model(model, X_train_t, variant.density_lambda, args.density_k, args.density_reference_cap, args.random_state)
        threshold, validation_objective, validation_metrics = _calibrate_variant_threshold(
            model,
            X_beta_cal_t,
            np.asarray(y_beta_cal, dtype=int),
            families_beta_cal.reset_index(drop=True),
            unseen_attack_families,
            variant.density_lambda,
            density_cache,
            args.density_k,
        )
        validation_rows.append(
            {
                "dataset": dataset_config.name,
                "phase": "fp2_part3_validation",
                "protocol": "m2_frozen",
                "variant": variant.label,
                "sample_weight_strategy": variant.sample_weight_strategy,
                "prototype_aggregation": variant.prototype_aggregation,
                "prototype_softmax_alpha": variant.prototype_softmax_alpha,
                "n_normal_representatives": variant.n_normal_representatives,
                "beta_strategy": variant.beta_strategy,
                "density_lambda": variant.density_lambda,
                "threshold_metric_label": "balanced_accuracy",
                "threshold": float(threshold),
                "validation_objective": float(validation_objective),
                **validation_metrics,
            }
        )
        test_rows.extend(
            [
                _evaluate_ocmm_variant_scenario(
                    dataset_config.name,
                    "fp2_part3_test",
                    variant,
                    model,
                    threshold,
                    density_cache,
                    args.density_k,
                    "standard",
                    X_test_standard_t,
                    np.asarray(split.y_test_standard, dtype=int),
                    split.test_standard_attack_families,
                    unseen_attack_families,
                ),
                _evaluate_ocmm_variant_scenario(
                    dataset_config.name,
                    "fp2_part3_test",
                    variant,
                    model,
                    threshold,
                    density_cache,
                    args.density_k,
                    "unseen",
                    X_test_unseen_t,
                    np.asarray(split.y_test_unseen, dtype=int),
                    split.test_unseen_attack_families,
                    unseen_attack_families,
                ),
            ]
        )
        variant_payloads.append(
            {
                "variant": variant.label,
                "config": variant.__dict__,
                "threshold": float(threshold),
                "validation_metrics": validation_metrics,
            }
        )

    return validation_rows, test_rows, {"variants": variant_payloads, "rows": test_rows[-4:]}


def _run_part4_frozen(
    dataset_config: DatasetConfig,
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    seed_validation_rows: list[dict[str, object]] = []
    seed_test_rows: list[dict[str, object]] = []

    for seed in args.adaptive_random_states:
        seed_args = argparse.Namespace(**vars(args))
        seed_args.random_state = int(seed)
        split = _load_split_for_dataset(dataset_config.name, seed_args)
        validation_rows, test_rows = _run_part4_seed(dataset_config, split, seed_args)
        seed_validation_rows.extend(validation_rows)
        seed_test_rows.extend(test_rows)

    summary_rows = _summarize_part4_rows(seed_test_rows)
    payload = {
        "random_states": [int(value) for value in args.adaptive_random_states],
        "seed_rows": seed_test_rows,
        "summary_rows": summary_rows,
    }
    return seed_validation_rows, summary_rows, payload


def _run_part4_seed(
    dataset_config: DatasetConfig,
    split: OneClassDatasetSplit,
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    preprocessor = dataset_config.preprocessor_factory(split.X_train_normal)
    X_train_t = preprocessor.transformer.fit_transform(split.X_train_normal)
    X_val_t = preprocessor.transformer.transform(split.X_val)
    X_test_standard_t = preprocessor.transformer.transform(split.X_test_standard)
    X_test_unseen_t = preprocessor.transformer.transform(split.X_test_unseen)
    unseen_attack_families = tuple(split.metadata["unseen_attack_families"])

    X_beta_aux_t, X_beta_cal_t, y_beta_aux, y_beta_cal, _, families_beta_cal = _build_beta_split(
        X_val_t,
        np.asarray(split.y_val, dtype=int),
        split.val_attack_families.reset_index(drop=True),
        args,
    )

    variants = [
        _baseline_part4_variant(dataset_config.name),
        _frozen_part4_variant(dataset_config.name, args),
        _adaptive_part4_variant(dataset_config.name, args),
    ]

    validation_rows: list[dict[str, object]] = []
    test_rows: list[dict[str, object]] = []
    for variant in variants:
        beta = _compute_variant_beta(variant, X_train_t, X_beta_aux_t, y_beta_aux, args.random_state)
        model = _make_part4_model(variant, beta, args.random_state, args)
        if isinstance(model, AdaptiveOCMMWHPCDetector):
            model.fit(X_train_t, X_val=X_beta_cal_t, y_val=np.asarray(y_beta_cal, dtype=int))
            threshold = float(model.threshold_)
            validation_metrics = _metrics_from_scores(
                np.asarray(y_beta_cal, dtype=int),
                model.score_samples(X_beta_cal_t),
                (model.score_samples(X_beta_cal_t) < threshold).astype(int),
                families_beta_cal.reset_index(drop=True),
                unseen_attack_families,
            )
            density_cache = None
        else:
            model.fit(X_train_t)
            density_cache = _build_density_cache_for_model(model, X_train_t, variant.density_lambda, args.density_k, args.density_reference_cap, args.random_state)
            threshold, validation_objective, validation_metrics = _calibrate_variant_threshold(
                model,
                X_beta_cal_t,
                np.asarray(y_beta_cal, dtype=int),
                families_beta_cal.reset_index(drop=True),
                unseen_attack_families,
                variant.density_lambda,
                density_cache,
                args.density_k,
            )
            validation_rows.append(
                {
                    "dataset": dataset_config.name,
                    "phase": "fp2_part4_validation",
                    "protocol": "m2_frozen",
                    "random_state": args.random_state,
                    "variant": variant.label,
                    "model_family": variant.model_family,
                    "selected_m": int(variant.n_normal_representatives),
                    "threshold": float(threshold),
                    "validation_objective": float(validation_objective),
                    **validation_metrics,
                }
            )

        selected_m = int(getattr(model, "n_normal_representatives_", variant.n_normal_representatives))
        if isinstance(model, AdaptiveOCMMWHPCDetector):
            validation_rows.append(
                {
                    "dataset": dataset_config.name,
                    "phase": "fp2_part4_validation",
                    "protocol": "m2_frozen",
                    "random_state": args.random_state,
                    "variant": variant.label,
                    "model_family": variant.model_family,
                    "selected_m": selected_m,
                    "threshold": float(threshold),
                    "validation_objective": float(getattr(model, "adaptive_structure_summary_", {}).get("objective_value", 0.0)),
                    **validation_metrics,
                }
            )

        test_rows.extend(
            [
                _evaluate_part4_variant_scenario(
                    dataset_config.name,
                    variant,
                    model,
                    threshold,
                    density_cache,
                    args.density_k,
                    args.random_state,
                    selected_m,
                    "standard",
                    X_test_standard_t,
                    np.asarray(split.y_test_standard, dtype=int),
                    split.test_standard_attack_families,
                    unseen_attack_families,
                ),
                _evaluate_part4_variant_scenario(
                    dataset_config.name,
                    variant,
                    model,
                    threshold,
                    density_cache,
                    args.density_k,
                    args.random_state,
                    selected_m,
                    "unseen",
                    X_test_unseen_t,
                    np.asarray(split.y_test_unseen, dtype=int),
                    split.test_unseen_attack_families,
                    unseen_attack_families,
                ),
            ]
        )

    return validation_rows, test_rows


def _run_full_custom(args: argparse.Namespace) -> None:
    dataset_configs = _dataset_configs()
    validation_rows: list[dict[str, object]] = []
    test_rows: list[dict[str, object]] = []
    payload: dict[str, object] = {
        "mode": "full",
        "protocol": {
            "full_protocol": args.full_protocol,
            "datasets": args.datasets,
            "validation_size": args.validation_size,
            "test_normal_size": args.test_normal_size,
            "random_state": args.random_state,
            "n_normal_representatives": args.n_normal_representatives,
            "prototype_aggregations": args.prototype_aggregations,
            "prototype_softmax_alphas": args.prototype_softmax_alphas,
            "sample_weight_strategies": args.sample_weight_strategies,
            "candidate_m": args.candidate_m,
        },
        "datasets": {},
    }

    for dataset_name in args.datasets:
        split = _load_split_for_dataset(dataset_name, args)
        dataset_payload, dataset_validation_rows, dataset_test_rows = _run_dataset_custom(dataset_configs[dataset_name], split, args)
        payload["datasets"][dataset_name] = dataset_payload
        validation_rows.extend(dataset_validation_rows)
        test_rows.extend(dataset_test_rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_rows(args.output.with_name("fp2_full_validation.csv"), validation_rows)
    _write_rows(args.output.with_name("fp2_full_test.csv"), test_rows)
    print(json.dumps({"summary_json": str(args.output), "validation_rows": len(validation_rows), "test_rows": len(test_rows)}, indent=2))


def _load_split_for_dataset(dataset_name: str, args: argparse.Namespace) -> OneClassDatasetSplit:
    if dataset_name == "unsw-nb15":
        train_df, test_df = load_unsw_nb15_raw_frames(args.unsw_raw_dir)
        return build_unsw_nb15_one_class_split(
            train_df,
            test_df,
            validation_size=args.validation_size,
            random_state=args.random_state,
            seen_attack_families=UNSW_NB15_SEEN_ATTACK_FAMILIES,
            unseen_attack_families=UNSW_NB15_UNSEEN_ATTACK_FAMILIES,
            auxiliary_attack_families=UNSW_NB15_AUXILIARY_ATTACK_FAMILIES,
        )
    if dataset_name == "nsl-kdd":
        train_df, test_df = load_nsl_kdd_raw_frames(args.nsl_raw_dir)
        return build_nsl_kdd_one_class_split(
            train_df,
            test_df,
            validation_size=args.validation_size,
            random_state=args.random_state,
        )
    frames = load_cic_ids2017_part3_frames(args.cic_raw_dir)
    return build_cic_ids2017_one_class_split(
        frames,
        validation_size=args.validation_size,
        test_normal_size=args.test_normal_size,
        random_state=args.random_state,
        seen_attack_families=CIC_IDS2017_SEEN_ATTACK_FAMILIES,
        unseen_attack_families=CIC_IDS2017_UNSEEN_ATTACK_FAMILIES,
        auxiliary_attack_families=CIC_IDS2017_AUXILIARY_ATTACK_FAMILIES,
    )


def _run_dataset_custom(
    dataset_config: DatasetConfig,
    split: OneClassDatasetSplit,
    args: argparse.Namespace,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    preprocessor = dataset_config.preprocessor_factory(split.X_train_normal)
    X_train_t = preprocessor.transformer.fit_transform(split.X_train_normal)
    X_val_t = preprocessor.transformer.transform(split.X_val)
    X_test_standard_t = preprocessor.transformer.transform(split.X_test_standard)
    X_test_unseen_t = preprocessor.transformer.transform(split.X_test_unseen)

    validation_rows: list[dict[str, object]] = []
    for strategy in args.sample_weight_strategies:
        for aggregation in args.prototype_aggregations:
            alpha_grid = args.prototype_softmax_alphas if aggregation == "softmax" else [None]
            for n_representatives in args.n_normal_representatives:
                for softmax_alpha in alpha_grid:
                    model = _make_ocmm_model_custom(
                        sample_weight_strategy=strategy,
                        prototype_aggregation=aggregation,
                        n_normal_representatives=n_representatives,
                        prototype_softmax_alpha=softmax_alpha,
                        args=args,
                    )
                    model.fit(X_train_t)
                    model.calibrate_threshold(X_val_t, split.y_val)
                    val_scores = model.score_samples(X_val_t)
                    val_pred = (model.predict(X_val_t) == model.anomaly_label).astype(int)
                    metrics = compute_one_class_metrics(
                        split.y_val,
                        val_scores,
                        val_pred,
                        attack_families=split.val_attack_families,
                        unseen_attack_families=tuple(split.metadata["unseen_attack_families"]),
                    )
                    validation_rows.append(
                        {
                            "dataset": dataset_config.name,
                            "phase": "fp2_part3_validation",
                            "sample_weight_strategy": strategy,
                            "prototype_aggregation": aggregation,
                            "prototype_softmax_alpha": softmax_alpha,
                            "n_normal_representatives": n_representatives,
                            "threshold": float(model.threshold_),
                            **metrics,
                        }
                    )

    selected = select_best_one_class_row(validation_rows)
    selected_model = _make_ocmm_model_custom(
        sample_weight_strategy=str(selected["sample_weight_strategy"]),
        prototype_aggregation=str(selected["prototype_aggregation"]),
        n_normal_representatives=int(selected["n_normal_representatives"]),
        prototype_softmax_alpha=None if selected["prototype_softmax_alpha"] is None else float(selected["prototype_softmax_alpha"]),
        args=args,
    )
    selected_model.fit(X_train_t)
    selected_model.calibrate_threshold(X_val_t, split.y_val)

    test_rows = [
        _evaluate_ocmm_scenario_custom(dataset_config.name, selected_model, "standard", X_test_standard_t, split.y_test_standard, split.test_standard_attack_families, selected, tuple(split.metadata["unseen_attack_families"])),
        _evaluate_ocmm_scenario_custom(dataset_config.name, selected_model, "unseen", X_test_unseen_t, split.y_test_unseen, split.test_unseen_attack_families, selected, tuple(split.metadata["unseen_attack_families"])),
    ]

    adaptive = AdaptiveOCMMWHPCDetector(
        structure_selection_strategy="grid",
        candidate_n_normal_representatives=tuple(int(value) for value in args.candidate_m),
        prototype_aggregation=str(selected["prototype_aggregation"]),
        prototype_softmax_alpha=10.0 if selected["prototype_softmax_alpha"] is None else float(selected["prototype_softmax_alpha"]),
        sample_weight_strategy=str(selected["sample_weight_strategy"]),
        sample_weight_temperature=args.temperature,
        sample_weight_boundary_gamma=args.boundary_gamma,
        sample_weight_boundary_scale=args.boundary_scale,
        normalize=True,
        score_metric="cosine",
        threshold_selection_metric="balanced_accuracy",
        random_state=args.random_state,
    )
    adaptive.fit(X_train_t, X_val=X_val_t, y_val=split.y_val)
    test_rows.extend(
        [
            _evaluate_adaptive_scenario_custom(dataset_config.name, adaptive, "standard", X_test_standard_t, split.y_test_standard, split.test_standard_attack_families, tuple(split.metadata["unseen_attack_families"])),
            _evaluate_adaptive_scenario_custom(dataset_config.name, adaptive, "unseen", X_test_unseen_t, split.y_test_unseen, split.test_unseen_attack_families, tuple(split.metadata["unseen_attack_families"])),
        ]
    )

    dataset_payload = {
        "split_metadata": split.metadata,
        "part3": {"selected": selected, "validation_rows": validation_rows},
        "part4": {"adaptive_selected_representatives": int(adaptive.n_normal_representatives_), "adaptive_structure_summary": getattr(adaptive, "adaptive_structure_summary_", {})},
        "test": {"rows": test_rows},
    }
    return dataset_payload, validation_rows, test_rows


def _make_ocmm_model_custom(
    *,
    sample_weight_strategy: str,
    prototype_aggregation: str,
    n_normal_representatives: int,
    prototype_softmax_alpha: float | None,
    args: argparse.Namespace,
) -> OCMMWHPCDetector:
    return OCMMWHPCDetector(
        n_normal_representatives=n_normal_representatives,
        representative_partition_strategy="kmeans",
        prototype_aggregation=prototype_aggregation,
        prototype_softmax_alpha=10.0 if prototype_softmax_alpha is None else prototype_softmax_alpha,
        sample_weight_strategy=sample_weight_strategy,
        sample_weight_temperature=args.temperature,
        sample_weight_boundary_gamma=args.boundary_gamma,
        sample_weight_boundary_scale=args.boundary_scale,
        feature_weight=None,
        normalize=True,
        score_metric="cosine",
        threshold_selection_metric="f1_anomalous",
        random_state=args.random_state,
    )


def _evaluate_ocmm_scenario_custom(
    dataset_name: str,
    model: OCMMWHPCDetector,
    scenario_name: str,
    X: np.ndarray,
    y_true: np.ndarray,
    attack_families: pd.Series,
    selected: dict[str, object],
    unseen_attack_families: tuple[str, ...],
) -> dict[str, object]:
    scores = model.score_samples(X)
    y_pred = (model.predict(X) == model.anomaly_label).astype(int)
    metrics = compute_one_class_metrics(y_true, scores, y_pred, attack_families=attack_families, unseen_attack_families=unseen_attack_families)
    return {
        "dataset": dataset_name,
        "phase": "fp2_part3_test",
        "scenario": scenario_name,
        "sample_weight_strategy": selected["sample_weight_strategy"],
        "prototype_aggregation": selected["prototype_aggregation"],
        "prototype_softmax_alpha": selected["prototype_softmax_alpha"],
        "n_normal_representatives": selected["n_normal_representatives"],
        "threshold": float(model.threshold_),
        **metrics,
    }


def _evaluate_adaptive_scenario_custom(
    dataset_name: str,
    model: AdaptiveOCMMWHPCDetector,
    scenario_name: str,
    X: np.ndarray,
    y_true: np.ndarray,
    attack_families: pd.Series,
    unseen_attack_families: tuple[str, ...],
) -> dict[str, object]:
    scores = model.score_samples(X)
    y_pred = (model.predict(X) == model.anomaly_label).astype(int)
    metrics = compute_one_class_metrics(y_true, scores, y_pred, attack_families=attack_families, unseen_attack_families=unseen_attack_families)
    return {
        "dataset": dataset_name,
        "phase": "fp2_part4_test",
        "scenario": scenario_name,
        "sample_weight_strategy": model.sample_weight_strategy,
        "prototype_aggregation": model.prototype_aggregation,
        "n_normal_representatives": int(model.n_normal_representatives_),
        "threshold": float(model.threshold_),
        **metrics,
    }


def _dataset_configs() -> dict[str, DatasetConfig]:
    return {
        "unsw-nb15": DatasetConfig("unsw-nb15", make_unsw_preprocessor),
        "nsl-kdd": DatasetConfig("nsl-kdd", make_nsl_kdd_preprocessor),
        "cic-ids2017": DatasetConfig("cic-ids2017", make_cic_ids2017_preprocessor),
    }


def _baseline_part3_variant(dataset_name: str) -> Part3Variant:
    if dataset_name == "nsl-kdd":
        return Part3Variant("ocwhpc_baseline", "uniform", "max", None, 1, None, None, "identity", 0.0)
    return Part3Variant("ocwhpc_baseline", "uniform", "softmax", 5.0, 1, None, None, "identity", 0.0)


def _fallback_frozen_part3_variant(dataset_name: str) -> Part3Variant:
    if dataset_name == "unsw-nb15":
        return Part3Variant("frozen_ocmmwhpc", "intra_class_core", "softmax", 5.0, 1, None, None, "mutual_information", 0.0)
    if dataset_name == "nsl-kdd":
        return Part3Variant("frozen_ocmmwhpc", "optimized_margin", "softmax", 5.0, 2, None, None, "identity", 0.4)
    return Part3Variant("frozen_ocmmwhpc", "center_boundary", "softmax", 5.0, 5, 0.5, 8.0, "identity", 0.4)


def _frozen_part3_variant(dataset_name: str, args: argparse.Namespace) -> Part3Variant:
    payload = _load_json_payload(args.m2_final_thresholds)
    if payload is None:
        return _fallback_frozen_part3_variant(dataset_name)

    rows = [
        row
        for row in payload.get("datasets", {}).get(dataset_name, {}).get("rows", [])
        if row.get("scenario") == "unseen" and row.get("threshold_metric_label") == "balanced_accuracy"
    ]
    if not rows:
        return _fallback_frozen_part3_variant(dataset_name)

    selected = max(
        rows,
        key=lambda row: (
            float(row["f1_anomalous"]),
            float(row["balanced_accuracy"]),
            float(row["anomaly_recall"]),
        ),
    )
    boundary_scale = None
    boundary_gamma = None
    if selected["sample_weight_strategy"] == "center_boundary":
        boundary_scale = 0.5
        boundary_gamma = 8.0
    return Part3Variant(
        "frozen_ocmmwhpc",
        str(selected["sample_weight_strategy"]),
        str(selected["prototype_aggregation"]),
        None if selected["prototype_softmax_alpha"] in ("", None) else float(selected["prototype_softmax_alpha"]),
        int(selected["n_normal_representatives"]),
        boundary_scale,
        boundary_gamma,
        str(selected["beta_strategy"]),
        float(selected["density_lambda"]),
    )


def _baseline_part4_variant(dataset_name: str) -> Part4Variant:
    baseline = _baseline_part3_variant(dataset_name)
    return Part4Variant(
        label="ocwhpc_baseline",
        model_family="ocwhpc",
        sample_weight_strategy=baseline.sample_weight_strategy,
        prototype_aggregation=baseline.prototype_aggregation,
        prototype_softmax_alpha=baseline.prototype_softmax_alpha,
        n_normal_representatives=baseline.n_normal_representatives,
        candidate_n_normal_representatives=(baseline.n_normal_representatives,),
        boundary_scale=baseline.boundary_scale,
        boundary_gamma=baseline.boundary_gamma,
        beta_strategy=baseline.beta_strategy,
        density_lambda=baseline.density_lambda,
    )


def _frozen_part4_variant(dataset_name: str, args: argparse.Namespace) -> Part4Variant:
    frozen = _frozen_part3_variant(dataset_name, args)
    return Part4Variant(
        label="frozen_ocmmwhpc",
        model_family="frozen_ocmmwhpc",
        sample_weight_strategy=frozen.sample_weight_strategy,
        prototype_aggregation=frozen.prototype_aggregation,
        prototype_softmax_alpha=frozen.prototype_softmax_alpha,
        n_normal_representatives=frozen.n_normal_representatives,
        candidate_n_normal_representatives=(frozen.n_normal_representatives,),
        boundary_scale=frozen.boundary_scale,
        boundary_gamma=frozen.boundary_gamma,
        beta_strategy=frozen.beta_strategy,
        density_lambda=frozen.density_lambda,
    )


def _adaptive_part4_variant(dataset_name: str, args: argparse.Namespace) -> Part4Variant:
    frozen = _frozen_part4_variant(dataset_name, args)
    return Part4Variant(
        label="adaptive_ocmmwhpc",
        model_family="adaptive_grid",
        sample_weight_strategy=frozen.sample_weight_strategy,
        prototype_aggregation=frozen.prototype_aggregation,
        prototype_softmax_alpha=frozen.prototype_softmax_alpha,
        n_normal_representatives=1,
        candidate_n_normal_representatives=tuple(int(value) for value in args.candidate_m),
        boundary_scale=frozen.boundary_scale,
        boundary_gamma=frozen.boundary_gamma,
        beta_strategy=frozen.beta_strategy,
        density_lambda=frozen.density_lambda,
        structure_selection_strategy="grid",
        structure_selection_metric="f_beta_anomalous",
        structure_selection_beta=2.0,
        structure_complexity_penalty=float(args.adaptive_complexity_penalty),
    )


def _build_beta_split(
    X_val_t: np.ndarray,
    y_val: np.ndarray,
    attack_families: pd.Series,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.Series, pd.Series]:
    X_beta_aux_t, X_beta_cal_t, y_beta_aux, y_beta_cal, families_beta_aux, families_beta_cal = train_test_split(
        X_val_t,
        np.asarray(y_val, dtype=int),
        attack_families,
        test_size=args.beta_calibration_size,
        random_state=args.random_state,
        stratify=np.asarray(y_val, dtype=int),
    )
    X_beta_aux_t, y_beta_aux = _cap_supervised_beta_split(
        np.asarray(X_beta_aux_t, dtype=float),
        np.asarray(y_beta_aux, dtype=int),
        args.beta_max_supervised_samples,
        args.random_state,
    )
    return X_beta_aux_t, np.asarray(X_beta_cal_t, dtype=float), y_beta_aux, np.asarray(y_beta_cal, dtype=int), families_beta_aux, families_beta_cal


def _cap_supervised_beta_split(
    X: np.ndarray,
    y: np.ndarray,
    max_samples: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    if X.shape[0] <= max_samples:
        return X, y
    sampled_indices, _ = train_test_split(
        np.arange(X.shape[0]),
        train_size=max_samples,
        random_state=random_state,
        stratify=y,
    )
    sampled_indices = np.sort(sampled_indices)
    return X[sampled_indices], y[sampled_indices]


def _compute_variant_beta(
    variant: Part3Variant | Part4Variant,
    X_train_t: np.ndarray,
    X_beta_aux_t: np.ndarray,
    y_beta_aux: np.ndarray,
    random_state: int,
) -> np.ndarray | None:
    if variant.beta_strategy == "identity":
        return None
    if variant.beta_strategy in UNSUPERVISED_BETA_STRATEGIES:
        return compute_feature_weights(variant.beta_strategy, X_train_t, y=None, random_state=random_state)
    return compute_feature_weights(variant.beta_strategy, X_beta_aux_t, y_beta_aux, random_state=random_state)


def _make_ocmm_variant_model(
    variant: Part3Variant,
    beta: np.ndarray | None,
    random_state: int,
) -> OCMMWHPCDetector:
    return OCMMWHPCDetector(
        n_normal_representatives=variant.n_normal_representatives,
        representative_partition_strategy="kmeans",
        prototype_aggregation=variant.prototype_aggregation,
        prototype_softmax_alpha=10.0 if variant.prototype_softmax_alpha is None else variant.prototype_softmax_alpha,
        sample_weight_strategy=variant.sample_weight_strategy,
        sample_weight_temperature=2.0,
        sample_weight_boundary_gamma=0.5 if variant.boundary_gamma is None else variant.boundary_gamma,
        sample_weight_boundary_scale=0.1 if variant.boundary_scale is None else variant.boundary_scale,
        feature_weight=beta,
        normalize=True,
        score_metric="cosine",
        threshold_selection_metric="f1_anomalous",
        random_state=random_state,
    )


def _make_part4_model(
    variant: Part4Variant,
    beta: np.ndarray | None,
    random_state: int,
    args: argparse.Namespace,
) -> OCMMWHPCDetector | AdaptiveOCMMWHPCDetector:
    common_kwargs = {
        "representative_partition_strategy": "kmeans",
        "representative_strategy": "weighted_mean",
        "prototype_aggregation": variant.prototype_aggregation,
        "prototype_softmax_alpha": 10.0 if variant.prototype_softmax_alpha is None else variant.prototype_softmax_alpha,
        "sample_weight_strategy": variant.sample_weight_strategy,
        "sample_weight_temperature": 2.0,
        "sample_weight_boundary_gamma": 0.5 if variant.boundary_gamma is None else variant.boundary_gamma,
        "sample_weight_boundary_scale": 0.1 if variant.boundary_scale is None else variant.boundary_scale,
        "feature_weight": beta,
        "normalize": True,
        "score_metric": "cosine",
        "threshold_selection_metric": "balanced_accuracy",
        "random_state": random_state,
    }
    if variant.model_family.startswith("adaptive"):
        return AdaptiveOCMMWHPCDetector(
            structure_selection_strategy="grid" if variant.structure_selection_strategy is None else variant.structure_selection_strategy,
            candidate_n_normal_representatives=variant.candidate_n_normal_representatives,
            max_normal_representatives=max(variant.candidate_n_normal_representatives),
            structure_complexity_penalty=variant.structure_complexity_penalty,
            structure_min_gain=0.001,
            hierarchical_min_mode_size=512,
            structure_selection_metric=variant.structure_selection_metric,
            structure_selection_beta=variant.structure_selection_beta,
            density_lambda=variant.density_lambda,
            density_k=args.density_k,
            density_reference_cap=args.density_reference_cap,
            n_normal_representatives=1,
            **common_kwargs,
        )
    return OCMMWHPCDetector(
        n_normal_representatives=variant.n_normal_representatives,
        **common_kwargs,
    )


def _build_density_cache_for_model(
    model: OCMMWHPCDetector,
    X_train_t: np.ndarray,
    density_lambda: float,
    density_k: int,
    density_reference_cap: int,
    random_state: int,
) -> ModeDensityCache | None:
    if np.isclose(density_lambda, 0.0):
        return None
    return _build_mode_density_cache(
        X_train_t,
        np.asarray(model.normal_partition_labels_, dtype=int),
        density_k,
        density_reference_cap,
        random_state,
    )


def _build_mode_density_cache(
    X_train_t: np.ndarray,
    partition_labels: np.ndarray,
    k: int,
    reference_cap: int,
    random_state: int,
) -> ModeDensityCache:
    rng = np.random.default_rng(random_state)
    neighbors_by_mode: dict[int, NearestNeighbors] = {}
    reference_distances_by_mode: dict[int, np.ndarray] = {}

    for mode_index in sorted(np.unique(partition_labels).tolist()):
        mode_mask = partition_labels == mode_index
        X_mode = np.asarray(X_train_t[mode_mask], dtype=float)
        if X_mode.shape[0] > reference_cap:
            sampled_indices = np.sort(rng.choice(X_mode.shape[0], size=reference_cap, replace=False))
            X_mode = X_mode[sampled_indices]

        neighbor_count = min(max(1, k), X_mode.shape[0])
        neighbors = NearestNeighbors(n_neighbors=neighbor_count, metric="euclidean")
        neighbors.fit(X_mode)
        neighbors_by_mode[int(mode_index)] = neighbors
        reference_distances_by_mode[int(mode_index)] = _reference_mean_distances(X_mode, k)

    return ModeDensityCache(
        neighbors_by_mode=neighbors_by_mode,
        reference_distances_by_mode=reference_distances_by_mode,
    )


def _reference_mean_distances(X_mode: np.ndarray, k: int) -> np.ndarray:
    if X_mode.shape[0] <= 1:
        return np.zeros(X_mode.shape[0], dtype=float)
    neighbor_count = min(k + 1, X_mode.shape[0])
    neighbors = NearestNeighbors(n_neighbors=neighbor_count, metric="euclidean")
    neighbors.fit(X_mode)
    distances, _ = neighbors.kneighbors(X_mode)
    if neighbor_count == 1:
        return distances[:, 0]
    return np.mean(distances[:, 1:], axis=1)


def _score_ocmm_variant(
    model: OCMMWHPCDetector,
    X: np.ndarray,
    density_cache: ModeDensityCache | None,
    density_k: int,
    density_lambda: float,
) -> np.ndarray:
    local_scores = model.score_local_samples(X)
    similarity = _aggregate_normal_scores(local_scores, model.prototype_aggregation, model.prototype_softmax_alpha)
    if np.isclose(density_lambda, 0.0) or density_cache is None:
        return similarity
    residual = _compute_density_residuals(local_scores, X, density_cache, density_k)
    return similarity - (density_lambda * residual)


def _aggregate_normal_scores(local_scores: np.ndarray, aggregation: str, softmax_alpha: float) -> np.ndarray:
    aggregated = aggregate_prototype_scores(
        {"normal": np.asarray(local_scores, dtype=float)},
        aggregation=aggregation,
        softmax_alpha=softmax_alpha,
    )
    return aggregated[:, 0]


def _compute_density_residuals(
    local_scores: np.ndarray,
    X_query_t: np.ndarray,
    density_cache: ModeDensityCache,
    k: int,
) -> np.ndarray:
    winners = np.argmax(local_scores, axis=1)
    residuals = np.zeros(X_query_t.shape[0], dtype=float)

    for mode_index in sorted(np.unique(winners).tolist()):
        mode_mask = winners == mode_index
        neighbors = density_cache.neighbors_by_mode[int(mode_index)]
        reference_distances = np.sort(density_cache.reference_distances_by_mode[int(mode_index)])
        n_neighbors = min(max(1, k), neighbors.n_samples_fit_)
        distances, _ = neighbors.kneighbors(np.asarray(X_query_t[mode_mask], dtype=float), n_neighbors=n_neighbors)
        mean_distances = np.mean(distances, axis=1)
        if reference_distances.size == 0:
            residuals[mode_mask] = 0.0
            continue
        ranks = np.searchsorted(reference_distances, mean_distances, side="right")
        residuals[mode_mask] = ranks / reference_distances.size

    return residuals


def _calibrate_variant_threshold(
    model: OCMMWHPCDetector,
    X_val: np.ndarray,
    y_val: np.ndarray,
    attack_families: pd.Series,
    unseen_attack_families: tuple[str, ...],
    density_lambda: float,
    density_cache: ModeDensityCache | None,
    density_k: int,
) -> tuple[float, float, dict[str, float]]:
    scores = _score_ocmm_variant(model, X_val, density_cache, density_k, density_lambda)
    model.threshold_selection_metric = "balanced_accuracy"
    threshold, validation_objective = model._fit_threshold(scores, np.asarray(y_val, dtype=int))
    y_pred = (scores < threshold).astype(int)
    validation_metrics = _metrics_from_scores(
        np.asarray(y_val, dtype=int),
        scores,
        y_pred,
        attack_families,
        unseen_attack_families,
    )
    return float(threshold), float(validation_objective), validation_metrics


def _metrics_from_scores(
    y_true: np.ndarray,
    scores: np.ndarray,
    y_pred: np.ndarray,
    attack_families: pd.Series,
    unseen_attack_families: tuple[str, ...],
) -> dict[str, float]:
    metrics = compute_one_class_metrics(
        np.asarray(y_true, dtype=int),
        np.asarray(scores, dtype=float),
        np.asarray(y_pred, dtype=int),
        attack_families=attack_families,
        unseen_attack_families=unseen_attack_families,
    )
    return {key: float(value) for key, value in metrics.items()}


def _load_json_payload(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _build_m2_reference_alignment(
    dataset_name: str,
    part3_payload: dict[str, object],
    part4_payload: dict[str, object],
    args: argparse.Namespace | None = None,
) -> dict[str, object]:
    reference_payload = _load_json_payload(
        DEFAULT_M2_PUBLISHED_REFERENCE_PATH if args is None else args.m2_published_reference
    )
    if reference_payload is None:
        return {}
    reference = reference_payload.get("datasets", {}).get(dataset_name)
    if reference is None:
        return {}

    current_part3_by_variant = {
        row["variant"]: row
        for row in part3_payload["rows"]
        if row["scenario"] == "unseen"
    }
    current_part3_standard = {
        row["variant"]: row
        for row in part3_payload["rows"]
        if row["scenario"] == "standard"
    }
    current_part4_by_variant = {
        row["variant"]: row
        for row in part4_payload["summary_rows"]
        if row["scenario"] == "unseen"
    }

    part3_alignment: dict[str, object] = {}
    for variant, ref_metrics in reference["part3"].items():
        unseen_row = current_part3_by_variant.get(variant)
        standard_row = current_part3_standard.get(variant)
        if unseen_row is None or standard_row is None:
            continue
        current = {
            "standard_f1_anomalous": float(standard_row["f1_anomalous"]),
            "unseen_f1_anomalous": float(unseen_row["f1_anomalous"]),
        }
        part3_alignment[variant] = {
            "reference": ref_metrics,
            "current": current,
            "delta": {
                key: current[key] - float(ref_metrics[key])
                for key in ref_metrics
            },
        }

    part4_alignment: dict[str, object] = {}
    for variant, ref_metrics in reference["part4"].items():
        current_row = current_part4_by_variant.get(variant)
        if current_row is None:
            continue
        current = {
            "selected_m_mean": float(current_row["selected_m_mean"]),
            "f1_anomalous_mean": float(current_row["f1_anomalous_mean"]),
            "balanced_accuracy_mean": float(current_row["balanced_accuracy_mean"]),
        }
        part4_alignment[variant] = {
            "reference": ref_metrics,
            "current": current,
            "delta": {
                key: current[key] - float(ref_metrics[key])
                for key in ref_metrics
            },
        }

    return {
        "reference_source": "published_m2_tables",
        "part3": part3_alignment,
        "part4": part4_alignment,
    }


def _evaluate_ocmm_variant_scenario(
    dataset_name: str,
    phase: str,
    variant: Part3Variant,
    model: OCMMWHPCDetector,
    threshold: float,
    density_cache: ModeDensityCache | None,
    density_k: int,
    scenario_name: str,
    X: np.ndarray,
    y_true: np.ndarray,
    attack_families: pd.Series,
    unseen_attack_families: tuple[str, ...],
) -> dict[str, object]:
    scores = _score_ocmm_variant(model, X, density_cache, density_k, variant.density_lambda)
    y_pred = (scores < threshold).astype(int)
    metrics = _metrics_from_scores(y_true, scores, y_pred, attack_families, unseen_attack_families)
    return {
        "dataset": dataset_name,
        "phase": phase,
        "protocol": "m2_frozen",
        "variant": variant.label,
        "scenario": scenario_name,
        "sample_weight_strategy": variant.sample_weight_strategy,
        "prototype_aggregation": variant.prototype_aggregation,
        "prototype_softmax_alpha": variant.prototype_softmax_alpha,
        "n_normal_representatives": variant.n_normal_representatives,
        "beta_strategy": variant.beta_strategy,
        "density_lambda": variant.density_lambda,
        "threshold_metric_label": "balanced_accuracy",
        "threshold": float(threshold),
        **metrics,
    }


def _evaluate_part4_variant_scenario(
    dataset_name: str,
    variant: Part4Variant,
    model: OCMMWHPCDetector | AdaptiveOCMMWHPCDetector,
    threshold: float,
    density_cache: ModeDensityCache | None,
    density_k: int,
    random_state: int,
    selected_m: int,
    scenario_name: str,
    X: np.ndarray,
    y_true: np.ndarray,
    attack_families: pd.Series,
    unseen_attack_families: tuple[str, ...],
) -> dict[str, object]:
    if isinstance(model, AdaptiveOCMMWHPCDetector):
        scores = model.score_samples(X)
    else:
        scores = _score_ocmm_variant(model, X, density_cache, density_k, variant.density_lambda)
    y_pred = (scores < threshold).astype(int)
    metrics = _metrics_from_scores(y_true, scores, y_pred, attack_families, unseen_attack_families)
    return {
        "dataset": dataset_name,
        "phase": "fp2_part4_test",
        "protocol": "m2_frozen",
        "aggregation_level": "seed",
        "random_state": int(random_state),
        "variant": variant.label,
        "model_family": variant.model_family,
        "scenario": scenario_name,
        "sample_weight_strategy": variant.sample_weight_strategy,
        "prototype_aggregation": variant.prototype_aggregation,
        "prototype_softmax_alpha": variant.prototype_softmax_alpha,
        "beta_strategy": variant.beta_strategy,
        "density_lambda": variant.density_lambda,
        "selected_m": int(selected_m),
        "structure_selection_metric": variant.structure_selection_metric,
        "structure_selection_beta": variant.structure_selection_beta,
        "structure_complexity_penalty": variant.structure_complexity_penalty,
        "threshold_metric_label": "balanced_accuracy",
        "threshold": float(threshold),
        **metrics,
    }


def _summarize_part4_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    if not rows:
        return []
    frame = pd.DataFrame(rows)
    summary_rows: list[dict[str, object]] = []
    group_columns = ["dataset", "variant", "model_family", "scenario"]
    metric_columns = [
        "selected_m",
        "threshold",
        "f1_anomalous",
        "balanced_accuracy",
        "anomaly_recall",
        "normal_acceptance_rate",
    ]
    for group_values, group_frame in frame.groupby(group_columns, dropna=False):
        row = dict(zip(group_columns, group_values))
        for carry_column in [
            "sample_weight_strategy",
            "prototype_aggregation",
            "prototype_softmax_alpha",
            "beta_strategy",
            "density_lambda",
            "structure_selection_metric",
            "structure_selection_beta",
            "structure_complexity_penalty",
            "threshold_metric_label",
        ]:
            values = group_frame[carry_column].dropna().tolist()
            row[carry_column] = values[0] if values else None
        row["phase"] = "fp2_part4_test"
        row["protocol"] = "m2_frozen"
        row["aggregation_level"] = "summary"
        row["random_state"] = " ".join(str(int(value)) for value in group_frame["random_state"].tolist())
        for metric_column in metric_columns:
            values = group_frame[metric_column].astype(float).to_numpy()
            row[f"{metric_column}_mean"] = float(np.mean(values))
            row[f"{metric_column}_std"] = float(np.std(values, ddof=0))
        summary_rows.append(row)
    return summary_rows


def _validate_frozen_args(args: argparse.Namespace) -> None:
    if args.validation_size <= 0.0 or args.validation_size >= 1.0:
        raise ValueError("validation_size must be between 0 and 1.")
    if args.beta_calibration_size <= 0.0 or args.beta_calibration_size >= 1.0:
        raise ValueError("beta_calibration_size must be between 0 and 1.")
    if args.beta_max_supervised_samples < 1:
        raise ValueError("beta_max_supervised_samples must be positive.")
    if args.density_k < 1:
        raise ValueError("density_k must be at least 1.")
    if args.density_reference_cap < 1:
        raise ValueError("density_reference_cap must be at least 1.")
    if args.adaptive_complexity_penalty < 0.0:
        raise ValueError("adaptive_complexity_penalty must be non-negative.")
    if not args.adaptive_random_states:
        raise ValueError("adaptive_random_states must not be empty.")
    if any(int(value) < 1 for value in args.candidate_m):
        raise ValueError("All candidate_m values must be at least 1.")


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    pd.DataFrame(rows).to_csv(path, index=False)


if __name__ == "__main__":
    main()
