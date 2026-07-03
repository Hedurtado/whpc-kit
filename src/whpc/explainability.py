from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted

from .one_class_multimodal import OCMMWHPCClassifier


def explain_ocmmwhpc_decisions(
    model: OCMMWHPCClassifier,
    X: np.ndarray,
    *,
    y_true: np.ndarray | None = None,
    attack_family: Sequence[object] | None = None,
    feature_names: Sequence[str] | None = None,
    feature_groups: Sequence[str] | None = None,
    sample_ids: Sequence[object] | None = None,
    top_k_features: int = 5,
) -> pd.DataFrame:
    """Build local, model-native explanations for OC-MM-WHPC decisions."""

    check_is_fitted(model, ["normal_representatives_", "normal_representatives_normalized_"])
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if top_k_features <= 0:
        raise ValueError("top_k_features must be positive.")

    n_samples, n_features = X.shape
    names = _resolve_feature_names(feature_names, n_features)
    groups = _resolve_feature_groups(feature_groups, names)
    ids = _resolve_sample_ids(sample_ids, n_samples)
    families = _resolve_optional_sequence(attack_family, n_samples, default="")
    y = None if y_true is None else np.asarray(y_true, dtype=int)
    if y is not None and y.shape[0] != n_samples:
        raise ValueError("y_true must have one entry per sample.")

    X_model = model._prepare_X(X)
    local_scores = model.score_local_samples(X)
    aggregate_scores = model.score_samples(X)
    predicted_labels = model.predict(X)
    predicted_anomalous = predicted_labels == model.anomaly_label
    top_indices = np.argmax(local_scores, axis=1)
    top_scores = local_scores[np.arange(n_samples), top_indices]
    if local_scores.shape[1] > 1:
        sorted_scores = np.sort(local_scores, axis=1)
        runner_up_scores = sorted_scores[:, -2]
        representative_gap = sorted_scores[:, -1] - sorted_scores[:, -2]
    else:
        runner_up_scores = np.full(n_samples, math.nan, dtype=float)
        representative_gap = np.full(n_samples, math.inf, dtype=float)

    threshold = float(model.threshold_) if hasattr(model, "threshold_") else math.nan
    threshold_margin = aggregate_scores - threshold if math.isfinite(threshold) else np.full(n_samples, math.nan)

    rows: list[dict[str, Any]] = []
    for row_index in range(n_samples):
        prototype_index = int(top_indices[row_index])
        contributions = _prototype_feature_contributions(model, X_model[row_index], prototype_index)
        abs_total = float(np.sum(np.abs(contributions)))
        top_positive = _format_top_features(contributions, names, top_k_features, largest=True)
        top_negative = _format_top_features(contributions, names, top_k_features, largest=False)
        top_abs_indices = np.argsort(np.abs(contributions), kind="mergesort")[::-1][:top_k_features]
        grouped = _group_contributions(contributions, groups)
        group_names = list(grouped.keys())
        group_values = np.asarray(list(grouped.values()), dtype=float)
        top_abs_group_indices = np.argsort(np.abs(group_values), kind="mergesort")[::-1][:top_k_features]
        group_abs_total = float(np.sum(np.abs(group_values)))
        top_abs_share = _safe_ratio(float(np.sum(np.abs(contributions[top_abs_indices]))), abs_total)
        top1_abs_share = _safe_ratio(float(np.max(np.abs(contributions))) if contributions.size else 0.0, abs_total)
        row: dict[str, Any] = {
            "sample_id": ids[row_index],
            "true_label": int(y[row_index]) if y is not None else "",
            "attack_family": str(families[row_index]),
            "predicted_label": str(predicted_labels[row_index]),
            "predicted_anomalous": bool(predicted_anomalous[row_index]),
            "is_correct": bool(predicted_anomalous[row_index] == bool(y[row_index])) if y is not None else "",
            "normality_score": float(aggregate_scores[row_index]),
            "threshold": threshold,
            "signed_threshold_margin": float(threshold_margin[row_index]),
            "absolute_threshold_margin": float(abs(threshold_margin[row_index])),
            "winning_representative": prototype_index,
            "winning_representative_score": float(top_scores[row_index]),
            "runner_up_representative_score": float(runner_up_scores[row_index]),
            "representative_gap": float(representative_gap[row_index]),
            "feature_contribution_l1": abs_total,
            "top1_abs_contribution_share": top1_abs_share,
            "topk_abs_contribution_share": top_abs_share,
            "top_abs_features": _format_feature_indices(contributions, names, top_abs_indices),
            "top_positive_features": top_positive,
            "top_negative_features": top_negative,
            "group_contribution_l1": group_abs_total,
            "topk_abs_group_share": _safe_ratio(
                float(np.sum(np.abs(group_values[top_abs_group_indices]))),
                group_abs_total,
            ),
            "top_abs_feature_groups": _format_feature_indices(group_values, group_names, top_abs_group_indices),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_ocmmwhpc_representatives(
    model: OCMMWHPCClassifier,
    *,
    feature_names: Sequence[str] | None = None,
    feature_groups: Sequence[str] | None = None,
    top_k_features: int = 10,
) -> pd.DataFrame:
    """Summarize normal representatives through dominant feature weights."""

    check_is_fitted(model, ["normal_representatives_"])
    representatives = np.asarray(model.normal_representatives_, dtype=float)
    names = _resolve_feature_names(feature_names, representatives.shape[1])
    groups = _resolve_feature_groups(feature_groups, names)
    rows = []
    for representative_index, representative in enumerate(representatives):
        abs_total = float(np.sum(np.abs(representative)))
        top_abs_indices = np.argsort(np.abs(representative), kind="mergesort")[::-1][:top_k_features]
        grouped = _group_contributions(representative, groups)
        group_names = list(grouped.keys())
        group_values = np.asarray(list(grouped.values()), dtype=float)
        top_abs_group_indices = np.argsort(np.abs(group_values), kind="mergesort")[::-1][:top_k_features]
        rows.append(
            {
                "representative": int(representative_index),
                "l2_norm": float(np.linalg.norm(representative)),
                "l1_abs_sum": abs_total,
                "top1_abs_share": _safe_ratio(
                    float(np.max(np.abs(representative))) if representative.size else 0.0,
                    abs_total,
                ),
                "topk_abs_share": _safe_ratio(float(np.sum(np.abs(representative[top_abs_indices]))), abs_total),
                "top_abs_features": _format_feature_indices(representative, names, top_abs_indices),
                "top_abs_feature_groups": _format_feature_indices(group_values, group_names, top_abs_group_indices),
            }
        )
    return pd.DataFrame(rows)


def explanation_summary(explanations: pd.DataFrame) -> dict[str, float]:
    """Compute aggregate explanation diagnostics from local explanation rows."""

    required = {
        "absolute_threshold_margin",
        "representative_gap",
        "top1_abs_contribution_share",
        "topk_abs_contribution_share",
    }
    missing = required.difference(explanations.columns)
    if missing:
        raise ValueError(f"Missing explanation columns: {sorted(missing)!r}")

    summary = {
        "n_explained": float(len(explanations)),
        "mean_absolute_threshold_margin": _nanmean(explanations["absolute_threshold_margin"]),
        "median_absolute_threshold_margin": _nanmedian(explanations["absolute_threshold_margin"]),
        "mean_representative_gap": _nanmean(explanations["representative_gap"]),
        "median_representative_gap": _nanmedian(explanations["representative_gap"]),
        "mean_top1_abs_contribution_share": _nanmean(explanations["top1_abs_contribution_share"]),
        "mean_topk_abs_contribution_share": _nanmean(explanations["topk_abs_contribution_share"]),
    }
    if "topk_abs_group_share" in explanations.columns:
        summary["mean_topk_abs_group_share"] = _nanmean(explanations["topk_abs_group_share"])
    if "is_correct" in explanations.columns and explanations["is_correct"].dtype != object:
        correct = explanations["is_correct"].astype(bool)
        summary["accuracy"] = float(np.mean(correct)) if len(correct) else math.nan
        summary["mean_margin_correct"] = _nanmean(explanations.loc[correct, "absolute_threshold_margin"])
        summary["mean_margin_error"] = _nanmean(explanations.loc[~correct, "absolute_threshold_margin"])
        summary["mean_gap_correct"] = _nanmean(explanations.loc[correct, "representative_gap"])
        summary["mean_gap_error"] = _nanmean(explanations.loc[~correct, "representative_gap"])
    return summary


def _prototype_feature_contributions(
    model: OCMMWHPCClassifier,
    x_model: np.ndarray,
    prototype_index: int,
) -> np.ndarray:
    if model.score_metric == "inner_product":
        representative = np.asarray(model.normal_representatives_[prototype_index], dtype=float)
        return x_model * representative

    representative = np.asarray(model.normal_representatives_normalized_[prototype_index], dtype=float)
    x_norm = float(np.linalg.norm(x_model))
    if x_norm <= 1e-15:
        return np.zeros_like(x_model, dtype=float)
    return (x_model * representative) / x_norm


def _format_top_features(
    contributions: np.ndarray,
    feature_names: list[str],
    top_k: int,
    *,
    largest: bool,
) -> str:
    if largest:
        order = np.argsort(contributions, kind="mergesort")[::-1]
    else:
        order = np.argsort(contributions, kind="mergesort")
    selected = order[:top_k]
    return _format_feature_indices(contributions, feature_names, selected)


def _format_feature_indices(values: np.ndarray, feature_names: list[str], indices: np.ndarray) -> str:
    return ";".join(f"{feature_names[int(index)]}:{float(values[int(index)]):.6g}" for index in indices)


def _group_contributions(contributions: np.ndarray, feature_groups: list[str]) -> dict[str, float]:
    grouped: dict[str, float] = {}
    for group, value in zip(feature_groups, contributions, strict=True):
        grouped[group] = grouped.get(group, 0.0) + float(value)
    return grouped


def _resolve_feature_names(feature_names: Sequence[str] | None, n_features: int) -> list[str]:
    if feature_names is None:
        return [f"feature_{index}" for index in range(n_features)]
    names = [str(name) for name in feature_names]
    if len(names) != n_features:
        raise ValueError(f"feature_names must have {n_features} entries, got {len(names)}.")
    return names


def _resolve_feature_groups(feature_groups: Sequence[str] | None, feature_names: Sequence[str]) -> list[str]:
    names = [str(name) for name in feature_names]
    if feature_groups is None:
        return names
    groups = [str(group) for group in feature_groups]
    if len(groups) != len(names):
        raise ValueError(f"feature_groups must have {len(names)} entries, got {len(groups)}.")
    return groups


def _resolve_sample_ids(sample_ids: Sequence[object] | None, n_samples: int) -> list[object]:
    if sample_ids is None:
        return list(range(n_samples))
    ids = list(sample_ids)
    if len(ids) != n_samples:
        raise ValueError(f"sample_ids must have {n_samples} entries, got {len(ids)}.")
    return ids


def _resolve_optional_sequence(values: Sequence[object] | None, n_samples: int, *, default: object) -> list[object]:
    if values is None:
        return [default] * n_samples
    resolved = list(values)
    if len(resolved) != n_samples:
        raise ValueError(f"Optional sequence must have {n_samples} entries, got {len(resolved)}.")
    return resolved


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 1e-15:
        return math.nan
    return float(numerator / denominator)


def _nanmean(values: pd.Series) -> float:
    array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if array.size == 0 or np.all(np.isnan(array)):
        return math.nan
    return float(np.nanmean(array))


def _nanmedian(values: pd.Series) -> float:
    array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if array.size == 0 or np.all(np.isnan(array)):
        return math.nan
    return float(np.nanmedian(array))
