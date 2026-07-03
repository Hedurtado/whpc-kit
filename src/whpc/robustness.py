from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score


def add_gaussian_noise(X: np.ndarray, *, std: float, rng: np.random.Generator) -> np.ndarray:
    if std < 0.0:
        raise ValueError("std must be non-negative.")
    X = np.asarray(X, dtype=float)
    if std == 0.0:
        return X.copy()
    return X + rng.normal(loc=0.0, scale=std, size=X.shape)


def random_feature_mask(
    X: np.ndarray,
    *,
    fraction: float,
    rng: np.random.Generator,
    fill_value: float = 0.0,
) -> np.ndarray:
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be between 0 and 1.")
    X = np.asarray(X, dtype=float)
    if fraction == 0.0:
        return X.copy()
    mask = rng.random(X.shape) < fraction
    perturbed = X.copy()
    perturbed[mask] = fill_value
    return perturbed


def top_feature_mask(
    X: np.ndarray,
    top_feature_indices: Sequence[Sequence[int]],
    *,
    top_k: int,
    fill_value: float = 0.0,
) -> np.ndarray:
    if top_k < 0:
        raise ValueError("top_k must be non-negative.")
    X = np.asarray(X, dtype=float)
    if top_k == 0:
        return X.copy()
    if len(top_feature_indices) != X.shape[0]:
        raise ValueError("top_feature_indices must have one entry per sample.")
    perturbed = X.copy()
    for row_index, indices in enumerate(top_feature_indices):
        selected = [int(index) for index in list(indices)[:top_k]]
        if selected:
            perturbed[row_index, selected] = fill_value
    return perturbed


def directional_feature_shift(
    X: np.ndarray,
    directions: np.ndarray,
    *,
    epsilon: float,
) -> np.ndarray:
    """Apply a fixed-direction perturbation per sample.

    This is an adversarial-lite stressor, not a certified attack. Directions
    should be precomputed from the fitted model evidence.
    """

    if epsilon < 0.0:
        raise ValueError("epsilon must be non-negative.")
    X = np.asarray(X, dtype=float)
    directions = np.asarray(directions, dtype=float)
    if directions.shape != X.shape:
        raise ValueError("directions must have the same shape as X.")
    if epsilon == 0.0:
        return X.copy()
    return X + float(epsilon) * directions


def parse_top_feature_indices(explanations: pd.DataFrame, feature_names: Sequence[str], *, column: str) -> list[list[int]]:
    if column not in explanations.columns:
        raise ValueError(f"Missing explanation column: {column}")
    name_to_index = {str(name): index for index, name in enumerate(feature_names)}
    rows = []
    for serialized in explanations[column].astype(str):
        indices = []
        for item in serialized.split(";"):
            if not item:
                continue
            name = item.split(":", 1)[0]
            if name in name_to_index:
                indices.append(name_to_index[name])
        rows.append(indices)
    return rows


def topk_jaccard(left: Sequence[str], right: Sequence[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set
    if not union:
        return math.nan
    return float(len(left_set & right_set) / len(union))


def parse_ranked_names(serialized: Any) -> list[str]:
    names = []
    for item in str(serialized).split(";"):
        if not item:
            continue
        names.append(item.split(":", 1)[0])
    return names


def robustness_metrics(
    y_true: np.ndarray,
    clean_scores: np.ndarray,
    clean_pred_anomalous: np.ndarray,
    perturbed_scores: np.ndarray,
    perturbed_pred_anomalous: np.ndarray,
    *,
    threshold: float,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    clean_scores = np.asarray(clean_scores, dtype=float)
    clean_pred = np.asarray(clean_pred_anomalous, dtype=int)
    perturbed_scores = np.asarray(perturbed_scores, dtype=float)
    perturbed_pred = np.asarray(perturbed_pred_anomalous, dtype=int)
    if not (
        y_true.shape == clean_scores.shape == clean_pred.shape == perturbed_scores.shape == perturbed_pred.shape
    ):
        raise ValueError("All metric arrays must have the same shape.")

    clean_margin = clean_scores - threshold
    perturbed_margin = perturbed_scores - threshold
    return {
        "f1_anomalous": float(f1_score(y_true, perturbed_pred, pos_label=1, zero_division=0)),
        "anomaly_recall": float(recall_score(y_true, perturbed_pred, pos_label=1, zero_division=0)),
        "balanced_accuracy": _balanced_accuracy(y_true, perturbed_pred),
        "prediction_flip_rate": float(np.mean(clean_pred != perturbed_pred)),
        "clean_predicted_anomaly_rate": float(np.mean(clean_pred == 1)),
        "perturbed_predicted_anomaly_rate": float(np.mean(perturbed_pred == 1)),
        "mean_abs_score_shift": float(np.mean(np.abs(clean_scores - perturbed_scores))),
        "mean_abs_margin_shift": float(np.mean(np.abs(clean_margin - perturbed_margin))),
        "mean_signed_score_shift": float(np.mean(perturbed_scores - clean_scores)),
        "mean_signed_margin_shift": float(np.mean(perturbed_margin - clean_margin)),
    }


def explanation_jaccard_summary(clean_explanations: pd.DataFrame, perturbed_explanations: pd.DataFrame) -> dict[str, float]:
    if clean_explanations.shape[0] != perturbed_explanations.shape[0]:
        raise ValueError("Explanation frames must have the same number of rows.")
    feature_scores = []
    group_scores = []
    for clean_row, perturbed_row in zip(
        clean_explanations.itertuples(index=False),
        perturbed_explanations.itertuples(index=False),
        strict=True,
    ):
        feature_scores.append(
            topk_jaccard(
                parse_ranked_names(getattr(clean_row, "top_abs_features")),
                parse_ranked_names(getattr(perturbed_row, "top_abs_features")),
            )
        )
        group_scores.append(
            topk_jaccard(
                parse_ranked_names(getattr(clean_row, "top_abs_feature_groups")),
                parse_ranked_names(getattr(perturbed_row, "top_abs_feature_groups")),
            )
        )
    return {
        "mean_topk_feature_jaccard": _nanmean(feature_scores),
        "mean_topk_group_jaccard": _nanmean(group_scores),
    }


def _balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return math.nan
    return float(balanced_accuracy_score(y_true, y_pred))


def _nanmean(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    if array.size == 0 or np.all(np.isnan(array)):
        return math.nan
    return float(np.nanmean(array))
