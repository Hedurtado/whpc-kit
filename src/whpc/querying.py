from __future__ import annotations

import math

import numpy as np


def compute_query_scores(
    support: dict[str, np.ndarray],
    predicted_labels: np.ndarray,
    *,
    alarm_active: bool,
    unknown_label: str = "unknown",
) -> np.ndarray:
    """Rank samples for feedback using available open-world support signals."""

    labels = np.asarray(predicted_labels, dtype=object)
    absolute_margin = np.asarray(support["absolute_threshold_margin"], dtype=float)
    representative_gap = np.asarray(support["representative_gap"], dtype=float)
    risk_count = np.asarray(support["unknown_risk_condition_count"], dtype=float)
    base_is_anomalous = np.asarray(support["base_is_anomalous"], dtype=bool)

    margin_uncertainty = 1.0 / (1.0 + np.maximum(absolute_margin, 0.0))
    finite_gap = np.isfinite(representative_gap)
    gap_uncertainty = np.zeros_like(representative_gap, dtype=float)
    gap_uncertainty[finite_gap] = 1.0 / (1.0 + np.maximum(representative_gap[finite_gap], 0.0))

    rejected = labels == unknown_label
    alarm_bonus = bool(alarm_active) & base_is_anomalous
    return (
        (2.0 * rejected.astype(float))
        + (risk_count / 3.0)
        + margin_uncertainty
        + gap_uncertainty
        + (0.5 * alarm_bonus.astype(float))
    )


def compute_accepted_risk_query_scores(
    support: dict[str, np.ndarray],
    predicted_labels: np.ndarray,
    *,
    alarm_active: bool,
    unknown_label: str = "unknown",
) -> np.ndarray:
    """Rank accepted samples that carry open-world risk evidence."""

    labels = np.asarray(predicted_labels, dtype=object)
    absolute_margin = np.asarray(support["absolute_threshold_margin"], dtype=float)
    representative_gap = np.asarray(support["representative_gap"], dtype=float)
    risk_count = np.asarray(support["unknown_risk_condition_count"], dtype=float)
    unknown_risk = np.asarray(support["unknown_risk_mask"], dtype=bool)
    base_is_anomalous = np.asarray(support["base_is_anomalous"], dtype=bool)

    margin_uncertainty = 1.0 / (1.0 + np.maximum(absolute_margin, 0.0))
    finite_gap = np.isfinite(representative_gap)
    gap_uncertainty = np.zeros_like(representative_gap, dtype=float)
    gap_uncertainty[finite_gap] = 1.0 / (1.0 + np.maximum(representative_gap[finite_gap], 0.0))

    accepted = labels != unknown_label
    alarm_bonus = bool(alarm_active) & base_is_anomalous
    scores = (
        (2.0 * unknown_risk.astype(float))
        + risk_count
        + margin_uncertainty
        + gap_uncertainty
        + (0.5 * alarm_bonus.astype(float))
    )
    scores[~accepted] = -math.inf
    return scores


def query_candidate_mask(
    support: dict[str, np.ndarray],
    predicted_labels: np.ndarray,
    *,
    alarm_active: bool,
    unknown_label: str = "unknown",
) -> np.ndarray:
    labels = np.asarray(predicted_labels, dtype=object)
    base_is_anomalous = np.asarray(support["base_is_anomalous"], dtype=bool)
    rejected_by_unknown_risk = np.asarray(
        support.get("rejected_by_unknown_risk", np.zeros(labels.shape[0], dtype=bool)),
        dtype=bool,
    )
    return (
        (labels == unknown_label)
        | np.asarray(support["unknown_risk_mask"], dtype=bool)
        | rejected_by_unknown_risk
        | np.asarray(support["rejected_by_margin"], dtype=bool)
        | np.asarray(support["rejected_by_anomaly_support"], dtype=bool)
        | (bool(alarm_active) & base_is_anomalous)
    )


def accepted_risk_candidate_mask(
    support: dict[str, np.ndarray],
    predicted_labels: np.ndarray,
    *,
    alarm_active: bool,
    unknown_label: str = "unknown",
) -> np.ndarray:
    labels = np.asarray(predicted_labels, dtype=object)
    accepted = labels != unknown_label
    base_is_anomalous = np.asarray(support["base_is_anomalous"], dtype=bool)
    risk_count = np.asarray(support["unknown_risk_condition_count"], dtype=int)
    return accepted & (
        np.asarray(support["unknown_risk_mask"], dtype=bool)
        | (risk_count > 0)
        | (bool(alarm_active) & base_is_anomalous)
    )


def select_top_k_queries(scores: np.ndarray, candidate_mask: np.ndarray, budget: int) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    candidates = np.asarray(candidate_mask, dtype=bool)
    if scores.ndim != 1 or candidates.ndim != 1 or scores.shape[0] != candidates.shape[0]:
        raise ValueError("scores and candidate_mask must be 1D arrays with the same length.")
    if budget < 0:
        raise ValueError("budget must be non-negative.")
    selected = np.zeros(scores.shape[0], dtype=bool)
    candidate_indices = np.flatnonzero(candidates)
    if budget == 0 or candidate_indices.size == 0:
        return selected
    ordered = candidate_indices[np.argsort(scores[candidate_indices], kind="mergesort")[::-1]]
    selected[ordered[:budget]] = True
    return selected


def select_dual_top_k_queries(
    primary_scores: np.ndarray,
    primary_candidate_mask: np.ndarray,
    secondary_scores: np.ndarray,
    secondary_candidate_mask: np.ndarray,
    budget: int,
    *,
    primary_budget_fraction: float = 0.5,
) -> np.ndarray:
    primary_scores = np.asarray(primary_scores, dtype=float)
    primary_candidates = np.asarray(primary_candidate_mask, dtype=bool)
    secondary_scores = np.asarray(secondary_scores, dtype=float)
    secondary_candidates = np.asarray(secondary_candidate_mask, dtype=bool)
    if budget < 0:
        raise ValueError("budget must be non-negative.")
    if not 0.0 <= primary_budget_fraction <= 1.0:
        raise ValueError("primary_budget_fraction must be between 0 and 1.")
    if primary_scores.ndim != 1 or secondary_scores.ndim != 1:
        raise ValueError("scores must be 1D arrays.")
    if primary_candidates.ndim != 1 or secondary_candidates.ndim != 1:
        raise ValueError("candidate masks must be 1D arrays.")
    if not (
        primary_scores.shape[0]
        == secondary_scores.shape[0]
        == primary_candidates.shape[0]
        == secondary_candidates.shape[0]
    ):
        raise ValueError("All score and candidate arrays must have the same length.")

    selected = np.zeros(primary_scores.shape[0], dtype=bool)
    if budget == 0:
        return selected

    primary_budget = int(math.ceil(budget * primary_budget_fraction))
    primary_selected = select_top_k_queries(primary_scores, primary_candidates, primary_budget)
    selected |= primary_selected

    remaining = budget - int(np.sum(selected))
    if remaining > 0:
        secondary_selected = select_top_k_queries(secondary_scores, secondary_candidates & ~selected, remaining)
        selected |= secondary_selected

    remaining = budget - int(np.sum(selected))
    if remaining > 0:
        primary_fill = select_top_k_queries(primary_scores, primary_candidates & ~selected, remaining)
        selected |= primary_fill

    return selected


def select_random_queries(candidate_mask: np.ndarray, budget: int, random_state: int | np.random.Generator) -> np.ndarray:
    candidates = np.asarray(candidate_mask, dtype=bool)
    if candidates.ndim != 1:
        raise ValueError("candidate_mask must be a 1D array.")
    if budget < 0:
        raise ValueError("budget must be non-negative.")
    selected = np.zeros(candidates.shape[0], dtype=bool)
    candidate_indices = np.flatnonzero(candidates)
    if budget == 0 or candidate_indices.size == 0:
        return selected
    rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
    chosen = rng.choice(candidate_indices, size=min(budget, candidate_indices.size), replace=False)
    selected[chosen] = True
    return selected


def apply_oracle_feedback(
    y_true: np.ndarray,
    predicted_labels: np.ndarray,
    queried_mask: np.ndarray,
    *,
    normal_label: str = "normal",
    anomaly_label: str = "anomalous",
) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=int)
    labels = np.asarray(predicted_labels, dtype=object).copy()
    queried = np.asarray(queried_mask, dtype=bool)
    if y_true.ndim != 1 or labels.ndim != 1 or queried.ndim != 1:
        raise ValueError("y_true, predicted_labels, and queried_mask must be 1D arrays.")
    if not (y_true.shape[0] == labels.shape[0] == queried.shape[0]):
        raise ValueError("All feedback inputs must have the same number of samples.")
    labels[queried & (y_true == 0)] = normal_label
    labels[queried & (y_true == 1)] = anomaly_label
    return labels


def split_queried_feedback(
    X: np.ndarray,
    y_true: np.ndarray,
    queried_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    queried = np.asarray(queried_mask, dtype=bool)
    if X.ndim != 2 or y_true.ndim != 1 or queried.ndim != 1:
        raise ValueError("X must be 2D, and y_true and queried_mask must be 1D.")
    if X.shape[0] != y_true.shape[0] or X.shape[0] != queried.shape[0]:
        raise ValueError("X, y_true, and queried_mask must have the same number of rows.")
    queried_X = X[queried]
    queried_y = y_true[queried]
    return queried_X[queried_y == 0], queried_X[queried_y == 1]


def append_with_cap(
    base: np.ndarray,
    rows: np.ndarray,
    *,
    cap: int,
    random_state: int,
) -> np.ndarray:
    base = np.asarray(base, dtype=float)
    rows = np.asarray(rows, dtype=float)
    if base.ndim != 2 or rows.ndim != 2:
        raise ValueError("base and rows must be 2D arrays.")
    if base.shape[1] != rows.shape[1]:
        raise ValueError("base and rows must have the same number of features.")
    if cap <= 0:
        raise ValueError("cap must be positive.")
    if rows.shape[0] == 0:
        return base.copy()
    updated = np.vstack([base, rows])
    if updated.shape[0] <= cap:
        return updated
    rng = np.random.default_rng(random_state)
    keep_indices = rng.choice(updated.shape[0], size=cap, replace=False)
    return updated[np.sort(keep_indices)]


def compute_feedback_metrics(
    y_true: np.ndarray,
    predicted_labels: np.ndarray,
    corrected_labels: np.ndarray,
    unknown_mask: np.ndarray,
    queried_mask: np.ndarray,
    *,
    normal_label: str = "normal",
    anomaly_label: str = "anomalous",
    unknown_label: str = "unknown",
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    before = np.asarray(predicted_labels, dtype=object)
    after = np.asarray(corrected_labels, dtype=object)
    unknown = np.asarray(unknown_mask, dtype=bool)
    queried = np.asarray(queried_mask, dtype=bool)
    if not (y_true.shape[0] == before.shape[0] == after.shape[0] == unknown.shape[0] == queried.shape[0]):
        raise ValueError("All feedback metric inputs must have the same number of samples.")

    known = ~unknown
    accepted_before = before != unknown_label
    accepted_after = after != unknown_label
    rejected_before = ~accepted_before
    rejected_after = ~accepted_after
    harmful_before = accepted_before & unknown
    harmful_after = accepted_after & unknown & ~queried
    binary_correct_after = ((after == normal_label) & (y_true == 0)) | ((after == anomaly_label) & (y_true == 1))
    return {
        "query_rate": _safe_mean(queried),
        "n_queried": float(np.sum(queried)),
        "queried_unknown": float(np.sum(queried & unknown)),
        "queried_known": float(np.sum(queried & known)),
        "queried_rejected": float(np.sum(queried & rejected_before)),
        "queried_harmful_accept": float(np.sum(queried & harmful_before)),
        "query_precision_unknown": _conditional_mean(unknown, queried),
        "query_precision_rejected": _conditional_mean(rejected_before, queried),
        "query_precision_harmful_accept": _conditional_mean(harmful_before, queried),
        "unknown_query_recall": _conditional_mean(queried, unknown),
        "rejected_query_recall": _conditional_mean(queried, rejected_before),
        "harmful_accept_query_recall": _conditional_mean(queried, harmful_before),
        "known_query_rate": _conditional_mean(queried, known),
        "reject_rate_before_feedback": _safe_mean(rejected_before),
        "reject_rate_after_feedback": _safe_mean(rejected_after),
        "reject_rate_reduction": _safe_mean(rejected_before) - _safe_mean(rejected_after),
        "harmful_accept_rate_before_feedback": _conditional_mean(harmful_before, unknown),
        "harmful_accept_rate_after_feedback": _conditional_mean(harmful_after, unknown),
        "harmful_accept_reduction": _rate_delta(harmful_before, harmful_after, unknown),
        "value_per_query": _value_per_query(harmful_before, harmful_after, queried),
        "coverage_after_feedback": _safe_mean(accepted_after),
        "coverage_gain_after_feedback": _safe_mean(accepted_after) - _safe_mean(accepted_before),
        "accepted_accuracy_after_feedback": _conditional_mean(binary_correct_after, accepted_after),
    }


def _safe_mean(mask: np.ndarray) -> float:
    if mask.size == 0:
        return math.nan
    return float(np.mean(mask))


def _conditional_mean(values: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return math.nan
    return float(np.mean(values[mask]))


def _rate_delta(before: np.ndarray, after: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return math.nan
    return float(np.mean(before[mask]) - np.mean(after[mask]))


def _value_per_query(before: np.ndarray, after: np.ndarray, queried: np.ndarray) -> float:
    n_queried = int(np.sum(queried))
    if n_queried == 0:
        return math.nan
    return float((np.sum(before) - np.sum(after)) / n_queried)
