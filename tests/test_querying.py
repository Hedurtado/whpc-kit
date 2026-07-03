from __future__ import annotations

import numpy as np

from whpc import (
    accepted_risk_candidate_mask,
    apply_oracle_feedback,
    compute_accepted_risk_query_scores,
    compute_feedback_metrics,
    query_candidate_mask,
    select_dual_top_k_queries,
    select_random_queries,
    select_top_k_queries,
    split_queried_feedback,
)


def test_select_top_k_queries_respects_candidate_mask_and_budget():
    scores = np.array([0.1, 0.9, 0.8, 0.7])
    candidates = np.array([True, False, True, True])
    selected = select_top_k_queries(scores, candidates, budget=2)
    assert selected.tolist() == [False, False, True, True]


def test_query_candidate_mask_handles_missing_unknown_risk_reject_key():
    support = {
        "base_is_anomalous": np.array([False, True, True]),
        "unknown_risk_mask": np.array([False, False, False]),
        "rejected_by_margin": np.array([False, False, False]),
        "rejected_by_anomaly_support": np.array([False, False, False]),
    }
    labels = np.array(["normal", "anomalous", "unknown"], dtype=object)
    selected = query_candidate_mask(support, labels, alarm_active=True)
    assert selected.tolist() == [False, True, True]


def test_accepted_risk_candidate_mask_excludes_rejected_samples():
    support = {
        "base_is_anomalous": np.array([True, True, True, False]),
        "unknown_risk_mask": np.array([True, True, False, False]),
        "unknown_risk_condition_count": np.array([1, 1, 0, 0]),
    }
    labels = np.array(["anomalous", "unknown", "anomalous", "normal"], dtype=object)
    selected = accepted_risk_candidate_mask(support, labels, alarm_active=False)
    assert selected.tolist() == [True, False, False, False]


def test_feedback_metrics_count_harmful_accept_reduction_from_queries():
    y_true = np.array([0, 1, 1, 1])
    unknown = np.array([False, True, True, False])
    labels = np.array(["normal", "anomalous", "unknown", "anomalous"], dtype=object)
    queried = np.array([False, True, False, False])
    corrected = apply_oracle_feedback(y_true, labels, queried)
    metrics = compute_feedback_metrics(y_true, labels, corrected, unknown, queried)
    assert np.isclose(metrics["query_rate"], 0.25)
    assert np.isclose(metrics["harmful_accept_rate_before_feedback"], 0.5)
    assert np.isclose(metrics["harmful_accept_rate_after_feedback"], 0.0)


def test_random_dual_and_split_helpers_smoke():
    candidates = np.array([True, True, True, False, True])
    first = select_random_queries(candidates, budget=2, random_state=7)
    second = select_random_queries(candidates, budget=2, random_state=7)
    assert first.tolist() == second.tolist()

    selected = select_dual_top_k_queries(
        np.array([0.9, 0.8, 0.1, 0.0]),
        np.array([True, True, False, False]),
        np.array([0.1, 0.2, 0.9, 0.8]),
        np.array([False, False, True, True]),
        budget=3,
        primary_budget_fraction=0.5,
    )
    assert selected.tolist() == [True, True, True, False]

    X = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    y = np.array([0, 1, 1])
    queried = np.array([True, False, True])
    normal_rows, anomaly_rows = split_queried_feedback(X, y, queried)
    assert normal_rows.tolist() == [[1.0, 0.0]]
    assert anomaly_rows.tolist() == [[-1.0, 0.0]]


def test_accepted_risk_query_scores_penalize_rejected_samples():
    support = {
        "absolute_threshold_margin": np.array([0.1, 0.1]),
        "representative_gap": np.array([0.2, 0.2]),
        "unknown_risk_condition_count": np.array([1, 1]),
        "unknown_risk_mask": np.array([True, True]),
        "base_is_anomalous": np.array([True, True]),
    }
    labels = np.array(["anomalous", "unknown"], dtype=object)
    scores = compute_accepted_risk_query_scores(support, labels, alarm_active=False)
    assert np.isfinite(scores[0])
    assert scores[1] == -np.inf
