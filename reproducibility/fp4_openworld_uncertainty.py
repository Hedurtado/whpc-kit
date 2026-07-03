from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke reproducibility entry point for FP4/M4.")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--output", type=Path, default=Path("reproducibility/artifacts/fp4_smoke_summary.json"))
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
    if args.mode == "full":
        raise NotImplementedError("Full FP4 reproduction is not implemented yet in whpc-kit.")

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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
