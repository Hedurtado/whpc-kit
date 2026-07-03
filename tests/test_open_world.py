from __future__ import annotations

import numpy as np

from whpc import OCMMWHPCClassifier, OpenWorldOCMMWHPCClassifier, compute_open_world_metrics


def make_fitted_open_world_detector(**kwargs) -> OpenWorldOCMMWHPCClassifier:
    base = OCMMWHPCClassifier(
        n_normal_representatives=1,
        representative_partition_strategy="kmeans",
        threshold=0.60,
        normalize=True,
        random_state=0,
    )
    detector = OpenWorldOCMMWHPCClassifier(base_detector=base, **kwargs)
    detector.fit(np.array([[1.0, 0.0], [0.95, 0.05], [0.90, 0.10]]))
    return detector


def test_open_world_rejects_borderline_threshold_decisions():
    detector = make_fitted_open_world_detector(min_decision_margin=0.05)
    labels = detector.predict(np.array([[1.0, 0.0], [0.60, 0.80], [-1.0, 0.0]]))
    assert labels.tolist() == ["normal", "unknown", "anomalous"]


def test_open_world_alarm_rejects_anomalous_samples_only_by_default():
    detector = make_fitted_open_world_detector(min_decision_margin=0.0, reject_on_alarm=True)
    labels = detector.predict(np.array([[1.0, 0.0], [-1.0, 0.0]]), alarm_mask=np.array([True, True]))
    assert labels.tolist() == ["normal", "unknown"]


def test_open_world_metrics_separate_unknown_rejection_and_harmful_accepts():
    metrics = compute_open_world_metrics(
        y_true=np.array([0, 1, 1, 1]),
        predicted_labels=np.array(["normal", "anomalous", "unknown", "anomalous"], dtype=object),
        unknown_mask=np.array([False, False, True, True]),
    )
    assert np.isclose(metrics["coverage"], 0.75)
    assert np.isclose(metrics["unknown_rejection_recall"], 0.5)
    assert np.isclose(metrics["harmful_accept_rate"], 0.5)
