from __future__ import annotations

import numpy as np
import pytest

from whpc.drift_monitoring import DriftAlarmConfig, compute_block_drift_signals, evaluate_drift_alarm


def test_compute_block_drift_signals_reports_expected_rates():
    signals = compute_block_drift_signals(
        normality_scores=np.array([0.80, 0.72, 0.50, 0.49, 0.20]),
        y_pred_anomalous=np.array([0, 0, 1, 1, 1]),
        accepted_update_mask=np.array([True, True, False, False, False]),
        threshold=0.50,
        reference_score_mean=0.70,
        borderline_margin=0.02,
    )
    assert signals.n_samples == 5
    assert np.isclose(signals.mean_normality_score, 0.542)
    assert np.isclose(signals.score_drop, 0.158)
    assert np.isclose(signals.predicted_anomaly_rate, 0.6)
    assert np.isclose(signals.borderline_rate, 0.4)
    assert np.isclose(signals.accepted_update_rate, 0.4)


def test_evaluate_drift_alarm_activates_when_min_conditions_match():
    signals = compute_block_drift_signals(
        normality_scores=np.array([0.42, 0.41, 0.40, 0.39]),
        y_pred_anomalous=np.array([1, 1, 1, 1]),
        accepted_update_mask=np.array([False, False, False, False]),
        threshold=0.40,
        reference_score_mean=0.60,
        borderline_margin=0.02,
    )
    alarm = evaluate_drift_alarm(
        signals,
        DriftAlarmConfig(
            anomaly_rate_threshold=0.75,
            score_drop_threshold=0.10,
            borderline_rate_threshold=0.50,
            accepted_update_floor=0.01,
            min_active_conditions=2,
        ),
    )
    assert alarm.is_alarm is True
    assert alarm.active_conditions == 4


def test_compute_block_drift_signals_validates_inputs():
    with pytest.raises(ValueError, match="same length"):
        compute_block_drift_signals(
            normality_scores=np.array([0.9, 0.8]),
            y_pred_anomalous=np.array([0]),
            accepted_update_mask=np.array([True, True]),
            threshold=0.5,
            reference_score_mean=0.7,
        )
