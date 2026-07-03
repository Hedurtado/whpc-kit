from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DriftSignals:
    n_samples: int
    mean_normality_score: float
    reference_score_mean: float
    score_drop: float
    predicted_anomaly_rate: float
    borderline_rate: float
    accepted_update_rate: float


@dataclass(frozen=True)
class DriftAlarmConfig:
    anomaly_rate_threshold: float = 0.40
    score_drop_threshold: float = 0.05
    borderline_rate_threshold: float = 0.10
    accepted_update_floor: float = 0.001
    min_active_conditions: int = 2


@dataclass(frozen=True)
class DriftAlarm:
    is_alarm: bool
    active_conditions: int
    anomaly_rate_condition: bool
    score_drop_condition: bool
    borderline_rate_condition: bool
    accepted_update_condition: bool


def compute_block_drift_signals(
    normality_scores: np.ndarray,
    y_pred_anomalous: np.ndarray,
    accepted_update_mask: np.ndarray,
    *,
    threshold: float,
    reference_score_mean: float,
    borderline_margin: float = 0.02,
) -> DriftSignals:
    scores = _as_1d_float_array(normality_scores, name="normality_scores")
    predictions = _as_1d_int_array(y_pred_anomalous, name="y_pred_anomalous")
    accepted = _as_1d_bool_array(accepted_update_mask, name="accepted_update_mask")
    _validate_same_length(scores, predictions, accepted)
    if scores.size == 0:
        raise ValueError("drift signals require at least one sample.")
    if borderline_margin < 0.0:
        raise ValueError("borderline_margin must be non-negative.")

    mean_score = float(np.mean(scores))
    threshold = float(threshold)
    reference_score_mean = float(reference_score_mean)
    return DriftSignals(
        n_samples=int(scores.size),
        mean_normality_score=mean_score,
        reference_score_mean=reference_score_mean,
        score_drop=max(0.0, reference_score_mean - mean_score),
        predicted_anomaly_rate=float(np.mean(predictions == 1)),
        borderline_rate=float(np.mean(np.abs(scores - threshold) <= borderline_margin)),
        accepted_update_rate=float(np.mean(accepted)),
    )


def evaluate_drift_alarm(signals: DriftSignals, config: DriftAlarmConfig | None = None) -> DriftAlarm:
    if config is None:
        config = DriftAlarmConfig()
    if config.min_active_conditions <= 0:
        raise ValueError("min_active_conditions must be positive.")

    anomaly_rate_condition = signals.predicted_anomaly_rate >= config.anomaly_rate_threshold
    score_drop_condition = signals.score_drop >= config.score_drop_threshold
    borderline_rate_condition = signals.borderline_rate >= config.borderline_rate_threshold
    accepted_update_condition = signals.accepted_update_rate <= config.accepted_update_floor
    active_conditions = sum(
        (
            anomaly_rate_condition,
            score_drop_condition,
            borderline_rate_condition,
            accepted_update_condition,
        )
    )
    return DriftAlarm(
        is_alarm=active_conditions >= config.min_active_conditions,
        active_conditions=int(active_conditions),
        anomaly_rate_condition=bool(anomaly_rate_condition),
        score_drop_condition=bool(score_drop_condition),
        borderline_rate_condition=bool(borderline_rate_condition),
        accepted_update_condition=bool(accepted_update_condition),
    )


def _as_1d_float_array(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _as_1d_int_array(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=int)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")
    return array


def _as_1d_bool_array(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=bool)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")
    return array


def _validate_same_length(*arrays: np.ndarray) -> None:
    lengths = {array.shape[0] for array in arrays}
    if len(lengths) != 1:
        raise ValueError("normality_scores, y_pred_anomalous, and accepted_update_mask must have the same length.")
