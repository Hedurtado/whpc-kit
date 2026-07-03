from __future__ import annotations

import math

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import f1_score
from sklearn.utils.validation import check_array, check_is_fitted

from .one_class_multimodal import OCMMWHPCClassifier


class OpenWorldOCMMWHPCClassifier(BaseEstimator, ClassifierMixin):
    """Reject-capable decision layer for OC-MM-WHPC.

    The wrapped detector still produces the binary one-class decision. This
    layer only decides whether that decision is sufficiently supported for
    deployment, or should be reported as unknown.
    """

    def __init__(
        self,
        base_detector: OCMMWHPCClassifier | None = None,
        min_decision_margin: float = 0.02,
        min_representative_gap: float = 0.0,
        anomaly_support_quantile: float = 0.05,
        unknown_risk_quantile: float = 0.10,
        unknown_risk_min_conditions: int = 1,
        reject_on_alarm: bool = True,
        alarm_reject_mode: str = "anomalous",
        unknown_label: str = "unknown",
    ) -> None:
        self.base_detector = base_detector
        self.min_decision_margin = min_decision_margin
        self.min_representative_gap = min_representative_gap
        self.anomaly_support_quantile = anomaly_support_quantile
        self.unknown_risk_quantile = unknown_risk_quantile
        self.unknown_risk_min_conditions = unknown_risk_min_conditions
        self.reject_on_alarm = reject_on_alarm
        self.alarm_reject_mode = alarm_reject_mode
        self.unknown_label = unknown_label

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "OpenWorldOCMMWHPCClassifier":
        self._validate_configuration()
        detector = OCMMWHPCClassifier() if self.base_detector is None else clone(self.base_detector)
        self.detector_ = detector.fit(X, y)
        self.n_features_in_ = self.detector_.n_features_in_
        self.classes_ = np.asarray(
            [self.detector_.normal_label, self.detector_.anomaly_label, self.unknown_label],
            dtype=object,
        )
        return self

    def calibrate_threshold(self, X_val: np.ndarray, y_val: np.ndarray) -> float:
        check_is_fitted(self, ["detector_"])
        threshold = self.detector_.calibrate_threshold(X_val, y_val)
        self._calibrate_anomaly_support(X_val, y_val)
        return threshold

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["detector_"])
        return self.detector_.score_samples(X)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["detector_"])
        return self.detector_.decision_function(X)

    def decision_support(self, X: np.ndarray, alarm_mask: np.ndarray | None = None) -> dict[str, np.ndarray]:
        check_is_fitted(self, ["detector_"])
        check_is_fitted(self.detector_, ["threshold_"])
        X = check_array(X, dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but OpenWorldOCMMWHPCClassifier was fitted with {self.n_features_in_}."
            )

        scores = self.detector_.score_samples(X)
        signed_margin = scores - float(self.detector_.threshold_)
        absolute_margin = np.abs(signed_margin)
        base_labels = self.detector_.predict(X)
        base_is_anomalous = base_labels == self.detector_.anomaly_label
        representative_gap = _representative_gap(self.detector_, X)
        alarm = _validate_alarm_mask(alarm_mask, scores.shape[0])

        rejected_by_margin = absolute_margin < float(self.min_decision_margin)
        rejected_by_representative_gap = representative_gap < float(self.min_representative_gap)
        rejected_by_anomaly_support = np.zeros(scores.shape[0], dtype=bool)
        if hasattr(self, "anomaly_score_floor_"):
            rejected_by_anomaly_support = base_is_anomalous & (scores < float(self.anomaly_score_floor_))
        low_validation_anomaly_score = np.zeros(scores.shape[0], dtype=bool)
        low_validation_anomaly_gap = np.zeros(scores.shape[0], dtype=bool)
        low_validation_anomaly_margin = np.zeros(scores.shape[0], dtype=bool)
        if hasattr(self, "unknown_risk_score_floor_"):
            low_validation_anomaly_score = base_is_anomalous & (scores < float(self.unknown_risk_score_floor_))
        if hasattr(self, "unknown_risk_gap_floor_"):
            low_validation_anomaly_gap = base_is_anomalous & (representative_gap < float(self.unknown_risk_gap_floor_))
        if hasattr(self, "unknown_risk_margin_floor_"):
            low_validation_anomaly_margin = base_is_anomalous & (absolute_margin < float(self.unknown_risk_margin_floor_))
        unknown_risk_condition_count = (
            low_validation_anomaly_score.astype(int)
            + low_validation_anomaly_gap.astype(int)
            + low_validation_anomaly_margin.astype(int)
        )
        unknown_risk_mask = base_is_anomalous & (unknown_risk_condition_count >= int(self.unknown_risk_min_conditions))
        rejected_by_alarm = np.zeros(scores.shape[0], dtype=bool)
        if self.reject_on_alarm:
            if self.alarm_reject_mode == "all":
                rejected_by_alarm = alarm
            elif self.alarm_reject_mode == "anomalous":
                rejected_by_alarm = alarm & base_is_anomalous
            else:
                raise ValueError("alarm_reject_mode must be 'anomalous' or 'all'.")

        reject_mask = rejected_by_margin | rejected_by_representative_gap | rejected_by_anomaly_support | rejected_by_alarm
        return {
            "normality_score": scores,
            "signed_threshold_margin": signed_margin,
            "absolute_threshold_margin": absolute_margin,
            "representative_gap": representative_gap,
            "base_is_anomalous": base_is_anomalous,
            "alarm_mask": alarm,
            "rejected_by_margin": rejected_by_margin,
            "rejected_by_representative_gap": rejected_by_representative_gap,
            "rejected_by_anomaly_support": rejected_by_anomaly_support,
            "low_validation_anomaly_score": low_validation_anomaly_score,
            "low_validation_anomaly_gap": low_validation_anomaly_gap,
            "low_validation_anomaly_margin": low_validation_anomaly_margin,
            "unknown_risk_condition_count": unknown_risk_condition_count,
            "unknown_risk_mask": unknown_risk_mask,
            "rejected_by_alarm": rejected_by_alarm,
            "reject_mask": reject_mask,
        }

    def predict(self, X: np.ndarray, alarm_mask: np.ndarray | None = None) -> np.ndarray:
        support = self.decision_support(X, alarm_mask=alarm_mask)
        labels = self.detector_.predict(X).astype(object)
        labels[support["reject_mask"]] = self.unknown_label
        return labels

    def _validate_configuration(self) -> None:
        if self.min_decision_margin < 0.0:
            raise ValueError("min_decision_margin must be non-negative.")
        if self.min_representative_gap < 0.0:
            raise ValueError("min_representative_gap must be non-negative.")
        if not 0.0 <= self.anomaly_support_quantile <= 1.0:
            raise ValueError("anomaly_support_quantile must be between 0 and 1.")
        if not 0.0 <= self.unknown_risk_quantile <= 1.0:
            raise ValueError("unknown_risk_quantile must be between 0 and 1.")
        if self.unknown_risk_min_conditions < 1 or self.unknown_risk_min_conditions > 3:
            raise ValueError("unknown_risk_min_conditions must be between 1 and 3.")
        if self.alarm_reject_mode not in {"anomalous", "all"}:
            raise ValueError("alarm_reject_mode must be 'anomalous' or 'all'.")
        if self.unknown_label in {"", None}:
            raise ValueError("unknown_label must be a non-empty string.")

    def _calibrate_anomaly_support(self, X_val: np.ndarray, y_val: np.ndarray) -> None:
        y_encoded = self.detector_._encode_anomaly_targets(y_val)
        anomaly_mask = y_encoded == 1
        if not np.any(anomaly_mask):
            return
        scores = self.detector_.score_samples(X_val)
        anomaly_scores = scores[anomaly_mask]
        self.anomaly_score_floor_ = float(np.quantile(anomaly_scores, self.anomaly_support_quantile))
        anomaly_margins = np.abs(anomaly_scores - float(self.detector_.threshold_))
        anomaly_gaps = _representative_gap(self.detector_, X_val[anomaly_mask])
        self.unknown_risk_score_floor_ = float(np.quantile(anomaly_scores, self.unknown_risk_quantile))
        self.unknown_risk_gap_floor_ = float(np.quantile(anomaly_gaps, self.unknown_risk_quantile))
        self.unknown_risk_margin_floor_ = float(np.quantile(anomaly_margins, self.unknown_risk_quantile))


def compute_open_world_metrics(
    y_true: np.ndarray,
    predicted_labels: np.ndarray,
    unknown_mask: np.ndarray,
    *,
    normal_label: str = "normal",
    anomaly_label: str = "anomalous",
    unknown_label: str = "unknown",
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    labels = np.asarray(predicted_labels, dtype=object)
    unknown_mask = np.asarray(unknown_mask, dtype=bool)
    if y_true.ndim != 1 or labels.ndim != 1 or unknown_mask.ndim != 1:
        raise ValueError("y_true, predicted_labels, and unknown_mask must be 1D arrays.")
    if not (y_true.shape[0] == labels.shape[0] == unknown_mask.shape[0]):
        raise ValueError("All open-world metric inputs must have the same number of samples.")

    accepted = labels != unknown_label
    rejected = ~accepted
    known_mask = ~unknown_mask
    predicted_anomalous = labels == anomaly_label
    binary_correct = ((labels == normal_label) & (y_true == 0)) | ((labels == anomaly_label) & (y_true == 1))

    accepted_known = accepted & known_mask
    accepted_all = accepted
    return {
        "coverage": _safe_mean(accepted),
        "reject_rate": _safe_mean(rejected),
        "unknown_rejection_recall": _conditional_mean(rejected, unknown_mask),
        "harmful_accept_rate": _conditional_mean(accepted, unknown_mask),
        "known_reject_rate": _conditional_mean(rejected, known_mask),
        "known_acceptance_rate": _conditional_mean(accepted, known_mask),
        "accepted_known_accuracy": _conditional_mean(binary_correct, accepted_known),
        "accepted_binary_accuracy": _conditional_mean(binary_correct, accepted_all),
        "accepted_f1_anomalous": _accepted_f1(y_true, predicted_anomalous, accepted_all),
        "n_samples": float(y_true.shape[0]),
        "n_unknown": float(np.sum(unknown_mask)),
        "n_rejected": float(np.sum(rejected)),
        "n_harmful_accept": float(np.sum(accepted & unknown_mask)),
    }


def _representative_gap(detector: OCMMWHPCClassifier, X: np.ndarray) -> np.ndarray:
    local_scores = detector.score_local_samples(X)
    if local_scores.shape[1] < 2:
        return np.full(local_scores.shape[0], np.inf, dtype=float)
    sorted_scores = np.sort(local_scores, axis=1)
    return sorted_scores[:, -1] - sorted_scores[:, -2]


def _validate_alarm_mask(alarm_mask: np.ndarray | None, n_samples: int) -> np.ndarray:
    if alarm_mask is None:
        return np.zeros(n_samples, dtype=bool)
    alarm = np.asarray(alarm_mask, dtype=bool)
    if alarm.ndim != 1:
        raise ValueError("alarm_mask must be a 1D array when provided.")
    if alarm.shape[0] != n_samples:
        raise ValueError("alarm_mask must have one entry per sample.")
    return alarm


def _safe_mean(mask: np.ndarray) -> float:
    if mask.size == 0:
        return math.nan
    return float(np.mean(mask))


def _conditional_mean(values: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return math.nan
    return float(np.mean(values[mask]))


def _accepted_f1(y_true: np.ndarray, predicted_anomalous: np.ndarray, accepted: np.ndarray) -> float:
    if not np.any(accepted):
        return math.nan
    return float(f1_score(y_true[accepted], predicted_anomalous[accepted].astype(int), pos_label=1, zero_division=0))
