from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import f1_score
from sklearn.utils.validation import check_array, check_is_fitted

from .multimodal_aggregation import aggregate_prototype_scores
from .multimodal_partition import PARTITION_STRATEGIES, partition_class_samples
from .representatives import REPRESENTATIVE_STRATEGIES, build_class_representative
from .weights import (
    center_boundary_sample_weights,
    intra_class_core_sample_weights,
    local_typicality_margin_sample_weights,
    mix_with_uniform_sample_weights,
    optimized_margin_sample_weights,
    uniform_sample_weights,
)


def _normalize_rows(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    safe_norms = np.where(norms > 1e-15, norms, 1.0)
    return X / safe_norms


def _normalize_vector(x: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(x))
    if norm <= 1e-15:
        return np.zeros_like(x, dtype=float)
    return x / norm


class OCMMWHPCClassifier(BaseEstimator):
    def __init__(
        self,
        n_normal_representatives: int = 4,
        representative_partition_strategy: str = "kmeans",
        representative_strategy: str = "weighted_mean",
        prototype_aggregation: str = "max",
        prototype_softmax_alpha: float = 10.0,
        sample_weight_strategy: str = "uniform",
        sample_weight_temperature: float = 1.0,
        sample_weight_mix_alpha: float = 1.0,
        sample_weight_boundary_gamma: float = 0.5,
        sample_weight_boundary_scale: float = 0.1,
        sample_weight_typicality_k: int = 15,
        sample_weight_typicality_reference_cap: int = 2048,
        sample_weight_typicality_gamma: float = 1.0,
        sample_weight_typicality_separation_lambda: float = 1.0,
        feature_weight: np.ndarray | None = None,
        normalize: bool = True,
        score_metric: str = "cosine",
        threshold: float | None = None,
        threshold_selection_metric: str = "f1_anomalous",
        threshold_metric_beta: float = 1.0,
        threshold_min_normal_acceptance: float | None = None,
        normal_label: str = "normal",
        anomaly_label: str = "anomalous",
        random_state: int | None = None,
    ) -> None:
        self.n_normal_representatives = n_normal_representatives
        self.representative_partition_strategy = representative_partition_strategy
        self.representative_strategy = representative_strategy
        self.prototype_aggregation = prototype_aggregation
        self.prototype_softmax_alpha = prototype_softmax_alpha
        self.sample_weight_strategy = sample_weight_strategy
        self.sample_weight_temperature = sample_weight_temperature
        self.sample_weight_mix_alpha = sample_weight_mix_alpha
        self.sample_weight_boundary_gamma = sample_weight_boundary_gamma
        self.sample_weight_boundary_scale = sample_weight_boundary_scale
        self.sample_weight_typicality_k = sample_weight_typicality_k
        self.sample_weight_typicality_reference_cap = sample_weight_typicality_reference_cap
        self.sample_weight_typicality_gamma = sample_weight_typicality_gamma
        self.sample_weight_typicality_separation_lambda = sample_weight_typicality_separation_lambda
        self.feature_weight = feature_weight
        self.normalize = normalize
        self.score_metric = score_metric
        self.threshold = threshold
        self.threshold_selection_metric = threshold_selection_metric
        self.threshold_metric_beta = threshold_metric_beta
        self.threshold_min_normal_acceptance = threshold_min_normal_acceptance
        self.normal_label = normal_label
        self.anomaly_label = anomaly_label
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "OCMMWHPCClassifier":
        X = check_array(X, dtype=float)
        self._validate_configuration()
        self._validate_optional_y(y, X.shape[0])

        X_model = self._prepare_X(X)
        self.n_features_in_ = X_model.shape[1]

        partition_labels, partition_indices = partition_class_samples(
            X_model,
            n_partitions=self.n_normal_representatives,
            strategy=self.representative_partition_strategy,
            random_state=self.random_state,
        )

        representatives = []
        representatives_normalized = []
        sample_weights = []

        for mode_index, local_indices in enumerate(partition_indices):
            X_mode = X_model[local_indices]
            X_other = np.delete(X_model, local_indices, axis=0)
            weights = self._resolve_sample_weights(X_mode, X_other, mode_index)
            representative = build_class_representative(
                X_mode,
                weights=weights,
                representative_strategy=self.representative_strategy,
                normalize=False,
            )
            representatives.append(representative)
            representatives_normalized.append(_normalize_vector(representative))
            sample_weights.append(weights)

        self.normal_partition_labels_ = partition_labels
        self.normal_sample_weights_ = sample_weights
        self.normal_representatives_ = np.vstack(representatives)
        self.normal_representatives_normalized_ = np.vstack(representatives_normalized)
        self.n_normal_representatives_ = self.normal_representatives_.shape[0]

        if self.threshold is not None:
            self.threshold_ = float(self.threshold)

        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        return self._compute_scores(X)

    def score_local_samples(self, X: np.ndarray) -> np.ndarray:
        return self._compute_local_scores(X)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        scores = self.score_samples(X)
        if hasattr(self, "threshold_"):
            return scores - self.threshold_
        return scores

    def calibrate_threshold(self, X_val: np.ndarray, y_val: np.ndarray) -> float:
        check_is_fitted(self, ["normal_representatives_", "normal_representatives_normalized_"])
        y_true = self._encode_anomaly_targets(y_val)
        scores = self.score_samples(X_val)

        threshold, best_metric_value = self._fit_threshold(scores, y_true)
        self.threshold_ = threshold
        self.calibration_summary_ = {
            "threshold": threshold,
            "metric": self.threshold_selection_metric,
            "best_metric_value": best_metric_value,
            "n_validation_samples": int(scores.shape[0]),
            "n_candidate_thresholds": int(np.unique(scores).shape[0] + 1),
        }
        return threshold

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["normal_representatives_", "normal_representatives_normalized_", "threshold_"])
        scores = self.score_samples(X)
        labels = np.full(scores.shape[0], self.normal_label, dtype=object)
        labels[scores < self.threshold_] = self.anomaly_label
        return labels

    def predict_labels(self, X: np.ndarray) -> np.ndarray:
        return self.predict(X)

    def _compute_local_scores(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["normal_representatives_", "normal_representatives_normalized_"])
        X = check_array(X, dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but OCMMWHPCClassifier was fitted with {self.n_features_in_}."
            )

        X_model = self._prepare_X(X)
        x_norms = np.linalg.norm(X_model, axis=1)

        if self.score_metric == "inner_product":
            return X_model @ self.normal_representatives_.T

        dot_products = X_model @ self.normal_representatives_normalized_.T
        return np.divide(
            dot_products,
            x_norms.reshape(-1, 1),
            out=np.zeros_like(dot_products),
            where=x_norms.reshape(-1, 1) > 1e-15,
        )

    def _compute_scores(self, X: np.ndarray) -> np.ndarray:
        local_scores = self._compute_local_scores(X)
        aggregated = aggregate_prototype_scores(
            {"normal": local_scores},
            aggregation=self.prototype_aggregation,
            softmax_alpha=self.prototype_softmax_alpha,
        )
        return aggregated[:, 0]

    def _prepare_X(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        X_weighted = self._apply_feature_weights(X)
        if self.normalize:
            return _normalize_rows(X_weighted)
        return X_weighted

    def _apply_feature_weights(self, X: np.ndarray) -> np.ndarray:
        if self.feature_weight is None:
            return X.copy()

        feature_weight = np.asarray(self.feature_weight, dtype=float)
        if feature_weight.ndim != 1:
            raise ValueError("feature_weight must be a 1D array.")
        if feature_weight.shape[0] != X.shape[1]:
            raise ValueError("feature_weight must have one entry per feature.")
        if np.any(feature_weight < 0.0):
            raise ValueError("feature_weight must be non-negative.")
        return X * feature_weight

    def _resolve_sample_weights(
        self,
        X_mode: np.ndarray,
        X_other: np.ndarray,
        mode_index: int,
    ) -> np.ndarray:
        if self.sample_weight_strategy == "uniform":
            return uniform_sample_weights(X_mode.shape[0])
        if self.sample_weight_strategy == "intra_class_core":
            return intra_class_core_sample_weights(
                X_mode,
                temperature=self.sample_weight_temperature,
            )
        if self.sample_weight_strategy == "mixed_intra_class_core":
            core_weights = intra_class_core_sample_weights(
                X_mode,
                temperature=self.sample_weight_temperature,
            )
            return mix_with_uniform_sample_weights(core_weights, alpha=self.sample_weight_mix_alpha)
        if self.sample_weight_strategy == "optimized_margin":
            return optimized_margin_sample_weights(
                X_mode,
                X_other,
                temperature=self.sample_weight_temperature,
            )
        if self.sample_weight_strategy == "center_boundary":
            return center_boundary_sample_weights(
                X_mode,
                X_other,
                temperature=self.sample_weight_temperature,
                boundary_gamma=self.sample_weight_boundary_gamma,
                boundary_scale=self.sample_weight_boundary_scale,
            )
        if self.sample_weight_strategy == "local_typicality_margin":
            return local_typicality_margin_sample_weights(
                X_mode,
                X_other,
                temperature=self.sample_weight_temperature,
                typicality_k=self.sample_weight_typicality_k,
                typicality_reference_cap=self.sample_weight_typicality_reference_cap,
                compactness_gamma=self.sample_weight_typicality_gamma,
                separation_lambda=self.sample_weight_typicality_separation_lambda,
                random_state=self.random_state,
            )

        raise ValueError(
            "sample_weight_strategy must be 'uniform', 'intra_class_core', 'mixed_intra_class_core', "
            "'optimized_margin', 'center_boundary', or 'local_typicality_margin'; "
            f"got {self.sample_weight_strategy!r} for mode {mode_index}."
        )

    def _fit_threshold(self, scores: np.ndarray, y_true: np.ndarray) -> tuple[float, float]:
        scores = np.asarray(scores, dtype=float)
        y_true = np.asarray(y_true, dtype=int)
        if scores.ndim != 1:
            raise ValueError("scores must be a 1D array.")
        if y_true.shape[0] != scores.shape[0]:
            raise ValueError("scores and y_true must have the same number of samples.")

        order = np.argsort(scores, kind="mergesort")
        sorted_scores = scores[order]
        sorted_labels = y_true[order]
        total_positives = int(np.sum(sorted_labels))
        total_negatives = int(sorted_labels.shape[0] - total_positives)

        best_threshold = float(sorted_scores[0])
        best_metric_value = -np.inf
        true_positives = 0
        false_positives = 0
        idx = 0

        while idx < sorted_scores.shape[0]:
            threshold = float(sorted_scores[idx])
            current_metric_value = self._threshold_objective_value(
                true_positives=true_positives,
                false_positives=false_positives,
                total_positives=total_positives,
                total_negatives=total_negatives,
            )
            if current_metric_value > best_metric_value or (
                np.isclose(current_metric_value, best_metric_value, atol=1e-12, rtol=1e-9) and threshold < best_threshold
            ):
                best_threshold = threshold
                best_metric_value = current_metric_value

            group_end = idx
            while group_end < sorted_scores.shape[0] and sorted_scores[group_end] == sorted_scores[idx]:
                group_end += 1

            group_labels = sorted_labels[idx:group_end]
            true_positives += int(np.sum(group_labels))
            false_positives += int(group_labels.shape[0] - np.sum(group_labels))
            idx = group_end

        final_threshold = float(np.nextafter(sorted_scores[-1], np.inf))
        final_metric_value = self._threshold_objective_value(
            true_positives=true_positives,
            false_positives=false_positives,
            total_positives=total_positives,
            total_negatives=total_negatives,
        )
        if final_metric_value > best_metric_value or (
            np.isclose(final_metric_value, best_metric_value, atol=1e-12, rtol=1e-9) and final_threshold < best_threshold
        ):
            best_threshold = final_threshold
            best_metric_value = final_metric_value

        return best_threshold, float(best_metric_value)

    def _fit_f1_anomalous_threshold(self, scores: np.ndarray, y_true: np.ndarray) -> tuple[float, float]:
        if self.threshold_selection_metric != "f1_anomalous":
            raise ValueError("threshold_selection_metric must be 'f1_anomalous'.")
        return self._fit_threshold(scores, y_true)

    def _threshold_objective_value(
        self,
        true_positives: int,
        false_positives: int,
        total_positives: int,
        total_negatives: int,
    ) -> float:
        false_negatives = total_positives - true_positives
        true_negatives = total_negatives - false_positives
        normal_acceptance_rate = 0.0 if total_negatives == 0 else true_negatives / total_negatives

        if self.threshold_selection_metric == "f1_anomalous":
            return _fbeta_from_prefix_counts(
                true_positives=true_positives,
                false_positives=false_positives,
                false_negatives=false_negatives,
                beta=1.0,
            )
        if self.threshold_selection_metric == "balanced_accuracy":
            tpr = 0.0 if total_positives == 0 else true_positives / total_positives
            tnr = 0.0 if total_negatives == 0 else true_negatives / total_negatives
            return float((tpr + tnr) / 2.0)
        if self.threshold_selection_metric == "gmean":
            tpr = 0.0 if total_positives == 0 else true_positives / total_positives
            tnr = 0.0 if total_negatives == 0 else true_negatives / total_negatives
            return float(np.sqrt(tpr * tnr))
        if self.threshold_selection_metric == "f_beta_anomalous":
            return _fbeta_from_prefix_counts(
                true_positives=true_positives,
                false_positives=false_positives,
                false_negatives=false_negatives,
                beta=self.threshold_metric_beta,
            )
        if self.threshold_selection_metric == "constrained_f1_anomalous":
            min_acceptance = 0.0 if self.threshold_min_normal_acceptance is None else self.threshold_min_normal_acceptance
            if normal_acceptance_rate < min_acceptance:
                return -np.inf
            return _fbeta_from_prefix_counts(
                true_positives=true_positives,
                false_positives=false_positives,
                false_negatives=false_negatives,
                beta=1.0,
            )

        raise ValueError(
            "threshold_selection_metric must be one of "
            "'f1_anomalous', 'balanced_accuracy', 'gmean', 'f_beta_anomalous', or 'constrained_f1_anomalous'."
        )

    def _encode_anomaly_targets(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError("y_val must be a 1D array.")
        if y.size == 0:
            raise ValueError("y_val must not be empty.")

        unique_values = np.unique(y)
        if unique_values.shape[0] < 2:
            raise ValueError("y_val must contain both normal and anomalous labels for threshold calibration.")

        if self.normal_label in unique_values or self.anomaly_label in unique_values:
            if self.normal_label not in unique_values or self.anomaly_label not in unique_values:
                raise ValueError("y_val must contain both configured normal_label and anomaly_label.")
            return (y == self.anomaly_label).astype(int)

        if np.issubdtype(y.dtype, np.bool_) or np.issubdtype(y.dtype, np.number):
            normalized = y.astype(float)
            if set(np.unique(normalized)).issubset({0.0, 1.0}):
                return normalized.astype(int)
            if set(np.unique(normalized)).issubset({-1.0, 1.0}):
                return (normalized == 1.0).astype(int)

        raise ValueError(
            "Unsupported y_val labels for threshold calibration. Use configured string labels or binary numeric labels."
        )

    def _validate_optional_y(self, y: np.ndarray | None, n_samples: int) -> None:
        if y is None:
            return

        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError("y must be a 1D array when provided.")
        if y.shape[0] != n_samples:
            raise ValueError("X and y must contain the same number of samples.")
        if np.unique(y).shape[0] > 1:
            raise ValueError("OCMMWHPCClassifier.fit accepts only normal samples; y must contain a single label.")

    def _validate_configuration(self) -> None:
        if not isinstance(self.n_normal_representatives, int):
            raise ValueError("n_normal_representatives must be an integer.")
        if self.n_normal_representatives < 1:
            raise ValueError("n_normal_representatives must be at least 1.")
        if self.representative_partition_strategy not in PARTITION_STRATEGIES:
            raise ValueError(
                "representative_partition_strategy must be one of "
                f"{sorted(PARTITION_STRATEGIES)!r}."
            )
        if self.representative_strategy not in REPRESENTATIVE_STRATEGIES:
            raise ValueError(
                "representative_strategy must be one of "
                f"{sorted(REPRESENTATIVE_STRATEGIES)!r}."
            )
        if self.score_metric not in {"cosine", "inner_product"}:
            raise ValueError("score_metric must be 'cosine' or 'inner_product'.")
        if self.threshold_selection_metric not in {
            "f1_anomalous",
            "balanced_accuracy",
            "gmean",
            "f_beta_anomalous",
            "constrained_f1_anomalous",
        }:
            raise ValueError(
                "threshold_selection_metric must be one of "
                "'f1_anomalous', 'balanced_accuracy', 'gmean', 'f_beta_anomalous', or 'constrained_f1_anomalous'."
            )
        if self.threshold_metric_beta <= 0.0:
            raise ValueError("threshold_metric_beta must be positive.")
        if self.threshold_min_normal_acceptance is not None and not 0.0 <= self.threshold_min_normal_acceptance <= 1.0:
            raise ValueError("threshold_min_normal_acceptance must be between 0 and 1 when provided.")
        if self.prototype_aggregation not in {"max", "softmax"}:
            raise ValueError("prototype_aggregation must be 'max' or 'softmax'.")
        if self.prototype_softmax_alpha <= 0.0:
            raise ValueError("prototype_softmax_alpha must be positive.")
        if self.sample_weight_temperature <= 0.0:
            raise ValueError("sample_weight_temperature must be positive.")
        if not 0.0 <= self.sample_weight_mix_alpha <= 1.0:
            raise ValueError("sample_weight_mix_alpha must be between 0 and 1.")
        if self.sample_weight_boundary_gamma < 0.0:
            raise ValueError("sample_weight_boundary_gamma must be non-negative.")
        if self.sample_weight_boundary_scale <= 0.0:
            raise ValueError("sample_weight_boundary_scale must be positive.")
        if self.sample_weight_typicality_k < 1:
            raise ValueError("sample_weight_typicality_k must be at least 1.")
        if self.sample_weight_typicality_reference_cap < 1:
            raise ValueError("sample_weight_typicality_reference_cap must be at least 1.")
        if self.sample_weight_typicality_gamma < 0.0:
            raise ValueError("sample_weight_typicality_gamma must be non-negative.")
        if self.sample_weight_typicality_separation_lambda < 0.0:
            raise ValueError("sample_weight_typicality_separation_lambda must be non-negative.")
        if self.threshold is not None and not np.isfinite(self.threshold):
            raise ValueError("threshold must be finite when provided.")


def _f1_from_prefix_counts(
    true_positives: int,
    false_positives: int,
    total_positives: int,
) -> float:
    false_negatives = total_positives - true_positives
    return _fbeta_from_prefix_counts(
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        beta=1.0,
    )


def _fbeta_from_prefix_counts(
    true_positives: int,
    false_positives: int,
    false_negatives: int,
    beta: float,
) -> float:
    beta_squared = beta * beta
    denominator = ((1.0 + beta_squared) * true_positives) + false_positives + (beta_squared * false_negatives)
    if denominator == 0.0:
        return 0.0
    return float(((1.0 + beta_squared) * true_positives) / denominator)
