from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from .multimodal_aggregation import aggregate_prototype_scores
from .multimodal_partition import partition_class_samples
from .representatives import REPRESENTATIVE_STRATEGIES, build_class_representative
from .weights import (
    center_boundary_sample_weights,
    intra_class_core_sample_weights,
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


class MMWHPCClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        n_representatives_per_class: int | dict[object, int] = 2,
        representative_partition_strategy: str = "kmeans",
        representative_strategy: str = "weighted_mean",
        prototype_aggregation: str = "max",
        prototype_softmax_alpha: float = 10.0,
        sample_weight_strategy: str = "uniform",
        sample_weight_by_class_and_mode: dict[object, list[np.ndarray]] | None = None,
        sample_weight_temperature: float = 1.0,
        sample_weight_mix_alpha: float = 1.0,
        sample_weight_boundary_gamma: float = 0.5,
        sample_weight_boundary_scale: float = 0.1,
        feature_weight: np.ndarray | None = None,
        score_metric: str = "cosine",
        normalize: bool = False,
        tie_break: str = "prior",
        random_state: int | None = None,
    ) -> None:
        self.n_representatives_per_class = n_representatives_per_class
        self.representative_partition_strategy = representative_partition_strategy
        self.representative_strategy = representative_strategy
        self.prototype_aggregation = prototype_aggregation
        self.prototype_softmax_alpha = prototype_softmax_alpha
        self.sample_weight_strategy = sample_weight_strategy
        self.sample_weight_by_class_and_mode = sample_weight_by_class_and_mode
        self.sample_weight_temperature = sample_weight_temperature
        self.sample_weight_mix_alpha = sample_weight_mix_alpha
        self.sample_weight_boundary_gamma = sample_weight_boundary_gamma
        self.sample_weight_boundary_scale = sample_weight_boundary_scale
        self.feature_weight = feature_weight
        self.score_metric = score_metric
        self.normalize = normalize
        self.tie_break = tie_break
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MMWHPCClassifier":
        X, y = check_X_y(X, y, dtype=float)
        self._validate_configuration()

        X_model = self._prepare_X(X)
        self.classes_, inverse = np.unique(y, return_inverse=True)
        self.n_features_in_ = X_model.shape[1]
        self.class_priors_ = {}
        self.n_representatives_per_class_ = {}
        self.class_partition_labels_ = {}
        self.class_sample_weights_ = {}
        self.class_representatives_ = {}
        self.class_representatives_normalized_ = {}

        class_counts = np.bincount(inverse)
        n_samples = X_model.shape[0]

        for class_index, class_label in enumerate(self.classes_):
            X_class = X_model[inverse == class_index]
            n_representatives = self._resolve_n_representatives(class_label, X_class.shape[0])
            partition_labels, partition_indices = partition_class_samples(
                X_class,
                n_partitions=n_representatives,
                strategy=self.representative_partition_strategy,
                random_state=self.random_state,
            )

            representatives = []
            representatives_normalized = []
            sample_weights = []

            for mode_index, local_indices in enumerate(partition_indices):
                X_mode = X_class[local_indices]
                X_other = np.delete(X_model, np.flatnonzero(inverse == class_index)[local_indices], axis=0)
                weights = self._resolve_sample_weights(X_mode, X_other, class_label, mode_index)
                representative = build_class_representative(
                    X_mode,
                    weights=weights,
                    representative_strategy=self.representative_strategy,
                    normalize=False,
                )
                representatives.append(representative)
                representatives_normalized.append(_normalize_vector(representative))
                sample_weights.append(weights)

            self.class_priors_[class_label] = class_counts[class_index] / n_samples
            self.n_representatives_per_class_[class_label] = n_representatives
            self.class_partition_labels_[class_label] = partition_labels
            self.class_sample_weights_[class_label] = sample_weights
            self.class_representatives_[class_label] = np.vstack(representatives)
            self.class_representatives_normalized_[class_label] = np.vstack(representatives_normalized)

        return self

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        return self._compute_scores(X)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return self.predict_scores(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self._compute_scores(X)
        predictions = []

        for row_scores in scores:
            best_score = np.max(row_scores)
            candidates = np.flatnonzero(np.isclose(row_scores, best_score, atol=1e-12, rtol=1e-9))

            if candidates.size > 1 and self.tie_break == "prior":
                priors = np.array([self.class_priors_[self.classes_[idx]] for idx in candidates])
                best_prior = np.max(priors)
                candidates = candidates[np.isclose(priors, best_prior, atol=1e-12, rtol=1e-9)]

            predictions.append(self.classes_[int(candidates[0])])

        return np.asarray(predictions, dtype=self.classes_.dtype)

    def _compute_local_scores(self, X: np.ndarray) -> dict[object, np.ndarray]:
        check_is_fitted(self, ["classes_", "class_representatives_", "class_representatives_normalized_"])
        X = check_array(X, dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but MMWHPCClassifier was fitted with {self.n_features_in_}."
            )

        X_model = self._prepare_X(X)
        local_scores_by_class: dict[object, np.ndarray] = {}
        x_norms = np.linalg.norm(X_model, axis=1)

        for class_label in self.classes_:
            representatives = self.class_representatives_[class_label]
            representatives_normalized = self.class_representatives_normalized_[class_label]
            if self.score_metric == "inner_product":
                local_scores = X_model @ representatives.T
            else:
                dot_products = X_model @ representatives_normalized.T
                local_scores = np.divide(
                    dot_products,
                    x_norms.reshape(-1, 1),
                    out=np.zeros_like(dot_products),
                    where=x_norms.reshape(-1, 1) > 1e-15,
                )
            local_scores_by_class[class_label] = local_scores

        return local_scores_by_class

    def _compute_scores(self, X: np.ndarray) -> np.ndarray:
        local_scores_by_class = self._compute_local_scores(X)
        ordered_scores = {class_label: local_scores_by_class[class_label] for class_label in self.classes_}
        return aggregate_prototype_scores(
            ordered_scores,
            aggregation=self.prototype_aggregation,
            softmax_alpha=self.prototype_softmax_alpha,
        )

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

    def _resolve_n_representatives(self, class_label: object, n_samples_in_class: int) -> int:
        if isinstance(self.n_representatives_per_class, int):
            n_representatives = self.n_representatives_per_class
        elif isinstance(self.n_representatives_per_class, dict):
            if class_label not in self.n_representatives_per_class:
                raise ValueError(f"Missing representative count for class label {class_label!r}.")
            n_representatives = self.n_representatives_per_class[class_label]
        else:
            raise ValueError("n_representatives_per_class must be an integer or a dict keyed by class label.")

        if not isinstance(n_representatives, int):
            raise ValueError("Each representative count must be an integer.")
        if n_representatives < 1:
            raise ValueError("n_representatives_per_class must be at least 1.")
        if n_representatives > n_samples_in_class:
            raise ValueError("n_representatives_per_class cannot exceed samples in a class.")
        return n_representatives

    def _resolve_sample_weights(
        self,
        X_mode: np.ndarray,
        X_other: np.ndarray,
        class_label: object,
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
        if self.sample_weight_strategy == "provided":
            raise ValueError(
                "sample_weight_strategy='provided' is not implemented for MMWHPCClassifier Version 1."
            )

        raise ValueError(
            "sample_weight_strategy must be 'uniform', 'intra_class_core', 'mixed_intra_class_core', "
            "'optimized_margin', 'center_boundary', or 'provided'."
        )

    def _validate_configuration(self) -> None:
        if self.score_metric not in {"cosine", "inner_product"}:
            raise ValueError("score_metric must be 'cosine' or 'inner_product'.")
        if self.tie_break not in {"prior", "first"}:
            raise ValueError("tie_break must be 'prior' or 'first'.")
        if self.representative_partition_strategy != "kmeans":
            raise ValueError("representative_partition_strategy must be 'kmeans'.")
        if self.representative_strategy not in REPRESENTATIVE_STRATEGIES:
            raise ValueError(
                "representative_strategy must be one of "
                f"{sorted(REPRESENTATIVE_STRATEGIES)!r}."
            )
        if self.prototype_aggregation not in {"max", "softmax"}:
            raise ValueError("prototype_aggregation must be 'max' or 'softmax'.")
        if self.prototype_softmax_alpha <= 0.0:
            raise ValueError("prototype_softmax_alpha must be positive.")
        if self.sample_weight_strategy not in {
            "uniform",
            "intra_class_core",
            "mixed_intra_class_core",
            "optimized_margin",
            "center_boundary",
            "provided",
        }:
            raise ValueError(
                "sample_weight_strategy must be 'uniform', 'intra_class_core', 'mixed_intra_class_core', "
                "'optimized_margin', 'center_boundary', or 'provided'."
            )
        if self.sample_weight_temperature <= 0.0:
            raise ValueError("sample_weight_temperature must be positive.")
        if self.sample_weight_mix_alpha < 0.0 or self.sample_weight_mix_alpha > 1.0:
            raise ValueError("sample_weight_mix_alpha must be between 0 and 1.")
        if self.sample_weight_boundary_gamma < 0.0:
            raise ValueError("sample_weight_boundary_gamma must be non-negative.")
        if self.sample_weight_boundary_scale <= 0.0:
            raise ValueError("sample_weight_boundary_scale must be positive.")
