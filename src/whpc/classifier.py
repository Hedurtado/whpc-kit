from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from .representatives import REPRESENTATIVE_STRATEGIES, build_class_representative
from .weights import (
    angular_separation_sample_weights_by_class,
    center_boundary_sample_weights,
    intra_class_core_sample_weights,
    mix_with_uniform_sample_weights,
    normalize_sample_weights,
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


class WHPCClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        sample_weight_strategy: str = "uniform",
        representative_strategy: str = "weighted_mean",
        sample_weight_by_class: dict[object, np.ndarray] | None = None,
        sample_weight_temperature: float = 1.0,
        sample_weight_mix_alpha: float = 1.0,
        sample_weight_boundary_gamma: float = 0.5,
        sample_weight_boundary_scale: float = 0.1,
        sample_weight_separation_lambda: float = 0.5,
        sample_weight_separation_iterations: int = 3,
        feature_weight: np.ndarray | None = None,
        score_metric: str = "cosine",
        normalize: bool = False,
        tie_break: str = "prior",
        random_state: int | None = None,
    ) -> None:
        self.sample_weight_strategy = sample_weight_strategy
        self.representative_strategy = representative_strategy
        self.sample_weight_by_class = sample_weight_by_class
        self.sample_weight_temperature = sample_weight_temperature
        self.sample_weight_mix_alpha = sample_weight_mix_alpha
        self.sample_weight_boundary_gamma = sample_weight_boundary_gamma
        self.sample_weight_boundary_scale = sample_weight_boundary_scale
        self.sample_weight_separation_lambda = sample_weight_separation_lambda
        self.sample_weight_separation_iterations = sample_weight_separation_iterations
        self.feature_weight = feature_weight
        self.score_metric = score_metric
        self.normalize = normalize
        self.tie_break = tie_break
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "WHPCClassifier":
        X, y = check_X_y(X, y, dtype=float)
        self._validate_configuration()

        X_model = self._prepare_X(X)
        self.classes_, inverse = np.unique(y, return_inverse=True)
        self.n_features_in_ = X_model.shape[1]
        self.class_priors_ = {}
        self.class_sample_weights_ = {}
        self.class_representatives_ = {}
        self.class_representatives_normalized_ = {}

        class_counts = np.bincount(inverse)
        n_samples = X_model.shape[0]
        X_by_class = [X_model[inverse == class_index] for class_index in range(self.classes_.shape[0])]

        if self.sample_weight_strategy == "angular_separation":
            weights_by_class = angular_separation_sample_weights_by_class(
                X_by_class,
                temperature=self.sample_weight_temperature,
                boundary_gamma=self.sample_weight_boundary_gamma,
                boundary_scale=self.sample_weight_boundary_scale,
                separation_lambda=self.sample_weight_separation_lambda,
                n_iter=self.sample_weight_separation_iterations,
            )
        else:
            weights_by_class = []

        for class_index, class_label in enumerate(self.classes_):
            X_class = X_by_class[class_index]
            X_other = X_model[inverse != class_index]
            if self.sample_weight_strategy == "angular_separation":
                weights = weights_by_class[class_index]
            else:
                weights = self._resolve_sample_weights(X_class, X_other, class_label)
            representative = build_class_representative(
                X_class,
                weights=weights,
                representative_strategy=self.representative_strategy,
                normalize=False,
            )

            self.class_priors_[class_label] = class_counts[class_index] / n_samples
            self.class_sample_weights_[class_label] = weights
            self.class_representatives_[class_label] = representative
            self.class_representatives_normalized_[class_label] = _normalize_vector(representative)

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

    def _compute_scores(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["classes_", "class_representatives_", "class_representatives_normalized_"])
        X = check_array(X, dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but WHPCClassifier was fitted with {self.n_features_in_}."
            )

        X_model = self._prepare_X(X)
        scores = np.zeros((X_model.shape[0], self.classes_.shape[0]), dtype=float)

        for class_index, class_label in enumerate(self.classes_):
            representative = self.class_representatives_[class_label]
            representative_normalized = self.class_representatives_normalized_[class_label]

            if self.score_metric == "inner_product":
                scores[:, class_index] = X_model @ representative
            else:
                dot_products = X_model @ representative_normalized
                x_norms = np.linalg.norm(X_model, axis=1)
                denominator = x_norms
                scores[:, class_index] = np.divide(
                    dot_products,
                    denominator,
                    out=np.zeros_like(dot_products),
                    where=denominator > 1e-15,
                )

        return scores

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
        X_class: np.ndarray,
        X_other: np.ndarray,
        class_label: object,
    ) -> np.ndarray:
        if self.sample_weight_strategy == "uniform":
            return uniform_sample_weights(X_class.shape[0])
        if self.sample_weight_strategy == "intra_class_core":
            return intra_class_core_sample_weights(
                X_class,
                temperature=self.sample_weight_temperature,
            )
        if self.sample_weight_strategy == "mixed_intra_class_core":
            core_weights = intra_class_core_sample_weights(
                X_class,
                temperature=self.sample_weight_temperature,
            )
            return mix_with_uniform_sample_weights(core_weights, alpha=self.sample_weight_mix_alpha)
        if self.sample_weight_strategy == "optimized_margin":
            return optimized_margin_sample_weights(
                X_class,
                X_other,
                temperature=self.sample_weight_temperature,
            )
        if self.sample_weight_strategy == "center_boundary":
            return center_boundary_sample_weights(
                X_class,
                X_other,
                temperature=self.sample_weight_temperature,
                boundary_gamma=self.sample_weight_boundary_gamma,
                boundary_scale=self.sample_weight_boundary_scale,
            )
        if self.sample_weight_strategy == "provided":
            if self.sample_weight_by_class is None:
                raise ValueError("sample_weight_by_class must be provided when sample_weight_strategy='provided'.")
            if class_label not in self.sample_weight_by_class:
                raise ValueError(f"Missing sample weights for class {class_label!r}.")

            weights = np.asarray(self.sample_weight_by_class[class_label], dtype=float)
            if weights.shape != (X_class.shape[0],):
                raise ValueError(
                    f"Sample weights for class {class_label!r} must have shape {(X_class.shape[0],)}, "
                    f"got {weights.shape}."
                )
            return normalize_sample_weights(weights)

        raise ValueError(
            "sample_weight_strategy must be 'uniform', 'intra_class_core', 'mixed_intra_class_core', "
            "'optimized_margin', 'center_boundary', 'angular_separation', or 'provided'."
        )

    def _validate_configuration(self) -> None:
        if self.score_metric not in {"cosine", "inner_product"}:
            raise ValueError("score_metric must be 'cosine' or 'inner_product'.")
        if self.tie_break not in {"prior", "first"}:
            raise ValueError("tie_break must be 'prior' or 'first'.")
        if self.representative_strategy not in REPRESENTATIVE_STRATEGIES:
            raise ValueError(
                "representative_strategy must be one of "
                f"{sorted(REPRESENTATIVE_STRATEGIES)!r}."
            )
        if self.sample_weight_strategy not in {
            "uniform",
            "intra_class_core",
            "mixed_intra_class_core",
            "optimized_margin",
            "center_boundary",
            "angular_separation",
            "provided",
        }:
            raise ValueError(
                "sample_weight_strategy must be 'uniform', 'intra_class_core', 'mixed_intra_class_core', "
                "'optimized_margin', 'center_boundary', 'angular_separation', or 'provided'."
            )
        if self.sample_weight_temperature <= 0.0:
            raise ValueError("sample_weight_temperature must be positive.")
        if self.sample_weight_mix_alpha < 0.0 or self.sample_weight_mix_alpha > 1.0:
            raise ValueError("sample_weight_mix_alpha must be between 0 and 1.")
        if self.sample_weight_boundary_gamma < 0.0:
            raise ValueError("sample_weight_boundary_gamma must be non-negative.")
        if self.sample_weight_boundary_scale <= 0.0:
            raise ValueError("sample_weight_boundary_scale must be positive.")
        if self.sample_weight_separation_lambda < 0.0:
            raise ValueError("sample_weight_separation_lambda must be non-negative.")
        if self.sample_weight_separation_iterations < 1:
            raise ValueError("sample_weight_separation_iterations must be at least 1.")
