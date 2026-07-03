from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_array, check_is_fitted

from .classifier import WHPCClassifier
from .scores import max_score
from .thresholds import fit_threshold


class WHPCOpenSetClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        base_classifier: WHPCClassifier | None = None,
        sample_weight_strategy: str = "uniform",
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
        threshold: float | None = None,
        threshold_strategy: str = "quantile",
        quantile: float = 0.05,
        anomaly_label: str = "anomaly",
        random_state: int | None = None,
    ) -> None:
        self.base_classifier = base_classifier
        self.sample_weight_strategy = sample_weight_strategy
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
        self.threshold = threshold
        self.threshold_strategy = threshold_strategy
        self.quantile = quantile
        self.anomaly_label = anomaly_label
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "WHPCOpenSetClassifier":
        classifier = self.base_classifier
        if classifier is None:
            classifier = WHPCClassifier(
                sample_weight_strategy=self.sample_weight_strategy,
                sample_weight_by_class=self.sample_weight_by_class,
                sample_weight_temperature=self.sample_weight_temperature,
                sample_weight_mix_alpha=self.sample_weight_mix_alpha,
                sample_weight_boundary_gamma=self.sample_weight_boundary_gamma,
                sample_weight_boundary_scale=self.sample_weight_boundary_scale,
                sample_weight_separation_lambda=self.sample_weight_separation_lambda,
                sample_weight_separation_iterations=self.sample_weight_separation_iterations,
                feature_weight=self.feature_weight,
                score_metric=self.score_metric,
                normalize=self.normalize,
                tie_break=self.tie_break,
                random_state=self.random_state,
            )
        else:
            classifier = clone(classifier)

        self.classifier_ = classifier.fit(X, y)
        train_scores = self.classifier_.predict_scores(X)
        train_max_scores = max_score(train_scores)
        self.threshold_ = (
            float(self.threshold)
            if self.threshold is not None
            else fit_threshold(train_max_scores, strategy=self.threshold_strategy, quantile=self.quantile)
        )
        self.classes_ = np.asarray(list(self.classifier_.classes_) + [self.anomaly_label], dtype=object)
        self.n_features_in_ = self.classifier_.n_features_in_
        return self

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["classifier_", "threshold_"])
        return self.classifier_.predict_scores(X)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        scores = self.predict_scores(X)
        return max_score(scores) - self.threshold_

    def is_anomaly(self, X: np.ndarray) -> np.ndarray:
        scores = self.predict_scores(X)
        return max_score(scores) < self.threshold_

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["classifier_", "threshold_"])
        X = check_array(X, dtype=float)
        labels = self.classifier_.predict(X).astype(object)
        labels[self.is_anomaly(X)] = self.anomaly_label
        return labels
