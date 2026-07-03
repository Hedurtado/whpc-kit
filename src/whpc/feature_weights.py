from __future__ import annotations

import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.utils.validation import check_X_y, check_array


FEATURE_WEIGHT_STRATEGIES = (
    "identity",
    "variance",
    "mean_activation",
    "correlation",
    "mutual_information",
)


def identity_feature_weights(n_features: int) -> np.ndarray:
    if n_features <= 0:
        raise ValueError("n_features must be positive.")
    return np.ones(n_features, dtype=float)


def normalize_feature_weights(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1:
        raise ValueError("weights must be a 1D array.")
    if weights.size == 0:
        raise ValueError("weights must not be empty.")
    if not np.all(np.isfinite(weights)):
        raise ValueError("weights must be finite.")
    if np.any(weights < 0.0):
        raise ValueError("weights must be non-negative.")

    total = float(np.sum(weights))
    if total <= 1e-15:
        return np.full(weights.size, 1.0 / weights.size, dtype=float)
    return weights / total


def variance_feature_weights(X: np.ndarray) -> np.ndarray:
    X = check_array(X, dtype=float)
    scores = np.var(X, axis=0)
    return normalize_feature_weights(scores)


def mean_activation_feature_weights(X: np.ndarray) -> np.ndarray:
    X = check_array(X, dtype=float)
    scores = np.mean(np.abs(X), axis=0)
    return normalize_feature_weights(scores)


def correlation_feature_weights(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    X, y = check_X_y(X, y, dtype=float)
    y_centered = y - np.mean(y)
    y_norm = float(np.linalg.norm(y_centered))
    if y_norm <= 1e-15:
        return np.full(X.shape[1], 1.0 / X.shape[1], dtype=float)

    X_centered = X - np.mean(X, axis=0, keepdims=True)
    x_norms = np.linalg.norm(X_centered, axis=0)
    numerators = np.abs(X_centered.T @ y_centered)
    denominators = x_norms * y_norm
    scores = np.divide(
        numerators,
        denominators,
        out=np.zeros_like(numerators, dtype=float),
        where=denominators > 1e-15,
    )
    return normalize_feature_weights(scores)


def mutual_information_feature_weights(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int | None = 42,
) -> np.ndarray:
    X, y = check_X_y(X, y, dtype=float)
    scores = mutual_info_classif(X, y, discrete_features=False, random_state=random_state)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    scores = np.maximum(scores, 0.0)
    return normalize_feature_weights(scores)


def compute_feature_weights(
    strategy: str,
    X: np.ndarray,
    y: np.ndarray | None = None,
    random_state: int | None = 42,
) -> np.ndarray:
    X_checked = check_array(X, dtype=float)

    if strategy == "identity":
        return identity_feature_weights(X_checked.shape[1])
    if strategy == "variance":
        return variance_feature_weights(X_checked)
    if strategy == "mean_activation":
        return mean_activation_feature_weights(X_checked)
    if strategy == "correlation":
        if y is None:
            raise ValueError("y is required for correlation feature weights.")
        return correlation_feature_weights(X_checked, y)
    if strategy == "mutual_information":
        if y is None:
            raise ValueError("y is required for mutual_information feature weights.")
        return mutual_information_feature_weights(X_checked, y, random_state=random_state)

    raise ValueError(f"Unknown feature-weight strategy: {strategy!r}.")
