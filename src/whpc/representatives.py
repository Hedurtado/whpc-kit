from __future__ import annotations

import numpy as np

from .weights import normalize_sample_weights, uniform_sample_weights

REPRESENTATIVE_STRATEGIES = {
    "weighted_mean",
    "weighted_medoid",
    "max_weight_sample",
}


def _normalize_rows(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    safe_norms = np.where(norms > 1e-15, norms, 1.0)
    return X / safe_norms


def build_class_representative(
    X_class: np.ndarray,
    weights: np.ndarray | None = None,
    representative_strategy: str = "weighted_mean",
    normalize: bool = True,
) -> np.ndarray:
    X_class = np.asarray(X_class, dtype=float)
    if X_class.ndim != 2:
        raise ValueError("X_class must be a 2D array.")
    if X_class.shape[0] == 0:
        raise ValueError("X_class must contain at least one sample.")
    if representative_strategy not in REPRESENTATIVE_STRATEGIES:
        raise ValueError(
            "representative_strategy must be one of "
            f"{sorted(REPRESENTATIVE_STRATEGIES)!r}; got {representative_strategy!r}."
        )

    if weights is None:
        weights = uniform_sample_weights(X_class.shape[0])
    else:
        weights = normalize_sample_weights(weights)

    if representative_strategy == "weighted_mean":
        representative = weights @ X_class
    elif representative_strategy == "weighted_medoid":
        X_normalized = _normalize_rows(X_class)
        weighted_center = weights @ X_normalized
        center_norm = float(np.linalg.norm(weighted_center))
        if center_norm <= 1e-15:
            representative = X_class[int(np.argmax(weights))].copy()
        else:
            medoid_scores = X_normalized @ (weighted_center / center_norm)
            representative = X_class[int(np.argmax(medoid_scores))].copy()
    else:
        representative = X_class[int(np.argmax(weights))].copy()

    if not normalize:
        return representative

    norm = float(np.linalg.norm(representative))
    if norm <= 1e-15:
        return np.zeros(X_class.shape[1], dtype=float)
    return representative / norm
