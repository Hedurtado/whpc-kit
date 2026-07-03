from __future__ import annotations

import numpy as np


def fit_threshold(
    scores: np.ndarray,
    strategy: str = "quantile",
    quantile: float = 0.05,
) -> float:
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        raise ValueError("scores must not be empty.")

    if strategy == "quantile":
        if not 0.0 <= quantile <= 1.0:
            raise ValueError("quantile must be in [0, 1].")
        return float(np.quantile(scores, quantile))
    if strategy == "fixed":
        raise ValueError("Use a numeric threshold directly when strategy='fixed'.")

    raise ValueError(f"Unsupported threshold strategy: {strategy}")


def apply_threshold(max_scores: np.ndarray, threshold: float) -> np.ndarray:
    max_scores = np.asarray(max_scores, dtype=float)
    return max_scores >= float(threshold)


def fit_residual_threshold(
    residuals: np.ndarray,
    strategy: str = "quantile",
    quantile: float = 0.95,
) -> float:
    residuals = np.asarray(residuals, dtype=float)
    if residuals.size == 0:
        raise ValueError("residuals must not be empty.")

    if strategy != "quantile":
        raise ValueError(f"Unsupported residual threshold strategy: {strategy}")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1].")
    return float(np.quantile(residuals, quantile))
