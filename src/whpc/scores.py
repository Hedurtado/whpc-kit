from __future__ import annotations

import numpy as np


def inner_product_score(x: np.ndarray, v: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    v = np.asarray(v, dtype=float)
    return float(np.dot(x, v))


def cosine_score(x: np.ndarray, v: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    v = np.asarray(v, dtype=float)
    x_norm = float(np.linalg.norm(x))
    v_norm = float(np.linalg.norm(v))
    if x_norm <= 1e-15 or v_norm <= 1e-15:
        return 0.0
    return float(np.dot(x, v) / (x_norm * v_norm))


def margin_score(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if scores.ndim == 1:
        ordered = np.sort(scores)
        if ordered.size == 0:
            return np.array(0.0)
        if ordered.size == 1:
            return np.array(ordered[-1])
        return np.array(ordered[-1] - ordered[-2])

    if scores.shape[1] == 0:
        return np.zeros(scores.shape[0], dtype=float)
    if scores.shape[1] == 1:
        return scores[:, 0].copy()

    ordered = np.sort(scores, axis=1)
    return ordered[:, -1] - ordered[:, -2]


def max_score(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if scores.ndim == 1:
        return np.array(np.max(scores))
    return np.max(scores, axis=1)
