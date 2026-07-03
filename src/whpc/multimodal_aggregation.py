from __future__ import annotations

import numpy as np


def aggregate_prototype_scores(
    scores_by_class: dict[object, np.ndarray],
    aggregation: str = "max",
    softmax_alpha: float = 10.0,
) -> np.ndarray:
    if not scores_by_class:
        raise ValueError("scores_by_class must not be empty.")
    if aggregation not in {"max", "softmax"}:
        raise ValueError("aggregation must be 'max' or 'softmax'.")
    if softmax_alpha <= 0.0:
        raise ValueError("softmax_alpha must be positive.")

    aggregated_columns = []
    n_samples = None
    for class_label, local_scores in scores_by_class.items():
        local_scores = np.asarray(local_scores, dtype=float)
        if local_scores.ndim != 2:
            raise ValueError(f"Scores for class {class_label!r} must be a 2D array.")
        if local_scores.shape[1] == 0:
            raise ValueError(f"Scores for class {class_label!r} must contain at least one prototype.")

        if n_samples is None:
            n_samples = local_scores.shape[0]
        elif local_scores.shape[0] != n_samples:
            raise ValueError("All class score matrices must have the same number of rows.")

        if aggregation == "max":
            aggregated_columns.append(np.max(local_scores, axis=1))
        else:
            scaled = softmax_alpha * local_scores
            row_max = np.max(scaled, axis=1, keepdims=True)
            logsumexp = np.log(np.sum(np.exp(scaled - row_max), axis=1)) + row_max.ravel()
            aggregated_columns.append(logsumexp / softmax_alpha)

    return np.column_stack(aggregated_columns)
