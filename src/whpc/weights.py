from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


def normalize_sample_weights(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1:
        raise ValueError("weights must be a 1D array.")
    if weights.size == 0:
        raise ValueError("weights must not be empty.")
    if np.any(weights < 0.0):
        raise ValueError("weights must be non-negative.")

    total = float(np.sum(weights))
    if total <= 1e-15:
        raise ValueError("weights must have positive sum.")
    return weights / total


def uniform_sample_weights(n_samples: int) -> np.ndarray:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    return np.full(n_samples, 1.0 / n_samples, dtype=float)


def mix_with_uniform_sample_weights(weights: np.ndarray, alpha: float) -> np.ndarray:
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError("alpha must be between 0 and 1.")

    normalized_weights = normalize_sample_weights(weights)
    uniform_weights = uniform_sample_weights(normalized_weights.size)
    return normalize_sample_weights((1.0 - alpha) * uniform_weights + alpha * normalized_weights)


def optimized_margin_sample_weights(
    X_class: np.ndarray,
    X_other: np.ndarray,
    temperature: float = 1.0,
) -> np.ndarray:
    X_class = np.asarray(X_class, dtype=float)
    X_other = np.asarray(X_other, dtype=float)

    if X_class.ndim != 2:
        raise ValueError("X_class must be a 2D array.")
    if X_class.shape[0] == 0:
        raise ValueError("X_class must contain at least one sample.")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")
    if X_other.ndim != 2:
        raise ValueError("X_other must be a 2D array.")
    if X_other.shape[1] != X_class.shape[1]:
        raise ValueError("X_other must have the same number of features as X_class.")

    if X_class.shape[0] == 1 or X_other.shape[0] == 0:
        return uniform_sample_weights(X_class.shape[0])

    class_sum = np.sum(X_class, axis=0, keepdims=True)
    leave_one_out_centers = (class_sum - X_class) / (X_class.shape[0] - 1)
    other_center = np.mean(X_other, axis=0, keepdims=True)

    positive_alignment = _rowwise_cosine_similarity(X_class, leave_one_out_centers)
    negative_alignment = _rowwise_cosine_similarity(X_class, np.repeat(other_center, X_class.shape[0], axis=0))
    utilities = positive_alignment - negative_alignment
    return _softmax_on_simplex(utilities / temperature)


def intra_class_core_sample_weights(
    X_class: np.ndarray,
    temperature: float = 1.0,
) -> np.ndarray:
    X_class = np.asarray(X_class, dtype=float)

    if X_class.ndim != 2:
        raise ValueError("X_class must be a 2D array.")
    if X_class.shape[0] == 0:
        raise ValueError("X_class must contain at least one sample.")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")
    if X_class.shape[0] == 1:
        return uniform_sample_weights(X_class.shape[0])

    class_sum = np.sum(X_class, axis=0, keepdims=True)
    leave_one_out_centers = (class_sum - X_class) / (X_class.shape[0] - 1)
    utilities = _rowwise_cosine_similarity(X_class, leave_one_out_centers)
    return _softmax_on_simplex(utilities / temperature)


def center_boundary_sample_weights(
    X_class: np.ndarray,
    X_other: np.ndarray,
    temperature: float = 1.0,
    boundary_gamma: float = 0.5,
    boundary_scale: float = 0.1,
) -> np.ndarray:
    X_class = np.asarray(X_class, dtype=float)
    X_other = np.asarray(X_other, dtype=float)

    if X_class.ndim != 2:
        raise ValueError("X_class must be a 2D array.")
    if X_class.shape[0] == 0:
        raise ValueError("X_class must contain at least one sample.")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")
    if boundary_gamma < 0.0:
        raise ValueError("boundary_gamma must be non-negative.")
    if boundary_scale <= 0.0:
        raise ValueError("boundary_scale must be positive.")
    if X_other.ndim != 2:
        raise ValueError("X_other must be a 2D array.")
    if X_other.shape[1] != X_class.shape[1]:
        raise ValueError("X_other must have the same number of features as X_class.")

    if X_class.shape[0] == 1 or X_other.shape[0] == 0:
        return uniform_sample_weights(X_class.shape[0])

    utilities = _center_boundary_utilities(
        X_class,
        X_other,
        boundary_gamma=boundary_gamma,
        boundary_scale=boundary_scale,
    )
    return _softmax_on_simplex(utilities / temperature)


def local_typicality_margin_sample_weights(
    X_class: np.ndarray,
    X_other: np.ndarray,
    temperature: float = 1.0,
    typicality_k: int = 15,
    typicality_reference_cap: int = 2048,
    compactness_gamma: float = 1.0,
    separation_lambda: float = 1.0,
    random_state: int | None = None,
) -> np.ndarray:
    X_class = np.asarray(X_class, dtype=float)
    X_other = np.asarray(X_other, dtype=float)

    if X_class.ndim != 2:
        raise ValueError("X_class must be a 2D array.")
    if X_class.shape[0] == 0:
        raise ValueError("X_class must contain at least one sample.")
    if X_other.ndim != 2:
        raise ValueError("X_other must be a 2D array.")
    if X_other.shape[1] != X_class.shape[1]:
        raise ValueError("X_other must have the same number of features as X_class.")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")
    if typicality_k < 1:
        raise ValueError("typicality_k must be at least 1.")
    if typicality_reference_cap < 1:
        raise ValueError("typicality_reference_cap must be at least 1.")
    if compactness_gamma < 0.0:
        raise ValueError("compactness_gamma must be non-negative.")
    if separation_lambda < 0.0:
        raise ValueError("separation_lambda must be non-negative.")

    if X_class.shape[0] == 1:
        return uniform_sample_weights(X_class.shape[0])

    local_typicality = _local_typicality_scores(
        X_class,
        typicality_k=typicality_k,
        reference_cap=typicality_reference_cap,
        random_state=random_state,
    )
    class_sum = np.sum(X_class, axis=0, keepdims=True)
    leave_one_out_centers = (class_sum - X_class) / (X_class.shape[0] - 1)
    core_alignment = _rowwise_cosine_similarity(X_class, leave_one_out_centers)

    if X_other.shape[0] == 0:
        other_alignment = np.zeros(X_class.shape[0], dtype=float)
    else:
        other_center = np.mean(X_other, axis=0, keepdims=True)
        other_alignment = _rowwise_cosine_similarity(X_class, np.repeat(other_center, X_class.shape[0], axis=0))

    utilities = local_typicality + compactness_gamma * core_alignment - separation_lambda * other_alignment
    return _softmax_on_simplex(utilities / temperature)


def angular_separation_sample_weights_by_class(
    X_by_class: list[np.ndarray],
    temperature: float = 1.0,
    boundary_gamma: float = 0.5,
    boundary_scale: float = 0.1,
    separation_lambda: float = 0.5,
    n_iter: int = 3,
) -> list[np.ndarray]:
    _validate_X_by_class(X_by_class)
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")
    if boundary_gamma < 0.0:
        raise ValueError("boundary_gamma must be non-negative.")
    if boundary_scale <= 0.0:
        raise ValueError("boundary_scale must be positive.")
    if separation_lambda < 0.0:
        raise ValueError("separation_lambda must be non-negative.")
    if n_iter < 1:
        raise ValueError("n_iter must be at least 1.")

    local_utilities = []
    for class_index, X_class in enumerate(X_by_class):
        X_other = _stack_other_classes(X_by_class, class_index)
        if X_class.shape[0] == 1 or X_other.shape[0] == 0:
            local_utilities.append(np.zeros(X_class.shape[0], dtype=float))
        else:
            local_utilities.append(
                _center_boundary_utilities(
                    X_class,
                    X_other,
                    boundary_gamma=boundary_gamma,
                    boundary_scale=boundary_scale,
                )
            )

    weights_by_class = [_softmax_on_simplex(utilities / temperature) for utilities in local_utilities]
    if separation_lambda == 0.0 or len(X_by_class) == 1:
        return weights_by_class

    for _ in range(n_iter):
        representatives = [
            _normalize_vector(weights @ X_class) for weights, X_class in zip(weights_by_class, X_by_class, strict=True)
        ]
        next_weights = []
        for class_index, X_class in enumerate(X_by_class):
            other_representatives = [
                representative
                for other_index, representative in enumerate(representatives)
                if other_index != class_index
            ]
            other_alignment = _max_alignment_to_representatives(X_class, other_representatives)
            utilities = local_utilities[class_index] - separation_lambda * other_alignment
            next_weights.append(_softmax_on_simplex(utilities / temperature))
        weights_by_class = next_weights

    return weights_by_class


def _center_boundary_utilities(
    X_class: np.ndarray,
    X_other: np.ndarray,
    boundary_gamma: float,
    boundary_scale: float,
) -> np.ndarray:
    class_sum = np.sum(X_class, axis=0, keepdims=True)
    leave_one_out_centers = (class_sum - X_class) / (X_class.shape[0] - 1)
    other_center = np.mean(X_other, axis=0, keepdims=True)

    core_alignment = _rowwise_cosine_similarity(X_class, leave_one_out_centers)
    other_alignment = _rowwise_cosine_similarity(X_class, np.repeat(other_center, X_class.shape[0], axis=0))
    margins = core_alignment - other_alignment
    boundary_score = np.where(margins >= 0.0, np.exp(-np.abs(margins) / boundary_scale), 0.0)
    return core_alignment + boundary_gamma * boundary_score


def _validate_X_by_class(X_by_class: list[np.ndarray]) -> None:
    if not X_by_class:
        raise ValueError("X_by_class must contain at least one class.")
    n_features = None
    for X_class in X_by_class:
        X_class = np.asarray(X_class, dtype=float)
        if X_class.ndim != 2:
            raise ValueError("Every class array must be 2D.")
        if X_class.shape[0] == 0:
            raise ValueError("Every class array must contain at least one sample.")
        if n_features is None:
            n_features = X_class.shape[1]
        elif X_class.shape[1] != n_features:
            raise ValueError("Every class array must have the same number of features.")


def _stack_other_classes(X_by_class: list[np.ndarray], class_index: int) -> np.ndarray:
    other_classes = [X_class for idx, X_class in enumerate(X_by_class) if idx != class_index]
    if not other_classes:
        return np.empty((0, X_by_class[class_index].shape[1]), dtype=float)
    return np.vstack(other_classes)


def _local_typicality_scores(
    X_class: np.ndarray,
    typicality_k: int,
    reference_cap: int,
    random_state: int | None,
) -> np.ndarray:
    n_samples = X_class.shape[0]
    if n_samples <= 1:
        return np.zeros(n_samples, dtype=float)

    if n_samples <= reference_cap:
        reference = X_class
        reference_is_full = True
    else:
        rng = np.random.default_rng(random_state)
        reference_indices = rng.choice(n_samples, size=reference_cap, replace=False)
        reference = X_class[np.sort(reference_indices)]
        reference_is_full = False

    n_neighbors = min(typicality_k + 1 if reference_is_full else typicality_k, reference.shape[0])
    if n_neighbors < 1:
        return np.zeros(n_samples, dtype=float)

    neighbors = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric="cosine",
        algorithm="brute",
    )
    neighbors.fit(reference)
    distances, _ = neighbors.kneighbors(X_class, return_distance=True)

    if reference_is_full and distances.shape[1] > 1:
        distances = distances[:, 1:]

    if distances.shape[1] == 0:
        return np.zeros(n_samples, dtype=float)

    similarities = 1.0 - distances
    return np.mean(similarities, axis=1)


def _normalize_vector(x: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(x))
    if norm <= 1e-15:
        return np.zeros_like(x, dtype=float)
    return x / norm


def _max_alignment_to_representatives(
    X_class: np.ndarray,
    representatives: list[np.ndarray],
) -> np.ndarray:
    if not representatives:
        return np.zeros(X_class.shape[0], dtype=float)

    alignment_columns = []
    for representative in representatives:
        tiled_representative = np.repeat(representative.reshape(1, -1), X_class.shape[0], axis=0)
        alignment_columns.append(_rowwise_cosine_similarity(X_class, tiled_representative))
    return np.max(np.column_stack(alignment_columns), axis=1)


def _softmax_on_simplex(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    if logits.ndim != 1:
        raise ValueError("logits must be a 1D array.")

    shifted = logits - np.max(logits)
    exp_logits = np.exp(shifted)
    return normalize_sample_weights(exp_logits)


def _rowwise_cosine_similarity(X_left: np.ndarray, X_right: np.ndarray) -> np.ndarray:
    X_left = np.asarray(X_left, dtype=float)
    X_right = np.asarray(X_right, dtype=float)
    if X_left.shape != X_right.shape:
        raise ValueError("X_left and X_right must have the same shape.")

    dot_products = np.sum(X_left * X_right, axis=1)
    left_norms = np.linalg.norm(X_left, axis=1)
    right_norms = np.linalg.norm(X_right, axis=1)
    denominators = left_norms * right_norms
    return np.divide(
        dot_products,
        denominators,
        out=np.zeros(X_left.shape[0], dtype=float),
        where=denominators > 1e-15,
    )
