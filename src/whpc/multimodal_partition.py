from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

PARTITION_STRATEGIES = {"kmeans", "cosine_farthest"}


def _normalize_rows(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    safe_norms = np.where(norms > 1e-15, norms, 1.0)
    return X / safe_norms


def partition_class_samples(
    X_class: np.ndarray,
    n_partitions: int,
    strategy: str = "kmeans",
    random_state: int | None = None,
) -> tuple[np.ndarray, list[np.ndarray]]:
    X_class = np.asarray(X_class, dtype=float)
    if X_class.ndim != 2:
        raise ValueError("X_class must be a 2D array.")
    if X_class.shape[0] == 0:
        raise ValueError("X_class must contain at least one sample.")
    if n_partitions < 1:
        raise ValueError("n_partitions must be at least 1.")
    if n_partitions > X_class.shape[0]:
        raise ValueError("n_partitions cannot exceed the number of class samples.")
    if strategy not in PARTITION_STRATEGIES:
        raise ValueError(f"strategy must be one of {sorted(PARTITION_STRATEGIES)!r}.")

    if n_partitions == 1:
        labels = np.zeros(X_class.shape[0], dtype=int)
        return labels, [np.arange(X_class.shape[0], dtype=int)]

    if strategy == "cosine_farthest":
        return _cosine_farthest_partition(X_class, n_partitions)

    kmeans = KMeans(
        n_clusters=n_partitions,
        n_init=10,
        random_state=random_state,
    )
    labels = kmeans.fit_predict(X_class)
    partitions = [np.flatnonzero(labels == partition_index) for partition_index in range(n_partitions)]

    if any(partition.size == 0 for partition in partitions):
        raise ValueError("Partitioning produced an empty prototype group.")

    return labels.astype(int, copy=False), partitions


def _cosine_farthest_partition(
    X_class: np.ndarray,
    n_partitions: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    X_normalized = _normalize_rows(X_class)
    global_center = np.mean(X_normalized, axis=0, keepdims=True)
    global_center = _normalize_rows(global_center)[0]
    first_seed = int(np.argmin(X_normalized @ global_center))

    seed_indices = [first_seed]
    max_similarity_to_seeds = X_normalized @ X_normalized[first_seed]

    while len(seed_indices) < n_partitions:
        candidate_scores = max_similarity_to_seeds.copy()
        candidate_scores[np.asarray(seed_indices, dtype=int)] = np.inf
        next_seed = int(np.argmin(candidate_scores))
        seed_indices.append(next_seed)
        max_similarity_to_seeds = np.maximum(max_similarity_to_seeds, X_normalized @ X_normalized[next_seed])

    seed_matrix = X_normalized[np.asarray(seed_indices, dtype=int)]
    labels = np.argmax(X_normalized @ seed_matrix.T, axis=1).astype(int, copy=False)
    partitions = [np.flatnonzero(labels == partition_index) for partition_index in range(n_partitions)]
    return labels, partitions
