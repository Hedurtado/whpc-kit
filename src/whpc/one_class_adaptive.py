from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_array, check_is_fitted

from .multimodal_aggregation import aggregate_prototype_scores
from .multimodal_partition import partition_class_samples
from .one_class_multimodal import OCMMWHPCClassifier, _fbeta_from_prefix_counts, _normalize_vector
from .representatives import build_class_representative


@dataclass(frozen=True)
class AdaptiveStructureEvaluation:
    objective_value: float
    structure_metric_value: float
    structure_threshold: float
    structure_robustness: float
    threshold: float
    threshold_metric_value: float
    n_representatives: int
    partition_indices: tuple[np.ndarray, ...]
    partition_labels: np.ndarray
    representatives: np.ndarray
    representatives_normalized: np.ndarray
    sample_weights: tuple[np.ndarray, ...]
    density_reference_by_mode: tuple[np.ndarray, ...] | None
    density_neighbors_by_mode: tuple[NearestNeighbors, ...] | None


@dataclass(frozen=True)
class HierarchicalSplitCandidate:
    leaf_position: int
    local_gain: float
    compactness_reduction: float
    child_separation: float
    child_indices: tuple[np.ndarray, np.ndarray]
    evaluation: AdaptiveStructureEvaluation


class AdaptiveOCMMWHPCClassifier(OCMMWHPCClassifier):
    def __init__(
        self,
        *,
        structure_selection_strategy: str = "grid",
        candidate_n_normal_representatives: tuple[int, ...] = (1, 2, 3, 4, 5),
        max_normal_representatives: int | None = None,
        structure_complexity_penalty: float = 0.0,
        structure_min_gain: float = 0.0,
        hierarchical_min_mode_size: int = 32,
        structure_selection_metric: str | None = None,
        structure_selection_beta: float | None = None,
        structure_selection_min_normal_acceptance: float | None = None,
        structure_selection_robustness_weight: float = 0.0,
        structure_selection_robustness_tolerance: float = 0.0,
        hierarchical_local_min_gain: float = 0.0,
        hierarchical_compactness_weight: float = 1.0,
        hierarchical_separation_weight: float = 0.1,
        density_lambda: float = 0.0,
        density_k: int = 15,
        density_reference_cap: int = 10000,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.structure_selection_strategy = structure_selection_strategy
        self.candidate_n_normal_representatives = candidate_n_normal_representatives
        self.max_normal_representatives = max_normal_representatives
        self.structure_complexity_penalty = structure_complexity_penalty
        self.structure_min_gain = structure_min_gain
        self.hierarchical_min_mode_size = hierarchical_min_mode_size
        self.structure_selection_metric = structure_selection_metric
        self.structure_selection_beta = structure_selection_beta
        self.structure_selection_min_normal_acceptance = structure_selection_min_normal_acceptance
        self.structure_selection_robustness_weight = structure_selection_robustness_weight
        self.structure_selection_robustness_tolerance = structure_selection_robustness_tolerance
        self.hierarchical_local_min_gain = hierarchical_local_min_gain
        self.hierarchical_compactness_weight = hierarchical_compactness_weight
        self.hierarchical_separation_weight = hierarchical_separation_weight
        self.density_lambda = density_lambda
        self.density_k = density_k
        self.density_reference_cap = density_reference_cap

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        *,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> "AdaptiveOCMMWHPCClassifier":
        X = check_array(X, dtype=float)
        X_val = check_array(X_val, dtype=float)
        self._validate_configuration()
        self._validate_adaptive_configuration()
        self._validate_optional_y(y, X.shape[0])

        X_model = self._prepare_X(X)
        X_val_model = self._prepare_X(X_val)
        self.n_features_in_ = X_model.shape[1]
        y_val_encoded = self._encode_anomaly_targets(y_val)

        if self.structure_selection_strategy == "grid":
            selected = self._fit_grid_structure(X_model, X_val_model, y_val_encoded)
        elif self.structure_selection_strategy == "hierarchical_legacy":
            selected = self._fit_hierarchical_legacy_structure(X_model, X_val_model, y_val_encoded)
        else:
            selected = self._fit_hierarchical_structure(X_model, X_val_model, y_val_encoded)

        self.normal_partition_labels_ = selected.partition_labels
        self.normal_partition_indices_ = [indices.copy() for indices in selected.partition_indices]
        self.normal_sample_weights_ = [weights.copy() for weights in selected.sample_weights]
        self.normal_representatives_ = selected.representatives.copy()
        self.normal_representatives_normalized_ = selected.representatives_normalized.copy()
        self.n_normal_representatives_ = selected.n_representatives
        self.threshold_ = float(selected.threshold)
        self.adaptive_structure_summary_ = {
            "strategy": self.structure_selection_strategy,
            "objective_value": float(selected.objective_value),
            "structure_selection_metric": self._resolve_structure_selection_metric(),
            "structure_selection_beta": float(self._resolve_structure_selection_beta()),
            "structure_selection_min_normal_acceptance": self._resolve_structure_selection_min_normal_acceptance(),
            "structure_selection_metric_value": float(selected.structure_metric_value),
            "structure_selection_threshold": float(selected.structure_threshold),
            "structure_selection_robustness": float(selected.structure_robustness),
            "threshold_selection_metric": self.threshold_selection_metric,
            "threshold_selection_beta": float(self.threshold_metric_beta),
            "threshold_min_normal_acceptance": self.threshold_min_normal_acceptance,
            "threshold_metric_value": float(selected.threshold_metric_value),
            "threshold": float(selected.threshold),
            "validation_metric_value": float(selected.structure_metric_value),
            "n_normal_representatives": int(selected.n_representatives),
            "structure_complexity_penalty": float(self.structure_complexity_penalty),
            "structure_min_gain": float(self.structure_min_gain),
            "structure_selection_robustness_weight": float(self.structure_selection_robustness_weight),
            "structure_selection_robustness_tolerance": float(self.structure_selection_robustness_tolerance),
            "hierarchical_local_min_gain": float(self.hierarchical_local_min_gain),
            "hierarchical_compactness_weight": float(self.hierarchical_compactness_weight),
            "hierarchical_separation_weight": float(self.hierarchical_separation_weight),
        }

        if selected.density_reference_by_mode is not None and selected.density_neighbors_by_mode is not None:
            self.density_reference_distances_by_mode_ = [row.copy() for row in selected.density_reference_by_mode]
            self.density_neighbors_by_mode_ = list(selected.density_neighbors_by_mode)

        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        local_scores, X_model = self._compute_local_scores_with_prepared_X(X)
        similarity = self._aggregate_local_scores(local_scores)
        if np.isclose(self.density_lambda, 0.0):
            return similarity
        residual = self._compute_density_residuals(local_scores, X_model)
        return similarity - (self.density_lambda * residual)

    def score_local_samples(self, X: np.ndarray) -> np.ndarray:
        local_scores, _ = self._compute_local_scores_with_prepared_X(X)
        return local_scores

    def _compute_local_scores_with_prepared_X(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        check_is_fitted(self, ["normal_representatives_", "normal_representatives_normalized_"])
        X = check_array(X, dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but AdaptiveOCMMWHPCClassifier was fitted with {self.n_features_in_}."
            )
        X_model = self._prepare_X(X)
        if self.score_metric == "inner_product":
            return X_model @ self.normal_representatives_.T, X_model

        x_norms = np.linalg.norm(X_model, axis=1)
        dot_products = X_model @ self.normal_representatives_normalized_.T
        local_scores = np.divide(
            dot_products,
            x_norms.reshape(-1, 1),
            out=np.zeros_like(dot_products),
            where=x_norms.reshape(-1, 1) > 1e-15,
        )
        return local_scores, X_model

    def _aggregate_local_scores(self, local_scores: np.ndarray) -> np.ndarray:
        aggregated = aggregate_prototype_scores(
            {"normal": np.asarray(local_scores, dtype=float)},
            aggregation=self.prototype_aggregation,
            softmax_alpha=self.prototype_softmax_alpha,
        )
        return aggregated[:, 0]

    def _fit_grid_structure(
        self,
        X_model: np.ndarray,
        X_val_model: np.ndarray,
        y_val_encoded: np.ndarray,
    ) -> AdaptiveStructureEvaluation:
        best: AdaptiveStructureEvaluation | None = None
        for n_representatives in self.candidate_n_normal_representatives:
            partition_labels, partition_indices = partition_class_samples(
                X_model,
                n_partitions=int(n_representatives),
                strategy=self.representative_partition_strategy,
                random_state=self.random_state,
            )
            evaluation = self._evaluate_partition(
                X_model,
                X_val_model,
                y_val_encoded,
                partition_labels=partition_labels,
                partition_indices=partition_indices,
            )
            if best is None or evaluation.objective_value > best.objective_value:
                best = evaluation
        if best is None:
            raise RuntimeError("Grid adaptive fit did not evaluate any candidate structure.")
        return best

    def _fit_hierarchical_structure(
        self,
        X_model: np.ndarray,
        X_val_model: np.ndarray,
        y_val_encoded: np.ndarray,
    ) -> AdaptiveStructureEvaluation:
        current_partition_indices = [np.arange(X_model.shape[0], dtype=int)]
        current_labels = np.zeros(X_model.shape[0], dtype=int)
        best_current = self._evaluate_partition(
            X_model,
            X_val_model,
            y_val_encoded,
            partition_labels=current_labels,
            partition_indices=current_partition_indices,
        )
        max_representatives = (
            max(self.candidate_n_normal_representatives)
            if self.max_normal_representatives is None
            else self.max_normal_representatives
        )

        while len(current_partition_indices) < max_representatives:
            candidate_queue: list[tuple[float, float, int, HierarchicalSplitCandidate]] = []
            for leaf_position, leaf_indices in enumerate(current_partition_indices):
                candidate = self._evaluate_best_first_leaf_split(
                    X_model,
                    X_val_model,
                    y_val_encoded,
                    current_partition_indices=current_partition_indices,
                    current_objective=best_current.objective_value,
                    leaf_position=leaf_position,
                    leaf_indices=leaf_indices,
                )
                if candidate is None:
                    continue
                global_gain = candidate.evaluation.objective_value - best_current.objective_value
                if global_gain <= self.structure_min_gain:
                    continue
                heapq.heappush(
                    candidate_queue,
                    (-float(global_gain), -float(candidate.local_gain), int(leaf_position), candidate),
                )

            if not candidate_queue:
                break

            _, _, _, best_candidate = heapq.heappop(candidate_queue)
            next_partition_indices = self._replace_partition_leaf(
                current_partition_indices,
                leaf_position=best_candidate.leaf_position,
                child_indices=best_candidate.child_indices,
            )
            current_partition_indices = [indices.copy() for indices in next_partition_indices]
            best_current = best_candidate.evaluation

        return best_current

    def _fit_hierarchical_legacy_structure(
        self,
        X_model: np.ndarray,
        X_val_model: np.ndarray,
        y_val_encoded: np.ndarray,
    ) -> AdaptiveStructureEvaluation:
        current_partition_indices = [np.arange(X_model.shape[0], dtype=int)]
        current_labels = np.zeros(X_model.shape[0], dtype=int)
        best_current = self._evaluate_partition(
            X_model,
            X_val_model,
            y_val_encoded,
            partition_labels=current_labels,
            partition_indices=current_partition_indices,
        )
        max_representatives = (
            max(self.candidate_n_normal_representatives)
            if self.max_normal_representatives is None
            else self.max_normal_representatives
        )

        while len(current_partition_indices) < max_representatives:
            best_candidate: AdaptiveStructureEvaluation | None = None
            best_candidate_indices: list[np.ndarray] | None = None

            for mode_position, mode_indices in enumerate(current_partition_indices):
                if mode_indices.shape[0] < max(2, self.hierarchical_min_mode_size):
                    continue
                local_labels, local_partitions = partition_class_samples(
                    X_model[mode_indices],
                    n_partitions=2,
                    strategy="kmeans",
                    random_state=self.random_state,
                )
                child_sizes = [part.shape[0] for part in local_partitions]
                if min(child_sizes) < max(1, self.hierarchical_min_mode_size // 2):
                    continue

                candidate_partition_indices = []
                for existing_mode_position, existing_indices in enumerate(current_partition_indices):
                    if existing_mode_position == mode_position:
                        continue
                    candidate_partition_indices.append(existing_indices.copy())
                candidate_partition_indices.extend([mode_indices[local_partitions[0]], mode_indices[local_partitions[1]]])
                candidate_partition_indices = sorted(candidate_partition_indices, key=lambda arr: int(arr[0]))

                candidate_labels = np.empty(X_model.shape[0], dtype=int)
                for new_mode_index, indices in enumerate(candidate_partition_indices):
                    candidate_labels[indices] = new_mode_index

                candidate = self._evaluate_partition(
                    X_model,
                    X_val_model,
                    y_val_encoded,
                    partition_labels=candidate_labels,
                    partition_indices=candidate_partition_indices,
                )
                if best_candidate is None or candidate.objective_value > best_candidate.objective_value:
                    best_candidate = candidate
                    best_candidate_indices = candidate_partition_indices

            if best_candidate is None:
                break
            if best_candidate.objective_value <= (best_current.objective_value + self.structure_min_gain):
                break

            best_current = best_candidate
            current_partition_indices = [indices.copy() for indices in best_candidate_indices]

        return best_current

    def _evaluate_best_first_leaf_split(
        self,
        X_model: np.ndarray,
        X_val_model: np.ndarray,
        y_val_encoded: np.ndarray,
        *,
        current_partition_indices: list[np.ndarray],
        current_objective: float,
        leaf_position: int,
        leaf_indices: np.ndarray,
    ) -> HierarchicalSplitCandidate | None:
        if leaf_indices.shape[0] < max(2, 2 * self.hierarchical_min_mode_size):
            return None

        local_labels, local_partitions = partition_class_samples(
            X_model[leaf_indices],
            n_partitions=2,
            strategy="kmeans",
            random_state=self.random_state,
        )
        child_sizes = [part.shape[0] for part in local_partitions]
        if min(child_sizes) < self.hierarchical_min_mode_size:
            return None

        left_indices = leaf_indices[local_partitions[0]]
        right_indices = leaf_indices[local_partitions[1]]
        local_gain, compactness_reduction, child_separation = self._compute_local_split_gain(
            X_model,
            parent_indices=leaf_indices,
            left_indices=left_indices,
            right_indices=right_indices,
        )
        if local_gain <= self.hierarchical_local_min_gain:
            return None

        candidate_partition_indices = self._replace_partition_leaf(
            current_partition_indices,
            leaf_position=leaf_position,
            child_indices=(left_indices, right_indices),
        )
        candidate_labels = self._labels_from_partition_indices(X_model.shape[0], candidate_partition_indices)
        candidate_evaluation = self._evaluate_partition(
            X_model,
            X_val_model,
            y_val_encoded,
            partition_labels=candidate_labels,
            partition_indices=candidate_partition_indices,
        )
        if candidate_evaluation.objective_value <= current_objective:
            return None
        return HierarchicalSplitCandidate(
            leaf_position=leaf_position,
            local_gain=float(local_gain),
            compactness_reduction=float(compactness_reduction),
            child_separation=float(child_separation),
            child_indices=(left_indices.copy(), right_indices.copy()),
            evaluation=candidate_evaluation,
        )

    def _compute_local_split_gain(
        self,
        X_model: np.ndarray,
        *,
        parent_indices: np.ndarray,
        left_indices: np.ndarray,
        right_indices: np.ndarray,
    ) -> tuple[float, float, float]:
        parent_representative = self._build_unweighted_local_representative(X_model[parent_indices])
        left_representative = self._build_unweighted_local_representative(X_model[left_indices])
        right_representative = self._build_unweighted_local_representative(X_model[right_indices])

        parent_compactness = self._local_compactness(X_model[parent_indices], parent_representative)
        left_compactness = self._local_compactness(X_model[left_indices], left_representative)
        right_compactness = self._local_compactness(X_model[right_indices], right_representative)
        weighted_child_compactness = (
            (left_indices.shape[0] / parent_indices.shape[0]) * left_compactness
            + (right_indices.shape[0] / parent_indices.shape[0]) * right_compactness
        )
        compactness_reduction = parent_compactness - weighted_child_compactness
        child_separation = 1.0 - self._cosine_between_vectors(left_representative, right_representative)
        local_gain = (
            self.hierarchical_compactness_weight * compactness_reduction
            + self.hierarchical_separation_weight * child_separation
        )
        return float(local_gain), float(compactness_reduction), float(child_separation)

    def _build_unweighted_local_representative(self, X_mode: np.ndarray) -> np.ndarray:
        return build_class_representative(
            X_mode,
            weights=None,
            representative_strategy=self.representative_strategy,
            normalize=False,
        )

    def _local_compactness(self, X_mode: np.ndarray, representative: np.ndarray) -> float:
        representative_normalized = _normalize_vector(representative)
        if np.linalg.norm(representative_normalized) <= 1e-15:
            return 1.0
        x_norms = np.linalg.norm(X_mode, axis=1)
        similarities = np.divide(
            X_mode @ representative_normalized,
            x_norms,
            out=np.zeros(X_mode.shape[0], dtype=float),
            where=x_norms > 1e-15,
        )
        return float(np.mean(1.0 - similarities))

    def _cosine_between_vectors(self, a: np.ndarray, b: np.ndarray) -> float:
        a_normalized = _normalize_vector(a)
        b_normalized = _normalize_vector(b)
        if np.linalg.norm(a_normalized) <= 1e-15 or np.linalg.norm(b_normalized) <= 1e-15:
            return 0.0
        return float(np.dot(a_normalized, b_normalized))

    def _replace_partition_leaf(
        self,
        partition_indices: list[np.ndarray],
        *,
        leaf_position: int,
        child_indices: tuple[np.ndarray, np.ndarray],
    ) -> list[np.ndarray]:
        candidate_partition_indices: list[np.ndarray] = []
        for existing_position, existing_indices in enumerate(partition_indices):
            if existing_position == leaf_position:
                candidate_partition_indices.extend([child_indices[0].copy(), child_indices[1].copy()])
            else:
                candidate_partition_indices.append(existing_indices.copy())
        return sorted(candidate_partition_indices, key=lambda arr: int(arr[0]))

    def _labels_from_partition_indices(
        self,
        n_samples: int,
        partition_indices: list[np.ndarray],
    ) -> np.ndarray:
        labels = np.empty(n_samples, dtype=int)
        for mode_index, indices in enumerate(partition_indices):
            labels[indices] = mode_index
        return labels

    def _evaluate_partition(
        self,
        X_model: np.ndarray,
        X_val_model: np.ndarray,
        y_val_encoded: np.ndarray,
        *,
        partition_labels: np.ndarray,
        partition_indices: list[np.ndarray],
    ) -> AdaptiveStructureEvaluation:
        representatives: list[np.ndarray] = []
        representatives_normalized: list[np.ndarray] = []
        sample_weights: list[np.ndarray] = []
        all_sample_mask = np.ones(X_model.shape[0], dtype=bool)
        for mode_index, local_indices in enumerate(partition_indices):
            X_mode = X_model[local_indices]
            other_mask = all_sample_mask.copy()
            other_mask[local_indices] = False
            X_other = X_model[other_mask]
            weights = self._resolve_sample_weights(X_mode, X_other, mode_index)
            representative = build_class_representative(
                X_mode,
                weights=weights,
                representative_strategy=self.representative_strategy,
                normalize=False,
            )
            representatives.append(representative)
            representatives_normalized.append(_normalize_vector(representative))
            sample_weights.append(weights)

        representatives_matrix = np.vstack(representatives)
        representatives_normalized_matrix = np.vstack(representatives_normalized)
        val_local_scores = self._compute_local_scores_from_state(X_val_model, representatives_matrix, representatives_normalized_matrix)
        density_reference_by_mode = None
        density_neighbors_by_mode = None
        if np.isclose(self.density_lambda, 0.0):
            val_scores = self._aggregate_local_scores(val_local_scores)
        else:
            density_neighbors_by_mode, density_reference_by_mode = self._build_density_cache(X_model, partition_indices)
            val_scores = self._compute_density_adjusted_scores(
                val_local_scores,
                X_val_model,
                density_neighbors_by_mode,
                density_reference_by_mode,
            )

        structure_threshold, structure_metric_value, structure_robustness = self._fit_structure_selection_threshold(
            val_scores,
            y_val_encoded,
        )
        threshold, threshold_metric_value = self._fit_threshold(val_scores, y_val_encoded)
        objective_value = (
            structure_metric_value
            + (self.structure_selection_robustness_weight * structure_robustness)
            - (self.structure_complexity_penalty * max(0, len(partition_indices) - 1))
        )
        return AdaptiveStructureEvaluation(
            objective_value=float(objective_value),
            structure_metric_value=float(structure_metric_value),
            structure_threshold=float(structure_threshold),
            structure_robustness=float(structure_robustness),
            threshold=float(threshold),
            threshold_metric_value=float(threshold_metric_value),
            n_representatives=len(partition_indices),
            partition_indices=tuple(indices.copy() for indices in partition_indices),
            partition_labels=partition_labels.copy(),
            representatives=representatives_matrix.copy(),
            representatives_normalized=representatives_normalized_matrix.copy(),
            sample_weights=tuple(weights.copy() for weights in sample_weights),
            density_reference_by_mode=None
            if density_reference_by_mode is None
            else tuple(row.copy() for row in density_reference_by_mode),
            density_neighbors_by_mode=None
            if density_neighbors_by_mode is None
            else tuple(density_neighbors_by_mode),
        )

    def _compute_local_scores_from_state(
        self,
        X_model: np.ndarray,
        representatives: np.ndarray,
        representatives_normalized: np.ndarray,
    ) -> np.ndarray:
        if self.score_metric == "inner_product":
            return X_model @ representatives.T

        x_norms = np.linalg.norm(X_model, axis=1)
        dot_products = X_model @ representatives_normalized.T
        return np.divide(
            dot_products,
            x_norms.reshape(-1, 1),
            out=np.zeros_like(dot_products),
            where=x_norms.reshape(-1, 1) > 1e-15,
        )

    def _build_density_cache(
        self,
        X_model: np.ndarray,
        partition_indices: list[np.ndarray],
    ) -> tuple[tuple[NearestNeighbors, ...], tuple[np.ndarray, ...]]:
        rng = np.random.default_rng(self.random_state)
        neighbors_by_mode: list[NearestNeighbors] = []
        reference_distances_by_mode: list[np.ndarray] = []
        for local_indices in partition_indices:
            X_mode = np.asarray(X_model[local_indices], dtype=float)
            if X_mode.shape[0] > self.density_reference_cap:
                sampled_indices = np.sort(rng.choice(X_mode.shape[0], size=self.density_reference_cap, replace=False))
                X_mode = X_mode[sampled_indices]
            neighbor_count = min(max(1, self.density_k), X_mode.shape[0])
            neighbors = NearestNeighbors(n_neighbors=neighbor_count, metric="euclidean")
            neighbors.fit(X_mode)
            neighbors_by_mode.append(neighbors)
            reference_distances_by_mode.append(self._reference_mean_distances(X_mode))
        return tuple(neighbors_by_mode), tuple(reference_distances_by_mode)

    def _reference_mean_distances(self, X_mode: np.ndarray) -> np.ndarray:
        if X_mode.shape[0] <= 1:
            return np.zeros(X_mode.shape[0], dtype=float)
        neighbor_count = min(self.density_k + 1, X_mode.shape[0])
        neighbors = NearestNeighbors(n_neighbors=neighbor_count, metric="euclidean")
        neighbors.fit(X_mode)
        distances, _ = neighbors.kneighbors(X_mode)
        if neighbor_count == 1:
            return distances[:, 0]
        return np.mean(distances[:, 1:], axis=1)

    def _compute_density_adjusted_scores(
        self,
        local_scores: np.ndarray,
        X_model: np.ndarray,
        neighbors_by_mode: tuple[NearestNeighbors, ...],
        reference_distances_by_mode: tuple[np.ndarray, ...],
    ) -> np.ndarray:
        similarity = self._aggregate_local_scores(local_scores)
        residual = self._compute_density_residuals_from_cache(
            local_scores,
            X_model,
            neighbors_by_mode,
            reference_distances_by_mode,
        )
        return similarity - (self.density_lambda * residual)

    def _compute_density_residuals(self, local_scores: np.ndarray, X_model: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["density_neighbors_by_mode_", "density_reference_distances_by_mode_"])
        return self._compute_density_residuals_from_cache(
            local_scores,
            X_model,
            tuple(self.density_neighbors_by_mode_),
            tuple(self.density_reference_distances_by_mode_),
        )

    def _compute_density_residuals_from_cache(
        self,
        local_scores: np.ndarray,
        X_model: np.ndarray,
        neighbors_by_mode: tuple[NearestNeighbors, ...],
        reference_distances_by_mode: tuple[np.ndarray, ...],
    ) -> np.ndarray:
        winners = np.argmax(local_scores, axis=1)
        residuals = np.zeros(X_model.shape[0], dtype=float)
        for mode_index in sorted(np.unique(winners).tolist()):
            mode_mask = winners == mode_index
            neighbors = neighbors_by_mode[int(mode_index)]
            reference_distances = np.sort(reference_distances_by_mode[int(mode_index)])
            n_neighbors = min(max(1, self.density_k), neighbors.n_samples_fit_)
            distances, _ = neighbors.kneighbors(np.asarray(X_model[mode_mask], dtype=float), n_neighbors=n_neighbors)
            mean_distances = np.mean(distances, axis=1)
            if reference_distances.size == 0:
                residuals[mode_mask] = 0.0
                continue
            ranks = np.searchsorted(reference_distances, mean_distances, side="right")
            residuals[mode_mask] = ranks / reference_distances.size
        return residuals

    def _fit_structure_selection_threshold(
        self,
        scores: np.ndarray,
        y_true: np.ndarray,
    ) -> tuple[float, float, float]:
        return self._fit_threshold_for_metric(
            scores,
            y_true,
            metric=self._resolve_structure_selection_metric(),
            beta=self._resolve_structure_selection_beta(),
            min_normal_acceptance=self._resolve_structure_selection_min_normal_acceptance(),
            robustness_tolerance=self.structure_selection_robustness_tolerance,
        )

    def _fit_threshold_for_metric(
        self,
        scores: np.ndarray,
        y_true: np.ndarray,
        *,
        metric: str,
        beta: float,
        min_normal_acceptance: float | None,
        robustness_tolerance: float,
    ) -> tuple[float, float, float]:
        scores = np.asarray(scores, dtype=float)
        y_true = np.asarray(y_true, dtype=int)
        if scores.ndim != 1:
            raise ValueError("scores must be a 1D array.")
        if y_true.shape[0] != scores.shape[0]:
            raise ValueError("scores and y_true must have the same number of samples.")

        order = np.argsort(scores, kind="mergesort")
        sorted_scores = scores[order]
        sorted_labels = y_true[order]
        total_positives = int(np.sum(sorted_labels))
        total_negatives = int(sorted_labels.shape[0] - total_positives)

        candidates: list[tuple[float, float]] = []
        true_positives = 0
        false_positives = 0
        idx = 0

        while idx < sorted_scores.shape[0]:
            threshold = float(sorted_scores[idx])
            metric_value = self._structure_threshold_objective_value(
                true_positives=true_positives,
                false_positives=false_positives,
                total_positives=total_positives,
                total_negatives=total_negatives,
                metric=metric,
                beta=beta,
                min_normal_acceptance=min_normal_acceptance,
            )
            candidates.append((threshold, metric_value))

            group_end = idx
            while group_end < sorted_scores.shape[0] and sorted_scores[group_end] == sorted_scores[idx]:
                group_end += 1
            group_labels = sorted_labels[idx:group_end]
            group_positive_count = int(np.sum(group_labels))
            true_positives += group_positive_count
            false_positives += int(group_labels.shape[0] - group_positive_count)
            idx = group_end

        final_threshold = float(np.nextafter(sorted_scores[-1], np.inf))
        final_metric_value = self._structure_threshold_objective_value(
            true_positives=true_positives,
            false_positives=false_positives,
            total_positives=total_positives,
            total_negatives=total_negatives,
            metric=metric,
            beta=beta,
            min_normal_acceptance=min_normal_acceptance,
        )
        candidates.append((final_threshold, final_metric_value))

        best_threshold, best_metric_value = max(candidates, key=lambda row: (float(row[1]), -float(row[0])))
        close_count = sum(
            1 for _, metric_value in candidates if metric_value >= (best_metric_value - robustness_tolerance)
        )
        robustness = 0.0 if not candidates else close_count / len(candidates)
        return float(best_threshold), float(best_metric_value), float(robustness)

    def _structure_threshold_objective_value(
        self,
        *,
        true_positives: int,
        false_positives: int,
        total_positives: int,
        total_negatives: int,
        metric: str,
        beta: float,
        min_normal_acceptance: float | None,
    ) -> float:
        false_negatives = total_positives - true_positives
        true_negatives = total_negatives - false_positives
        tpr = 0.0 if total_positives == 0 else true_positives / total_positives
        tnr = 0.0 if total_negatives == 0 else true_negatives / total_negatives

        if metric == "balanced_accuracy":
            return float((tpr + tnr) / 2.0)
        if metric == "gmean":
            return float(np.sqrt(tpr * tnr))
        if metric == "f1_anomalous":
            return _fbeta_from_prefix_counts(
                true_positives=true_positives,
                false_positives=false_positives,
                false_negatives=false_negatives,
                beta=1.0,
            )
        if metric == "f_beta_anomalous":
            return _fbeta_from_prefix_counts(
                true_positives=true_positives,
                false_positives=false_positives,
                false_negatives=false_negatives,
                beta=beta,
            )
        if metric == "constrained_f1_anomalous":
            minimum = 0.0 if min_normal_acceptance is None else min_normal_acceptance
            if tnr < minimum:
                return float("-inf")
            return _fbeta_from_prefix_counts(
                true_positives=true_positives,
                false_positives=false_positives,
                false_negatives=false_negatives,
                beta=1.0,
            )
        raise ValueError(
            "structure_selection_metric must be one of "
            "'f1_anomalous', 'balanced_accuracy', 'gmean', 'f_beta_anomalous', or 'constrained_f1_anomalous'."
        )

    def _resolve_structure_selection_metric(self) -> str:
        if self.structure_selection_metric is None:
            return self.threshold_selection_metric
        return self.structure_selection_metric

    def _resolve_structure_selection_beta(self) -> float:
        if self.structure_selection_beta is None:
            return self.threshold_metric_beta
        return self.structure_selection_beta

    def _resolve_structure_selection_min_normal_acceptance(self) -> float | None:
        if self.structure_selection_min_normal_acceptance is None:
            return self.threshold_min_normal_acceptance
        return self.structure_selection_min_normal_acceptance

    def _validate_adaptive_configuration(self) -> None:
        if self.structure_selection_strategy not in {"grid", "hierarchical", "hierarchical_legacy"}:
            raise ValueError("structure_selection_strategy must be 'grid', 'hierarchical', or 'hierarchical_legacy'.")
        if not self.candidate_n_normal_representatives:
            raise ValueError("candidate_n_normal_representatives must not be empty.")
        if any(int(value) < 1 for value in self.candidate_n_normal_representatives):
            raise ValueError("All candidate_n_normal_representatives values must be at least 1.")
        if self.max_normal_representatives is not None and self.max_normal_representatives < 1:
            raise ValueError("max_normal_representatives must be at least 1 when provided.")
        if self.structure_complexity_penalty < 0.0:
            raise ValueError("structure_complexity_penalty must be non-negative.")
        if self.structure_min_gain < 0.0:
            raise ValueError("structure_min_gain must be non-negative.")
        if self.hierarchical_min_mode_size < 2:
            raise ValueError("hierarchical_min_mode_size must be at least 2.")
        if self.hierarchical_local_min_gain < 0.0:
            raise ValueError("hierarchical_local_min_gain must be non-negative.")
        if self.hierarchical_compactness_weight < 0.0:
            raise ValueError("hierarchical_compactness_weight must be non-negative.")
        if self.hierarchical_separation_weight < 0.0:
            raise ValueError("hierarchical_separation_weight must be non-negative.")
        if self._resolve_structure_selection_metric() not in {
            "f1_anomalous",
            "balanced_accuracy",
            "gmean",
            "f_beta_anomalous",
            "constrained_f1_anomalous",
        }:
            raise ValueError(
                "structure_selection_metric must be one of "
                "'f1_anomalous', 'balanced_accuracy', 'gmean', 'f_beta_anomalous', or 'constrained_f1_anomalous'."
            )
        if self._resolve_structure_selection_beta() <= 0.0:
            raise ValueError("structure_selection_beta must be positive.")
        min_acceptance = self._resolve_structure_selection_min_normal_acceptance()
        if min_acceptance is not None and not 0.0 <= min_acceptance <= 1.0:
            raise ValueError("structure_selection_min_normal_acceptance must be between 0 and 1 when provided.")
        if self.structure_selection_robustness_weight < 0.0:
            raise ValueError("structure_selection_robustness_weight must be non-negative.")
        if self.structure_selection_robustness_tolerance < 0.0:
            raise ValueError("structure_selection_robustness_tolerance must be non-negative.")
        if self.density_lambda < 0.0:
            raise ValueError("density_lambda must be non-negative.")
        if self.density_k < 1:
            raise ValueError("density_k must be at least 1.")
        if self.density_reference_cap < 1:
            raise ValueError("density_reference_cap must be at least 1.")
