import numpy as np

from whpc.core.weights import (
    angular_separation_sample_weights_by_class,
    center_boundary_sample_weights,
    intra_class_core_sample_weights,
    local_typicality_margin_sample_weights,
    mix_with_uniform_sample_weights,
    normalize_sample_weights,
    optimized_margin_sample_weights,
    uniform_sample_weights,
)


def test_uniform_weights_sum_to_one():
    weights = uniform_sample_weights(4)
    assert np.all(weights >= 0.0)
    assert np.isclose(np.sum(weights), 1.0)


def test_normalize_sample_weights_preserves_convex_constraint():
    weights = normalize_sample_weights(np.array([1.0, 3.0, 6.0]))
    assert np.all(weights >= 0.0)
    assert np.isclose(np.sum(weights), 1.0)
    assert np.allclose(weights, np.array([0.1, 0.3, 0.6]))


def test_optimized_margin_weights_preserve_convex_constraint():
    X_class = np.array([[1.0, 0.0], [3.0, 0.0]])
    X_other = np.array([[0.0, 1.0], [0.0, 3.0]])
    weights = optimized_margin_sample_weights(X_class, X_other, temperature=1.0)
    assert np.all(weights >= 0.0)
    assert np.isclose(np.sum(weights), 1.0)


def test_optimized_margin_weights_downweight_internal_outlier():
    X_class = np.array([[1.0, 0.0], [1.0, 0.1], [0.0, 1.0]])
    X_other = np.array([[-1.0, 0.0], [-1.0, -0.1]])
    weights = optimized_margin_sample_weights(X_class, X_other, temperature=1.0)
    assert weights[2] < weights[0]
    assert weights[2] < weights[1]


def test_intra_class_core_weights_downweight_internal_outlier():
    X_class = np.array([[1.0, 0.0], [1.0, 0.1], [0.0, 1.0]])
    weights = intra_class_core_sample_weights(X_class, temperature=1.0)
    assert np.isclose(np.sum(weights), 1.0)
    assert weights[2] < weights[0]
    assert weights[2] < weights[1]


def test_mix_with_uniform_alpha_endpoints():
    weights = np.array([0.7, 0.2, 0.1])
    assert np.allclose(mix_with_uniform_sample_weights(weights, alpha=0.0), np.full(3, 1.0 / 3.0))
    assert np.allclose(mix_with_uniform_sample_weights(weights, alpha=1.0), weights)


def test_center_boundary_gamma_zero_matches_intra_class_core():
    X_class = np.array([[1.0, 0.0], [1.0, 0.1], [0.7, 0.4]])
    X_other = np.array([[0.0, 1.0], [0.1, 1.0], [0.2, 0.9]])
    center_boundary_weights = center_boundary_sample_weights(
        X_class,
        X_other,
        temperature=1.0,
        boundary_gamma=0.0,
        boundary_scale=0.5,
    )
    intra_class_weights = intra_class_core_sample_weights(X_class, temperature=1.0)
    assert np.allclose(center_boundary_weights, intra_class_weights)


def test_angular_separation_lambda_zero_matches_center_boundary():
    X_by_class = [
        np.array([[1.0, 0.0], [1.0, 0.1], [0.7, 0.4]]),
        np.array([[0.0, 1.0], [0.1, 1.0], [0.2, 0.9]]),
    ]
    weights_by_class = angular_separation_sample_weights_by_class(
        X_by_class,
        temperature=1.0,
        boundary_gamma=1.0,
        boundary_scale=0.5,
        separation_lambda=0.0,
        n_iter=2,
    )
    expected_class_zero = center_boundary_sample_weights(
        X_by_class[0],
        X_by_class[1],
        temperature=1.0,
        boundary_gamma=1.0,
        boundary_scale=0.5,
    )
    assert np.allclose(weights_by_class[0], expected_class_zero)


def test_local_typicality_margin_weights_preserve_convex_constraint():
    X_class = np.array([[1.0, 0.0], [1.0, 0.1], [0.7, 0.4]])
    X_other = np.array([[0.0, 1.0], [0.1, 1.0], [0.2, 0.9]])
    weights = local_typicality_margin_sample_weights(
        X_class,
        X_other,
        temperature=1.0,
        typicality_k=2,
        typicality_reference_cap=3,
        compactness_gamma=1.0,
        separation_lambda=1.0,
        random_state=0,
    )
    assert np.all(weights >= 0.0)
    assert np.isclose(np.sum(weights), 1.0)
