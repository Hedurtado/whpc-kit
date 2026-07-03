import numpy as np
import pytest

from whpc.core.representatives import build_class_representative


def test_representative_matches_uniform_average():
    X_class = np.array([[1.0, 0.0], [3.0, 0.0]])
    representative = build_class_representative(X_class, normalize=False)
    assert np.allclose(representative, np.array([2.0, 0.0]))


def test_representative_changes_with_non_uniform_weights():
    X_class = np.array([[1.0, 0.0], [3.0, 0.0]])
    representative = build_class_representative(X_class, weights=np.array([0.25, 0.75]), normalize=False)
    assert np.allclose(representative, np.array([2.5, 0.0]))


def test_weighted_medoid_returns_existing_sample():
    X_class = np.array([[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0]])
    representative = build_class_representative(
        X_class,
        weights=np.array([0.45, 0.45, 0.10]),
        representative_strategy="weighted_medoid",
        normalize=False,
    )
    assert any(np.allclose(representative, sample) for sample in X_class)
    assert any(np.allclose(representative, sample) for sample in X_class[:2])


def test_max_weight_sample_returns_heaviest_sample():
    X_class = np.array([[1.0, 0.0], [3.0, 0.0], [2.0, 1.0]])
    representative = build_class_representative(
        X_class,
        weights=np.array([0.2, 0.1, 0.7]),
        representative_strategy="max_weight_sample",
        normalize=False,
    )
    assert np.allclose(representative, np.array([2.0, 1.0]))


def test_invalid_representative_strategy_raises():
    X_class = np.array([[1.0, 0.0], [3.0, 0.0]])
    with pytest.raises(ValueError, match="representative_strategy"):
        build_class_representative(X_class, representative_strategy="invalid")
