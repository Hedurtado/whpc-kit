import numpy as np

from whpc.core.scores import cosine_score, inner_product_score, margin_score


def test_inner_product_score_for_aligned_vector():
    assert np.isclose(inner_product_score(np.array([1.0, 0.0]), np.array([2.0, 0.0])), 2.0)


def test_cosine_score_for_orthogonal_vector():
    assert np.isclose(cosine_score(np.array([1.0, 0.0]), np.array([0.0, 2.0])), 0.0)


def test_cosine_score_for_same_direction():
    assert np.isclose(cosine_score(np.array([1.0, 1.0]), np.array([2.0, 2.0])), 1.0)


def test_margin_score_from_score_matrix():
    scores = np.array([[0.9, 0.2, 0.1], [0.7, 0.68, 0.1]])
    margins = margin_score(scores)
    assert np.allclose(margins, np.array([0.7, 0.02]))
