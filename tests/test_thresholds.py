import numpy as np

from whpc.core.thresholds import apply_threshold, fit_threshold


def test_quantile_threshold_and_application():
    scores = np.array([0.2, 0.4, 0.8, 0.9])
    threshold = fit_threshold(scores, strategy="quantile", quantile=0.25)
    accepted = apply_threshold(np.array([0.1, 0.5, 0.95]), threshold)
    assert np.isclose(threshold, np.quantile(scores, 0.25))
    assert np.array_equal(accepted, np.array([False, True, True]))
