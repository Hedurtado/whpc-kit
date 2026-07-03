import numpy as np

from whpc import (
    AdaptiveOCMMWHPCClassifier,
    AdaptiveOCMMWHPCDetector,
    MMWHPCClassifier,
    OCMMWHPCClassifier,
    OCMMWHPCDetector,
    OpenWorldOCMMWHPCClassifier,
    OpenWorldWHPCDetector,
    WHPCClassifier,
)


def make_axis_dataset():
    X = np.array(
        [
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [0.0, 1.0],
            [0.0, 2.0],
            [0.0, 3.0],
        ]
    )
    y = np.array([0, 0, 0, 1, 1, 1])
    return X, y


def make_normal_only_bimodal_dataset():
    return np.array(
        [
            [2.0, 0.0],
            [3.0, 0.0],
            [-2.0, 0.0],
            [-3.0, 0.0],
        ]
    )


def test_supervised_whpc_predicts_training_geometry():
    X, y = make_axis_dataset()
    clf = WHPCClassifier()
    clf.fit(X, y)

    assert np.array_equal(clf.predict(X), y)
    assert clf.predict_scores(X[:2]).shape == (2, 2)


def test_multimodal_whpc_predicts_training_geometry():
    X, y = make_axis_dataset()
    clf = MMWHPCClassifier(n_representatives_per_class=2, random_state=0)
    clf.fit(X, y)

    assert np.array_equal(clf.predict(X), y)
    assert clf.predict_scores(X[:2]).shape == (2, 2)


def test_one_class_aliases_are_backwards_compatible():
    assert OCMMWHPCDetector is OCMMWHPCClassifier
    assert AdaptiveOCMMWHPCDetector is AdaptiveOCMMWHPCClassifier
    assert OpenWorldWHPCDetector is OpenWorldOCMMWHPCClassifier


def test_one_class_detector_scores_samples():
    X_normal = make_normal_only_bimodal_dataset()
    detector = OCMMWHPCDetector(n_normal_representatives=2, normalize=True, random_state=0)
    detector.fit(X_normal)

    scores = detector.score_samples(np.array([[4.0, 0.0], [0.0, 4.0], [-4.0, 0.0]]))

    assert scores.shape == (3,)
    assert scores[0] > scores[1]
    assert scores[2] > scores[1]
