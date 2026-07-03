from __future__ import annotations

import numpy as np

from whpc import MMWHPCClassifier, OCMMWHPCDetector, WHPCClassifier


def make_supervised_data() -> tuple[np.ndarray, np.ndarray]:
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


def make_one_class_data() -> tuple[np.ndarray, np.ndarray]:
    X_train_normal = np.array(
        [
            [2.0, 0.0],
            [3.0, 0.0],
            [-2.0, 0.0],
            [-3.0, 0.0],
        ]
    )
    X_eval = np.array(
        [
            [2.5, 0.0],
            [-2.5, 0.0],
            [0.0, 2.0],
            [0.0, -2.0],
        ]
    )
    return X_train_normal, X_eval


def main() -> None:
    X, y = make_supervised_data()

    whpc = WHPCClassifier()
    whpc.fit(X, y)
    print("WHPC predictions:", whpc.predict(X).tolist())

    mm_whpc = MMWHPCClassifier(n_representatives_per_class=2, random_state=0)
    mm_whpc.fit(X, y)
    print("MM-WHPC predictions:", mm_whpc.predict(X).tolist())

    X_train_normal, X_eval = make_one_class_data()
    oc_detector = OCMMWHPCDetector(n_normal_representatives=2, normalize=True, random_state=0)
    oc_detector.fit(X_train_normal)
    print("OC-MM-WHPC scores:", np.round(oc_detector.score_samples(X_eval), 4).tolist())


if __name__ == "__main__":
    main()
