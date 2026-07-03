from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from whpc import (
    OCMMWHPCDetector,
    add_gaussian_noise,
    explain_ocmmwhpc_decisions,
    explanation_jaccard_summary,
    explanation_summary,
    parse_top_feature_indices,
    robustness_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke reproducibility entry point for FP5/M5.")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--output", type=Path, default=Path("reproducibility/artifacts/fp5_smoke_summary.json"))
    return parser.parse_args()


def make_normal_only_bimodal_dataset() -> np.ndarray:
    return np.array(
        [
            [2.0, 0.0],
            [3.0, 0.0],
            [-2.0, 0.0],
            [-3.0, 0.0],
        ]
    )


def make_validation_dataset() -> tuple[np.ndarray, np.ndarray]:
    X_val = np.array(
        [
            [2.5, 0.0],
            [-2.5, 0.0],
            [0.0, 2.0],
            [0.0, -2.0],
        ]
    )
    y_val = np.array([0, 0, 1, 1])
    return X_val, y_val


def main() -> None:
    args = parse_args()
    if args.mode == "full":
        raise NotImplementedError("Full FP5 reproduction is not implemented yet in whpc-kit.")

    X_train_normal = make_normal_only_bimodal_dataset()
    X_eval, y_true = make_validation_dataset()
    feature_names = ["feature_0", "feature_1"]

    detector = OCMMWHPCDetector(n_normal_representatives=2, normalize=True, random_state=0)
    detector.fit(X_train_normal)
    detector.calibrate_threshold(X_eval, y_true)

    clean_scores = detector.score_samples(X_eval)
    clean_pred = (detector.predict(X_eval) == detector.anomaly_label).astype(int)
    clean_explanations = explain_ocmmwhpc_decisions(
        detector,
        X_eval,
        y_true=y_true,
        feature_names=feature_names,
        top_k_features=2,
    )
    explanation_diag = explanation_summary(clean_explanations)
    top_indices = parse_top_feature_indices(clean_explanations, feature_names, column="top_abs_features")

    rng = np.random.default_rng(0)
    X_perturbed = add_gaussian_noise(X_eval, std=0.05, rng=rng)
    perturbed_scores = detector.score_samples(X_perturbed)
    perturbed_pred = (detector.predict(X_perturbed) == detector.anomaly_label).astype(int)
    perturbed_explanations = explain_ocmmwhpc_decisions(
        detector,
        X_perturbed,
        y_true=y_true,
        feature_names=feature_names,
        top_k_features=2,
    )
    robustness_diag = robustness_metrics(
        y_true,
        clean_scores,
        clean_pred,
        perturbed_scores,
        perturbed_pred,
        threshold=float(detector.threshold_),
    )
    jaccard_diag = explanation_jaccard_summary(clean_explanations, perturbed_explanations)

    summary = {
        "mode": "smoke",
        "n_explained": explanation_diag["n_explained"],
        "mean_absolute_threshold_margin": explanation_diag["mean_absolute_threshold_margin"],
        "prediction_flip_rate": robustness_diag["prediction_flip_rate"],
        "mean_abs_score_shift": robustness_diag["mean_abs_score_shift"],
        "mean_topk_feature_jaccard": jaccard_diag["mean_topk_feature_jaccard"],
        "parsed_top_feature_lengths": [len(indices) for indices in top_indices],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
