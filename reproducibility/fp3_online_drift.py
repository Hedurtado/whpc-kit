from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from whpc import DriftAlarmConfig, compute_block_drift_signals, evaluate_drift_alarm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke reproducibility entry point for FP3/M3.")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--output", type=Path, default=Path("reproducibility/artifacts/fp3_smoke_summary.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "full":
        raise NotImplementedError("Full FP3 reproduction is not implemented yet in whpc-kit.")

    signals = compute_block_drift_signals(
        normality_scores=np.array([0.80, 0.72, 0.50, 0.49, 0.20]),
        y_pred_anomalous=np.array([0, 0, 1, 1, 1]),
        accepted_update_mask=np.array([True, True, False, False, False]),
        threshold=0.50,
        reference_score_mean=0.70,
        borderline_margin=0.02,
    )
    alarm = evaluate_drift_alarm(
        signals,
        DriftAlarmConfig(
            anomaly_rate_threshold=0.50,
            score_drop_threshold=0.10,
            borderline_rate_threshold=0.20,
            accepted_update_floor=0.45,
            min_active_conditions=2,
        ),
    )
    summary = {
        "mode": "smoke",
        "mean_normality_score": signals.mean_normality_score,
        "score_drop": signals.score_drop,
        "predicted_anomaly_rate": signals.predicted_anomaly_rate,
        "borderline_rate": signals.borderline_rate,
        "accepted_update_rate": signals.accepted_update_rate,
        "is_alarm": alarm.is_alarm,
        "active_conditions": alarm.active_conditions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
