from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WHPCSelector:
    """Heuristic advisory selector for the W-HPC family.

    The selector is intentionally lightweight in `v0.1`. It helps map a
    scenario description to a likely model family, but it is not a validated
    auto-ML or model-selection procedure and must not replace dataset-specific
    empirical evaluation.
    """

    labels_available: bool = True
    normal_only: bool = False
    streaming: bool = False
    drift_expected: bool = False
    unknown_attacks: bool = False
    need_explainability: bool = False
    need_robustness: bool = False
    internal_variability_high: bool = False
    adaptive_structure: bool = False
    human_analysts_available: bool = False

    def recommend_model_name(self) -> str:
        if self.streaming and self.drift_expected:
            return "DriftAwareWHPCDetector"
        if self.streaming:
            return "OnlineWHPCDetector"
        if self.normal_only and self.adaptive_structure:
            return "AdaptiveOCMMWHPCDetector"
        if self.normal_only:
            return "OCMMWHPCDetector"
        if self.unknown_attacks and self.human_analysts_available:
            return "OpenWorldWHPCDetector + ActiveQueryWHPCDetector"
        if self.unknown_attacks:
            return "OpenWorldWHPCDetector"
        if self.internal_variability_high:
            return "MMWHPCClassifier"
        return "WHPCClassifier"

    def reasons(self) -> list[str]:
        reasons: list[str] = []
        if self.normal_only:
            reasons.append("Only normal data is available for training.")
        if self.adaptive_structure:
            reasons.append("Automatic structure selection is desirable.")
        if self.streaming:
            reasons.append("Data arrives continuously.")
        if self.drift_expected:
            reasons.append("Distribution drift is expected.")
        if self.unknown_attacks:
            reasons.append("Unknown attacks may appear.")
        if self.human_analysts_available:
            reasons.append("Human analysts are available for selective feedback.")
        if self.internal_variability_high:
            reasons.append("Known classes have high internal variability.")
        if self.need_explainability:
            reasons.append("Explanations are required for audit or review.")
        if self.need_robustness:
            reasons.append("Robustness analysis is required.")
        if not self.labels_available:
            reasons.append("Standard supervised labels are not available.")
        if not reasons:
            reasons.append("A lightweight supervised baseline is appropriate.")
        return reasons

    def recommend(self) -> dict[str, object]:
        return {
            "advisory": True,
            "recommended_model": self.recommend_model_name(),
            "reasons": self.reasons(),
            "validation_note": "Use this recommendation as a starting point and confirm it with dataset-specific validation.",
        }
