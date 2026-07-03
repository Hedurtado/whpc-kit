from .classifier import WHPCClassifier
from .datasets import LoadedDataset, load_cic_ids2017, load_nsl_kdd, load_unsw_nb15
from .drift_monitoring import (
    DriftAlarm,
    DriftAlarmConfig,
    DriftSignals,
    compute_block_drift_signals,
    evaluate_drift_alarm,
)
from .explainability import (
    explain_ocmmwhpc_decisions,
    explanation_summary,
    summarize_ocmmwhpc_representatives,
)
from .feature_weights import (
    FEATURE_WEIGHT_STRATEGIES,
    compute_feature_weights,
    correlation_feature_weights,
    identity_feature_weights,
    mean_activation_feature_weights,
    mutual_information_feature_weights,
    normalize_feature_weights,
    variance_feature_weights,
)
from .multimodal_classifier import MMWHPCClassifier
from .one_class_adaptive import AdaptiveOCMMWHPCClassifier
from .one_class_multimodal import OCMMWHPCClassifier
from .open_world import OpenWorldOCMMWHPCClassifier, compute_open_world_metrics
from .querying import (
    accepted_risk_candidate_mask,
    append_with_cap,
    apply_oracle_feedback,
    compute_accepted_risk_query_scores,
    compute_feedback_metrics,
    compute_query_scores,
    query_candidate_mask,
    select_dual_top_k_queries,
    select_random_queries,
    select_top_k_queries,
    split_queried_feedback,
)
from .robustness import (
    add_gaussian_noise,
    directional_feature_shift,
    explanation_jaccard_summary,
    parse_top_feature_indices,
    random_feature_mask,
    robustness_metrics,
    top_feature_mask,
)
from .reports import build_model_card, render_model_card_markdown
from .selection import WHPCSelector
from .open_set import WHPCOpenSetClassifier
from .preprocessing import (
    PreprocessingSpec,
    make_cic_ids2017_preprocessor,
    make_nsl_kdd_preprocessor,
    make_unsw_preprocessor,
)

# Public detector aliases for the standalone kit API. The original research
# names remain exported for backwards compatibility.
OCMMWHPCDetector = OCMMWHPCClassifier
AdaptiveOCMMWHPCDetector = AdaptiveOCMMWHPCClassifier
OpenWorldWHPCDetector = OpenWorldOCMMWHPCClassifier

__all__ = [
    "LoadedDataset",
    "MMWHPCClassifier",
    "AdaptiveOCMMWHPCClassifier",
    "AdaptiveOCMMWHPCDetector",
    "OCMMWHPCClassifier",
    "OCMMWHPCDetector",
    "OpenWorldOCMMWHPCClassifier",
    "OpenWorldWHPCDetector",
    "DriftAlarm",
    "DriftAlarmConfig",
    "DriftSignals",
    "FEATURE_WEIGHT_STRATEGIES",
    "PreprocessingSpec",
    "WHPCClassifier",
    "WHPCOpenSetClassifier",
    "WHPCSelector",
    "build_model_card",
    "compute_block_drift_signals",
    "explain_ocmmwhpc_decisions",
    "explanation_summary",
    "compute_feature_weights",
    "compute_open_world_metrics",
    "accepted_risk_candidate_mask",
    "append_with_cap",
    "apply_oracle_feedback",
    "compute_accepted_risk_query_scores",
    "compute_feedback_metrics",
    "compute_query_scores",
    "query_candidate_mask",
    "select_dual_top_k_queries",
    "select_random_queries",
    "select_top_k_queries",
    "split_queried_feedback",
    "summarize_ocmmwhpc_representatives",
    "add_gaussian_noise",
    "directional_feature_shift",
    "explanation_jaccard_summary",
    "parse_top_feature_indices",
    "random_feature_mask",
    "robustness_metrics",
    "top_feature_mask",
    "correlation_feature_weights",
    "evaluate_drift_alarm",
    "identity_feature_weights",
    "load_cic_ids2017",
    "load_nsl_kdd",
    "load_unsw_nb15",
    "make_cic_ids2017_preprocessor",
    "make_nsl_kdd_preprocessor",
    "make_unsw_preprocessor",
    "mean_activation_feature_weights",
    "mutual_information_feature_weights",
    "normalize_feature_weights",
    "variance_feature_weights",
    "render_model_card_markdown",
]
