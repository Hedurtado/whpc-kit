from __future__ import annotations

from whpc.selection import WHPCSelector


def test_selector_prefers_adaptive_one_class_when_requested():
    selector = WHPCSelector(normal_only=True, adaptive_structure=True, labels_available=False)
    recommendation = selector.recommend()
    assert recommendation["advisory"] is True
    assert recommendation["recommended_model"] == "AdaptiveOCMMWHPCDetector"
    assert "validation" in recommendation["validation_note"].lower()


def test_selector_can_recommend_open_world_plus_querying():
    selector = WHPCSelector(unknown_attacks=True, human_analysts_available=True)
    recommendation = selector.recommend()
    assert recommendation["advisory"] is True
    assert recommendation["recommended_model"] == "OpenWorldWHPCDetector + ActiveQueryWHPCDetector"
