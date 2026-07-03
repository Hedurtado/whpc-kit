from __future__ import annotations

from whpc import WHPCClassifier
from whpc.reports import build_model_card, render_model_card_markdown


def test_model_card_smoke():
    model = WHPCClassifier()
    card = build_model_card(model, task_type="classification", notes=["synthetic smoke"])
    markdown = render_model_card_markdown(card)
    assert card["model_name"] == "WHPCClassifier"
    assert card["task_type"] == "classification"
    assert "WHPCClassifier" in markdown
