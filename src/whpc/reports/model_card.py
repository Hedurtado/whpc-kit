from __future__ import annotations

from typing import Any


def build_model_card(
    model: Any,
    *,
    task_type: str | None = None,
    data_summary: dict[str, Any] | None = None,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    params = model.get_params(deep=False) if hasattr(model, "get_params") else {}
    fitted = any(name.endswith("_") for name in vars(model))
    return {
        "model_name": model.__class__.__name__,
        "task_type": task_type or "unspecified",
        "fitted": bool(fitted),
        "parameters": params,
        "data_summary": data_summary or {},
        "notes": notes or [],
    }


def render_model_card_markdown(card: dict[str, Any]) -> str:
    lines = [
        f"# Model Card: {card['model_name']}",
        "",
        f"- task_type: {card['task_type']}",
        f"- fitted: {card['fitted']}",
        "",
        "## Parameters",
        "",
    ]
    parameters = card.get("parameters", {})
    if parameters:
        for key, value in sorted(parameters.items()):
            lines.append(f"- {key}: {value}")
    else:
        lines.append("- none")
    lines.extend(["", "## Notes", ""])
    notes = card.get("notes", [])
    if notes:
        for note in notes:
            lines.append(f"- {note}")
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)
