"""Shared catalog parsing helpers for OpenAI-compatible /models responses."""

from __future__ import annotations

from typing import Any

from ..types import ModelInfo


def _catalog_items(body: Any) -> list[dict[str, Any]]:
    items = body.get("data") if isinstance(body, dict) else body
    if not isinstance(items, list):
        raise ValueError("unexpected /models response shape")
    return [item for item in items if isinstance(item, dict)]


def _as_int(value: Any) -> int | None:
    """JSON numbers may arrive as float or numeric string; bool must not count as an int.

    Accepts ``8192``, ``8192.0``, ``"8192"``, and ``"8192.0"``. Non-integral strings
    (``"1.5"``) are rejected; non-integral floats still truncate via ``int()``.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    try:
        number = float(text)
    except ValueError:
        return None
    if not number.is_integer():
        return None
    return int(number)


def _catalog_model(provider: str, item: dict[str, Any], *, is_router: bool = False) -> ModelInfo:
    """A ``ModelInfo`` from an OpenRouter-style catalog entry (shared by OpenRouter-like gateways)."""
    raw = item.get("architecture")
    architecture: dict[str, Any] = raw if isinstance(raw, dict) else {}
    parameters = item.get("supported_parameters")
    context = item.get("context_length", item.get("context_window"))
    inputs = architecture.get("input_modalities")
    return ModelInfo(
        provider=provider,
        id=str(item["id"]),
        name=item.get("name") if isinstance(item.get("name"), str) else None,
        context_length=_as_int(context),
        input_modalities=frozenset(inputs) if isinstance(inputs, list) and inputs else None,
        # Empty list means "not reported" (same as missing), not "supports nothing".
        supported_parameters=frozenset(parameters) if isinstance(parameters, list) and parameters else None,
        is_router=is_router,
    )


def _is_zero(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        return False
    try:
        return float(value) == 0
    except ValueError:
        return False


def _is_zero_priced(pricing: Any) -> bool:
    if not isinstance(pricing, dict) or "prompt" not in pricing or "completion" not in pricing:
        return False
    return all(
        _is_zero(value) for key, value in pricing.items() if key != "discount" and isinstance(value, (str, int, float))
    )
