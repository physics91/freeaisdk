"""Parse OpenAI-style ``x-ratelimit-*`` response headers into RateSnapshot values."""

from __future__ import annotations

import re
import time
from collections.abc import Mapping

from .._http import Headers, parse_reset
from ..limits import LimitScope, RateSnapshot, Unit, Window

_LIMIT_HEADER_RE = re.compile(
    r"^x-ratelimit-(limit|remaining|reset)-(requests|tokens)(?:-(minute|hour|day|month))?$"
)
_WINDOWS: dict[str, Window] = {"minute": "minute", "hour": "hour", "day": "day", "month": "month"}


def _as_window(name: str | None) -> Window | None:
    return _WINDOWS.get(name) if name else None


def _number(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def rate_snapshots_from_headers(
    headers: Mapping[str, str],
    header_windows: Mapping[str, Window],
    header_scope: LimitScope,
) -> list[RateSnapshot]:
    """Limit state from ``x-ratelimit-{limit,remaining,reset}-{requests,tokens}[-window]`` headers."""
    found: dict[tuple[Unit, Window], dict[str, float]] = {}
    wall = time.time()
    for name, value in Headers(headers).items():
        match = _LIMIT_HEADER_RE.match(name.lower())
        if not match:
            continue
        kind, unit_name, window_name = match.groups()
        unit: Unit = "requests" if unit_name == "requests" else "tokens"
        window = _as_window(window_name) or header_windows.get(unit)
        if window is None:
            continue
        number = parse_reset(value.strip(), wall) if kind == "reset" else _number(value)
        if number is not None:
            found.setdefault((unit, window), {})[kind] = number
    return [
        RateSnapshot(
            window, unit, values["limit"], max(0.0, values["remaining"]), header_scope, values.get("reset")
        )
        for (unit, window), values in found.items()
        if values.get("limit", 0) > 0 and "remaining" in values
    ]
