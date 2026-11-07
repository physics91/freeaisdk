"""Pure helpers for mapping HTTP/error payloads to FailureKind inputs."""

from __future__ import annotations

import re
from typing import Any

from ..errors import FailureKind
from ._catalog import _as_int

_KIND_BY_STATUS = {
    400: FailureKind.BAD_REQUEST,
    401: FailureKind.AUTH,
    402: FailureKind.QUOTA,
    403: FailureKind.REFUSED,
    404: FailureKind.MODEL_GONE,
    413: FailureKind.BAD_REQUEST,
    422: FailureKind.BAD_REQUEST,
    429: FailureKind.RATE_LIMIT,
}
# Server wait hints only describe rate limits, quota resets and 503 back-pressure. Other failures
# (e.g. a Groq 500 that happens to carry the daily reset header) must not inherit them.
_HINTED_KINDS = frozenset({FailureKind.RATE_LIMIT, FailureKind.PROVIDER_RATE_LIMIT, FailureKind.QUOTA_EXHAUSTED})
_CONTEXT_RE = re.compile(
    r"context[ _-]?(?:length|window)|maximum context|too many tokens|prompt is too long|reduce the length"
)
_DAILY_RE = re.compile(r"per[ -]day|daily|\brpd\b|\btpd\b|perday")
_MONTHLY_RE = re.compile(r"per[ -]month|/ ?month|monthly")
_BAD_KEY_RE = re.compile(r"api key not valid|invalid api key|incorrect api key|invalid_api_key")
_NOT_ENTITLED_RE = re.compile(r"data policy|subscription|upgrade your plan|not available (?:on|for) your")


def _parse_code_int(code: Any) -> int | None:
    """Int/float/digit-string codes; ``None`` for bools and non-numeric labels."""
    parsed = _as_int(code)
    if parsed is None and isinstance(code, str) and code.isdigit():
        return int(code)
    return parsed


def _code_key(code: Any) -> str:
    """Normalize business/error codes so ``3036.0`` and ``"3036"`` share one lookup key."""
    parsed = _parse_code_int(code)
    if parsed is not None:
        return str(parsed)
    return "" if code is None else str(code)


def _effective_status(error: dict[str, Any], status: int) -> int:
    code = error.get("code")
    parsed = _parse_code_int(code)
    if parsed is not None and 100 <= parsed < 600:
        return parsed
    if status != 200:
        return status
    metadata = _metadata(error)
    labels = f"{code} {error.get('type')} {error.get('status')} {metadata.get('error_type')}".lower()
    if "rate_limit" in labels or "resource_exhausted" in labels:
        return 429
    return 502  # an error inside a 200 is an upstream failure


def _metadata(error: dict[str, Any]) -> dict[str, Any]:
    metadata = error.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _message(error: dict[str, Any], body: Any, code: int) -> str:
    message = error.get("message")
    if not isinstance(message, str) or not message:
        message = body.strip()[:200] if isinstance(body, str) and body.strip() else f"HTTP {code}"
    raw = _metadata(error).get("raw")
    if isinstance(raw, str) and raw and raw not in message:
        message = f"{message}: {raw}"
    return message[:500]
