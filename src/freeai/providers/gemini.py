"""Gemini free-tier provider."""

from __future__ import annotations

import re
from collections.abc import Mapping
from functools import lru_cache
from typing import Any

from .._http import Headers, seconds_until_midnight
from ..errors import FailureKind, ProviderFailure, Scope
from ..limits import rpd, rpm, tpm
from ..types import ModelInfo
from ._catalog import _catalog_items
from ._impl import _VERIFIED, Provider, Tier, Verdict

GEMINI_FREE_MODELS = (
    "gemini-3.8-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3-flash-preview",
)
_GEMINI_SOURCE = "scriptbyai.com/gemini-api-free-tier-limits (third party; Google publishes no free limits)"
_GEMINI_FLASH = ("gemini-*-flash", "gemini-*-flash-preview")
_GEMINI_FLASH_LITE = ("gemini-*-flash-lite", "gemini-*-flash-lite-preview")
_GEMINI_VERSION_RE = re.compile(r"gemini-(\d+)(?:\.(\d+))?")


@lru_cache(maxsize=64)
def _gemini_ignores_sampling(model_id: str) -> bool:
    match = _GEMINI_VERSION_RE.search(model_id)
    if not match:
        return False
    return (int(match.group(1)), int(match.group(2) or 0)) >= (3, 6)


def _gemini_quota_ids(error: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for detail in error.get("details") or ():
        if isinstance(detail, dict) and str(detail.get("@type", "")).endswith("QuotaFailure"):
            ids.extend(str(v.get("quotaId", "")) for v in detail.get("violations") or () if isinstance(v, dict))
    return ids


class Gemini(Provider):
    """Gemini API free tier: per project and per model, daily reset at midnight Pacific time.

    Google no longer publishes the free numbers, so the declared limits come from a third-party copy
    of AI Studio's table: Flash 5 RPM and 20 RPD, Flash-Lite 10 RPM and 500 RPD, 250K TPM each
    (preview models assumed to match Flash). Without them every limit is only learned from a 429.
    A billing-enabled project has higher limits: pass ``limits=[]`` or your own.

    Free usage may be used to improve Google products, and services offered to users in the EEA,
    Switzerland or the UK must use the paid tier.
    """

    tier: Tier = "free"
    trains_on_prompts = True
    key_env = "GEMINI_API_KEY"
    reset_tz = "America/Los_Angeles"
    reset_headers = ()
    default_models = GEMINI_FREE_MODELS
    default_limits = (
        rpm(5, models=_GEMINI_FLASH, source=_GEMINI_SOURCE, last_verified=_VERIFIED),
        rpd(20, models=_GEMINI_FLASH, source=_GEMINI_SOURCE, last_verified=_VERIFIED),
        rpm(10, models=_GEMINI_FLASH_LITE, source=_GEMINI_SOURCE, last_verified=_VERIFIED),
        rpd(500, models=_GEMINI_FLASH_LITE, source=_GEMINI_SOURCE, last_verified=_VERIFIED),
        tpm(250_000, models=(*_GEMINI_FLASH, *_GEMINI_FLASH_LITE), source=_GEMINI_SOURCE, last_verified=_VERIFIED),
    )

    def __init__(
        self,
        api_key: str | None = None,
        *,
        name: str = "gemini",
        base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai",
        **kwargs: Any,
    ) -> None:
        super().__init__(name, base_url, api_key=api_key, **kwargs)

    def _parse_catalog(self, body: Any) -> list[ModelInfo]:
        ids = [str(item["id"]).removeprefix("models/") for item in _catalog_items(body) if item.get("id")]
        return [self._as_model(model_id) for model_id in ids]

    def prepare_payload(self, model_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        body = super().prepare_payload(model_id, payload)
        if _gemini_ignores_sampling(model_id):
            for name in ("temperature", "top_p", "top_k"):
                body.pop(name, None)  # ignored by 3.6+, announced to become a 400
        return body

    def classify(self, status: int, headers: Mapping[str, str], body: Any) -> ProviderFailure:
        failure = super().classify(status, headers, body)
        if failure.kind is FailureKind.QUOTA_EXHAUSTED:
            # Per-day quota: RetryInfo may suggest seconds, but the counter only resets at midnight PT.
            failure.retry_after = max(failure.retry_after or 0.0, seconds_until_midnight(self.reset_tz))
        return failure

    def _verdict(self, code: int, headers: Headers, error: dict[str, Any], text: str) -> Verdict:
        if code == 403 and ("free_tier_verification_required" in text or error.get("status") == "PERMISSION_DENIED"):
            return FailureKind.AUTH, Scope.PROVIDER  # project-level: a new key in the same project won't help
        return super()._verdict(code, headers, error, text)

    def _rate_limit_verdict(self, headers: Headers, error: dict[str, Any], text: str) -> Verdict:
        quota_ids = " ".join(_gemini_quota_ids(error)).lower()
        if "perday" in quota_ids:
            return FailureKind.QUOTA_EXHAUSTED, Scope.MODEL
        if quota_ids:
            return FailureKind.RATE_LIMIT, Scope.MODEL
        return super()._rate_limit_verdict(headers, error, text)
