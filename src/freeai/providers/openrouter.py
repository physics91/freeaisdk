"""OpenRouter free-tier provider."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from .._http import Headers, embedded_header_names
from ..errors import FailureKind, Scope
from ..limits import RateSnapshot
from ..types import ModelInfo
from ._catalog import _catalog_items, _catalog_model, _is_zero_priced
from ._classify import _metadata
from ._impl import Provider, Tier, Verdict

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_PLATFORM_LIMIT_HEADERS = frozenset({"x-ratelimit-limit", "x-ratelimit-remaining", "x-ratelimit-reset"})
_STRICT_PARAMS = ("tools", "tool_choice", "response_format")


class OpenRouter(Provider):
    """OpenRouter with automatic discovery of free models.

    The free allowance is per account: 20 requests/minute and 50 or 1000 requests/day (shared by
    every ``:free`` model, failed attempts included). ``rpm`` defaults to 20 and the daily budget is
    synced from ``GET /key`` on first use and after a daily-limit 429.
    """

    tier: Tier = "free"
    key_env = "OPENROUTER_API_KEY"
    can_deny_training = True

    def __init__(
        self,
        api_key: str | None = None,
        *,
        include_routers: bool = True,
        include_zero_priced: bool = False,
        app_name: str | None = None,
        app_url: str | None = None,
        rpm: int | None = 20,
        name: str = "openrouter",
        base_url: str = OPENROUTER_BASE_URL,
        headers: Mapping[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        merged = dict(headers or {})
        if app_url:
            merged["HTTP-Referer"] = app_url
        if app_name:
            merged["X-Title"] = app_name
        self.include_routers = include_routers
        self.include_zero_priced = include_zero_priced
        super().__init__(name, base_url, api_key=api_key, rpm=rpm, headers=merged, **kwargs)

    def _catalog_paths(self) -> tuple[str, ...]:
        # /models/user already drops models the account's privacy settings would reject (404).
        return ("/models/user", "/models") if self.api_key else ("/models",)

    def _parse_catalog(self, body: Any) -> list[ModelInfo]:
        today = datetime.now(UTC).date().isoformat()
        models = [m for item in _catalog_items(body) if (m := self._free_model(item, today))]
        return sorted(models, key=lambda m: m.is_router)  # stable: keeps API order, routers last

    def _free_model(self, item: dict[str, Any], today: str) -> ModelInfo | None:
        if not isinstance(item.get("id"), str):
            return None
        model_id: str = item["id"]
        architecture = item.get("architecture") or {}
        if set(architecture.get("output_modalities") or ["text"]) != {"text"}:
            return None  # audio/image generators are not chat models
        expires = item.get("expiration_date")
        if isinstance(expires, str) and expires < today:
            return None
        endpoints = item.get("endpoints")
        if isinstance(endpoints, list) and endpoints and "/api/v1/chat/completions" not in endpoints:
            return None
        if not _is_zero_priced(item.get("pricing")):
            return None  # never risk a billed request, even for a ":free" id
        is_router = architecture.get("tokenizer") == "Router" or model_id.startswith("openrouter/")
        if is_router:
            if not self.include_routers:
                return None
        elif not (model_id.endswith(":free") or self.include_zero_priced):
            return None
        return _catalog_model(self.name, item, is_router=is_router)

    def prepare_payload(self, model_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        body = super().prepare_payload(model_id, payload)
        if any(body.get(name) is not None for name in _STRICT_PARAMS):
            # Without this OpenRouter may route to an endpoint that silently ignores tools/JSON mode.
            body["provider"] = {**(body.get("provider") or {}), "require_parameters": True}
        return body

    def deny_training(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        body = dict(payload)
        body["provider"] = {**(body.get("provider") or {}), "data_collection": "deny"}
        return body

    def _verdict(self, code: int, headers: Headers, error: dict[str, Any], text: str) -> Verdict:
        metadata = _metadata(error)
        if code == 402 and metadata.get("limit_source") == "openrouter_in_flight_budget":
            return FailureKind.PROVIDER_RATE_LIMIT, Scope.PROVIDER
        return super()._verdict(code, headers, error, text)

    def _rate_limit_verdict(self, headers: Headers, error: dict[str, Any], text: str) -> Verdict:
        # Message prefixes come first: account-limit samples also carry a provider_name.
        if "free-models-per-day" in text:
            return FailureKind.QUOTA_EXHAUSTED, Scope.PROVIDER
        if "free-models-per-" in text:
            return FailureKind.PROVIDER_RATE_LIMIT, Scope.PROVIDER
        if "limit_rpd/" in text:
            return FailureKind.QUOTA_EXHAUSTED, Scope.MODEL
        if "limit_rpm/" in text or "rate-limited upstream" in text:
            return FailureKind.RATE_LIMIT, Scope.MODEL
        metadata = _metadata(error)
        if metadata.get("provider_name") or metadata.get("provider_code") is not None:
            return FailureKind.RATE_LIMIT, Scope.MODEL  # upstream provider limit: another model may work
        names = {name.lower() for name in headers} | embedded_header_names(error)
        if names & _PLATFORM_LIMIT_HEADERS or not error:
            # Platform limit, or a 429 page without an OpenRouter error object (edge/DDoS).
            return FailureKind.PROVIDER_RATE_LIMIT, Scope.PROVIDER
        return FailureKind.RATE_LIMIT, Scope.MODEL

    def rate_snapshots(self, headers: Mapping[str, str]) -> list[RateSnapshot]:
        return []  # X-RateLimit-* carry no window; the daily state comes from GET /key

    def _quota_path(self) -> str | None:
        return "/key" if self.api_key else None

    def _parse_quota(self, body: Any) -> list[RateSnapshot]:
        data = body.get("data", body) if isinstance(body, dict) else {}
        daily = data.get("free_model_daily_requests") if isinstance(data, dict) else None
        if not isinstance(daily, dict) or not isinstance(daily.get("used"), int):
            return []
        limit = daily.get("limit")
        if not isinstance(limit, int) or limit <= 0:
            return []
        return [RateSnapshot("day", "requests", limit, max(0, limit - daily["used"]), "provider")]
