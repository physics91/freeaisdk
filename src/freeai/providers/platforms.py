"""Remaining free/trial/paid platform providers (non-OpenRouter/Groq/Gemini)."""

from __future__ import annotations

import dataclasses
import os
import re
import time
from collections.abc import Mapping
from datetime import datetime
from typing import Any, ClassVar

from .._http import Headers, _zone, error_text
from ..errors import FailureKind, FreeAIError, Scope
from ..limits import RateSnapshot, Window, per_month, rpd, rph, rpm, tpd, tph, tpm
from ..types import ModelInfo
from ._catalog import _catalog_items, _catalog_model, _is_zero
from ._classify import _MONTHLY_RE
from ._impl import _VERIFIED, Provider, Tier, Verdict
from ._ratelimit import _number

CEREBRAS_TRIAL_MODELS = ("gpt-oss-120b", "qwen-3.8-27b")
# Free-allocation models, cheapest in neurons first (Llama 3.3 70B burns the daily budget fastest).
CLOUDFLARE_FREE_MODELS = (
    "@cf/qwen/qwen3-30b-a3b-fp8",
    "@cf/openai/gpt-oss-20b",
    "@cf/openai/gpt-oss-120b",
    "@cf/qwen/qwen3.8-27b",
    "@cf/meta/llama-4-scout-17b-16e-instruct",
    "@cf/mistralai/mistral-small-3.1-24b-instruct",
    "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
)
ZAI_FREE_MODELS = ("glm-4.7-flash", "glm-4.5-flash", "glm-4.6v-flash")
# IDs present in /v1/models; whether each is a "Free Endpoint" is shown only on build.nvidia.com.
NVIDIA_FREE_MODELS = (
    "nvidia/nemotron-3-super-120b-a12b",
    "openai/gpt-oss-20b",
    "deepseek-ai/deepseek-v4.1-flash",
    "z-ai/glm-5.3-flash",
    "moonshotai/kimi-k2.6",
    "google/gemma-4-31b-it",
    "nvidia/nemotron-3-ultra-550b-a55b",
)
SAMBANOVA_FREE_MODELS = ("gpt-oss-120b", "Meta-Llama-3.3-70B-Instruct", "gemma-4-31B-it")
COHERE_TRIAL_MODELS = ("command-a-plus-05-2026", "command-a-03-2025", "command-r-plus-08-2024")
# The last three were found by ``freeai audit`` and answered an anonymous request each on 2026-11-10,
# every one with its own 2 per minute.
OVH_ANONYMOUS_MODELS = (
    "Qwen3.8-27B",
    "gpt-oss-120b",
    "gpt-oss-20b",
    "Qwen3.5-397B-A17B",
    "Meta-Llama-3_3-70B-Instruct",
    "Mistral-Small-3.2-24B-Instruct-2506",
    "Qwen3.6-27B",
    "Qwen2.5-VL-72B-Instruct",
    "Qwen3.5-9B",
)
_ZAI_RESET_RE = re.compile(r"reset at (\d{4}-\d{2}-\d{2}[ t]\d{2}:\d{2}(?::\d{2})?)")


class Cloudflare(Provider):
    """Cloudflare Workers AI: 10,000 neurons/day per account (resets 00:00 UTC), 300 RPM per model.

    Needs an API token and the account id (both from the dashboard). Cloudflare states it does not
    train on Workers AI inputs. Error ``3036`` means the daily neurons are gone, ``3040`` that no
    data center has capacity.
    """

    tier: Tier = "free"
    trains_on_prompts = False
    key_env = "CLOUDFLARE_API_TOKEN"
    uncounted: Window | None = "day"  # the neurons: their cost per request depends on model and length
    default_models = CLOUDFLARE_FREE_MODELS
    default_limits = (rpm(300, source="developers.cloudflare.com/workers-ai/platform/limits", last_verified=_VERIFIED),)
    business_codes: ClassVar[Mapping[str, Verdict]] = {
        "3036": (FailureKind.QUOTA_EXHAUSTED, Scope.PROVIDER),
        "3040": (FailureKind.UNAVAILABLE, Scope.MODEL),
    }

    def __init__(
        self,
        api_key: str | None = None,
        *,
        account_id: str | None = None,
        name: str = "cloudflare",
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        account = account_id or os.environ.get("CLOUDFLARE_ACCOUNT_ID")
        if base_url is None:
            if not account:
                raise FreeAIError("cloudflare: pass account_id= or set CLOUDFLARE_ACCOUNT_ID")
            base_url = f"https://api.cloudflare.com/client/v4/accounts/{account}/ai/v1"
        super().__init__(name, base_url, api_key=api_key, **kwargs)

    def _rate_limit_verdict(self, headers: Headers, error: dict[str, Any], text: str) -> Verdict:
        if "daily free allocation" in text or "neurons" in text:
            return FailureKind.QUOTA_EXHAUSTED, Scope.PROVIDER
        return super()._rate_limit_verdict(headers, error, text)


class ZAI(Provider):
    """Z.ai (Zhipu) GLM Flash models, free of charge.

    Business codes in ``error.code`` decide the outcome: ``1302`` rate limit, ``1305`` overload,
    ``1308``/``1310`` usage limit until ``next_flush_time``, ``1313`` fair-use throttling, ``1113``
    no balance. Free models reportedly allow one concurrent request (unverified), so
    ``max_concurrency`` defaults to 1; that slot count is per process, even with a shared store.
    Whether prompts are used for training is not stated.
    """

    tier: Tier = "free"
    key_env = "ZAI_API_KEY"
    reset_tz = "Asia/Shanghai"
    uncounted: Window | None = "day"  # the usage limit behind 1308/1310; its size and window are unpublished
    default_models = ZAI_FREE_MODELS
    max_concurrency = 1
    business_codes: ClassVar[Mapping[str, Verdict]] = {
        "1302": (FailureKind.RATE_LIMIT, Scope.MODEL),
        "1305": (FailureKind.UNAVAILABLE, Scope.MODEL),
        "1308": (FailureKind.QUOTA_EXHAUSTED, Scope.PROVIDER),
        "1310": (FailureKind.QUOTA_EXHAUSTED, Scope.PROVIDER),
        "1313": (FailureKind.PROVIDER_RATE_LIMIT, Scope.PROVIDER),
        "1113": (FailureKind.QUOTA, Scope.PROVIDER),
        "1301": (FailureKind.REFUSED, None),
        "1211": (FailureKind.MODEL_GONE, None),
    }

    def __init__(
        self,
        api_key: str | None = None,
        *,
        name: str = "zai",
        base_url: str = "https://api.z.ai/api/paas/v4",
        **kwargs: Any,
    ) -> None:
        super().__init__(name, base_url, api_key=api_key, **kwargs)

    def _quota_reset_hint(self, error: dict[str, Any], text: str | None = None) -> float:
        body = text if text is not None else error_text(error)
        match = _ZAI_RESET_RE.search(body)
        if match:
            try:
                moment = datetime.fromisoformat(match.group(1).replace(" ", "T")).replace(tzinfo=_zone(self.reset_tz))
                return max(0.0, moment.timestamp() - time.time())
            except ValueError:
                pass
        return super()._quota_reset_hint(error, body)


class Kilo(Provider):
    """Kilo AI Gateway: ``isFree`` models, usable anonymously (200 requests/hour per IP) or with a key.

    The catalog reports ``mayTrainOnYourPrompts`` per model; every free model is flagged as training
    on prompts (2026-11-06), so ``allow_training=False`` excludes them.
    """

    tier: Tier = "free"
    key_env = "KILO_API_KEY"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        name: str = "kilo",
        base_url: str = "https://ai-gateway.kilo.ai/api/v1",
        include_routers: bool = True,
        **kwargs: Any,
    ) -> None:
        self.include_routers = include_routers
        super().__init__(name, base_url, api_key=api_key, keyless=True, **kwargs)
        if not self.api_key and "limits" not in kwargs:
            self.limits.append(rph(200, scope="provider", source="kilo.ai/docs/gateway", last_verified=_VERIFIED))
        elif self.api_key and "uncounted" not in kwargs:
            self.uncounted = "hour"  # a key's allowance is not published; the anonymous one is hourly

    def _parse_catalog(self, body: Any) -> list[ModelInfo]:
        models: list[ModelInfo] = []
        for item in _catalog_items(body):
            if item.get("isFree") is not True or not isinstance(item.get("id"), str):
                continue
            is_router = item["id"].startswith(("kilo-auto/", "openrouter/"))
            if is_router and not self.include_routers:
                continue
            trains = item.get("mayTrainOnYourPrompts")
            model = _catalog_model(self.name, item, is_router=is_router)
            models.append(dataclasses.replace(model, trains_on_prompts=trains if isinstance(trains, bool) else None))
        return sorted(models, key=lambda m: m.is_router)


class Requesty(Provider):
    """Requesty router, $0 plan: zero-priced models only, 200 requests/day per account."""

    tier: Tier = "free"
    key_env = "REQUESTY_API_KEY"
    default_limits = (rpd(200, scope="provider", source="requesty.ai/pricing", last_verified=_VERIFIED),)

    def __init__(
        self,
        api_key: str | None = None,
        *,
        name: str = "requesty",
        base_url: str = "https://router.requesty.ai/v1",
        **kwargs: Any,
    ) -> None:
        super().__init__(name, base_url, api_key=api_key, **kwargs)

    def _parse_catalog(self, body: Any) -> list[ModelInfo]:
        return [
            _catalog_model(self.name, item)
            for item in _catalog_items(body)
            if isinstance(item.get("id"), str)
            and _is_zero(item.get("input_price"))
            and _is_zero(item.get("output_price"))
        ]


class NVIDIA(Provider):
    """NVIDIA API catalog (build.nvidia.com) trial endpoints: about 40 RPM per account.

    Trial use is logged and used to improve NVIDIA products; do not send personal or confidential
    data. Which catalog models are "Free Endpoint" is only shown on build.nvidia.com, so the model
    list is a curated snapshot.
    """

    tier: Tier = "free"
    trains_on_prompts = True
    key_env = "NVIDIA_API_KEY"
    default_models = NVIDIA_FREE_MODELS
    default_limits = (
        rpm(40, scope="provider", source="forums.developer.nvidia.com", last_verified=_VERIFIED, estimate=True),
    )

    def __init__(
        self,
        api_key: str | None = None,
        *,
        name: str = "nvidia",
        base_url: str = "https://integrate.api.nvidia.com/v1",
        **kwargs: Any,
    ) -> None:
        super().__init__(name, base_url, api_key=api_key, **kwargs)


class SambaNova(Provider):
    """SambaNova Cloud free tier (no payment method linked): per model 20 RPM, 20 RPD, 200K TPD."""

    tier: Tier = "free"
    key_env = "SAMBANOVA_API_KEY"
    header_windows: ClassVar[Mapping[str, Window]] = {"requests": "minute"}
    default_models = SAMBANOVA_FREE_MODELS
    default_limits = (
        rpm(20, source="docs.sambanova.ai/docs/en/models/rate-limits", last_verified=_VERIFIED),
        rpd(20, source="docs.sambanova.ai/docs/en/models/rate-limits", last_verified=_VERIFIED),
        tpd(200_000, source="docs.sambanova.ai/docs/en/models/rate-limits", last_verified=_VERIFIED),
    )

    def __init__(
        self,
        api_key: str | None = None,
        *,
        name: str = "sambanova",
        base_url: str = "https://api.sambanova.ai/v1",
        **kwargs: Any,
    ) -> None:
        super().__init__(name, base_url, api_key=api_key, **kwargs)


class Cohere(Provider):
    """Cohere trial key: 1,000 calls/month and 20 chat requests/minute per model; not for production."""

    tier: Tier = "trial"
    key_env = "COHERE_API_KEY"
    default_models = COHERE_TRIAL_MODELS
    default_limits = (
        rpm(20, source="docs.cohere.com/docs/rate-limits", last_verified=_VERIFIED),
        per_month(1000, scope="provider", source="docs.cohere.com/docs/rate-limits", last_verified=_VERIFIED),
    )

    def __init__(
        self,
        api_key: str | None = None,
        *,
        name: str = "cohere",
        base_url: str = "https://api.cohere.ai/compatibility/v1",
        **kwargs: Any,
    ) -> None:
        super().__init__(name, base_url, api_key=api_key, **kwargs)

    def _rate_limit_verdict(self, headers: Headers, error: dict[str, Any], text: str) -> Verdict:
        if "trial key" in text and _MONTHLY_RE.search(text):
            return FailureKind.QUOTA_EXHAUSTED, Scope.PROVIDER
        return super()._rate_limit_verdict(headers, error, text)


class OVH(Provider):
    """OVHcloud AI Endpoints, anonymous access: 2 requests/minute per IP and model, no key needed.

    OVHcloud states prompts are never used for training and are processed in the EU.
    """

    tier: Tier = "free"
    trains_on_prompts = False
    default_models = OVH_ANONYMOUS_MODELS
    default_limits = (rpm(2, source="help.ovhcloud.com AI Endpoints capabilities", last_verified=_VERIFIED),)

    def rate_snapshots(self, headers: Mapping[str, str]) -> list[RateSnapshot]:
        """Kong-style headers: ``x-ratelimit-{limit,remaining}-minute`` and ``ratelimit-reset``.

        The minute is fixed: it resets all at once ``ratelimit-reset`` seconds later (measured
        2026-11-06: 44, 43, 42 s over three requests, then 429 with ``Retry-After: 42``).
        """
        found = Headers(headers)
        limit = _number(found.get("x-ratelimit-limit-minute", ""))
        remaining = _number(found.get("x-ratelimit-remaining-minute", ""))
        reset = _number(found.get("ratelimit-reset", ""))
        if not limit or remaining is None or reset is None:
            return super().rate_snapshots(headers)
        return [RateSnapshot("minute", "requests", limit, max(0.0, remaining), self.header_scope, reset, rolling=False)]

    def __init__(
        self,
        api_key: str | None = None,
        *,
        name: str = "ovh",
        base_url: str = "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1",
        **kwargs: Any,
    ) -> None:
        super().__init__(name, base_url, api_key=api_key, keyless=True, **kwargs)


class Cerebras(Provider):
    """Cerebras: no permanent free tier since 2026-07-16; a $5, 30-day trial needs a payment method.

    Trial limits per model: 5 RPM, 30K TPM, 1M TPH, 1M TPD; ``-minute``/``-day`` limit headers
    keep the counters exact.
    """

    tier: Tier = "trial"
    key_env = "CEREBRAS_API_KEY"
    default_models = CEREBRAS_TRIAL_MODELS
    default_limits = (
        rpm(5, source="inference-docs.cerebras.ai/support/rate-limits", last_verified=_VERIFIED),
        tpm(30_000, source="inference-docs.cerebras.ai/support/rate-limits", last_verified=_VERIFIED),
        tph(1_000_000, source="inference-docs.cerebras.ai/support/rate-limits", last_verified=_VERIFIED),
        tpd(1_000_000, source="inference-docs.cerebras.ai/support/rate-limits", last_verified=_VERIFIED),
    )

    def __init__(
        self,
        api_key: str | None = None,
        *,
        name: str = "cerebras",
        base_url: str = "https://api.cerebras.ai/v1",
        **kwargs: Any,
    ) -> None:
        super().__init__(name, base_url, api_key=api_key, **kwargs)

    def prepare_payload(self, model_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        body = super().prepare_payload(model_id, payload)
        if "max_completion_tokens" in body:
            body.pop("max_tokens", None)
        return body


class Mistral(Provider):
    """Mistral: the official docs say Free mode has no API access; API keys are pay-as-you-go."""

    tier: Tier = "paid"
    key_env = "MISTRAL_API_KEY"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        name: str = "mistral",
        base_url: str = "https://api.mistral.ai/v1",
        **kwargs: Any,
    ) -> None:
        super().__init__(name, base_url, api_key=api_key, **kwargs)
