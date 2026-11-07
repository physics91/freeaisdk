"""Providers: OpenAI-compatible chat endpoints and the free-tier presets.

A provider does I/O and translates each platform's wire format: it adapts payloads, classifies
errors into :class:`FailureKind`, and reports limit state as :class:`RateSnapshot`. Which model to
try next, cooldowns and quota accounting live in the routing core.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import os
import threading
import time
from collections.abc import Iterable, Mapping
from typing import Any, ClassVar, Literal

from .._http import (
    AsyncClient,
    Client,
    Headers,
    error_object,
    error_text,
    retry_hint,
    seconds_until_midnight,
)
from ..errors import FailureKind, ProviderFailure, Scope
from ..limits import (
    Limit,
    LimitScope,
    RateSnapshot,
    Window,
    seconds_until,
)
from ..types import ModelInfo
from ._catalog_mixin import CatalogMixin
from ._classify import (
    _BAD_KEY_RE,
    _CONTEXT_RE,
    _DAILY_RE,
    _HINTED_KINDS,
    _KIND_BY_STATUS,
    _MONTHLY_RE,
    _NOT_ENTITLED_RE,
    _effective_status,
    _message,
)
from ._quota_mixin import QuotaMixin
from ._ratelimit import rate_snapshots_from_headers
from ._request_mixin import Completion as Completion
from ._request_mixin import RequestMixin
from ._request_mixin import SnapshotSink as SnapshotSink
from ._transport_mixin import CONNECTION_LIMITS as CONNECTION_LIMITS
from ._transport_mixin import TransportMixin

Tier = Literal["free", "trial", "paid", "unknown"]
Verdict = tuple[FailureKind, Scope | None]


class Provider(CatalogMixin, TransportMixin, RequestMixin, QuotaMixin):
    """Any OpenAI-compatible ``/chat/completions`` endpoint.

    Pass ``models`` to pin the model list; otherwise ``GET {base_url}/models`` is used, which lists
    everything the key can see (not necessarily only free-tier models). ``tier`` declares whether
    the account is free; only gateways with prices in their catalog can prove it per model.

    ``limits`` declares known allowances (see :mod:`freeai.limits`); ``rpm``/``rpd`` are shortcuts
    for account-wide request limits. Limits learned from response headers are added at run time.

    The HTTP clients it creates allow :data:`CONNECTION_LIMITS` (1000 connections, 100 kept alive);
    pass ``http_client``/``async_http_client`` to use your own.
    """

    tier: Tier = "unknown"
    trains_on_prompts: bool | None = None
    can_deny_training: bool = False  # the platform honors a per-request "do not train" flag
    reset_tz: str = "UTC"
    reset_headers: tuple[str, ...] = ("x-ratelimit-reset",)
    # Window of unsuffixed ``x-ratelimit-{limit,remaining}-{requests,tokens}`` headers. Unknown by
    # default: OpenAI means per minute, Groq means per day for requests.
    header_windows: ClassVar[Mapping[str, Window]] = {}
    header_scope: LimitScope = "model"
    # Platform-specific error codes (``error.code``) that override the HTTP status.
    business_codes: ClassVar[Mapping[str, Verdict]] = {}
    default_models: tuple[str | ModelInfo, ...] | None = None
    default_limits: tuple[Limit, ...] = ()
    drop_params: frozenset[str] = frozenset()
    key_env: str | None = None
    max_concurrency: int | None = None
    # Token windows are charged an estimate until the platform answers: prompt characters / 4 plus
    # max_tokens (512 when the caller sets none), plus the chat template and system text the
    # platform adds to every prompt when it counts that too.
    prompt_overhead_tokens: int = 0

    def __init__(
        self,
        name: str,
        base_url: str,
        *,
        api_key: str | None = None,
        api_key_env: str | None = None,
        models: Iterable[str | ModelInfo] | None = None,
        limits: Iterable[Limit] | None = None,
        rpm: int | None = None,
        rpd: int | None = None,
        max_concurrency: int | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float = 60.0,
        catalog_ttl: float = 3600.0,
        http_client: Client | None = None,
        async_http_client: AsyncClient | None = None,
        keyless: bool = False,
        tier: Tier | None = None,
        trains_on_prompts: bool | None = None,
        drop_params: Iterable[str] | None = None,
        reset_tz: str | None = None,
    ) -> None:
        self.name = name
        self.base_url = base_url.rstrip("/")
        env = api_key_env or self.key_env
        self.api_key = api_key if api_key is not None else (os.environ.get(env) if env else None)
        self.keyless = keyless
        if tier is not None:
            self.tier = tier
        if trains_on_prompts is not None:
            self.trains_on_prompts = trains_on_prompts
        if drop_params is not None:
            self.drop_params = frozenset(drop_params)
        if reset_tz is not None:
            self.reset_tz = reset_tz
        if max_concurrency is not None:
            self.max_concurrency = max_concurrency
        declared = list(self.default_limits if limits is None else limits)
        if rpm is not None:
            declared.append(Limit(rpm, "minute", scope="provider", source="rpm argument"))
        if rpd is not None:
            declared.append(Limit(rpd, "day", scope="provider", source="rpd argument"))
        self.limits = declared
        self.timeout = timeout
        self.catalog_ttl = catalog_ttl
        chosen = models if models is not None else self.default_models
        self._static = None if chosen is None else tuple(self._as_model(m) for m in chosen)
        self._catalog: tuple[ModelInfo, ...] | None = None
        self._catalog_at = -math.inf
        self._extra_headers = dict(headers or {})
        self._http = http_client
        self._ahttp = async_http_client
        self._owns_http = http_client is None
        self._owns_ahttp = async_http_client is None
        # Discovery I/O uses the flight locks; ``_catalog_lock`` only covers cache read/write.
        # Order when both are needed: flight lock, then ``_catalog_lock`` (never reverse).
        self._catalog_lock = threading.Lock()
        self._sync_catalog_lock = threading.Lock()
        self._acatalog_lock: asyncio.Lock | None = None
        self._acatalog_loop: asyncio.AbstractEventLoop | None = None
        self._client_lock = threading.Lock()
        # Freeze the account namespace at construction; changing name/api_key later is unsupported.
        if not self.api_key:
            self._state_id = self.name
        else:
            self._state_id = f"{self.name}#{hashlib.sha256(self.api_key.encode()).hexdigest()[:12]}"

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, base_url={self.base_url!r}, tier={self.tier!r})"

    @property
    def state_id(self) -> str:
        """Namespace for shared state: cooldowns and quotas belong to an account, i.e. to a key.

        Computed once from the ``name`` and ``api_key`` used at construction.
        """
        return self._state_id

    # -- HTTP GET/headers: TransportMixin; catalog/chat: CatalogMixin / RequestMixin ------

    # -- quota fetch: QuotaMixin; header snapshots stay here (used by classify / chat I/O) --

    def rate_snapshots(self, headers: Mapping[str, str]) -> list[RateSnapshot]:
        """Limit state from ``x-ratelimit-{limit,remaining,reset}-{requests,tokens}[-window]`` headers."""
        return rate_snapshots_from_headers(headers, self.header_windows, self.header_scope)

    # -- error classification --------------------------------------------------------------

    def classify(self, status: int, headers: Mapping[str, str], body: Any) -> ProviderFailure:
        headers = headers if isinstance(headers, Headers) else Headers(headers)
        error = error_object(body)
        code = _effective_status(error, status)
        text = error_text(error)
        kind, scope = self._verdict(code, headers, error, text)
        hint = None
        if kind in _HINTED_KINDS or code == 503:
            hint = retry_hint(headers, error, reset_headers=self.reset_headers)
        if kind is FailureKind.QUOTA_EXHAUSTED and hint is None:
            hint = self._quota_reset_hint(error, text)
        return ProviderFailure(
            kind,
            f"{self.name}: {_message(error, body, code)}",
            status=code,
            retry_after=hint,
            scope=scope,
            snapshots=self.rate_snapshots(headers),
        )

    def _verdict(self, code: int, headers: Headers, error: dict[str, Any], text: str) -> Verdict:
        business = self.business_codes.get(str(error.get("code")))
        if business is not None:
            return business
        if code == 429:
            return self._rate_limit_verdict(headers, error, text)
        if code in (400, 403) and _BAD_KEY_RE.search(text):
            return FailureKind.AUTH, None
        if code in (400, 413, 422) and _CONTEXT_RE.search(text):
            return FailureKind.CONTEXT_EXCEEDED, None
        if code in (403, 404) and _NOT_ENTITLED_RE.search(text):
            return FailureKind.NOT_ENTITLED, None
        return _KIND_BY_STATUS.get(code, FailureKind.UNAVAILABLE), None

    def _rate_limit_verdict(self, headers: Headers, error: dict[str, Any], text: str) -> Verdict:
        # Most OpenAI-compatible free tiers (Groq, Cerebras, Gemini) limit per model.
        if _DAILY_RE.search(text) or _MONTHLY_RE.search(text):
            return FailureKind.QUOTA_EXHAUSTED, None
        return FailureKind.RATE_LIMIT, None

    def _quota_reset_hint(self, error: dict[str, Any], text: str | None = None) -> float:
        """Without a server hint, an exhausted allowance lasts until the next reset boundary."""
        body = text if text is not None else error_text(error)
        if _MONTHLY_RE.search(body):
            return seconds_until("month", time.time(), self.reset_tz)
        return seconds_until_midnight(self.reset_tz)


_VERIFIED = "2026-11-06"
