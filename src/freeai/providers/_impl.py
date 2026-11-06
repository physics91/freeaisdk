"""Providers: OpenAI-compatible chat endpoints and the free-tier presets.

A provider does I/O and translates each platform's wire format: it adapts payloads, classifies
errors into :class:`FailureKind`, and reports limit state as :class:`RateSnapshot`. Which model to
try next, cooldowns and quota accounting live in the routing core.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
import os
import re
import threading
import time
from collections.abc import AsyncGenerator, Callable, Generator, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, ClassVar, Literal

from .._http import (
    AsyncClient,
    Client,
    EventReader,
    Headers,
    HTTPError,
    Limits,
    Timeout,
    TransportError,
    _zone,
    decode,
    error_object,
    error_text,
    is_event_stream,
    logger,
    parse_reset,
    retry_hint,
    seconds_until_midnight,
)
from ..errors import FailureKind, FreeAIError, ProviderFailure, Scope
from ..limits import (
    Limit,
    LimitScope,
    RateSnapshot,
    Unit,
    Window,
    per_month,
    rpd,
    rph,
    rpm,
    seconds_until,
    tpd,
    tph,
    tpm,
)
from ..types import ModelInfo
from ._catalog import _catalog_items, _catalog_model, _is_zero

Tier = Literal["free", "trial", "paid", "unknown"]
Verdict = tuple[FailureKind, Scope | None]
SnapshotSink = Callable[[list[RateSnapshot]], None]

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
_LIMIT_HEADER_RE = re.compile(r"^x-ratelimit-(limit|remaining|reset)-(requests|tokens)(?:-(minute|hour|day|month))?$")
_STALE_CATALOG_RETRY = 60.0
_CONNECT_TIMEOUT = 5.0
# httpx defaults (100 connections, 20 kept alive) queue a busy provider's requests behind the pool and
# reconnect above 20 concurrent streams; these match the official OpenAI and Anthropic SDKs.
CONNECTION_LIMITS = Limits(max_connections=1000, max_keepalive_connections=100)


@dataclass(frozen=True, slots=True)
class Completion:
    """A successful non-streaming response plus the limit state its headers reported."""

    body: dict[str, Any]
    snapshots: list[RateSnapshot] = field(default_factory=list)


class Provider:
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
        self._catalog_lock = threading.Lock()
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

    # -- catalog ---------------------------------------------------------------------------

    def list_models(self, *, refresh: bool = False) -> list[ModelInfo]:
        """Public catalog copy; callers may mutate the returned list safely."""
        return list(self.models_snapshot(refresh=refresh))

    async def alist_models(self, *, refresh: bool = False) -> list[ModelInfo]:
        """Public catalog copy; callers may mutate the returned list safely."""
        return list(await self.amodels_snapshot(refresh=refresh))

    def models_snapshot(self, *, refresh: bool = False) -> tuple[ModelInfo, ...]:
        """Cached catalog for routing. Treat as immutable; use :meth:`list_models` to copy."""
        if self._static is not None:
            return self._static
        with self._catalog_lock:
            now = time.monotonic()
            if self._catalog_stale(now, refresh):
                try:
                    self._set_catalog(self._discover(), now)
                except (HTTPError, ValueError, KeyError, TypeError) as exc:
                    self._catalog_failed(exc, now)
            return self._catalog or ()

    async def amodels_snapshot(self, *, refresh: bool = False) -> tuple[ModelInfo, ...]:
        """Async variant of :meth:`models_snapshot`."""
        if self._static is not None:
            return self._static
        now = time.monotonic()
        if self._catalog_stale(now, refresh):
            try:
                self._set_catalog(await self._adiscover(), now)
            except (HTTPError, ValueError, KeyError, TypeError) as exc:
                self._catalog_failed(exc, now)
        return self._catalog or ()

    def invalidate(self) -> None:
        """Force rediscovery on the next ``list_models`` call."""
        self._catalog_at = -math.inf

    def _catalog_stale(self, now: float, refresh: bool) -> bool:
        return refresh or self._catalog is None or now - self._catalog_at > self.catalog_ttl

    def _set_catalog(self, catalog: list[ModelInfo], now: float) -> None:
        self._catalog = tuple(catalog)
        self._catalog_at = now

    def _catalog_failed(self, exc: Exception, now: float) -> None:
        if self._catalog is None:
            raise FreeAIError(f"{self.name}: model discovery failed: {exc}") from exc
        logger.warning("%s: model discovery failed, keeping stale catalog: %s", self.name, exc)
        self._catalog_at = now - self.catalog_ttl + _STALE_CATALOG_RETRY

    def _catalog_paths(self) -> tuple[str, ...]:
        """Catalog endpoints to try in order; later ones are fallbacks after an HTTP error."""
        return ("/models",)

    def _discover(self) -> list[ModelInfo]:
        paths = self._catalog_paths()
        for index, path in enumerate(paths):
            try:
                return self._parse_catalog(self._get_json(path))
            except HTTPError as exc:
                self._catalog_path_failed(path, index, paths, exc)
        return []

    async def _adiscover(self) -> list[ModelInfo]:
        paths = self._catalog_paths()
        for index, path in enumerate(paths):
            try:
                return self._parse_catalog(await self._aget_json(path))
            except HTTPError as exc:
                self._catalog_path_failed(path, index, paths, exc)
        return []

    def _catalog_path_failed(self, path: str, index: int, paths: tuple[str, ...], exc: HTTPError) -> None:
        if index == len(paths) - 1:
            raise
        logger.info("%s: %s unavailable (%s); trying the next catalog", self.name, path, exc)

    def _parse_catalog(self, body: Any) -> list[ModelInfo]:
        return [self._as_model(str(item["id"])) for item in _catalog_items(body) if item.get("id")]

    def _as_model(self, model: str | ModelInfo) -> ModelInfo:
        if isinstance(model, ModelInfo):
            trains = self.trains_on_prompts if model.trains_on_prompts is None else model.trains_on_prompts
            return dataclasses.replace(model, provider=self.name, trains_on_prompts=trains)
        return ModelInfo(self.name, model, trains_on_prompts=self.trains_on_prompts)

    def _get_json(self, path: str) -> Any:
        response = self._client().get(f"{self.base_url}{path}", headers=self._headers())
        response.raise_for_status()
        return response.json()

    async def _aget_json(self, path: str) -> Any:
        response = await self._async_client().get(f"{self.base_url}{path}", headers=self._headers())
        response.raise_for_status()
        return response.json()

    # -- requests --------------------------------------------------------------------------

    def prepare_payload(self, model_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Adapt a request to this provider (drop parameters it rejects, set the model)."""
        body = {key: value for key, value in payload.items() if key not in self.drop_params}
        body["model"] = model_id
        return body

    def deny_training(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Ask the platform not to train on this request (only where ``can_deny_training``)."""
        return dict(payload)

    def complete(self, model_id: str, payload: Mapping[str, Any], *, timeout: float | None = None) -> Completion:
        self._require_key()
        url, body, headers = self._request(model_id, payload, stream=False)
        try:
            response = self._client().post(url, json=body, headers=headers, **self._timeout_args(timeout, None))
        except TransportError as exc:
            raise self._transport_failure(exc) from exc
        return self._completion(response.status_code, response.headers, decode(response))

    async def acomplete(self, model_id: str, payload: Mapping[str, Any], *, timeout: float | None = None) -> Completion:
        self._require_key()
        url, body, headers = self._request(model_id, payload, stream=False)
        try:
            response = await self._async_client().post(
                url, json=body, headers=headers, **self._timeout_args(timeout, None)
            )
        except TransportError as exc:
            raise self._transport_failure(exc) from exc
        return self._completion(response.status_code, response.headers, decode(response))

    def stream(
        self,
        model_id: str,
        payload: Mapping[str, Any],
        *,
        timeout: float | None = None,
        first_token_timeout: float | None = None,
        on_snapshots: SnapshotSink | None = None,
    ) -> Generator[dict[str, Any], None, None]:
        self._require_key()
        url, body, req_headers = self._request(model_id, payload, stream=True)
        try:
            with self._client().stream(
                "POST", url, json=body, headers=req_headers, **self._timeout_args(timeout, first_token_timeout)
            ) as response:
                status, resp_headers = response.status_code, response.headers
                if status != 200 or not is_event_stream(resp_headers):
                    response.read()
                    events = self._stream_preamble(status, resp_headers, decode(response), on_snapshots)
                    if events is None:
                        raise RuntimeError("non-SSE stream preamble returned no events")
                    yield from events
                    return
                self._stream_preamble(status, resp_headers, None, on_snapshots)
                reader = EventReader(
                    self.name, resp_headers, first_token_timeout, classify=self.classify
                )
                for line in response.iter_lines():
                    yield from reader.feed(line)
                    if reader.done:
                        return
                yield from reader.finish()
        except TransportError as exc:
            raise self._transport_failure(exc) from exc

    async def astream(
        self,
        model_id: str,
        payload: Mapping[str, Any],
        *,
        timeout: float | None = None,
        first_token_timeout: float | None = None,
        on_snapshots: SnapshotSink | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        self._require_key()
        url, body, req_headers = self._request(model_id, payload, stream=True)
        try:
            async with self._async_client().stream(
                "POST", url, json=body, headers=req_headers, **self._timeout_args(timeout, first_token_timeout)
            ) as response:
                status, resp_headers = response.status_code, response.headers
                if status != 200 or not is_event_stream(resp_headers):
                    await response.aread()
                    events = self._stream_preamble(status, resp_headers, decode(response), on_snapshots)
                    if events is None:
                        raise RuntimeError("non-SSE stream preamble returned no events")
                    for event in events:
                        yield event
                    return
                self._stream_preamble(status, resp_headers, None, on_snapshots)
                reader = EventReader(
                    self.name, resp_headers, first_token_timeout, classify=self.classify
                )
                async for line in response.aiter_lines():
                    for event in reader.feed(line):
                        yield event
                    if reader.done:
                        return
                for event in reader.finish():
                    yield event
        except TransportError as exc:
            raise self._transport_failure(exc) from exc

    def _stream_preamble(
        self,
        status: int,
        headers: Headers,
        body: Any | None,
        on_snapshots: SnapshotSink | None,
    ) -> list[dict[str, Any]] | None:
        """Shared classify / snapshot / JSON-body handling for :meth:`stream` and :meth:`astream`.

        Callers must read and decode ``body`` before calling when ``status != 200`` or the response
        is not an event stream. Returns the JSON-as-stream events, or ``None`` to continue with SSE.
        """
        if status != 200:
            raise self.classify(status, headers, body)
        if on_snapshots is not None:
            on_snapshots(self.rate_snapshots(headers))
        if not is_event_stream(headers):
            return list(self._json_as_stream(headers, body))
        return None

    def _request(
        self, model_id: str, payload: Mapping[str, Any], *, stream: bool
    ) -> tuple[str, dict[str, Any], dict[str, str]]:
        body = self.prepare_payload(model_id, payload)
        if stream:
            body["stream"] = True
        return f"{self.base_url}/chat/completions", body, self._headers()

    def _completion(self, status: int, headers: Headers, body: Any) -> Completion:
        # A 200 can still carry only an error object (OpenRouter documents this).
        if status != 200 or not isinstance(body, dict) or error_object(body) or not body.get("choices"):
            raise self.classify(status, headers, body)
        return Completion(body, self.rate_snapshots(headers))

    def _json_as_stream(self, headers: Headers, body: Any) -> Generator[dict[str, Any], None, None]:
        if not isinstance(body, dict) or error_object(body) or not body.get("choices"):
            raise self.classify(200, headers, body)
        choice = body["choices"][0] if isinstance(body["choices"][0], dict) else {}
        yield {
            "model": body.get("model"),
            "usage": body.get("usage"),
            "choices": [
                {"index": 0, "delta": choice.get("message") or {}, "finish_reason": choice.get("finish_reason")}
            ],
        }

    def _transport_failure(self, exc: Exception) -> ProviderFailure:
        return ProviderFailure(FailureKind.UNAVAILABLE, f"{self.name}: {type(exc).__name__}: {exc}")

    # -- quota -----------------------------------------------------------------------------

    def refresh_quota(self) -> list[RateSnapshot]:
        """Read server-side quota counters, if the platform exposes them. Never raises."""
        path = self._quota_path()
        if path is None:
            return []
        try:
            return self._parse_quota(self._get_json(path))
        except (HTTPError, ValueError, KeyError, TypeError) as exc:
            return self._quota_unavailable(exc)

    async def arefresh_quota(self) -> list[RateSnapshot]:
        path = self._quota_path()
        if path is None:
            return []
        try:
            return self._parse_quota(await self._aget_json(path))
        except (HTTPError, ValueError, KeyError, TypeError) as exc:
            return self._quota_unavailable(exc)

    def _quota_unavailable(self, exc: Exception) -> list[RateSnapshot]:
        logger.info("%s: could not read the quota: %s", self.name, exc)
        return []

    def _quota_path(self) -> str | None:
        return None

    def _parse_quota(self, body: Any) -> list[RateSnapshot]:
        return []

    def rate_snapshots(self, headers: Mapping[str, str]) -> list[RateSnapshot]:
        """Limit state from ``x-ratelimit-{limit,remaining,reset}-{requests,tokens}[-window]`` headers."""
        found: dict[tuple[Unit, Window], dict[str, float]] = {}
        wall = time.time()
        for name, value in Headers(headers).items():
            match = _LIMIT_HEADER_RE.match(name.lower())
            if not match:
                continue
            kind, unit_name, window_name = match.groups()
            unit: Unit = "requests" if unit_name == "requests" else "tokens"
            window = _as_window(window_name) or self.header_windows.get(unit)
            if window is None:
                continue
            number = parse_reset(value.strip(), wall) if kind == "reset" else _number(value)
            if number is not None:
                found.setdefault((unit, window), {})[kind] = number
        return [
            RateSnapshot(
                window, unit, values["limit"], max(0.0, values["remaining"]), self.header_scope, values.get("reset")
            )
            for (unit, window), values in found.items()
            if values.get("limit", 0) > 0 and "remaining" in values
        ]

    # -- lifecycle -------------------------------------------------------------------------

    def close(self) -> None:
        if self._owns_http and self._http is not None:
            self._http.close()
            self._http = None

    async def aclose(self) -> None:
        if self._owns_ahttp and self._ahttp is not None:
            await self._ahttp.aclose()
            self._ahttp = None
        self.close()

    def _client(self) -> Client:
        with self._client_lock:
            if self._http is None:
                self._http = Client(timeout=self._default_timeout(), limits=CONNECTION_LIMITS)
            return self._http

    def _async_client(self) -> AsyncClient:
        with self._client_lock:
            if self._ahttp is None:
                self._ahttp = AsyncClient(timeout=self._default_timeout(), limits=CONNECTION_LIMITS)
            return self._ahttp

    def _default_timeout(self) -> Timeout:
        return Timeout(self.timeout, connect=min(self.timeout, _CONNECT_TIMEOUT))

    def _timeout_args(self, timeout: float | None, first_token: float | None) -> dict[str, Any]:
        """Per-request timeout narrowed by the remaining deadline and the first-token timeout."""
        if timeout is None and first_token is None:
            return {}
        total = max(0.1, min(self.timeout, timeout if timeout is not None else math.inf))
        read = max(0.1, min(total, first_token if first_token is not None else math.inf))
        return {"timeout": Timeout(total, connect=min(total, _CONNECT_TIMEOUT), read=read)}

    # -- error classification --------------------------------------------------------------

    def classify(self, status: int, headers: Mapping[str, str], body: Any) -> ProviderFailure:
        headers = Headers(headers)
        error = error_object(body)
        code = _effective_status(error, status)
        kind, scope = self._verdict(code, headers, error, error_text(error))
        hint = None
        if kind in _HINTED_KINDS or code == 503:
            hint = retry_hint(headers, error, reset_headers=self.reset_headers)
        if kind is FailureKind.QUOTA_EXHAUSTED and hint is None:
            hint = self._quota_reset_hint(error)
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

    def _quota_reset_hint(self, error: dict[str, Any]) -> float:
        """Without a server hint, an exhausted allowance lasts until the next reset boundary."""
        if _MONTHLY_RE.search(error_text(error)):
            return seconds_until("month", time.time(), self.reset_tz)
        return seconds_until_midnight(self.reset_tz)

    # -- helpers ---------------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        headers = dict(self._extra_headers)
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _require_key(self) -> None:
        if not self.api_key and not self.keyless:
            raise ProviderFailure(FailureKind.AUTH, f"{self.name}: missing API key")


# -- Presets with a known free tier (verified 2026-11-06, see reports/) -------------------------

_VERIFIED = "2026-11-06"
GROQ_FREE_MODELS = ("openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b")
GEMINI_FREE_MODELS = (
    "gemini-3.8-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3-flash-preview",
)
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
OVH_ANONYMOUS_MODELS = (
    "Qwen3.8-27B",
    "gpt-oss-120b",
    "gpt-oss-20b",
    "Qwen3.5-397B-A17B",
    "Meta-Llama-3_3-70B-Instruct",
    "Mistral-Small-3.2-24B-Instruct-2506",
)
_GEMINI_SOURCE = "scriptbyai.com/gemini-api-free-tier-limits (third party; Google publishes no free limits)"
_GEMINI_FLASH = ("gemini-*-flash", "gemini-*-flash-preview")
_GEMINI_FLASH_LITE = ("gemini-*-flash-lite", "gemini-*-flash-lite-preview")
_GEMINI_VERSION_RE = re.compile(r"gemini-(\d+)(?:\.(\d+))?")
_ZAI_RESET_RE = re.compile(r"reset at (\d{4}-\d{2}-\d{2}[ t]\d{2}:\d{2}(?::\d{2})?)")


class Groq(Provider):
    """Groq free plan: per-model limits (30 RPM, 1K RPD, 8K TPM, 200K TPD for the free models).

    ``x-ratelimit-*-requests`` headers describe the *daily* window and ``*-tokens`` the minute
    window; they overwrite the local counters when a request is accepted. Neither resets at a
    calendar boundary: ``x-ratelimit-reset-requests`` is 86.4 s per request used today. Measured on
    2026-11-06 (``reports/load-test/groq-sequence-real.jsonl``, ``groq-daily-refill-real.jsonl``),
    the daily count gets one request back per whole 86.4 s since the previous request, and the
    remainder is dropped when a request arrives: requests 13 s apart got nothing back, requests
    300 s apart always got 3. Under steady traffic the daily count therefore does not refill. The
    SDK treats the window as rolling and resyncs it from every response; once it is spent, the next
    request waits 86.4 s (plus ``EDGE_MARGIN``) after the last reading, when Groq has refilled one.
    ``retry-after`` (sent with 429s) is the only retry hint used.

    Accepting a request pre-charges its prompt (chat template included) plus ``max_tokens`` against
    the minute window and refunds the unused output when it ends. Measured on 2026-11-06 on the
    three free models (``reports/load-test/groq-reserve-probe.jsonl``): the template added 15 to 74
    tokens to a short prompt, hence ``prompt_overhead_tokens``. Without ``max_tokens`` the output
    reserve was 2 to 961 tokens: on gpt-oss mostly the length of a recent completion, on qwen with
    no pattern. It cannot be predicted, so the SDK keeps its generic 512; set ``max_tokens`` on Groq
    for an exact pre-charge. qwen also limits output tokens per minute (1000): one of 12 qwen
    requests without ``max_tokens`` got a 429 "Request too large" against it, while a request with
    ``max_tokens`` 1500 was accepted, so what triggers it is unknown. As Groq words it as a problem
    of the request, it skips the model for that request only.
    """

    tier: Tier = "free"
    trains_on_prompts = False
    key_env = "GROQ_API_KEY"
    reset_headers = ()
    header_windows: ClassVar[Mapping[str, Window]] = {"requests": "day", "tokens": "minute"}
    prompt_overhead_tokens = 80
    default_models = GROQ_FREE_MODELS
    default_limits = (
        rpm(30, source="console.groq.com/docs/rate-limits", last_verified=_VERIFIED),
        rpd(1000, source="console.groq.com/docs/rate-limits", last_verified=_VERIFIED),
        tpm(8000, source="console.groq.com/docs/rate-limits", last_verified=_VERIFIED),
        tpd(200_000, source="console.groq.com/docs/rate-limits", last_verified=_VERIFIED),
    )
    drop_params = frozenset({"logprobs", "logit_bias", "top_logprobs"})

    def __init__(
        self,
        api_key: str | None = None,
        *,
        name: str = "groq",
        base_url: str = "https://api.groq.com/openai/v1",
        **kwargs: Any,
    ) -> None:
        super().__init__(name, base_url, api_key=api_key, **kwargs)

    def prepare_payload(self, model_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        body = super().prepare_payload(model_id, payload)
        body["messages"] = [_without(message, "name") for message in body.get("messages", [])]
        return body

    def rate_snapshots(self, headers: Mapping[str, str]) -> list[RateSnapshot]:
        return [dataclasses.replace(s, rolling=True) if s.per == "day" else s for s in super().rate_snapshots(headers)]

    def _rate_limit_verdict(self, headers: Headers, error: dict[str, Any], text: str) -> Verdict:
        # "Request too large for model ... on tokens per minute (TPM)" or "... on output tokens per
        # minute (OTPM)" (a 429 seen on 2026-11-06): Groq says this request itself is too large and
        # asks to reduce it. Skip the model for this request instead of benching it for everyone.
        if "request too large" in text:
            return FailureKind.BAD_REQUEST, Scope.REQUEST
        return super()._rate_limit_verdict(headers, error, text)


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


class Cloudflare(Provider):
    """Cloudflare Workers AI: 10,000 neurons/day per account (resets 00:00 UTC), 300 RPM per model.

    Needs an API token and the account id (both from the dashboard). Cloudflare states it does not
    train on Workers AI inputs. Error ``3036`` means the daily neurons are gone, ``3040`` that no
    data center has capacity.
    """

    tier: Tier = "free"
    trains_on_prompts = False
    key_env = "CLOUDFLARE_API_TOKEN"
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

    def _quota_reset_hint(self, error: dict[str, Any]) -> float:
        match = _ZAI_RESET_RE.search(error_text(error))
        if match:
            try:
                moment = datetime.fromisoformat(match.group(1).replace(" ", "T")).replace(tzinfo=_zone(self.reset_tz))
                return max(0.0, moment.timestamp() - time.time())
            except ValueError:
                pass
        return super()._quota_reset_hint(error)


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
    default_limits = (rpm(40, scope="provider", source="forums.developer.nvidia.com", last_verified=_VERIFIED),)

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


# -- helpers -----------------------------------------------------------------------------------


_WINDOWS: dict[str, Window] = {"minute": "minute", "hour": "hour", "day": "day", "month": "month"}


def _as_window(name: str | None) -> Window | None:
    return _WINDOWS.get(name) if name else None


def _number(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def _effective_status(error: dict[str, Any], status: int) -> int:
    code = error.get("code")
    if isinstance(code, int) and not isinstance(code, bool) and 100 <= code < 600:
        return code
    if isinstance(code, str) and code.isdigit() and 100 <= int(code) < 600:
        return int(code)
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


def _without(message: Any, key: str) -> Any:
    if isinstance(message, dict) and key in message:
        return {k: v for k, v in message.items() if k != key}
    return message


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
