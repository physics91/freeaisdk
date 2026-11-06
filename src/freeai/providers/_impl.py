"""Providers: OpenAI-compatible chat endpoints and the free-tier presets.

A provider does I/O and translates each platform's wire format: it adapts payloads, classifies
errors into :class:`FailureKind`, and reports limit state as :class:`RateSnapshot`. Which model to
try next, cooldowns and quota accounting live in the routing core.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import math
import os
import re
import threading
import time
from collections.abc import AsyncGenerator, Callable, Generator, Iterable, Mapping
from dataclasses import dataclass, field
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
    seconds_until,
)
from ..types import ModelInfo
from ._catalog import _catalog_items
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

Tier = Literal["free", "trial", "paid", "unknown"]
Verdict = tuple[FailureKind, Scope | None]
SnapshotSink = Callable[[list[RateSnapshot]], None]

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

    # -- catalog ---------------------------------------------------------------------------

    def list_models(self, *, refresh: bool = False) -> list[ModelInfo]:
        """Public catalog copy; callers may mutate the returned list safely."""
        return list(self.models_snapshot(refresh=refresh))

    async def alist_models(self, *, refresh: bool = False) -> list[ModelInfo]:
        """Public catalog copy; callers may mutate the returned list safely."""
        return list(await self.amodels_snapshot(refresh=refresh))

    def models_snapshot(self, *, refresh: bool = False) -> tuple[ModelInfo, ...]:
        """Cached catalog for routing. Treat as immutable; use :meth:`list_models` to copy.

        Concurrent cold calls share one discovery (single-flight) without holding
        ``_catalog_lock`` across network I/O.
        """
        if self._static is not None:
            return self._static
        with self._sync_catalog_lock:
            now = time.monotonic()
            if self._catalog_stale(now, refresh):
                try:
                    catalog = self._discover()
                except (HTTPError, ValueError, KeyError, TypeError) as exc:
                    with self._catalog_lock:
                        self._catalog_failed(exc, time.monotonic())
                        return self._catalog or ()
                with self._catalog_lock:
                    self._set_catalog(catalog, time.monotonic())
                    return self._catalog or ()
            with self._catalog_lock:
                return self._catalog or ()

    async def amodels_snapshot(self, *, refresh: bool = False) -> tuple[ModelInfo, ...]:
        """Async variant of :meth:`models_snapshot`.

        Concurrent cold calls on the same event loop share one discovery (single-flight).
        """
        if self._static is not None:
            return self._static
        async with self._async_catalog_lock():
            now = time.monotonic()
            if self._catalog_stale(now, refresh):
                try:
                    catalog = await self._adiscover()
                except (HTTPError, ValueError, KeyError, TypeError) as exc:
                    with self._catalog_lock:
                        self._catalog_failed(exc, time.monotonic())
                        return self._catalog or ()
                with self._catalog_lock:
                    self._set_catalog(catalog, time.monotonic())
                    return self._catalog or ()
            with self._catalog_lock:
                return self._catalog or ()

    def _async_catalog_lock(self) -> asyncio.Lock:
        """Event-loop-bound lock for async catalog single-flight; recreate when the loop changes."""
        loop = asyncio.get_running_loop()
        with self._catalog_lock:
            if self._acatalog_lock is None or self._acatalog_loop is not loop:
                self._acatalog_lock = asyncio.Lock()
                self._acatalog_loop = loop
            return self._acatalog_lock

    def invalidate(self) -> None:
        """Force rediscovery on the next ``list_models`` call."""
        self._catalog_at = -math.inf

    def catalog_may_fetch(self) -> bool:
        """True when the next :meth:`models_snapshot` may perform network I/O."""
        if self._static is not None:
            return False
        return self._catalog_stale(time.monotonic(), refresh=False)

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

    def quota_may_fetch(self) -> bool:
        """True when :meth:`refresh_quota` may perform network I/O."""
        return self._quota_path() is not None

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


_VERIFIED = "2026-11-06"

# -- helpers -----------------------------------------------------------------------------------


_WINDOWS: dict[str, Window] = {"minute": "minute", "hour": "hour", "day": "day", "month": "month"}


def _as_window(name: str | None) -> Window | None:
    return _WINDOWS.get(name) if name else None


def _number(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def _without(message: Any, key: str) -> Any:
    if isinstance(message, dict) and key in message:
        return {k: v for k, v in message.items() if k != key}
    return message

