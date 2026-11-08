"""Provider mixin: lazy httpx clients, timeouts, headers, and close lifecycle."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from .._http import AsyncClient, Client, Limits, Timeout
from ..errors import FailureKind, ProviderFailure

_CONNECT_TIMEOUT = 5.0
# httpx defaults (100 connections, 20 kept alive) queue a busy provider's requests behind the pool and
# reconnect above 20 concurrent streams; these match the official OpenAI and Anthropic SDKs.
CONNECTION_LIMITS = Limits(max_connections=1000, max_keepalive_connections=100)


class TransportMixin:
    """HTTP client lifecycle and shared GET helpers.

    Expects Provider fields set in ``__init__`` (``timeout``, ``base_url``, clients, locks, keys).
    """

    name: str
    base_url: str
    api_key: str | None
    keyless: bool
    timeout: float
    _extra_headers: dict[str, str]
    _http: Client | None
    _ahttp: AsyncClient | None
    _owns_http: bool
    _owns_ahttp: bool
    _client_lock: threading.Lock
    _headers_cached: Mapping[str, str] | None

    def _headers(self) -> Mapping[str, str]:
        """Stable auth/extra headers. Built once and frozen so callers cannot poison the cache."""
        cached = getattr(self, "_headers_cached", None)
        if isinstance(cached, Mapping):
            return cached
        headers = dict(self._extra_headers)
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        frozen = MappingProxyType(headers)
        self._headers_cached = frozen
        return frozen

    def _require_key(self) -> None:
        if not self.api_key and not self.keyless:
            raise ProviderFailure(FailureKind.AUTH, f"{self.name}: missing API key")

    def _get_json(self, path: str) -> Any:
        response = self._client().get(f"{self.base_url}{path}", headers=self._headers())
        response.raise_for_status()
        return response.json()

    async def _aget_json(self, path: str) -> Any:
        response = await self._async_client().get(f"{self.base_url}{path}", headers=self._headers())
        response.raise_for_status()
        return response.json()

    def close(self) -> None:
        http = self._http
        if self._owns_http and http is not None:
            http.close()
            self._http = None

    async def aclose(self) -> None:
        ahttp = self._ahttp
        if self._owns_ahttp and ahttp is not None:
            await ahttp.aclose()
            self._ahttp = None
        self.close()

    def _client(self) -> Client:
        with self._client_lock:
            http = self._http
            if http is None:
                http = Client(timeout=self._default_timeout(), limits=CONNECTION_LIMITS)
                self._http = http
            return http

    def _async_client(self) -> AsyncClient:
        with self._client_lock:
            ahttp = self._ahttp
            if ahttp is None:
                ahttp = AsyncClient(timeout=self._default_timeout(), limits=CONNECTION_LIMITS)
                self._ahttp = ahttp
            return ahttp

    def _default_timeout(self) -> Timeout:
        return Timeout(self.timeout, connect=min(self.timeout, _CONNECT_TIMEOUT))

    def _timeout_args(self, timeout: float | None, first_token: float | None) -> dict[str, Any]:
        """Per-request timeout narrowed by the remaining deadline and the first-token timeout."""
        if timeout is None and first_token is None:
            return {}
        # Respect short deadline leftovers; only floor when using the provider default.
        total = max(0.1, self.timeout) if timeout is None else (min(self.timeout, max(0.0, timeout)) or 1e-6)
        read = total if first_token is None else (min(total, max(0.0, first_token)) or 1e-6)
        return {"timeout": Timeout(total, connect=min(total, _CONNECT_TIMEOUT), read=read)}
