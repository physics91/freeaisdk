"""Provider mixin: lazy httpx clients, timeouts, and close lifecycle."""

from __future__ import annotations

import math
from typing import Any

from .._http import AsyncClient, Client, Limits, Timeout

_CONNECT_TIMEOUT = 5.0
# httpx defaults (100 connections, 20 kept alive) queue a busy provider's requests behind the pool and
# reconnect above 20 concurrent streams; these match the official OpenAI and Anthropic SDKs.
CONNECTION_LIMITS = Limits(max_connections=1000, max_keepalive_connections=100)


class TransportMixin:
    """HTTP client lifecycle. Expects Provider fields set in ``__init__`` (``timeout``, clients, locks)."""

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
