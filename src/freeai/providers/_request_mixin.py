"""Provider mixin: chat completions request/stream I/O.

Depends on Provider supplying ``_client`` / ``_async_client`` / ``_timeout_args`` (TransportMixin),
``_headers`` / ``_require_key``, and ``classify`` / ``rate_snapshots``.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable, Generator, Mapping
from dataclasses import dataclass, field
from typing import Any

from .._http import Headers, TransportError, decode, error_object
from .._sse import EventReader, is_event_stream
from ..errors import FailureKind, ProviderFailure
from ..limits import RateSnapshot

SnapshotSink = Callable[[list[RateSnapshot]], None]


@dataclass(frozen=True, slots=True)
class Completion:
    """A successful non-streaming response plus the limit state its headers reported."""

    body: dict[str, Any]
    snapshots: list[RateSnapshot] = field(default_factory=list)


class RequestMixin:
    """OpenAI-compatible ``/chat/completions`` send path."""

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
                reader = EventReader(self.name, resp_headers, first_token_timeout, classify=self.classify)
                yield from reader.consume(response.iter_lines())
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
                reader = EventReader(self.name, resp_headers, first_token_timeout, classify=self.classify)
                async for event in reader.aconsume(response.aiter_lines()):
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
    ) -> tuple[str, dict[str, Any], Mapping[str, str]]:
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
