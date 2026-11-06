"""HTTP boundary and wire-format helpers: retry hints, error bodies, server-sent events.

Every other module imports the HTTP library through this one, so swapping it is a one-file change.
"""

from __future__ import annotations

import email.utils
import json
import logging
import re
import time
from collections.abc import AsyncIterable, AsyncIterator, Callable, Iterable, Iterator, Mapping
from datetime import UTC, datetime, timedelta, timezone, tzinfo
from typing import Any

import httpx

from .errors import FailureKind, ProviderFailure

Client = httpx.Client
AsyncClient = httpx.AsyncClient
Headers = httpx.Headers
Response = httpx.Response
Timeout = httpx.Timeout
Limits = httpx.Limits
HTTPError = httpx.HTTPError
TransportError = httpx.TransportError

logger = logging.getLogger("freeai")
logger.addHandler(logging.NullHandler())

# Durations such as "2m59.56s", "7.66s" or "520ms" (Groq's x-ratelimit-reset-* format).
_DURATION = re.compile(
    r"^(?:(?P<h>\d+(?:\.\d+)?)h)?(?:(?P<m>\d+(?:\.\d+)?)m(?!s))?"
    r"(?:(?P<s>\d+(?:\.\d+)?)s)?(?:(?P<ms>\d+(?:\.\d+)?)ms)?$"
)
# "Please retry in 40.15s", "Please try again in 2m59.56s", "retry after 30 seconds".
_RETRY_IN = re.compile(
    r"(?:retry|try again)\s+(?:in|after)\s+(\d+(?:\.\d+)?(?:h|ms|m|s)(?:\d+(?:\.\d+)?(?:h|ms|m|s))*)"
)
_RETRY_IN_WORDS = re.compile(r"(?:retry|try again)\s+(?:in|after)\s+(\d+(?:\.\d+)?)\s*(?:seconds?|secs?)\b")


def retry_hint(
    headers: Mapping[str, str],
    error: Mapping[str, Any],
    *,
    reset_headers: Iterable[str] = ("x-ratelimit-reset",),
    now: float | None = None,
) -> float | None:
    """Seconds the server asked us to wait, from headers, body-embedded headers, or the error body.

    ``reset_headers`` lists the reset headers that describe the limit that was hit; platforms such
    as Groq send reset headers for unrelated windows on every response, so they pass ``()``.
    """
    wall = time.time() if now is None else now
    reset_headers = tuple(reset_headers)
    for source in (Headers(headers), _embedded_headers(error)):
        delay = _header_delay(source, reset_headers, wall)
        if delay is not None:
            return delay
    return _body_delay(error)


def _header_delay(headers: Headers, reset_headers: tuple[str, ...], wall: float) -> float | None:
    if value := headers.get("retry-after-ms"):
        try:
            return max(0.0, float(value) / 1000)
        except ValueError:
            pass
    if value := headers.get("retry-after"):
        delay = _parse_retry_after(value.strip(), wall)
        if delay is not None:
            return delay
    for name in reset_headers:
        if value := headers.get(name):
            delay = parse_reset(value.strip(), wall)
            if delay is not None:
                return delay
    return None


def _embedded_headers(error: Mapping[str, Any]) -> Headers:
    """OpenRouter copies the limit headers into ``error.metadata.headers`` of the JSON body."""
    metadata = error.get("metadata")
    embedded = metadata.get("headers") if isinstance(metadata, dict) else None
    if not isinstance(embedded, dict):
        return Headers()
    return Headers({str(k): str(v) for k, v in embedded.items() if v is not None})


def embedded_header_names(error: Mapping[str, Any]) -> set[str]:
    return {name.lower() for name in _embedded_headers(error)}


def _body_delay(error: Mapping[str, Any]) -> float | None:
    for detail in error.get("details") or ():  # Google RPC status details (Gemini)
        if isinstance(detail, dict) and str(detail.get("@type", "")).endswith("RetryInfo"):
            delay = _parse_proto_duration(detail.get("retryDelay"))
            if delay is not None:
                return delay
    text = error_text(error)
    if match := _RETRY_IN.search(text):
        return _parse_duration(match.group(1))
    if match := _RETRY_IN_WORDS.search(text):
        return float(match.group(1))
    return None


def _parse_proto_duration(value: Any) -> float | None:
    if isinstance(value, str) and value.endswith("s"):
        try:
            return max(0.0, float(value[:-1]))
        except ValueError:
            return None
    if isinstance(value, dict):
        try:
            return max(0.0, float(value.get("seconds", 0)) + float(value.get("nanos", 0)) / 1e9)
        except (TypeError, ValueError):
            return None
    return None


def _parse_retry_after(value: str, wall: float) -> float | None:
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        return max(0.0, email.utils.parsedate_to_datetime(value).timestamp() - wall)
    except (TypeError, ValueError):
        return None


def parse_reset(value: str, wall: float) -> float | None:
    """Seconds until a reset given as epoch ms, epoch s, relative seconds, or a duration string."""
    try:
        number = float(value)
    except ValueError:
        return _parse_duration(value)
    if number > 1e12:  # epoch milliseconds (OpenRouter)
        return max(0.0, number / 1000 - wall)
    if number > 1e9:  # epoch seconds
        return max(0.0, number - wall)
    return max(0.0, number)  # relative seconds


def _parse_duration(value: str) -> float | None:
    match = _DURATION.match(value)
    if not value or not match or not any(match.groupdict().values()):
        return None
    parts = {k: float(v) for k, v in match.groupdict().items() if v}
    return parts.get("h", 0) * 3600 + parts.get("m", 0) * 60 + parts.get("s", 0) + parts.get("ms", 0) / 1000


def error_object(body: Any) -> dict[str, Any]:
    """Normalize the error shapes seen across platforms to one dict with ``message``/``code``/...

    Handles ``{"error": {...}}`` (OpenAI, OpenRouter, Gemini), a list wrapper (Gemini compat),
    ``{"error": "text"}``, flat bodies (Cerebras), ``{"object": "error"}`` and ``{"detail": ...}``
    (Mistral), and Cloudflare's ``{"success": false, "errors": [...]}`` envelope.
    """
    if isinstance(body, list):
        body = body[0] if body else None
    if not isinstance(body, dict):
        return {}
    error = body.get("error")
    if isinstance(error, dict):
        return error
    if isinstance(error, str):
        return {"message": error, **({"code": body["code"]} if "code" in body else {})}
    errors = body.get("errors")
    if body.get("success") is False and isinstance(errors, list) and errors and isinstance(errors[0], dict):
        return errors[0]
    detail = body.get("detail")
    if isinstance(detail, dict):
        nested = detail.get("error")
        return nested if isinstance(nested, dict) else detail
    if detail is not None:
        return {"message": detail if isinstance(detail, str) else json.dumps(detail)[:500]}
    # Flat bodies: Cerebras ({"message", "type", "code"}), Cohere ({"message"} only).
    flat = body.get("object") == "error" or "type" in body or "code" in body or isinstance(body.get("message"), str)
    if "choices" not in body and "message" in body and flat:
        return body
    return {}


def error_text(error: Mapping[str, Any]) -> str:
    """Message plus OpenRouter's upstream ``metadata.raw``, lower-cased for matching."""
    metadata = error.get("metadata")
    raw = metadata.get("raw") if isinstance(metadata, dict) else None
    return f"{error.get('message') or ''} {raw if isinstance(raw, str) else ''}".lower()


def decode(response: Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return response.text


class SSEDecoder:
    """Incremental server-sent events parser (WHATWG rules) yielding each event's JSON ``data``.

    Feed it one line at a time (sync or async). Multi-line ``data`` fields are joined with newlines,
    comments are skipped and ``[DONE]`` sets :attr:`done`. An ``event: error`` frame is surfaced as
    ``{"error": ...}``. Call :meth:`flush` at end of input: several servers omit the final blank line.
    """

    def __init__(self) -> None:
        self.done = False
        self._data: list[str] = []
        self._event = ""

    def feed(self, raw: str) -> list[Any]:
        if self.done:
            return []
        line = raw.rstrip("\r")
        if line == "":
            return self._dispatch()
        if line.startswith(":"):
            return []
        name, _, value = line.partition(":")
        value = value.removeprefix(" ")
        if name == "data":
            self._data.append(value)
        elif name == "event":
            self._event = value
        return []

    def flush(self) -> list[Any]:
        return [] if self.done else self._dispatch()

    def _dispatch(self) -> list[Any]:
        data, event = self._data, self._event
        self._data, self._event = [], ""
        items: list[Any] = []
        for item in _decode_event(data, event):
            if item is _DONE:
                self.done = True
                break
            items.append(item)
        return items


def iter_sse(lines: Iterable[str]) -> Iterator[Any]:
    """Parse server-sent events from an iterable of lines (see :class:`SSEDecoder`)."""
    decoder = SSEDecoder()
    for line in lines:
        yield from decoder.feed(line)
        if decoder.done:
            return
    yield from decoder.flush()


def is_event_stream(headers: Headers) -> bool:
    return "text/event-stream" in headers.get("content-type", "text/event-stream")


class EventReader:
    """Turns SSE lines into chunk dicts; raises on error events and on a missed first-token deadline.

    ``classify`` is the provider's error policy (mid-stream errors arrive with HTTP 200).
    """

    def __init__(
        self,
        name: str,
        headers: Headers,
        first_token_timeout: float | None,
        *,
        classify: Callable[[int, Headers, Any], ProviderFailure],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._name = name
        self._headers = headers
        self._classify = classify
        self._decoder = SSEDecoder()
        self._clock = clock
        self._started = clock()
        self._first_token_timeout = first_token_timeout
        self._seen = False

    @property
    def done(self) -> bool:
        return self._decoder.done

    def feed(self, line: str) -> list[dict[str, Any]]:
        events = self._accept(self._decoder.feed(line))
        # Keep-alive comments (": OPENROUTER PROCESSING") keep the socket busy without output.
        timeout = self._first_token_timeout
        if not self._seen and timeout is not None and self._clock() - self._started > timeout:
            raise ProviderFailure(
                FailureKind.UNAVAILABLE,
                f"{self._name}: no streamed output within {self._first_token_timeout:g}s",
            )
        return events

    def finish(self) -> list[dict[str, Any]]:
        return self._accept(self._decoder.flush())

    def consume(self, lines: Iterable[str]) -> Iterator[dict[str, Any]]:
        """Feed SSE lines until ``done``, then flush; preserves feed → yield → done → finish order."""
        for line in lines:
            yield from self.feed(line)
            if self.done:
                return
        yield from self.finish()

    async def aconsume(self, lines: AsyncIterable[str]) -> AsyncIterator[dict[str, Any]]:
        """Async variant of :meth:`consume`."""
        async for line in lines:
            for event in self.feed(line):
                yield event
            if self.done:
                return
        for event in self.finish():
            yield event

    def _accept(self, items: list[Any]) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            # Mid-stream errors arrive as an SSE event while the HTTP status stays 200.
            if error_object(item):
                raise self._classify(200, self._headers, item)
            self._seen = True
            events.append(item)
        return events


_DONE = object()


def _decode_event(data: list[str], event: str) -> Iterator[Any]:
    if not data:
        return
    payload = "\n".join(data)
    if payload.strip() == "[DONE]":
        yield _DONE
        return
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        # Tolerate servers that separate events with single newlines.
        for line in data:
            if line.strip() == "[DONE]":
                yield _DONE
                return
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                logger.debug("skipping undecodable SSE data: %.200s", line)
        return
    if event == "error" and isinstance(parsed, dict) and "error" not in parsed:
        parsed = {"error": parsed}
    yield parsed


def seconds_until_midnight(zone: str, *, now: datetime | None = None) -> float:
    """Seconds until the next local midnight in ``zone`` (e.g. ``"America/Los_Angeles"``)."""
    tz = _zone(zone)
    current = (now or datetime.now(tz)).astimezone(tz)
    midnight = datetime.combine(current.date() + timedelta(days=1), datetime.min.time(), tzinfo=tz)
    return (midnight - current).total_seconds()


def _zone(zone: str) -> tzinfo:
    if zone.upper() == "UTC":
        return UTC
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo(zone)
    except Exception:  # no tz database (e.g. Windows without tzdata): fall back to a fixed offset
        logger.warning("time zone %s unavailable; install tzdata. Falling back to UTC-8.", zone)
        return timezone(timedelta(hours=-8))
