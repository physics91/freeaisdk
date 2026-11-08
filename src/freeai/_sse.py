"""Server-sent events: decoder, EventReader, and content-type helpers."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterable, AsyncIterator, Callable, Iterable, Iterator
from typing import Any

from ._http import Headers, error_object, logger
from .errors import FailureKind, ProviderFailure

_DONE = object()


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
    """True when ``Content-Type``'s media type is ``text/event-stream`` (case-insensitive).

    Missing or blank content types are not treated as SSE: a JSON 200 without ``Content-Type``
    must take the JSON-as-stream path instead of an empty event-stream parse.
    """
    raw: str | None = headers.get("content-type")
    if raw is None or not raw.strip():
        return False
    media_type = raw.split(";", 1)[0].strip().casefold()
    return media_type == "text/event-stream"


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
            # Role-only / empty preambles are not first tokens: match stream has_output.
            if _event_has_output(item):
                self._seen = True
            events.append(item)
        return events


def _event_has_output(event: dict[str, Any]) -> bool:
    """True when an SSE chunk carries user-visible or reasoning text / tool calls."""
    raw_choices = event.get("choices")
    choices = raw_choices if isinstance(raw_choices, list) else []
    choice = choices[0] if choices and isinstance(choices[0], dict) else {}
    raw_delta = choice.get("delta") if isinstance(choice, dict) else None
    delta = raw_delta if isinstance(raw_delta, dict) else {}
    content = delta.get("content")
    if isinstance(content, str) and content.strip():
        return True
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text = part.get("text")
                if isinstance(text, str) and text.strip():
                    return True
    reasoning = delta.get("reasoning") or delta.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning.strip():
        return True
    tools = delta.get("tool_calls")
    return isinstance(tools, list) and any(isinstance(item, dict) for item in tools)


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
