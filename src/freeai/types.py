"""Public data types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .errors import FailureKind, Scope

Message = dict[str, Any]


@dataclass(frozen=True, slots=True)
class Requirements:
    """Capabilities a model must have to serve a request."""

    parameters: frozenset[str] = frozenset()
    input_modalities: frozenset[str] = frozenset()
    min_context: int | None = None

    def describe(self) -> str:
        items = [*sorted(self.parameters), *(f"input:{m}" for m in sorted(self.input_modalities))]
        if self.min_context:
            items.append(f"context>={self.min_context}")
        return ", ".join(items) or "none"


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """A model a provider can serve. ``None`` capabilities mean "unknown, assume supported"."""

    provider: str
    id: str
    name: str | None = None
    context_length: int | None = None
    input_modalities: frozenset[str] | None = None
    supported_parameters: frozenset[str] | None = None
    is_router: bool = False
    trains_on_prompts: bool | None = None  # None: unknown or varies by endpoint

    @property
    def key(self) -> str:
        return f"{self.provider}:{self.id}"

    def satisfies(self, requirements: Requirements) -> bool:
        if self.supported_parameters is not None and not requirements.parameters <= self.supported_parameters:
            return False
        if self.input_modalities is not None and not requirements.input_modalities <= self.input_modalities:
            return False
        if requirements.min_context and self.context_length is not None:
            return self.context_length >= requirements.min_context
        return True


@dataclass(frozen=True, slots=True)
class CatalogDrift:
    """How a provider's pinned model list differs from what the platform lists right now.

    ``unpinned`` models are only listed by the platform: a catalog without prices cannot show that
    the free plan may use them, so they are reported and never used on their own.
    """

    provider: str
    pinned: tuple[str, ...]
    missing: tuple[str, ...] = ()  # pinned, but the platform no longer lists them
    unpinned: tuple[str, ...] = ()  # listed by the platform, but not pinned
    error: str | None = None  # the catalog could not be read

    @property
    def changed(self) -> bool:
        return bool(self.missing or self.unpinned)


@dataclass(frozen=True, slots=True)
class Attempt:
    """One request sent to one model, successful or not."""

    provider: str
    model: str
    ok: bool
    latency: float
    kind: FailureKind | None = None
    status: int | None = None
    message: str | None = None
    cooldown: float = 0.0
    scope: Scope | None = None
    served_model: str | None = None  # the model that actually answered (routers report the real one)
    usage: dict[str, Any] | None = None

    @property
    def key(self) -> str:
        return f"{self.provider}:{self.model}"


@dataclass(slots=True)
class ChatResponse:
    text: str
    provider: str
    model: str  # model that actually answered (a router such as openrouter/free reports the real one)
    requested_model: str
    message: Message
    finish_reason: str | None
    usage: dict[str, Any] | None
    raw: dict[str, Any]
    attempts: list[Attempt] = field(default_factory=list)

    @property
    def tool_calls(self) -> list[dict[str, Any]]:
        raw = self.message.get("tool_calls")
        if not isinstance(raw, list):
            return []
        return [item for item in raw if isinstance(item, dict)]

    def __str__(self) -> str:
        return self.text


@dataclass(frozen=True, slots=True)
class StreamChunk:
    text: str
    provider: str
    model: str
    reasoning: str = ""
    tool_calls: tuple[dict[str, Any], ...] = ()
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def has_output(self) -> bool:
        # Strip so whitespace-only text/reasoning deltas do not commit the stream
        # (chat EMPTY-fails-over on empty answers).
        return bool(self.text.strip() or self.reasoning.strip() or self.tool_calls)

    def __str__(self) -> str:
        return self.text
