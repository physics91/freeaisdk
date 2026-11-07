"""Request preparation: aliases, capability params, and payload shaping for Router."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from ._messages import Messages, _estimate_tokens, _input_modalities, _normalize_messages, _param_active
from ._routing_types import Request
from .limits import Cost
from .providers import Provider
from .types import ModelInfo, Requirements

# ``model="fast"`` etc.: preference patterns (by model-id naming convention), not a hard filter.
# Unmatched models stay as a fallback, so an alias never makes a request unservable.
ALIASES: dict[str, tuple[str, ...]] = {
    "fast": ("*flash-lite*", "*lightning*", "*nano*", "*-mini*", "*-20b*", "*8b*", "*30b-a3b*", "*flash*"),
    "smart": (
        "*ultra*",
        "*kimi*",
        "*deepseek*",
        "*397b*",
        "*120b*",
        "*super*",
        "*70b*",
        "*command-a-plus*",
        "gemini-3.8*",
    ),
    "coding": ("*coder*", "*code*", "*kimi*", "*deepseek*", "*glm*", "*qwen3*", "*gpt-oss-120b*"),
}

# Request parameters that only some models support; their presence narrows the candidates.
_CAPABILITY_PARAMS = ("tools", "tool_choice", "response_format", "reasoning")
_RESERVED_PARAMS = {"messages", "stream", "models"}
# Only choices[0] is read, so extra choices would just burn quota (and Groq/Cerebras reject n != 1).
_STRIPPED_PARAMS = {"n"}


class PreparePolicy:
    """Build a :class:`Request` and adapt payloads for training-denial providers."""

    __slots__ = ("_by_name", "aliases", "allow_training")

    def __init__(
        self,
        *,
        aliases: Mapping[str, Sequence[str]],
        allow_training: bool,
        by_name: Mapping[str, Provider],
    ) -> None:
        self.aliases = aliases
        self.allow_training = allow_training
        self._by_name = by_name

    def prepare(
        self,
        messages: Messages,
        system: str | None,
        model: str | Sequence[str] | None,
        require: Iterable[str],
        min_context: int | None,
        params: dict[str, Any],
    ) -> Request:
        reserved = _RESERVED_PARAMS.intersection(params)
        if reserved:
            raise TypeError(f"unsupported keyword(s) {sorted(reserved)}: use model=[...] or stream()")
        chat_messages = _normalize_messages(messages, system)
        if _STRIPPED_PARAMS.intersection(params):
            params = {k: v for k, v in params.items() if k not in _STRIPPED_PARAMS}
        parameters = set(require) | {name for name in _CAPABILITY_PARAMS if _param_active(params.get(name))}
        requirements = Requirements(frozenset(parameters), _input_modalities(chat_messages), min_context)
        models: tuple[str, ...] = (model,) if isinstance(model, str) else tuple(model or ())
        prefer: tuple[str, ...] = ()
        if len(models) == 1 and models[0] in self.aliases:
            prefer = tuple(self.aliases[models[0]])
            models = ()
        elif len(models) == 1 and models[0] in ("auto", "freeai", "freeai/auto"):
            models = ()
        payload = {**params, "messages": chat_messages}
        return Request(payload, requirements, models, Cost(1.0, _estimate_tokens(payload)), prefer)

    def payload_for(self, model: ModelInfo, request: Request) -> dict[str, Any]:
        provider = self._by_name[model.provider]
        if not self.allow_training and provider.can_deny_training:
            return provider.deny_training(request.payload)
        return request.payload
