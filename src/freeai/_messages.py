"""Message normalization and OpenAI-style response/chunk parsing (no routing I/O)."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from ._routing_types import _usage_number
from .errors import FailureKind, ProviderFailure
from .types import ChatResponse, Message, ModelInfo, StreamChunk

Messages = str | Sequence[Message]

_PART_MODALITY = {"image_url": "image", "input_audio": "audio", "file": "file", "video_url": "video"}


def _param_active(value: Any) -> bool:
    """True when a request param meaningfully asks for a capability.

    Empty ``tools=[]`` / ``tool_choice=""`` / ``response_format={}``, falsy scalars such as
    ``reasoning=False`` / ``0``, ``tool_choice="none"``, and the OpenAI default
    ``response_format={"type": "text"}`` must not narrow candidates or force OpenRouter
    ``require_parameters``.
    """
    if isinstance(value, str) and value.strip().casefold() == "none":
        return False
    if isinstance(value, Mapping):
        if set(value) <= {"type"} and str(value.get("type", "")).casefold() in {"", "text", "none"}:
            return False
        return bool(value)
    return bool(value)


_CHARS_PER_TOKEN = 4  # rough size estimate, refined by ``usage`` after each response
_TOKENS_PER_ATTACHMENT = 1000  # images/audio/files count as a flat amount instead of their base64 size
_DEFAULT_OUTPUT_TOKENS = 512  # output reserve when the caller sets no max_tokens


def _tool_call_items(value: Any) -> tuple[dict[str, Any], ...]:
    """Only a list of dicts counts; a bare dict would become ``tuple(dict)`` key chars otherwise."""
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, dict))


def _response_str(value: Any) -> str | None:
    """API ``model`` / ``finish_reason`` fields; non-strings are treated as missing."""
    return value if isinstance(value, str) else None


def to_response(model: ModelInfo, data: dict[str, Any], *, retry_on_empty: bool) -> ChatResponse:
    raw_choices = data.get("choices")
    choices = raw_choices if isinstance(raw_choices, list) else []
    choice = choices[0] if choices and isinstance(choices[0], dict) else None
    if choice is None or choice.get("finish_reason") == "error":
        raise ProviderFailure(FailureKind.UNAVAILABLE, f"{model.provider}: {model.id} returned an error choice")
    raw_message = choice.get("message")
    message = raw_message if isinstance(raw_message, dict) else {}
    text = content_text(message.get("content"))
    finish_reason = _response_str(choice.get("finish_reason"))
    if not text.strip() and not _tool_call_items(message.get("tool_calls")):
        if finish_reason == "length":
            raise output_limit_failure(model)
        if retry_on_empty:
            raise ProviderFailure(FailureKind.EMPTY, f"{model.provider}: {model.id} returned an empty completion")
    raw_usage = data.get("usage")
    return ChatResponse(
        text=text,
        provider=model.provider,
        model=_response_str(data.get("model")) or model.id,
        requested_model=model.id,
        message=message,
        finish_reason=finish_reason,
        usage=raw_usage if isinstance(raw_usage, Mapping) else None,
        raw=data,
    )


def to_chunk(model: ModelInfo, event: dict[str, Any]) -> StreamChunk:
    raw_choices = event.get("choices")
    choices = raw_choices if isinstance(raw_choices, list) else []
    choice = choices[0] if choices and isinstance(choices[0], dict) else {}
    if choice.get("finish_reason") == "error":
        raise ProviderFailure(FailureKind.UNAVAILABLE, f"{model.provider}: {model.id} stream ended with an error")
    raw_delta = choice.get("delta")
    delta = raw_delta if isinstance(raw_delta, dict) else {}
    reasoning = delta.get("reasoning") or delta.get("reasoning_content")
    raw_usage = event.get("usage")
    return StreamChunk(
        text=content_text(delta.get("content")),
        provider=model.provider,
        model=_response_str(event.get("model")) or model.id,
        reasoning=reasoning if isinstance(reasoning, str) else "",
        tool_calls=_tool_call_items(delta.get("tool_calls")),
        finish_reason=_response_str(choice.get("finish_reason")),
        usage=raw_usage if isinstance(raw_usage, Mapping) else None,
        raw=event,
    )


def content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Some platforms send ``{"type":"text","text":null}``; treat non-str as empty.
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


def empty_stream(model: ModelInfo) -> ProviderFailure:
    return ProviderFailure(FailureKind.EMPTY, f"{model.provider}: {model.id} streamed no output")


def output_limit_failure(model: ModelInfo) -> ProviderFailure:
    return ProviderFailure(
        FailureKind.OUTPUT_LIMIT,
        f"{model.provider}: {model.id} used the whole max_tokens budget (likely on reasoning) "
        "before answering; raise max_tokens or lower the reasoning effort",
    )


def stream_chunk_answered(chunk: StreamChunk) -> bool:
    """True when the chunk carries a user-visible answer (text or tool calls), not reasoning alone."""
    return bool(chunk.text.strip() or chunk.tool_calls)


def _normalize_messages(messages: Messages, system: str | None) -> list[Message]:
    if isinstance(messages, str):
        result: list[Message] = [{"role": "user", "content": messages}]
    else:
        result = []
        for index, message in enumerate(messages):
            if not isinstance(message, Mapping):
                raise TypeError(f"messages[{index}] must be a mapping, got {type(message).__name__}")
            result.append(dict(message))
    if system is not None:
        if not isinstance(system, str):
            raise TypeError(f"system must be a string, got {type(system).__name__}")
        result.insert(0, {"role": "system", "content": system})
    if not result:
        raise ValueError("messages must not be empty")
    return result


def _input_modalities(messages: list[Message]) -> frozenset[str]:
    found: set[str] = set()
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") in _PART_MODALITY:
                    found.add(_PART_MODALITY[part["type"]])
    return frozenset(found)


def _estimate_tokens(payload: Mapping[str, Any]) -> float:
    """Prompt plus output reserve, in tokens; attachments count as a flat amount."""
    chars = 0
    attachments = 0
    for message in payload.get("messages", []):
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str):
            chars += len(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text = part.get("text")
                    if isinstance(text, str):
                        chars += len(text)
                else:
                    attachments += 1
        if isinstance(message, dict) and message.get("tool_calls"):
            chars += len(json.dumps(message["tool_calls"], default=str))
    if payload.get("tools"):
        chars += len(json.dumps(payload["tools"], default=str))
    # ``0`` is a valid reserve (falsy); only a missing/invalid value falls back to the default.
    if (output := payload.get("max_completion_tokens")) is None:
        output = payload.get("max_tokens")
    parsed = _usage_number(output)
    if parsed is None:
        parsed = float(_DEFAULT_OUTPUT_TOKENS)
    return float(chars // _CHARS_PER_TOKEN + attachments * _TOKENS_PER_ATTACHMENT + parsed)
