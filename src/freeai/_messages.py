"""Message normalization and OpenAI-style response/chunk parsing (no routing I/O)."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from .errors import FailureKind, ProviderFailure
from .types import ChatResponse, Message, ModelInfo, StreamChunk

Messages = str | Sequence[Message]

_PART_MODALITY = {"image_url": "image", "input_audio": "audio", "file": "file", "video_url": "video"}


def _param_active(value: Any) -> bool:
    """True when a request param meaningfully asks for a capability.

    Empty ``tools=[]`` / ``tool_choice=""`` / ``response_format={}`` must not narrow
    candidates or force OpenRouter ``require_parameters``.
    """
    if isinstance(value, (str, list, dict, tuple, set)):
        return bool(value)
    return value is not None


_CHARS_PER_TOKEN = 4  # rough size estimate, refined by ``usage`` after each response
_TOKENS_PER_ATTACHMENT = 1000  # images/audio/files count as a flat amount instead of their base64 size
_DEFAULT_OUTPUT_TOKENS = 512  # output reserve when the caller sets no max_tokens


def to_response(model: ModelInfo, data: dict[str, Any], *, retry_on_empty: bool) -> ChatResponse:
    choices = data.get("choices") or []
    choice = choices[0] if choices and isinstance(choices[0], dict) else None
    if choice is None or choice.get("finish_reason") == "error":
        raise ProviderFailure(FailureKind.UNAVAILABLE, f"{model.provider}: {model.id} returned an error choice")
    message = choice.get("message") or {}
    text = content_text(message.get("content"))
    if not text.strip() and not message.get("tool_calls"):
        if choice.get("finish_reason") == "length":
            raise ProviderFailure(
                FailureKind.OUTPUT_LIMIT,
                f"{model.provider}: {model.id} used the whole max_tokens budget (likely on reasoning) "
                "before answering; raise max_tokens or lower the reasoning effort",
            )
        if retry_on_empty:
            raise ProviderFailure(FailureKind.EMPTY, f"{model.provider}: {model.id} returned an empty completion")
    return ChatResponse(
        text=text,
        provider=model.provider,
        model=data.get("model") or model.id,
        requested_model=model.id,
        message=message,
        finish_reason=choice.get("finish_reason"),
        usage=data.get("usage"),
        raw=data,
    )


def to_chunk(model: ModelInfo, event: dict[str, Any]) -> StreamChunk:
    choices = event.get("choices") or []
    choice = choices[0] if choices and isinstance(choices[0], dict) else {}
    if choice.get("finish_reason") == "error":
        raise ProviderFailure(FailureKind.UNAVAILABLE, f"{model.provider}: {model.id} stream ended with an error")
    delta = choice.get("delta") or {}
    reasoning = delta.get("reasoning") or delta.get("reasoning_content")
    return StreamChunk(
        text=content_text(delta.get("content")),
        provider=model.provider,
        model=event.get("model") or model.id,
        reasoning=reasoning if isinstance(reasoning, str) else "",
        tool_calls=tuple(delta.get("tool_calls") or ()),
        finish_reason=choice.get("finish_reason"),
        usage=event.get("usage"),
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


def _normalize_messages(messages: Messages, system: str | None) -> list[Message]:
    if isinstance(messages, str):
        result: list[Message] = [{"role": "user", "content": messages}]
    else:
        result = [dict(m) for m in messages]
    if system is not None:
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
    # ``0`` is a valid reserve (falsy); only a missing key falls back to the default.
    if (output := payload.get("max_completion_tokens")) is None:
        output = payload.get("max_tokens")
    if output is None or not isinstance(output, (int, float)) or isinstance(output, bool):
        output = _DEFAULT_OUTPUT_TOKENS
    return float(chars // _CHARS_PER_TOKEN + attachments * _TOKENS_PER_ATTACHMENT + output)
