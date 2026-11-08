"""OpenAI-SDK-shaped facade: ``client.chat.completions.create(...)`` on top of FreeAI.

Code written for the ``openai`` package can switch with a one-line change::

    from freeai.compat import OpenAI
    client = OpenAI()  # FreeAI.from_env() underneath
    reply = client.chat.completions.create(model="auto", messages=[{"role": "user", "content": "hi"}])
    print(reply["choices"][0]["message"]["content"])

Responses are plain dicts in the OpenAI wire format (not pydantic objects). ``model`` is the model
that answered; an ``x_freeai`` entry adds the provider, the catalog model requested and the attempt count.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import aclosing, closing
from typing import Any, Literal, overload

from .aio import AsyncFreeAI
from .client import FreeAI
from .types import ChatResponse, Message, ModelInfo, StreamChunk

__all__ = ["AsyncOpenAI", "OpenAI", "chunk_dict", "completion_dict", "model_dict", "parse_model"]


def parse_model(model: str | Sequence[str] | None) -> str | tuple[str, ...] | None:
    """``"auto"``/``None`` → any free model; ``"a,b"`` → ordered candidates; aliases pass through."""
    if model is None:
        return None
    if isinstance(model, str):
        parts = tuple(part.strip() for part in model.split(",") if part.strip())
        return parts[0] if len(parts) == 1 else (parts or None)
    return tuple(model)


def completion_dict(response: ChatResponse) -> dict[str, Any]:
    body = dict(response.raw)
    now = time.time()
    body.setdefault("id", f"freeai-{int(now * 1000)}")
    body.setdefault("created", int(now))
    body["object"] = "chat.completion"
    body["model"] = response.model
    body["x_freeai"] = _origin(response.provider, response.requested_model, len(response.attempts))
    return body


def chunk_dict(chunk: StreamChunk) -> dict[str, Any]:
    body = dict(chunk.raw)
    body.setdefault("object", "chat.completion.chunk")
    body["model"] = chunk.model
    body["x_freeai"] = {"provider": chunk.provider}
    return body


def model_dict(model: ModelInfo) -> dict[str, Any]:
    return {
        "id": model.key,
        "object": "model",
        "owned_by": model.provider,
        "context_length": model.context_length,
        "x_freeai": {"model": model.id, "is_router": model.is_router, "trains_on_prompts": model.trains_on_prompts},
    }


def _origin(provider: str, requested: str, attempts: int) -> dict[str, Any]:
    return {"provider": provider, "requested_model": requested, "attempts": attempts}


class _Completions:
    def __init__(self, ai: FreeAI) -> None:
        self._ai = ai

    @overload
    def create(
        self, *, messages: Sequence[Message], model: str | None = ..., stream: Literal[False] = ..., **params: Any
    ) -> dict[str, Any]: ...

    @overload
    def create(
        self, *, messages: Sequence[Message], model: str | None = ..., stream: Literal[True], **params: Any
    ) -> Iterator[dict[str, Any]]: ...

    def create(
        self, *, messages: Sequence[Message], model: str | None = None, stream: bool = False, **params: Any
    ) -> dict[str, Any] | Iterator[dict[str, Any]]:
        if stream:

            def generate() -> Iterator[dict[str, Any]]:
                with closing(self._ai.stream(messages, model=parse_model(model), **params)) as chunks:
                    for chunk in chunks:
                        yield chunk_dict(chunk)

            return generate()
        return completion_dict(self._ai.chat(messages, model=parse_model(model), **params))


class _AsyncCompletions:
    def __init__(self, ai: AsyncFreeAI) -> None:
        self._ai = ai

    async def create(
        self, *, messages: Sequence[Message], model: str | None = None, stream: bool = False, **params: Any
    ) -> dict[str, Any] | AsyncIterator[dict[str, Any]]:
        if stream:

            async def generate() -> AsyncIterator[dict[str, Any]]:
                async with aclosing(self._ai.stream(messages, model=parse_model(model), **params)) as chunks:
                    async for chunk in chunks:
                        yield chunk_dict(chunk)

            return generate()
        return completion_dict(await self._ai.chat(messages, model=parse_model(model), **params))


class _Chat:
    def __init__(self, completions: Any) -> None:
        self.completions = completions


class _Models:
    def __init__(self, ai: FreeAI) -> None:
        self._ai = ai

    def list(self) -> dict[str, Any]:
        return {"object": "list", "data": [model_dict(m) for m in self._ai.models()]}


class _AsyncModels:
    def __init__(self, ai: AsyncFreeAI) -> None:
        self._ai = ai

    async def list(self) -> dict[str, Any]:
        return {"object": "list", "data": [model_dict(m) for m in await self._ai.models()]}


class OpenAI:
    """Minimal ``openai.OpenAI`` look-alike backed by :class:`~freeai.FreeAI`."""

    def __init__(self, ai: FreeAI | None = None, **options: Any) -> None:
        self.freeai = ai if ai is not None else FreeAI.from_env(**options)
        self.chat = _Chat(_Completions(self.freeai))
        self.models = _Models(self.freeai)

    def close(self) -> None:
        self.freeai.close()

    def __enter__(self) -> OpenAI:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class AsyncOpenAI:
    """Minimal ``openai.AsyncOpenAI`` look-alike backed by :class:`~freeai.AsyncFreeAI`."""

    def __init__(self, ai: AsyncFreeAI | None = None, **options: Any) -> None:
        self.freeai = ai if ai is not None else AsyncFreeAI.from_env(**options)
        self.chat = _Chat(_AsyncCompletions(self.freeai))
        self.models = _AsyncModels(self.freeai)

    async def close(self) -> None:
        await self.freeai.aclose()

    async def __aenter__(self) -> AsyncOpenAI:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()
