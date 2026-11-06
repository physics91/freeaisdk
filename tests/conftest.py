from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from freeai import FreeAI, OpenRouter, Provider


def catalog_model(
    model_id: str,
    *,
    params: tuple[str, ...] = ("temperature", "max_tokens"),
    inputs: tuple[str, ...] = ("text",),
    outputs: tuple[str, ...] = ("text",),
    price: str = "0",
    tokenizer: str = "Other",
    expires: str | None = None,
    context: int = 32768,
) -> dict[str, Any]:
    return {
        "id": model_id,
        "name": model_id,
        "context_length": context,
        "expiration_date": expires,
        "pricing": {"prompt": price, "completion": price},
        "architecture": {"input_modalities": list(inputs), "output_modalities": list(outputs), "tokenizer": tokenizer},
        "supported_parameters": list(params),
        "endpoints": ["/api/v1/chat/completions"],
    }


CATALOG = {
    "data": [
        catalog_model("alpha/one:free", params=("tools", "tool_choice", "temperature")),
        catalog_model("beta/two:free", inputs=("text", "image")),
        catalog_model("gamma/three:free", context=8000),
        catalog_model("openrouter/free", tokenizer="Router", params=("tools", "temperature"), inputs=("text", "image")),
        catalog_model("paid/model", price="0.000001"),
        catalog_model("stealth/zero"),
        catalog_model("music/gen:free", outputs=("text", "audio")),
        catalog_model("old/expired:free", expires="2000-01-01"),
        catalog_model("nvidia/x-content-safety:free"),
    ]
}


def completion(text: str = "hello", model: str = "x") -> dict[str, Any]:
    return {
        "id": "gen-1",
        "model": model,
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
    }


def ok(text: str = "hello", model: str = "x", *, headers: dict[str, str] | None = None) -> httpx.Response:
    return httpx.Response(200, json=completion(text, model), headers=headers)


def error(
    status: int,
    message: str = "boom",
    *,
    headers: dict[str, str] | None = None,
    headers_in_body: dict[str, str] | None = None,
    **metadata: Any,
) -> httpx.Response:
    body: dict[str, Any] = {"error": {"code": status, "message": message}}
    if headers_in_body:
        metadata["headers"] = headers_in_body
    if metadata:
        body["error"]["metadata"] = metadata
    return httpx.Response(status, json=body, headers=headers)


def sse(*events: dict[str, Any] | str) -> httpx.Response:
    lines = [": OPENROUTER PROCESSING"]
    for event in events:
        lines.append(f"data: {event if isinstance(event, str) else json.dumps(event)}")
    lines.append("data: [DONE]")
    return httpx.Response(
        200, content=("\n\n".join(lines) + "\n\n").encode(), headers={"content-type": "text/event-stream"}
    )


def delta(text: str = "", *, role: bool = False, finish: str | None = None) -> dict[str, Any]:
    d: dict[str, Any] = {"content": text}
    if role:
        d["role"] = "assistant"
    return {"model": "x", "choices": [{"index": 0, "delta": d, "finish_reason": finish}]}


Reply = httpx.Response | Callable[[dict[str, Any]], httpx.Response]


class FakeServer:
    """Scripted OpenAI-compatible server. ``script[model]`` is a list of replies used in order;
    the last reply repeats once the list runs out."""

    def __init__(self, catalog: dict[str, Any] | None = None) -> None:
        self.catalog = catalog or CATALOG
        self.script: dict[str, list[Reply]] = {}
        self.calls: list[dict[str, Any]] = []
        self.catalog_fetches = 0
        self.paths: list[str] = []
        self.daily: dict[str, int] = {"used": 0, "limit": 1000, "remaining": 1000}
        self._served: dict[str, int] = defaultdict(int)

    def on(self, model: str, *replies: Reply) -> FakeServer:
        self.script[model] = list(replies)
        return self

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith(("/models", "/models/user")):
            self.catalog_fetches += 1
            self.paths.append(path)
            return httpx.Response(200, json=self.catalog)
        if path.endswith("/key"):
            self.paths.append(path)
            return httpx.Response(200, json={"data": {"free_model_daily_requests": self.daily}})
        body = json.loads(request.content)
        self.calls.append(body)
        replies = self.script.get(body["model"])
        if not replies:
            return ok(model=body["model"])
        index = min(self._served[body["model"]], len(replies) - 1)
        self._served[body["model"]] += 1
        reply = replies[index]
        return reply(body) if callable(reply) else reply

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self))

    def aclient(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self))

    @property
    def models_called(self) -> list[str]:
        return [c["model"] for c in self.calls]


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def server() -> FakeServer:
    return FakeServer()


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def make_ai(server: FakeServer, clock: FakeClock):
    def factory(*extra: Provider, openrouter: dict[str, Any] | None = None, **kwargs: Any) -> FreeAI:
        router = OpenRouter(api_key="sk-test", http_client=server.client(), **(openrouter or {}))
        kwargs.setdefault("max_wait", 0.0)
        return FreeAI([router, *extra], clock=clock, sleep=clock.sleep, **kwargs)

    return factory
