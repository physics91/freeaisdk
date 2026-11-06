"""AsyncFreeAI shares the routing core; these run the async transport path end to end (mocked HTTP)."""

import asyncio
from typing import Any

import pytest

from freeai import AsyncFreeAI, ExhaustedError, FailureKind, OpenRouter, StreamInterrupted

from .conftest import FakeClock, FakeServer, delta, error, sse

ALPHA, BETA = "alpha/one:free", "beta/two:free"


def make(server: FakeServer, clock: FakeClock, **kwargs: Any) -> AsyncFreeAI:
    async def sleep(seconds: float) -> None:
        clock.sleep(seconds)

    router = OpenRouter(api_key="sk-test", async_http_client=server.aclient())
    kwargs.setdefault("max_wait", 0.0)
    return AsyncFreeAI(router, clock=clock, sleep=sleep, **kwargs)


def run(coro):
    return asyncio.run(coro)


def test_async_chat_fails_over_and_syncs_quota(server, clock):
    server.on(ALPHA, error(429, "rate-limited upstream", headers={"Retry-After": "7"}, provider_name="X"))

    async def main():
        async with make(server, clock) as ai:
            return await ai.chat("hi"), ai.status()

    response, status = run(main())
    assert response.requested_model == BETA
    assert [a.kind for a in response.attempts] == [FailureKind.RATE_LIMIT, None]
    assert status["models"] == {f"openrouter:{ALPHA}": 7}
    assert "/api/v1/key" in server.paths and "/api/v1/models/user" in server.paths


def test_async_stream_and_interruption(server, clock):
    server.on(ALPHA, error(503, "no provider"))
    server.on(BETA, sse(delta(role=True), delta("o"), delta("k"), delta(finish="stop")))

    async def collect():
        ai = make(server, clock)
        try:
            return [chunk.text async for chunk in ai.stream("hi")]
        finally:
            await ai.aclose()

    assert "".join(run(collect())) == "ok"

    broken = FakeServer()
    failing = {
        "error": {"code": 502, "message": "crashed"},
        "choices": [{"index": 0, "delta": {}, "finish_reason": "error"}],
    }
    broken.on(ALPHA, sse(delta("partial"), failing))

    async def interrupted():
        ai = make(broken, clock)
        async for _ in ai.stream("hi"):
            pass

    with pytest.raises(StreamInterrupted) as info:
        run(interrupted())
    assert info.value.partial_text == "partial"


def test_async_validation_and_exhaustion(server, clock):
    ai = make(server, clock)
    with pytest.raises(TypeError):
        ai.stream("hi", stream=True)

    async def no_match():
        await ai.chat("hi", require=["web_search_options"])

    with pytest.raises(ExhaustedError, match="no free model matches"):
        run(no_match())
    models = run(ai.models())
    assert models[-1].is_router
