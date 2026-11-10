"""Waiting for quota: ``max_wait`` per call, ``batch()`` on both clients, and the proxy's bound on it."""

import asyncio
import math
import time
from typing import Any

import httpx
import pytest

from freeai import AsyncFreeAI, ExhaustedError, FailureKind, FreeAI, Provider, rpd, rpm
from freeai.__main__ import main

from .conftest import FakeServer, error, ok


def provider(server: FakeServer, **kwargs: Any) -> Provider:
    return Provider(
        "p",
        "https://p.example/v1",
        api_key="k",
        models=["m1"],
        http_client=server.client(),
        async_http_client=server.aclient(),
        **kwargs,
    )


def echo(body: dict[str, Any]) -> httpx.Response:
    """Answer with the prompt in upper case, so a result shows which prompt it belongs to."""
    prompt = body["messages"][-1]["content"]
    return error(400, "refused prompt") if prompt == "bad" else ok(prompt.upper())


def leftovers(ai: FreeAI | AsyncFreeAI) -> tuple[list[str], int]:
    """Reservations and concurrency slots still held (none once every request is over)."""
    with ai._router.store.transaction() as tx:
        return [key for key in tx if key.startswith("p:")], sum(ai._router._inflight.values())


def test_a_call_can_wait_longer_than_the_client_default(clock):
    ai = FreeAI(provider(FakeServer(), limits=[rpm(1)]), clock=clock, sleep=clock.sleep, max_wait=0.0)
    ai.chat("one")
    with pytest.raises(ExhaustedError) as info:
        ai.chat("two")
    assert info.value.next_available_in == pytest.approx(61.0)
    with pytest.raises(ExhaustedError):
        ai.chat("two", max_wait=30)  # still short of the 61 s
    assert clock.sleeps == []
    assert ai.chat("two", max_wait=120).text == "hello"
    assert clock.sleeps == [pytest.approx(61.0)]
    assert "".join(chunk.text for chunk in ai.stream("three", max_wait=math.inf)) == "hello"
    assert len(clock.sleeps) == 2
    assert ai.max_wait == 0.0  # the client's own default is untouched


def test_an_unlimited_wait_outlasts_a_daily_window_but_not_the_deadline(clock):
    clock.now = 20 * 86400.0 + 23 * 3600  # 23:00 UTC
    ai = FreeAI(provider(FakeServer(), limits=[rpd(1)]), clock=clock, sleep=clock.sleep, max_wait=0.0)
    ai.chat("one")
    ai.chat("two", max_wait=math.inf)
    assert clock.sleeps == [pytest.approx(3600)]  # until the day's allowance is back
    with pytest.raises(ExhaustedError) as info:
        ai.chat("three", max_wait=math.inf, deadline=600)
    assert info.value.next_available_in == pytest.approx(86400)
    assert len(clock.sleeps) == 1


def test_batch_answers_everything_in_order_and_waits_for_quota(clock):
    server = FakeServer().on("m1", echo)
    ai = FreeAI(provider(server, limits=[rpm(2)]), clock=clock, sleep=clock.sleep, max_wait=0.0)
    prompts = [f"p{i}" for i in range(6)]
    results = ai.batch(prompts, concurrency=1)
    assert [r.text for r in results if not isinstance(r, Exception)] == [p.upper() for p in prompts]
    assert len(clock.sleeps) == 4  # two fit the minute at once; each of the others waits for its turn
    assert leftovers(ai) == ([], 0)

    hasty = FreeAI(provider(FakeServer().on("m1", echo), limits=[rpm(2)]), clock=clock, sleep=clock.sleep)
    results = hasty.batch(prompts, concurrency=1, max_wait=0)
    assert [type(r).__name__ for r in results] == ["ChatResponse"] * 2 + ["ExhaustedError"] * 4


def test_batch_keeps_input_order_with_concurrent_workers(clock):
    server = FakeServer().on("m1", echo)
    ai = FreeAI(provider(server, limits=[rpm(30)]), clock=clock, sleep=clock.sleep, max_wait=0.0)
    prompts = [f"p{i}" for i in range(90)]  # three minutes' worth
    seen: list[int] = []
    results = ai.batch(prompts, concurrency=8, on_result=lambda index, result: seen.append(index), max_tokens=16)
    assert [r.text for r in results if not isinstance(r, Exception)] == [p.upper() for p in prompts]
    assert sorted(seen) == list(range(90))
    assert all(call["max_tokens"] == 16 for call in server.calls)
    assert leftovers(ai) == ([], 0)
    with pytest.raises(ValueError, match="concurrency"):
        ai.batch(prompts, concurrency=0)
    assert ai.batch([]) == []


def test_batch_reports_each_failure_in_place(clock):
    ai = FreeAI(provider(FakeServer().on("m1", echo)), clock=clock, sleep=clock.sleep)
    first, second, third = ai.batch(["a", "bad", "c"], concurrency=2)
    assert (first.text, third.text) == ("A", "C")  # type: ignore[union-attr]
    assert isinstance(second, ExhaustedError)
    assert second.attempts[0].kind is FailureKind.BAD_REQUEST


def test_batch_stops_and_wakes_what_is_waiting_on_any_other_error(clock):
    server = FakeServer()
    ai = FreeAI(provider(server, limits=[rpm(1)]), clock=clock)  # real sleeps: two requests wait 61 s

    def boom(index: int, result: object) -> None:
        raise RuntimeError("stop here")

    started = time.monotonic()
    with pytest.raises(RuntimeError, match="stop here"):
        ai.batch(["a", "b", "c"], concurrency=3, on_result=boom)
    assert time.monotonic() - started < 10  # they were woken instead of sleeping it out
    assert len(server.calls) == 1
    assert leftovers(ai) == ([], 0)

    with pytest.raises(TypeError, match="stream"):
        ai.batch(["a"], stream=True)  # a bad argument is no per-prompt failure


def test_async_batch_keeps_order_and_reports_failures_in_place(clock):
    server = FakeServer().on("m1", echo)

    async def sleep(seconds: float) -> None:
        clock.sleep(seconds)

    async def run() -> tuple[list[Any], list[int], tuple[list[str], int]]:
        seen: list[int] = []
        ai = AsyncFreeAI(provider(server, limits=[rpm(2)]), clock=clock, sleep=sleep, max_wait=0.0)
        async with ai:
            results = await ai.batch(
                ["a", "bad", "c", "d", "e"], concurrency=2, on_result=lambda index, _: seen.append(index)
            )
            return results, seen, leftovers(ai)

    results, seen, left = asyncio.run(run())
    assert [getattr(r, "text", type(r).__name__) for r in results] == ["A", "ExhaustedError", "C", "D", "E"]
    assert sorted(seen) == [0, 1, 2, 3, 4]
    assert clock.sleeps  # more than a minute's worth: it waited
    assert left == ([], 0)


def test_cancelling_an_async_batch_releases_requests_in_flight_and_waiting(clock):
    calls: list[str] = []

    async def stuck(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        await asyncio.Event().wait()  # never answers
        raise AssertionError("unreachable")

    async def run() -> tuple[int, tuple[list[str], int], int]:
        slow = Provider(
            "p",
            "https://p.example/v1",
            api_key="k",
            models=["m1"],
            limits=[rpm(2)],
            async_http_client=httpx.AsyncClient(transport=httpx.MockTransport(stuck)),
        )
        async with AsyncFreeAI(slow, clock=clock) as ai:  # real sleeps: the third request waits for quota
            task = asyncio.create_task(ai.batch(["a", "b", "c"], concurrency=3))
            for _ in range(200):
                if len(calls) == 2:
                    break
                await asyncio.sleep(0.01)
            held = leftovers(ai)[1]
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            return held, leftovers(ai), len(calls)

    held, left, sent = asyncio.run(run())
    assert (held, sent) == (2, 2)  # two were at the platform, the third was waiting for the minute
    assert left == ([], 0)


def test_cancelled_requests_give_their_quota_back(clock):
    server = FakeServer()

    async def stuck(request: httpx.Request) -> httpx.Response:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    async def run() -> int:
        slow = Provider(
            "p",
            "https://p.example/v1",
            api_key="k",
            models=["m1"],
            limits=[rpm(2)],
            async_http_client=httpx.AsyncClient(transport=httpx.MockTransport(stuck)),
        )
        async with AsyncFreeAI(slow, clock=clock, max_wait=0.0) as ai:
            task = asyncio.create_task(ai.batch(["a", "b"], concurrency=2))
            await asyncio.sleep(0.05)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            slow._ahttp = server.aclient()  # the platform answers again
            results = await ai.batch(["c", "d"], concurrency=2, max_wait=0)
            return sum(not isinstance(r, Exception) for r in results)

    assert asyncio.run(run()) == 2  # nothing of the minute's two was lost to the cancelled pair


def test_cli_passes_routing_options(monkeypatch, capsys):
    seen: dict[str, Any] = {}

    class _FakeAI:
        @classmethod
        def from_env(cls, **kwargs: Any) -> "_FakeAI":
            seen.update(kwargs)
            return cls()

        def __enter__(self) -> "_FakeAI":
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def models(self, **kwargs: Any) -> list[Any]:
            return []

    monkeypatch.setattr("freeai.__main__.FreeAI", _FakeAI)
    assert main(["--strategy", "quota", "--max-wait", "inf", "--adapt-limits", "models"]) == 0
    assert (seen["strategy"], seen["max_wait"], seen["adapt_limits"]) == ("quota", math.inf, True)
    seen.clear()
    assert main(["--strategy", "priority", "--no-adapt-limits", "models"]) == 0  # the behaviour before
    assert (seen["strategy"], seen["adapt_limits"]) == ("priority", False)
    seen.clear()
    assert main(["models"]) == 0
    assert not {"strategy", "max_wait", "adapt_limits"} & set(seen)  # defaults stay the client's
    assert seen["include_keyless"] is None  # OVHcloud's anonymous endpoints only
    assert main(["--keyless", "models"]) == 0 and seen["include_keyless"] is True  # Kilo's as well
    assert main(["--no-keyless", "models"]) == 0 and seen["include_keyless"] is False  # none
    with pytest.raises(SystemExit):
        main(["--strategy", "fastest", "models"])
    capsys.readouterr()


def test_a_stop_between_the_check_and_the_wait_is_not_slept_out(clock):
    import threading

    ai = FreeAI(provider(FakeServer(), limits=[rpm(1)]), clock=clock)  # real sleeps
    ai.chat("one")  # the next request has to wait 61 s
    stop = threading.Event()
    mark = ai._mark

    def stopped_right_here() -> tuple[int, int | None]:
        stop.set()  # what batch() does when it is stopped ...
        assert ai._wakeups is not None
        ai._wakeups.wake_all()
        return mark()  # ... just before this request notes where it is

    ai._mark = stopped_right_here  # type: ignore[method-assign]
    request = ai._request("two", None, None, (), None, {})
    started = time.monotonic()
    with pytest.raises(ExhaustedError, match="stopped"):
        ai._chat(request, None, None, math.inf, stop)
    assert time.monotonic() - started < 5


def test_async_batch_starts_nothing_when_its_prompts_cannot_be_read(clock):
    server = FakeServer()

    def prompts():
        yield "a"
        yield "b"
        raise RuntimeError("no more prompts")

    async def run() -> int:
        async with AsyncFreeAI(provider(server), clock=clock) as ai:
            with pytest.raises(RuntimeError, match="no more prompts"):
                await ai.batch(prompts())
            await asyncio.sleep(0.05)
            return len(server.calls)

    assert asyncio.run(run()) == 0


def test_batch_applies_an_iterator_of_requirements_to_every_prompt(clock):
    def tools_only() -> Provider:
        from freeai import ModelInfo

        plain_model = ModelInfo("p", "plain", supported_parameters=frozenset({"temperature"}))
        tool_model = ModelInfo("p", "tools", supported_parameters=frozenset({"temperature", "tools"}))
        server = FakeServer()
        return Provider(
            "p",
            "https://p.example/v1",
            api_key="k",
            models=[plain_model, tool_model],
            http_client=server.client(),
            async_http_client=server.aclient(),
        )

    results = FreeAI(tools_only(), clock=clock).batch(["a", "b", "c"], require=iter(["tools"]))
    assert [r.requested_model for r in results if not isinstance(r, Exception)] == ["tools"] * 3

    async def run() -> list[str]:
        async with AsyncFreeAI(tools_only(), clock=clock) as ai:
            answers = await ai.batch(["a", "b", "c"], require=iter(["tools"]))
            return [r.requested_model for r in answers if not isinstance(r, Exception)]

    assert asyncio.run(run()) == ["tools"] * 3
