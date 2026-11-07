"""Waiting requests wake up when another request frees quota, instead of sleeping out their delay."""

import asyncio
import json
import threading
import time

import httpx
import pytest

from freeai import AsyncFreeAI, FreeAI, Provider, SQLiteStore, rpm, tpm
from freeai._core import Wait

from .conftest import FakeClock, FakeServer, delta, ok, sse

# 600 TPM and a 512-token estimate per request: a second request would wait about 43 s. The first
# request's headers say only 10 tokens were used, so it can go as soon as they arrive.
LIGHT_USE = {"x-ratelimit-limit-tokens-minute": "600", "x-ratelimit-remaining-tokens-minute": "590"}


def _provider(**clients) -> Provider:
    return Provider("p", "https://p.example/v1", api_key="k", models=["m1"], limits=[tpm(600)], **clients)


def _stream_reply() -> httpx.Response:
    response = sse(delta("hi"), delta(finish="stop"))
    response.headers.update(LIGHT_USE)
    return response


def test_threads_waiting_for_quota_wake_when_headers_free_it():
    at_server, go_on, finished = threading.Event(), threading.Event(), threading.Event()

    def first(body: dict) -> httpx.Response:
        at_server.set()
        time.sleep(0.3)  # the second request starts waiting meanwhile
        return _stream_reply()

    server = FakeServer().on("m1", first, ok(model="m1"))
    streamed: list[str] = []

    def read(ai: FreeAI) -> None:
        chunks = ai.stream("hi")
        text = next(chunks).text  # headers came first: the platform has accepted this request
        go_on.wait(10)  # hold the stream open, so only its acceptance can wake the other request
        streamed.append(text + "".join(c.text for c in chunks))
        finished.set()

    with FreeAI(_provider(http_client=server.client()), max_wait=60) as ai:
        reader = threading.Thread(target=read, args=(ai,))
        reader.start()
        assert at_server.wait(5)
        started = time.monotonic()
        ai.chat("hi")
        waited = time.monotonic() - started
        assert not finished.is_set()
        go_on.set()
        reader.join(5)
    assert streamed == ["hi"]
    assert len(server.calls) == 2
    assert waited < 5  # not the ~43 s computed before the headers arrived


def test_tasks_waiting_for_quota_wake_when_headers_free_it():
    async def main() -> float:
        at_server, go_on = asyncio.Event(), asyncio.Event()
        calls: list[int] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            if len(calls) == 1:
                at_server.set()
                await asyncio.sleep(0.3)
                return _stream_reply()
            return ok(model="m1")

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        async with AsyncFreeAI(_provider(async_http_client=client), max_wait=60) as ai:

            async def read() -> str:
                chunks = ai.stream("hi")
                text = (await anext(chunks)).text
                await go_on.wait()  # held open: only its acceptance can wake the other request
                return text + "".join([c.text async for c in chunks])

            reader = asyncio.create_task(read())
            await asyncio.wait_for(at_server.wait(), 5)
            started = time.monotonic()
            await asyncio.wait_for(ai.chat("hi"), 10)
            waited = time.monotonic() - started
            assert not reader.done()
            go_on.set()
            assert await reader == "hi"
        assert len(calls) == 2
        return waited

    assert asyncio.run(main()) < 5


USED_UP = {"x-ratelimit-limit-tokens-minute": "600", "x-ratelimit-remaining-tokens-minute": "0"}


def _two_models(**clients) -> Provider:
    return Provider("p", "https://p.example/v1", api_key="k", models=["m1", "m2"], limits=[tpm(600)], **clients)


def _used_up(body: dict) -> httpx.Response:
    response = ok(model=body["model"])
    response.headers.update(USED_UP)
    return response


def test_every_group_of_waiting_threads_wakes():
    """A thread that can only use m1 waits longest; m2 frees up. The m2 waiter must not sleep on."""
    clock = FakeClock()
    at_server, headers_go, go_on = threading.Event(), threading.Event(), threading.Event()

    def held(body: dict) -> httpx.Response:
        at_server.set()
        assert headers_go.wait(10)
        return _stream_reply()

    server = FakeServer().on("m1", _used_up, ok(model="m1")).on("m2", held, ok(model="m2"))
    results: dict[str, str] = {}

    def read(ai: FreeAI) -> None:
        chunks = ai.stream("hi", model="m2")
        next(chunks)
        go_on.wait(10)  # held open: only its acceptance can wake anyone
        results["reader"] = "".join(c.text for c in chunks)

    def call(ai: FreeAI, model: str) -> None:
        results[model] = ai.chat("hi", model=model).text

    with FreeAI(_two_models(http_client=server.client()), clock=clock, max_wait=120) as ai:
        ai.chat("hi", model="m1")  # m1's minute is used up: about 52 s to wait
        # Daemon threads, released in ``finally``: the fake clock does not move by itself, so a
        # waiter left behind by a failed assertion would otherwise keep pytest from exiting.
        reader = threading.Thread(target=read, args=(ai,), daemon=True)
        first = threading.Thread(target=call, args=(ai, "m1"), daemon=True)  # the oldest waiter, for m1
        second = threading.Thread(target=call, args=(ai, "m2"), daemon=True)  # waits for the reader's m2 quota
        try:
            reader.start()
            assert at_server.wait(5)
            first.start()
            time.sleep(0.2)
            second.start()
            time.sleep(0.2)
            started = time.monotonic()
            headers_go.set()
            second.join(5)
            waited = time.monotonic() - started
            assert "m2" in results and first.is_alive()  # m1 is still used up
        finally:
            headers_go.set()
            go_on.set()
            clock.now += 61
            ai._router.changed()  # let the m1 waiter finish
            first.join(5)
            reader.join(5)
    assert waited < 5
    assert set(results) == {"m1", "m2", "reader"}


def test_every_group_of_waiting_tasks_wakes():
    async def main() -> float:
        clock = FakeClock()
        at_server, headers_go, go_on = asyncio.Event(), asyncio.Event(), asyncio.Event()
        served: dict[str, int] = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            model = json.loads(request.content)["model"]
            served[model] = served.get(model, 0) + 1
            if model == "m1":
                return _used_up({"model": model}) if served[model] == 1 else ok(model=model)
            if served[model] == 1:
                at_server.set()
                await headers_go.wait()
                return _stream_reply()
            return ok(model=model)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        async with AsyncFreeAI(_two_models(async_http_client=client), clock=clock, max_wait=120) as ai:

            async def read() -> str:
                chunks = ai.stream("hi", model="m2")
                text = (await anext(chunks)).text
                await go_on.wait()
                return text + "".join([c.text async for c in chunks])

            await ai.chat("hi", model="m1")  # m1's minute is used up
            reader = asyncio.create_task(read())
            await asyncio.wait_for(at_server.wait(), 5)
            first = asyncio.create_task(ai.chat("hi", model="m1"))
            await asyncio.sleep(0.2)
            second = asyncio.create_task(ai.chat("hi", model="m2"))
            await asyncio.sleep(0.2)
            started = time.monotonic()
            headers_go.set()
            await asyncio.wait_for(second, 5)
            waited = time.monotonic() - started
            assert not first.done()
            go_on.set()
            clock.now += 61
            ai._router.changed()
            await asyncio.wait_for(first, 5)
            assert await reader == "hi"
        return waited

    assert asyncio.run(main()) < 5


def _held_reader(other: FreeAI, at_server: threading.Event, go_on: threading.Event) -> threading.Thread:
    def read() -> None:
        chunks = other.stream("hi")
        next(chunks)
        go_on.wait(10)
        for _ in chunks:
            pass

    return threading.Thread(target=read, daemon=True)


def _other_process(store: SQLiteStore):
    """A second client on the same store, run from its own thread (its own SQLite connection)."""
    at_server, headers_go, go_on = threading.Event(), threading.Event(), threading.Event()

    def held(body: dict) -> httpx.Response:
        at_server.set()
        assert headers_go.wait(10)
        return _stream_reply()

    server = FakeServer().on("m1", held, ok(model="m1"))
    other = FreeAI(_provider(http_client=server.client()), store=store, max_wait=60)
    reader = _held_reader(other, at_server, go_on)
    reader.start()
    assert at_server.wait(5)  # its reservation is in the store; its headers are held back
    return other, reader, headers_go, go_on


def test_threads_wake_when_another_process_frees_quota(tmp_path):
    store = SQLiteStore(tmp_path / "state.db")
    other, reader, headers_go, go_on = _other_process(store)
    done: list[str] = []
    with FreeAI(_provider(http_client=FakeServer().client()), store=store, max_wait=60) as ai:
        waiter = threading.Thread(target=lambda: done.append(ai.chat("hi").text), daemon=True)
        waiter.start()
        time.sleep(0.3)  # waits about 43 s for the other client's reservation
        started = time.monotonic()
        headers_go.set()  # the other client's headers say only 10 tokens were used
        waiter.join(5)
        waited = time.monotonic() - started
    go_on.set()
    reader.join(5)
    other.close()
    store.close()
    assert len(done) == 1 and waited < 5  # answered (by the default reply), not after ~43 s


def test_tasks_wake_when_another_process_frees_quota(tmp_path):
    store = SQLiteStore(tmp_path / "state.db")
    other, reader, headers_go, go_on = _other_process(store)

    async def main() -> float:
        client = httpx.AsyncClient(transport=httpx.MockTransport(FakeServer()))
        async with AsyncFreeAI(_provider(async_http_client=client), store=store, max_wait=60) as ai:
            waiter = asyncio.create_task(ai.chat("hi"))
            await asyncio.sleep(0.3)
            started = time.monotonic()
            headers_go.set()
            await asyncio.wait_for(waiter, 5)
            return time.monotonic() - started

    try:
        assert asyncio.run(main()) < 5
    finally:
        go_on.set()
        reader.join(5)
        other.close()
        store.close()


def test_max_wait_counts_the_time_actually_waited():
    clock = FakeClock()
    server = FakeServer()
    provider = Provider(
        "p", "https://p.example/v1", api_key="k", models=["m1"], limits=[rpm(1)], http_client=server.client()
    )
    ai = FreeAI(provider, clock=clock, sleep=clock.sleep, max_wait=70)
    ai.chat("hi")  # the minute's only request: the next one is 61 s away
    router = ai._router
    request = router.prepare("hi", None, None, (), None, {})
    plan = router.plan(request, router.order(request, ai._pools(), turn=0), max_attempts=None, deadline=None)
    first = plan.next()
    assert isinstance(first, Wait) and first.delay == pytest.approx(61.0)
    clock.now += 1.0  # woken early by another request, but nothing was freed for this one
    second = plan.next()  # 1 s spent and 60 s to go fits max_wait; counting the 61 s planned would not
    assert isinstance(second, Wait) and second.delay == pytest.approx(60.0)
    clock.now += second.delay
    assert plan.next() == plan.ordered[0]


def test_sync_wake_skips_dead_waiters():
    from collections import deque

    from freeai.client import _Waiter, _Wakeups

    wakeups = _Wakeups()
    group = frozenset({"m1"})
    dead = _Waiter()
    dead.alive = False
    live = _Waiter()
    wakeups._waiting[group] = deque([dead, live])
    wakeups.wake()
    assert live.event.is_set() and not live.alive
    assert not dead.event.is_set()


def test_sync_wait_forwards_wake_absorbed_after_timeout():
    from freeai.client import _Wakeups

    wakeups = _Wakeups()
    group = frozenset({"m1"})
    mark = wakeups.mark()
    in_sleep = threading.Event()
    release_sleep = threading.Event()
    peer_done = threading.Event()
    calls = {"n": 0}
    real_sleep = wakeups._sleep

    def gated_sleep(event: threading.Event, delay: float, version: int | None) -> bool:
        calls["n"] += 1
        if calls["n"] == 1:
            in_sleep.set()
            assert release_sleep.wait(2)
            return False  # timed out; wake may have claimed this slot meanwhile
        return real_sleep(event, delay, version)

    wakeups._sleep = gated_sleep  # type: ignore[method-assign]

    def first() -> None:
        wakeups.wait(10.0, mark, group)

    def second() -> None:
        wakeups.wait(10.0, mark, group)
        peer_done.set()

    t1 = threading.Thread(target=first)
    t1.start()
    assert in_sleep.wait(2)
    t2 = threading.Thread(target=second)
    t2.start()
    time.sleep(0.05)  # second enqueues behind first
    wakeups.wake()  # claims the timed-out head
    release_sleep.set()  # first finally forwards to peer
    assert peer_done.wait(2), "absorbed wake must reach the next waiter"
    t1.join(2)
    t2.join(2)


def test_async_wake_skips_dead_and_forwards_after_timeout_claim():
    from collections import deque

    from freeai.aio import _Waiter, _Wakeups

    async def main() -> None:
        wakeups = _Wakeups()
        group = frozenset({"m1"})
        dead = _Waiter()
        dead.alive = False
        timed_out = _Waiter()
        peer = _Waiter()
        wakeups._waiting[group] = deque([dead, timed_out, peer])
        wakeups.wake()
        assert timed_out.future.done() and not timed_out.alive
        assert not peer.future.done()
        # timed-out wait() finally after wake claimed the slot
        claimed, timed_out_flag = True, True
        timed_out.alive = False
        if claimed and timed_out_flag and wakeups._waiting.get(group):
            wakeups.wake()
        assert peer.future.done()

    asyncio.run(main())
