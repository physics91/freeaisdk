"""Routing policy: privacy, response checks, health ordering, probes, retry budget, deadlines."""

import gc
import random

import httpx
import pytest

from freeai import ExhaustedError, FailureKind, FreeAI, Gemini, Groq, ModelInfo, Provider, RetryBudget
from freeai._core import Wait
from freeai._http import EventReader

from .conftest import FakeClock, FakeServer, error, ok, sse

ALPHA, BETA = "alpha/one:free", "beta/two:free"


def plain(server: FakeServer, name: str = "p", models=("m1", "m2", "m3"), **kwargs) -> Provider:
    return Provider(
        name, f"https://{name}.example/v1", api_key="k", models=list(models), http_client=server.client(), **kwargs
    )


def make(clock: FakeClock, *providers: Provider, **kwargs) -> FreeAI:
    kwargs.setdefault("max_wait", 0.0)
    return FreeAI(list(providers), clock=clock, sleep=clock.sleep, **kwargs)


def test_allow_training_false_filters_models_and_asks_openrouter(make_ai, server):
    gemini = Gemini(api_key="k", http_client=FakeServer().client())
    groq_server = FakeServer()
    groq = Groq(api_key="k", http_client=groq_server.client())
    ai = make_ai(gemini, groq, allow_training=False)
    providers = {m.provider for m in ai.models()}
    assert providers == {"openrouter", "groq"}  # Gemini trains on free-tier prompts

    ai.chat("hi", tools=[{"type": "function", "function": {"name": "w", "parameters": {}}}])
    assert server.calls[0]["provider"] == {"require_parameters": True, "data_collection": "deny"}

    with pytest.raises(ExhaustedError, match="allow_training=False"):
        make_ai(gemini, allow_training=False).chat("hi", model="gemini:*")


def test_reject_response_fails_over_without_cooldown(make_ai, server):
    server.on(ALPHA, ok("I cannot help with that"))
    ai = make_ai(reject_response=lambda r: "refusal" if "cannot" in r.text else None)
    response = ai.chat("hi")
    assert response.requested_model == BETA
    first = response.attempts[0]
    assert (first.kind, first.cooldown) == (FailureKind.REJECTED, 0)
    assert "refusal" in (first.message or "")
    assert ai.status()["models"] == {}


def test_health_strategy_sinks_failing_models(clock):
    server = FakeServer()
    server.on("m1", error(500, "down"))
    ai = make(clock, plain(server), strategy="health", rng=random.Random(7), cooldowns=None)
    for _ in range(6):
        clock.now += 60  # let cooldowns expire so m1 stays a candidate
        ai.chat("hi")
    health = ai.health()
    assert health["p:m1"]["success"] < 0.5
    server.calls.clear()
    picks = []
    for _ in range(20):
        clock.now += 60
        picks.append(ai.chat("hi").requested_model)
    assert picks.count("m1") == 0  # P2C never prefers the failing model over a healthy one


def test_router_success_credits_the_served_model(make_ai, server):
    server.on("openrouter/free", ok(model="meta/llama:free"))
    ai = make_ai()
    ai.chat("hi", model="openrouter/free")
    assert "openrouter:meta/llama:free" in ai.health()


def test_half_open_admits_one_probe(clock):
    server = FakeServer()
    server.on("m1", error(429, "slow", headers={"Retry-After": "5"}))
    ai = make(clock, plain(server, models=["m1"]), max_wait=10)
    with pytest.raises(ExhaustedError):
        ai.chat("hi", max_attempts=1)
    clock.now += 6  # cooldown over: next request is a probe
    router = ai._router
    probe = router.plan(router.prepare("hi", None, None, (), None, {}), ai.models(), max_attempts=None, deadline=None)
    other = router.plan(router.prepare("hi", None, None, (), None, {}), ai.models(), max_attempts=None, deadline=None)
    assert isinstance(probe.next(), ModelInfo)
    assert other.next() == Wait(0.25)  # the second caller waits for the probe's verdict
    probe.release(ai.models()[0])
    assert isinstance(other.next(), ModelInfo)


def test_max_concurrency_limits_inflight_requests(clock):
    server = FakeServer()
    ai = make(clock, plain(server, models=["m1"], max_concurrency=1), max_wait=1)
    router = ai._router
    request = router.prepare("hi", None, None, (), None, {})
    first = router.plan(request, ai.models(), max_attempts=None, deadline=None)
    second = router.plan(request, ai.models(), max_attempts=None, deadline=None)
    target = first.next()
    assert isinstance(target, ModelInfo)
    assert second.next() == Wait(0.25)
    first.succeeded(target, clock.now)
    assert isinstance(second.next(), ModelInfo)


def test_retry_budget_stops_failover_during_outages(clock):
    server = FakeServer()
    for model in ("m1", "m2", "m3"):
        server.on(model, error(503, "overloaded"))
    ai = make(clock, plain(server), retry_budget=RetryBudget(max_tokens=4), max_attempts=3)
    with pytest.raises(ExhaustedError):
        ai.chat("hi")
    clock.now += 600
    with pytest.raises(ExhaustedError, match="retry budget") as info:
        ai.chat("hi")
    assert len(info.value.attempts) == 1  # one try, no sweep across the remaining endpoints

    disabled = make(clock, plain(FakeServer()), retry_budget=False)
    assert disabled._router.retry_budget is None


def test_deadline_bounds_waiting_and_http_timeout(clock):
    timeouts = []

    def handler(request: httpx.Request) -> httpx.Response:
        timeouts.append(request.extensions.get("timeout"))
        return error(429, "slow", headers={"Retry-After": "20"})

    provider = Provider(
        "p",
        "https://p.example/v1",
        api_key="k",
        models=["m1"],
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    ai = make(clock, provider, max_wait=60)
    with pytest.raises(ExhaustedError, match="cooling down") as info:
        ai.chat("hi", deadline=10)
    assert info.value.next_available_in == pytest.approx(20)
    assert timeouts[0]["read"] == pytest.approx(10)
    assert clock.sleeps == []  # a 20 s wait cannot fit a 10 s deadline


def test_first_token_timeout_counts_only_data_events():
    fake = FakeClock()
    provider = Provider("p", "https://p.example")
    reader = EventReader("p", httpx.Headers(), 5.0, classify=provider.classify, clock=fake)
    assert reader.feed(": OPENROUTER PROCESSING") == []
    fake.now += 6
    with pytest.raises(Exception, match="no streamed output within 5s"):
        reader.feed(": OPENROUTER PROCESSING")

    fake2 = FakeClock()
    reader = EventReader("p", httpx.Headers(), 5.0, classify=provider.classify, clock=fake2)
    reader.feed('data: {"choices": []}')
    assert reader.feed("") == [{"choices": []}]
    fake2.now += 60
    assert reader.feed(": keepalive") == []  # output already started: no first-token deadline


def test_stream_early_exit_releases_the_slot(clock):
    server = FakeServer()
    chunks = [{"model": "x", "choices": [{"index": 0, "delta": {"content": c}}]} for c in "ab"]
    server.on("m1", sse(*chunks), ok())
    ai = make(clock, plain(server, models=["m1"], max_concurrency=1))
    for chunk in ai.stream("hi"):
        assert chunk.text == "a"
        break
    gc.collect()
    assert ai.chat("hi").text == "hello"  # the abandoned stream did not keep the only slot
