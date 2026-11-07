"""Routing policy: privacy, response checks, health ordering, probes, retry budget, deadlines."""

import gc
import random

import httpx
import pytest

from freeai import ExhaustedError, FailureKind, FreeAI, Gemini, Groq, ModelInfo, Provider, ProviderFailure, RetryBudget
from freeai._core import Wait
from freeai._messages import _estimate_tokens, content_text, to_response
from freeai._plan import Plan
from freeai._routing_types import _total_tokens
from freeai._sse import EventReader
from freeai.health import HealthStats
from freeai.providers._catalog import _catalog_model
from freeai.types import Requirements

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


@pytest.mark.parametrize("body", [{}, {"choices": []}, {"choices": [None]}, {"choices": ["x"]}])
def test_to_response_missing_choices_is_unavailable(body):
    model = ModelInfo(provider="p", id="m")
    with pytest.raises(ProviderFailure) as info:
        to_response(model, body, retry_on_empty=True)
    assert info.value.kind is FailureKind.UNAVAILABLE


def test_content_text_ignores_null_and_non_str_parts():
    assert content_text([{"type": "text", "text": None}, {"type": "text", "text": "hi"}]) == "hi"
    assert content_text([{"type": "text", "text": 1}]) == ""
    assert content_text("plain") == "plain"


def test_estimate_tokens_skips_null_text_parts():
    payload = {"messages": [{"role": "user", "content": [{"type": "text", "text": None}]}]}
    # Default output reserve only; null text must not become len("None").
    assert _estimate_tokens(payload) == float(512)


def test_estimate_tokens_honors_zero_max_tokens():
    base = {"messages": [{"role": "user", "content": "hi"}]}
    assert _estimate_tokens({**base, "max_tokens": 0}) == 0.0
    assert _estimate_tokens({**base, "max_completion_tokens": 0}) == 0.0
    assert _estimate_tokens(base) == float(512)


def test_catalog_empty_supported_parameters_means_unknown():
    model = _catalog_model("p", {"id": "x", "supported_parameters": []})
    assert model.supported_parameters is None
    assert model.satisfies(Requirements(parameters=frozenset({"tools"})))


def test_catalog_context_length_accepts_float_rejects_bool():
    assert _catalog_model("p", {"id": "x", "context_length": 128000.0}).context_length == 128000
    assert _catalog_model("p", {"id": "x", "context_window": 8192.0}).context_length == 8192
    assert _catalog_model("p", {"id": "x", "context_length": True}).context_length is None


def test_openrouter_quota_accepts_float_rejects_bool():
    from freeai import OpenRouter

    provider = OpenRouter(api_key="k")
    snaps = provider._parse_quota({"data": {"free_model_daily_requests": {"used": 5.0, "limit": 50.0}}})
    assert len(snaps) == 1 and snaps[0].remaining == 45
    assert provider._parse_quota({"data": {"free_model_daily_requests": {"used": True, "limit": 50}}}) == []


def test_needed_context_treats_zero_length_as_known():
    class _Req:
        cost = type("C", (), {"tokens": 99.0})()

    class _Holder:
        request = _Req()

    assert Plan._needed_context(_Holder(), ModelInfo(provider="p", id="m", context_length=0)) == 1
    assert Plan._needed_context(_Holder(), ModelInfo(provider="p", id="m")) == 99


def test_total_tokens_sums_partial_usage_fields():
    assert _total_tokens({"total_tokens": 15}) == 15.0
    assert _total_tokens({"prompt_tokens": 10, "completion_tokens": 5}) == 15.0
    assert _total_tokens({"prompt_tokens": 10}) == 10.0
    assert _total_tokens({"completion_tokens": 0}) == 0.0
    assert _total_tokens({}) is None
    assert _total_tokens(None) is None


def test_health_latency_seeds_from_first_success_after_failures():
    health = HealthStats(alpha=0.3)
    health.record("m", False, 0.5)
    health.record("m", True, 0.2)
    snap = health.snapshot()["m"]
    assert snap["latency"] == pytest.approx(0.2)
    assert snap["success"] == pytest.approx(0.3)
    assert health.score("m") == pytest.approx(0.3 / 1.2)
