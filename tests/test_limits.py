"""Quota windows, header snapshots and the shared state store."""

import asyncio
import fnmatch
import json
import math

import httpx
import pytest

from freeai import (
    OVH,
    AsyncFreeAI,
    ExhaustedError,
    FailureKind,
    FreeAI,
    Gemini,
    Groq,
    Limit,
    ModelInfo,
    Provider,
    RateSnapshot,
    Scope,
    SQLiteStore,
    rpd,
    rpm,
    tpd,
    tpm,
)
from freeai._core import Wait, _learned_key, _rank
from freeai.limits import (
    EDGE_MARGIN,
    Cost,
    cooldown_snapshot,
    pending_add,
    quota_charge,
    quota_observe,
    quota_ready_at,
    seconds_until,
)
from freeai.state import MemoryStore, _like_prefix

from .conftest import FakeClock, FakeServer, delta, error, ok, sse

DAY = 86400.0


def make_provider(server: FakeServer, *, name: str = "p", models=("m1", "m2"), **kwargs) -> Provider:
    return Provider(
        name, "https://p.example/v1", api_key="k", models=list(models), http_client=server.client(), **kwargs
    )


def make(server: FakeServer, clock: FakeClock, *providers: Provider, **kwargs) -> FreeAI:
    kwargs.setdefault("max_wait", 0.0)
    return FreeAI(list(providers) or [make_provider(server)], clock=clock, sleep=clock.sleep, **kwargs)


def test_rolling_window_is_gcra_and_uses_refilled_quota_a_margin_late():
    tx: dict = {}
    limits = [rpm(2)]
    cost = Cost(1, 0)
    for _ in range(2):  # a burst out of a whole window is not delayed
        assert quota_ready_at(tx, limits, "p", "m", cost, 1000.0, "UTC") == 1000.0
        quota_charge(tx, limits, "p", "m", cost, 1000.0, "UTC")
    ready = 1030.0 + EDGE_MARGIN  # the slot regained after 30 s, used a margin late
    assert quota_ready_at(tx, limits, "p", "m", cost, 1030.0, "UTC") > 1030.0  # not exactly at the edge
    assert quota_ready_at(tx, limits, "p", "m", cost, 1000.0, "UTC") == pytest.approx(ready)
    assert quota_ready_at(tx, limits, "p", "m", cost, 1030.0, "UTC") == pytest.approx(ready)
    assert quota_ready_at(tx, limits, "p", "m", cost, ready, "UTC") == ready
    quota_charge(tx, limits, "p", "m", cost, ready, "UTC")
    assert quota_ready_at(tx, limits, "p", "m", cost, ready, "UTC") == pytest.approx(1060.0 + EDGE_MARGIN)


def test_fixed_daily_window_resets_at_the_zone_boundary():
    tx: dict = {}
    limits = [rpd(1)]
    now = 1_700_000_000.0  # 2023-11-14T22:13:20Z
    quota_charge(tx, limits, "p", "m", Cost(), now, "UTC")
    ready = quota_ready_at(tx, limits, "p", "m", Cost(), now, "UTC")
    assert ready == pytest.approx(now + seconds_until("day", now, "UTC"))
    assert quota_ready_at(tx, limits, "p", "m", Cost(), ready, "UTC") == ready  # new day, fresh budget
    la = quota_ready_at(tx, limits, "p", "m", Cost(), now, "America/Los_Angeles")
    assert la != ready  # same counter, but Pacific midnight is a different instant


def test_server_reset_replaces_the_calendar_boundary():
    tx: dict = {}
    now = 1_700_000_000.0
    snapshot = RateSnapshot("day", "requests", limit=1000, remaining=0, reset_in=179.56)
    limit = quota_observe(tx, "p", "m", snapshot, now, "UTC")
    ready = now + 179.56 + EDGE_MARGIN  # the server's reset is rounded, so it is honored a margin late
    assert quota_ready_at(tx, [limit], "p", "m", Cost(), now, "UTC") == pytest.approx(ready, abs=1e-6)
    assert quota_ready_at(tx, [limit], "p", "m", Cost(), now + 180, "UTC") == pytest.approx(ready, abs=1e-6)
    assert quota_ready_at(tx, [limit], "p", "m", Cost(), ready, "UTC") == ready


def test_per_model_daily_limit_blocks_only_that_model(server, clock):
    ai = make(server, clock, make_provider(server, limits=[rpd(2)]))
    for _ in range(2):
        ai.chat("hi", model="m1")
    assert ai.chat("hi").requested_model == "m2"  # m1 is out for the day, m2 still has budget
    with pytest.raises(ExhaustedError, match="cooling down or rate limited") as info:
        ai.chat("hi", model="m1")
    assert info.value.next_available_in == pytest.approx(seconds_until("day", clock.now, "UTC"))


def test_provider_scope_limit_is_shared_by_models(server, clock):
    ai = make(server, clock, make_provider(server, rpd=1))
    ai.chat("hi")
    with pytest.raises(ExhaustedError):
        ai.chat("hi", model="m2")
    assert server.models_called == ["m1"]


def test_token_window_uses_estimate_then_usage(server, clock):
    # 1000 TPM: the 512-token output reserve plus the prompt fits once, then usage (4 tokens) refunds it.
    ai = make(server, clock, make_provider(server, models=["m1"], limits=[tpm(1000)]))
    ai.chat("hi")
    ai.chat("hi")  # without settling, two 512+ token estimates would exceed 1000 per minute
    assert len(server.calls) == 2
    assert clock.sleeps == []


def test_groq_headers_overwrite_counters(server, clock):
    # Groq gives a daily request back per 86.4 s since the previous request; reset is 86.4 s per
    # request used. Once the count is spent, the next request waits one interval after the reading.
    headers = {
        "x-ratelimit-limit-requests": "1000",
        "x-ratelimit-remaining-requests": "0",
        "x-ratelimit-reset-requests": "24h0m0s",
        "x-ratelimit-limit-tokens": "8000",
        "x-ratelimit-remaining-tokens": "7990",
    }
    groq = Groq(api_key="k", http_client=server.client())
    first, second = groq.list_models()[:2]
    server.on(first.id, ok(headers=headers))
    ai = make(server, clock, groq)
    ai.chat("hi", model=first.id)
    assert ai.chat("hi").requested_model == second.id  # the first model's daily window is spent
    with pytest.raises(ExhaustedError) as info:
        ai.chat("hi", model=first.id)
    assert info.value.next_available_in == pytest.approx(86.4 + EDGE_MARGIN, abs=1e-6)  # one refilled, not 24 h


def test_groq_request_too_large_skips_the_model_for_that_request_only(server, clock):
    # A 429 seen on 2026-11-06: one qwen request without max_tokens needed more output tokens than
    # the model's 1000 per minute. Waiting would not help, and the model still serves other requests.
    too_large = error(
        429,
        "Request too large for model `qwen/qwen3.8-27b` in organization `org_x` service tier `on_demand` on "
        "output tokens per minute (OTPM): Limit 1000, Requested 1206, please reduce your message size and try again.",
    )
    groq = Groq(api_key="k", http_client=server.client())
    first, second = groq.list_models()[:2]
    failure = groq.classify(429, {}, too_large.json())
    assert (failure.kind, failure.scope) == (FailureKind.BAD_REQUEST, Scope.REQUEST)
    busy = groq.classify(429, {"retry-after": "2"}, error(429, "Rate limit reached for model `m` on tokens").json())
    assert busy.kind is FailureKind.RATE_LIMIT  # an ordinary 429 still benches the model

    server.on(first.id, too_large, ok(model=first.id))
    ai = make(server, clock, groq)
    assert ai.chat("hi", model=[first.id, second.id]).requested_model == second.id
    assert ai.chat("hi", model=first.id).requested_model == first.id  # not cooled down


def test_groq_daily_window_is_read_as_a_rolling_window():
    # Headers recorded on 2026-11-06 (reports/load-test/groq-probe.jsonl): 2 used, 2 * 86.4 s to full.
    snapshots = Groq(api_key="k").rate_snapshots(
        {
            "x-ratelimit-limit-requests": "1000",
            "x-ratelimit-remaining-requests": "998",
            "x-ratelimit-reset-requests": "2m52.8s",
            "x-ratelimit-limit-tokens": "8000",
            "x-ratelimit-remaining-tokens": "7807",
            "x-ratelimit-reset-tokens": "1.447s",
        }
    )
    day = next(s for s in snapshots if s.per == "day")
    assert day.rolling is True and day.reset_in == pytest.approx(172.8)
    tx: dict = {}
    now = 1_700_000_000.0
    limit = quota_observe(tx, "groq", "m", day, now, "UTC")
    assert tx[limit.key("groq", "m")]["tat"] == pytest.approx(now + day.reset_in, abs=1e-6)  # GCRA agrees


def test_ovh_kong_headers_are_a_fixed_minute():
    headers = {
        "x-ratelimit-limit-minute": "2",
        "x-ratelimit-remaining-minute": "1",
        "ratelimit-limit": "2",
        "ratelimit-remaining": "1",
        "ratelimit-reset": "44",
    }  # as OVHcloud sent them on 2026-11-06
    [snapshot] = OVH().rate_snapshots(headers)
    assert (snapshot.per, snapshot.unit, snapshot.limit, snapshot.remaining) == ("minute", "requests", 2, 1)
    assert (snapshot.reset_in, snapshot.rolling) == (44, False)


def test_fixed_minute_window_waits_for_its_reset_and_repeats():
    tx: dict = {}
    now = 1000.0
    quota_observe(tx, "p", "m", RateSnapshot("minute", "requests", 2, 0, reset_in=44.0, rolling=False), now, "UTC")
    limits, cost = [rpm(2)], Cost(1, 0)  # the declared window shares the learned one's counter
    end = now + 44
    assert quota_ready_at(tx, limits, "p", "m", cost, now + 31, "UTC") == end + EDGE_MARGIN  # not GCRA's 30 s
    at = end + EDGE_MARGIN
    for _ in range(2):  # the next minute is whole
        assert quota_ready_at(tx, limits, "p", "m", cost, at, "UTC") == at
        quota_charge(tx, limits, "p", "m", cost, at, "UTC")
    assert quota_ready_at(tx, limits, "p", "m", cost, at, "UTC") == end + 60 + EDGE_MARGIN
    later = end + 190  # two quiet minutes later
    assert quota_ready_at(tx, limits, "p", "m", cost, later, "UTC") == later
    for _ in range(2):
        quota_charge(tx, limits, "p", "m", cost, later, "UTC")
    assert quota_ready_at(tx, limits, "p", "m", cost, later, "UTC") == end + 240 + EDGE_MARGIN


def test_fixed_minute_without_a_reset_stays_rolling():
    tx: dict = {}
    limit = quota_observe(tx, "p", "m", RateSnapshot("minute", "requests", 2, 0, rolling=False), 1000.0, "UTC")
    assert "tat" in tx[limit.key("p", "m")]


def _kong_minute(clock: FakeClock, limit: int, start: float):
    """OVHcloud-like platform: fixed minutes from ``start``, counted on acceptance, Kong headers."""
    counts: dict[int, int] = {}

    def reply(body: dict) -> httpx.Response:
        window = math.floor((clock.now - start) / 60)
        counts[window] = counts.get(window, 0) + 1
        left = start + (window + 1) * 60 - clock.now
        if counts[window] > limit:
            return error(429, "API rate limit exceeded", headers={"Retry-After": str(math.ceil(left))})
        response = ok(model=body["model"])
        remaining = str(limit - counts[window])
        response.headers.update(
            {
                "x-ratelimit-limit-minute": str(limit),
                "x-ratelimit-remaining-minute": remaining,
                "ratelimit-remaining": remaining,
                "ratelimit-reset": str(int(left)),  # whole seconds, rounded down: the worst case
            }
        )
        return response

    return reply


def test_ovh_requests_wait_for_the_fixed_minute(clock):
    server = FakeServer()
    ovh = OVH(http_client=server.client())
    model = ovh.list_models()[0].id
    server.on(model, _kong_minute(clock, 2, start=clock.now - 15.3))  # 44.7 s left in this minute
    ai = make(server, clock, ovh, max_wait=120)
    for _ in range(4):
        ai.chat("hi", model=model)
    assert len(server.calls) == 4  # no 429: a rolling window would have tried again after 31 s
    assert clock.sleeps == [pytest.approx(44 + EDGE_MARGIN)]


def _kong(remaining: int, reset: int, limit: int = 2) -> dict[str, str]:
    return {
        "x-ratelimit-limit-minute": str(limit),
        "x-ratelimit-remaining-minute": str(remaining),
        "ratelimit-reset": str(reset),
    }


def _ovh_plans(server: FakeServer, clock: FakeClock):
    ovh = OVH(http_client=server.client())
    model = ovh.list_models()[0].id
    ai = make(server, clock, ovh, max_wait=120)
    router = ai._router

    def plan():
        request = router.prepare("hi", None, model, (), None, {})
        return router.plan(request, router.order(request, ai._pools(), turn=0), max_attempts=None, deadline=None)

    return ovh, plan


def test_late_headers_of_a_fixed_minute_do_not_free_it(server, clock):
    ovh, plan = _ovh_plans(server, clock)
    first, second = plan(), plan()
    assert not isinstance(first.next(), Wait) and not isinstance(second.next(), Wait)
    # The platform counted the first request before the second, but the responses come back swapped,
    # and a third request goes out in between.
    second.admitted(ovh.rate_snapshots(_kong(2, 40, limit=4)))
    assert not isinstance(plan().next(), Wait)
    first.admitted(ovh.rate_snapshots(_kong(3, 40, limit=4)))
    assert not isinstance(plan().next(), Wait)  # the fourth and last request of this minute
    fifth = plan().next()
    assert isinstance(fifth, Wait) and fifth.delay == pytest.approx(40 + EDGE_MARGIN)


def test_late_headers_of_an_ended_fixed_minute_are_ignored(server, clock):
    ovh, plan = _ovh_plans(server, clock)
    late, current = plan(), plan()
    assert not isinstance(late.next(), Wait)
    clock.now += 0.5
    assert not isinstance(current.next(), Wait)  # the minute turned over while the first was on its way
    current.admitted(ovh.rate_snapshots(_kong(1, 59)))  # counted in the new minute
    late.admitted(ovh.rate_snapshots(_kong(0, 0)))  # counted in the old one, handled afterwards
    assert not isinstance(plan().next(), Wait)  # the new minute's second request
    fourth = plan().next()
    assert isinstance(fourth, Wait) and fourth.delay == pytest.approx(59 + EDGE_MARGIN)


def test_late_request_readings_of_a_rolling_window_are_ignored():
    tx: dict = {}
    now, limits, cost = 1000.0, [rpm(10)], Cost(1, 0)
    # The platform counted two concurrent requests as the 5th and the 6th of the window.
    key = quota_observe(tx, "p", "m", RateSnapshot("minute", "requests", 10, 4), now, "UTC").key("p", "m")
    quota_charge(tx, limits, "p", "m", cost, now, "UTC")  # a 7th goes out meanwhile
    pending_add(tx, "seventh", [(key, 1.0)], now + 300)
    quota_observe(tx, "p", "m", RateSnapshot("minute", "requests", 10, 5), now + 0.01, "UTC")  # the 5th's, late
    at = now + 0.02
    for _ in range(3):
        assert quota_ready_at(tx, limits, "p", "m", cost, at, "UTC") == at
        quota_charge(tx, limits, "p", "m", cost, at, "UTC")
    assert quota_ready_at(tx, limits, "p", "m", cost, at, "UTC") > at + 1  # an 11th would be one too many


def test_lower_token_readings_are_taken_unless_their_window_ended():
    """Platforms hand unused output back, so a token count can drop for real within a window."""
    tx: dict = {}
    now = 1000.0
    rolling = quota_observe(tx, "p", "m1", RateSnapshot("minute", "tokens", 8000, 2000), now, "UTC")
    quota_observe(tx, "p", "m1", RateSnapshot("minute", "tokens", 8000, 7000), now + 1, "UTC")
    assert tx[rolling.key("p", "m1")]["tat"] == pytest.approx(now + 1 + 1000 * 60 / 8000)

    def minute(remaining: float, reset: float) -> RateSnapshot:
        return RateSnapshot("minute", "tokens", 8000, remaining, reset_in=reset, rolling=False)

    key = quota_observe(tx, "p", "m2", minute(7000, 59.0), now, "UTC").key("p", "m2")  # a new minute
    quota_observe(tx, "p", "m2", minute(0, 0.0), now + 0.1, "UTC")  # the previous minute's, handled late
    assert tx[key]["used"] == 1000
    quota_observe(tx, "p", "m2", minute(7500, 58.0), now + 1, "UTC")  # unused output handed back
    assert tx[key]["used"] == 500


def test_same_window_token_until_does_not_move_backward():
    """Token seen may drop, but a same-window until must not reopen the day early."""
    tx: dict = {}
    now = 10_000.0
    key = quota_observe(tx, "p", "m", RateSnapshot("day", "tokens", 1000, 100, reset_in=3600), now, "UTC").key(
        "p", "m"
    )
    assert tx[key]["until"] == pytest.approx(now + 3600)
    quota_observe(tx, "p", "m", RateSnapshot("day", "tokens", 1000, 200, reset_in=-50), now, "UTC")
    assert tx[key]["seen"] == 800  # unused tokens handed back
    assert tx[key]["until"] == pytest.approx(now + 3600)
    assert quota_ready_at(tx, [tpd(1000)], "p", "m", Cost(0, 500), now, "UTC") > now


def test_month_readings_keep_seen_across_short_until_drift():
    """Month half-window is weeks, so a 2-day until drift is still the same calendar month."""
    tx: dict = {}
    now = 1_000_000.0
    key = quota_observe(
        tx,
        "p",
        "*",
        RateSnapshot("month", "requests", 1000, 10, scope="provider", reset_in=20 * 86400),
        now,
        "UTC",
    ).key("p", "*")
    assert tx[key]["seen"] == 990
    quota_observe(
        tx,
        "p",
        "*",
        RateSnapshot("month", "requests", 1000, 900, scope="provider", reset_in=20 * 86400 + 2 * 86400),
        now + 10,
        "UTC",
    )
    assert tx[key]["seen"] == 990


def test_rolling_request_served_survives_limit_amount_change():
    """A later header with a different limit amount must not forget requests already counted."""
    tx: dict = {}
    now = 1000.0
    key = quota_observe(tx, "p", "m", RateSnapshot("minute", "requests", 10, 5, rolling=True), now, "UTC").key(
        "p", "m"
    )
    served = tx[key]["served"]
    assert served == pytest.approx(now + 5 * 6.0)
    quota_observe(tx, "p", "m", RateSnapshot("minute", "requests", 60, 60, rolling=True), now + 1, "UTC")
    assert tx[key]["amount"] == 60
    assert tx[key]["served"] == pytest.approx(served)
    assert tx[key]["tat"] >= served


def test_groq_charges_its_chat_template_too(server, clock):
    groq = Groq(api_key="k", http_client=server.client())
    model = groq.list_models()[0].id
    ai = make(server, clock, groq)
    router, prompt = ai._router, "x" * 40  # 10 tokens by the generic estimate

    def cost(provider: Provider, **params) -> float:
        return router.cost(provider, router.prepare(prompt, None, model, (), None, params)).tokens

    assert cost(groq) == 10 + 80 + 512  # Groq's reserve without max_tokens is unpredictable (2-961 seen)
    assert cost(groq, max_tokens=100) == 10 + 80 + 100
    assert cost(groq, max_tokens=0) == 10 + 80  # zero output reserve still needs the chat template
    assert cost(make_provider(server)) == 10 + 512  # other platforms keep the generic estimate

    ai.chat(prompt, model=model)  # the fake reports 4 tokens of usage and no limit headers
    with router.store.transaction() as tx:
        minute = tx[f"q:{groq.state_id}:{model}:minute:tokens"]
        day = tx[f"q:{groq.state_id}:{model}:day:tokens"]
    assert minute["tat"] == pytest.approx(clock.now + 4 * 60 / 8000, abs=1e-6)  # settled from what was charged
    assert day["used"] == 4


def test_suffixed_headers_name_their_window():
    snapshots = Provider("c", "https://c.example").rate_snapshots(
        {
            "x-ratelimit-limit-requests-day": "14400",
            "x-ratelimit-remaining-requests-day": "14399",
            "x-ratelimit-limit-tokens-minute": "30000",
            "x-ratelimit-remaining-tokens-minute": "29000",
            "x-ratelimit-limit-requests": "30",  # unsuffixed: window unknown for a generic provider
            "x-ratelimit-remaining-requests": "29",
        }
    )
    assert sorted((s.per, s.unit, s.limit, s.remaining) for s in snapshots) == [
        ("day", "requests", 14400, 14399),
        ("minute", "tokens", 30000, 29000),
    ]


def test_limit_metadata_and_globs():
    limit = Limit(20, "day", models=("gpt-oss-*",), source="docs", last_verified="2026-11-06")
    assert limit.applies_to("gpt-oss-120b")
    assert not limit.applies_to("llama")
    assert limit.key("p", "gpt-oss-120b") == "q:p:gpt-oss-120b:day:requests"
    assert Limit(1, "minute", scope="provider").key("p", "x") == "q:p:*:minute:requests"


def test_observe_upserts_learned_limits_by_identity(clock):
    store = MemoryStore()
    provider = Provider("p", "https://p.example/v1", api_key="k", models=["m1"])
    ai = make(FakeServer(), clock, provider, store=store)
    first = [RateSnapshot("minute", "requests", 10, 9), RateSnapshot("day", "tokens", 1000, 900, scope="provider")]
    second = [RateSnapshot("minute", "requests", 20, 19)]  # same identity, new amount
    ai._router.observe("p", "m1", first)
    ai._router.observe("p", "m1", second)
    with store.transaction() as tx:
        registry = list(tx[_learned_key(provider.state_id)])
    assert registry == [
        {"per": "day", "unit": "tokens", "scope": "provider", "model": None, "amount": 1000},
        {"per": "minute", "unit": "requests", "scope": "model", "model": "m1", "amount": 20},
    ]


def test_compiled_globs_match_fnmatchcase():
    patterns = ("*flash*", "provider:exact", "a_b%", "*-mini*", "CASE", "")
    samples = (
        "xflashy",
        "flash",
        "provider:exact",
        "a_b%",
        "foo-mini-bar",
        "CASE",
        "case",
        "plain",
        "",
    )
    assert Limit(1, "minute").applies_to("anything")  # models is None
    for pattern in patterns:
        limit = Limit(1, "minute", models=(pattern,))
        for sample in samples:
            assert limit.applies_to(sample) is fnmatch.fnmatchcase(sample, pattern)
    models = [
        ModelInfo(provider="p", id="flash-lite"),
        ModelInfo(provider="p", id="other"),
        ModelInfo(provider="provider", id="exact"),
    ]
    for model in models:
        expected = next(
            (
                i
                for i, pattern in enumerate(patterns)
                if fnmatch.fnmatchcase(model.key, pattern) or fnmatch.fnmatchcase(model.id, pattern)
            ),
            len(patterns),
        )
        assert _rank(model, patterns) == expected
        assert _rank(model, []) == 0


def test_order_reuses_explicit_model_ranks(clock):
    """Explicit ``models`` filter and prefer-order share the same ``_rank`` values."""
    provider = make_provider(FakeServer(), name="p", models=["flash-lite", "other", "flash"])
    ai = make(FakeServer(), clock, provider)
    request = ai._router.prepare("hi", None, ["*flash*", "other"], (), None, {})
    ordered = ai._router.order(request, ai._pools(), turn=0)
    assert [m.id for m in ordered] == ["flash-lite", "flash", "other"]


def test_status_single_pass_reads_models_for_all_providers(clock):
    store = MemoryStore()
    first = make(FakeServer(), clock, make_provider(FakeServer(), name="a", models=["m1"]), store=store)
    second = make(FakeServer(), clock, make_provider(FakeServer(), name="b", models=["m2"]), store=store)
    with store.transaction() as tx:
        tx[f"cd:m:{first.providers[0].state_id}|m1"] = {"until": clock.now + 10, "failures": 1}
        tx[f"cd:m:{second.providers[0].state_id}|m2"] = {"until": clock.now + 20, "failures": 1}
        tx[f"cd:p:{first.providers[0].state_id}"] = {"until": clock.now + 5, "failures": 1}
    combined = FreeAI(
        [first.providers[0], second.providers[0]], clock=clock, sleep=clock.sleep, store=store, max_wait=0.0
    )
    assert combined.status() == {"models": {"a:m1": 10.0, "b:m2": 20.0}, "providers": {"a": 5.0}}


def test_sqlite_store_shares_cooldowns_between_clients(tmp_path, clock):
    path = tmp_path / "state.db"
    server = FakeServer()
    server.on("m1", error(429, "slow down", headers={"Retry-After": "40"}))
    first = make(server, clock, make_provider(server), store=SQLiteStore(path))
    assert first.chat("hi").requested_model == "m2"

    other = FakeServer()
    second = make(other, clock, make_provider(other), store=SQLiteStore(path))  # another process
    assert second.status()["models"] == {"p:m1": 40}
    assert second.chat("hi").requested_model == "m2"
    assert other.models_called == ["m2"]


def test_sqlite_store_keeps_infinite_auth_cooldown_and_isolates_keys(tmp_path, clock):
    path = tmp_path / "state.db"
    server = FakeServer()
    server.on("m1", error(401, "bad key"))
    with pytest.raises(ExhaustedError, match="check the API keys"):
        make(server, clock, make_provider(server, models=["m1"]), store=SQLiteStore(path)).chat("hi")

    reopened = make(server, clock, make_provider(server, models=["m1"]), store=SQLiteStore(path))
    assert math.isinf(reopened.status()["providers"]["p"])

    fresh = Provider("p", "https://p.example/v1", api_key="new-key", models=["m1"], http_client=FakeServer().client())
    assert make(server, clock, fresh, store=SQLiteStore(path)).status()["providers"] == {}  # a new key is a new account


def test_failure_kind_rejected_has_no_cooldown():
    from freeai import CooldownPolicy

    assert CooldownPolicy().duration(FailureKind.REJECTED, 3, None) == 0


def test_sqlite_flush_stores_compact_json(tmp_path):
    store = SQLiteStore(tmp_path / "compact.db")
    try:
        payload = {"until": 100.0, "failures": 2}
        with store.transaction() as tx:
            tx["k"] = payload
        row = store._conn().execute("SELECT value FROM freeai_state WHERE key = ?", ("k",)).fetchone()
        assert row is not None
        assert row[0] == json.dumps(payload, separators=(",", ":"))
        with store.transaction() as tx:
            assert tx["k"] == payload
    finally:
        store.close()


def test_sqlite_keys_with_prefix_matches_iter_and_escapes_like(tmp_path):
    store = SQLiteStore(tmp_path / "prefix.db")
    try:
        with store.transaction() as tx:
            tx["cd:m:sid|alpha"] = {"until": 100.0, "failures": 1}
            tx["cd:m:sid|beta"] = {"until": 100.0, "failures": 1}
            tx["cd:m:other|gamma"] = {"until": 100.0, "failures": 1}
            tx["cd:m:sid_x|delta"] = {"until": 100.0, "failures": 1}  # `_` must not be a LIKE wildcard
            tx["plain%one"] = {"until": 100.0, "failures": 1}
            tx["plainXone"] = {"until": 100.0, "failures": 1}
            tx["CaseKey"] = {"until": 100.0, "failures": 1}

        with store.transaction() as tx:
            prefix = "cd:m:sid|"
            assert list(tx.keys_with_prefix(prefix)) == ["cd:m:sid|alpha", "cd:m:sid|beta"]
            assert sorted(k for k in tx if k.startswith(prefix)) == list(tx.keys_with_prefix(prefix))
            assert [k for k, _ in tx.items_with_prefix(prefix)] == list(tx.keys_with_prefix(prefix))
            assert cooldown_snapshot(tx, prefix, now=50.0) == {"alpha": 50.0, "beta": 50.0}
            for key, entry in tx.items_with_prefix(prefix):
                assert entry == tx[key]

            assert list(tx.keys_with_prefix("plain%")) == ["plain%one"]
            assert "plainXone" not in tx.keys_with_prefix("plain%")
            assert list(tx.keys_with_prefix("case")) == []  # LIKE must not beat startswith on ASCII case
            assert _like_prefix("a_b%") == r"a\_b\%%"

            tx["cd:m:sid|dirty"] = {"until": 100.0, "failures": 1}
            del tx["cd:m:sid|alpha"]
            assert list(tx.keys_with_prefix(prefix)) == ["cd:m:sid|beta", "cd:m:sid|dirty"]
            assert sorted(k for k in tx if k.startswith(prefix)) == list(tx.keys_with_prefix(prefix))
            assert [k for k, _ in tx.items_with_prefix(prefix)] == list(tx.keys_with_prefix(prefix))
            assert cooldown_snapshot(tx, prefix, now=50.0) == {"beta": 50.0, "dirty": 50.0}
    finally:
        store.close()


def test_memory_keys_with_prefix_matches_iter():
    store = MemoryStore()
    with store.transaction() as tx:
        tx["cd:m:sid|alpha"] = {"until": 100.0, "failures": 1}
        tx["cd:m:sid|beta"] = {"until": 100.0, "failures": 1}
        tx["cd:m:other|gamma"] = {"until": 100.0, "failures": 1}
        tx["q:sid|*:minute:requests"] = {"used": 1}
        prefix = "cd:m:sid|"
        assert list(tx.keys_with_prefix(prefix)) == ["cd:m:sid|alpha", "cd:m:sid|beta"]
        assert sorted(k for k in tx if k.startswith(prefix)) == list(tx.keys_with_prefix(prefix))
        assert [k for k, _ in tx.items_with_prefix(prefix)] == list(tx.keys_with_prefix(prefix))
        assert cooldown_snapshot(tx, prefix, now=50.0) == {"alpha": 50.0, "beta": 50.0}
        for key, entry in tx.items_with_prefix(prefix):
            assert entry == tx[key]
        del tx["cd:m:sid|alpha"]
        tx["cd:m:sid|dirty"] = {"until": 100.0, "failures": 1}
        assert list(tx.keys_with_prefix(prefix)) == ["cd:m:sid|beta", "cd:m:sid|dirty"]
        assert cooldown_snapshot(tx, prefix, now=50.0) == {"beta": 50.0, "dirty": 50.0}


# -- header snapshots and requests the platform has not counted yet -----------------------------


def _counting(limit: int, on_request=None):
    """A platform that counts each request when it accepts it and reports what is left."""
    accepted: list[dict] = []

    def reply(body: dict) -> httpx.Response:
        if on_request is not None:
            on_request(accepted)
        accepted.append(body)
        left = {"x-ratelimit-limit-requests-minute": str(limit)}
        left["x-ratelimit-remaining-requests-minute"] = str(max(0, limit - len(accepted)))
        response = sse(delta("hi"), delta(finish="stop")) if body.get("stream") else ok(model=body["model"])
        response.headers.update(left)
        return response

    return reply


def _pending(ai) -> list[str]:
    """Reservations still recorded in the store (none once every request is over)."""
    with ai._router.store.transaction() as tx:
        return [key for key in tx if key.startswith("p:")]


def test_snapshot_keeps_reservations_the_platform_has_not_accepted(server, clock):
    # One request is still on its way (reserved here, not yet counted there) while two others are
    # answered. Overwriting the counter with the server's view would forget the first one.
    outcomes: list[str] = []
    entered: list[bool] = []

    def meanwhile(accepted: list[dict]) -> None:
        if entered:
            return  # one of the inner requests
        entered.append(True)
        for _ in range(3):
            try:
                ai.chat("inner")
                outcomes.append("sent")
            except ExhaustedError:
                outcomes.append("blocked")

    server.on("m1", _counting(3, meanwhile))
    ai = make(server, clock, make_provider(server, models=["m1"], limits=[rpm(3)]))
    ai.chat("outer")
    assert outcomes == ["sent", "sent", "blocked"]  # the outer request still holds the third slot
    assert _pending(ai) == []


def test_accepted_streams_are_counted_once(server, clock):
    # Stream headers mean the platform accepted (and counted) the request, so the next headers
    # already include it. Ending a stream must not re-apply its older headers either.
    server.on("m1", _counting(3))
    ai = make(server, clock, make_provider(server, models=["m1"], limits=[rpm(3)]))
    first, second = ai.stream("hi"), ai.stream("hi")
    next(first)
    next(second)
    ai.chat("hi")  # third of three
    for finish in (lambda: None, lambda: (list(first), list(second))):
        finish()
        with pytest.raises(ExhaustedError) as info:
            ai.chat("hi")
        assert info.value.next_available_in == pytest.approx(20.0 + EDGE_MARGIN)  # three used, not five or one
    assert _pending(ai) == []


def test_ending_a_stream_does_not_reapply_its_token_reading(server, clock):
    # A stale request count is ignored anyway (counts never drop), but token counts can drop for
    # real when unused output is handed back, so an older token reading would be taken as is.
    left = iter(["7000", "5000"])

    def reply(body: dict) -> httpx.Response:
        response = sse(delta("hi"), delta(finish="stop"))
        response.headers.update(
            {"x-ratelimit-limit-tokens-minute": "8000", "x-ratelimit-remaining-tokens-minute": next(left)}
        )
        return response

    server.on("m1", reply)
    provider = make_provider(server, models=["m1"], limits=[tpm(8000)])
    ai = make(server, clock, provider)
    key = f"q:{provider.state_id}:m1:minute:tokens"

    def tat() -> float:
        with ai._router.store.transaction() as tx:
            return float(tx[key]["tat"])

    first, second = ai.stream("hi"), ai.stream("hi")
    next(first)  # 1000 tokens used
    next(second)  # 3000 tokens used: the platform's latest count
    assert tat() == pytest.approx(clock.now + 3000 * 60 / 8000)
    list(first)  # ends after the second stream's headers arrived
    assert tat() == pytest.approx(clock.now + 3000 * 60 / 8000)  # not back to 1000 used
    list(second)
    assert tat() == pytest.approx(clock.now + 3000 * 60 / 8000)


def test_async_streams_report_acceptance_too(server, clock):
    server.on("m1", _counting(3))
    provider = Provider(
        "p", "https://p.example/v1", api_key="k", models=["m1"], limits=[rpm(3)], async_http_client=server.aclient()
    )

    async def main():
        async with AsyncFreeAI([provider], clock=clock, max_wait=0.0) as ai:
            first, second = ai.stream("hi"), ai.stream("hi")
            await anext(first)
            await anext(second)
            await ai.chat("hi")
            for stream in (first, second):
                async for _ in stream:
                    pass
            with pytest.raises(ExhaustedError) as info:
                await ai.chat("hi")
            return info.value.next_available_in, _pending(ai)

    wait, pending = asyncio.run(main())
    assert wait == pytest.approx(20.0 + EDGE_MARGIN)
    assert pending == []


def test_observe_adds_unaccepted_reservations_to_the_server_count():
    now = 1_700_000_000.0
    key = "q:p:m:minute:requests"
    tx: dict = {key: {"tat": now + 9 * 6, "amount": 10}}  # local estimate: 9 of 10
    pending_add(tx, "elsewhere", [(key, 3.0)], expires_at=now + 300)
    snapshot = RateSnapshot("minute", "requests", limit=10, remaining=8)
    quota_observe(tx, "p", "m", snapshot, now, "UTC")
    assert tx[key]["tat"] - now == pytest.approx(5 * 6)  # 2 counted by the server + 3 not accepted yet


def test_reservations_of_a_dead_process_expire():
    now = 1_700_000_000.0
    key = "q:p:m:minute:requests"
    tx: dict = {}
    pending_add(tx, "dead", [(key, 3.0)], expires_at=now + 300)
    snapshot = RateSnapshot("minute", "requests", limit=10, remaining=8)
    quota_observe(tx, "p", "m", snapshot, now + 299, "UTC")
    assert tx[key]["tat"] - (now + 299) == pytest.approx(5 * 6)
    quota_observe(tx, "p", "m", snapshot, now + 301, "UTC")
    assert tx[key]["tat"] - (now + 301) == pytest.approx(2 * 6)
    assert "p:" + key not in tx  # pruned


def test_corrupt_pending_map_is_replaced():
    now = 1_700_000_000.0
    key = "q:p:m:minute:requests"
    pending_key = "p:" + key
    tx: dict = {pending_key: ["not", "a", "map"]}
    pending_add(tx, "r1", [(key, 2.0)], expires_at=now + 300)
    assert tx[pending_key] == {"r1": [2.0, now + 300]}
    tx[pending_key] = "corrupt"
    snapshot = RateSnapshot("minute", "requests", limit=10, remaining=8)
    quota_observe(tx, "p", "m", snapshot, now, "UTC")
    assert pending_key not in tx
    assert tx[key]["tat"] - now == pytest.approx(2 * 6)


def test_corrupt_pending_rows_are_skipped():
    from freeai.limits import _pending_total

    now = 1_700_000_000.0
    key = "q:p:m:minute:requests"
    pending_key = "p:" + key
    tx: dict = {
        pending_key: {
            "ok": [3.0, now + 300],
            "short": [1.0],
            "text": "nope",
            "expired": [9.0, now - 1],
        }
    }
    assert _pending_total(tx, key, now) == 3.0
    assert tx[pending_key] == {"ok": [3.0, now + 300]}


def test_corrupt_quota_and_observe_rows_are_ignored():
    import threading

    from freeai._observe import Observe
    from freeai._routing_types import _learned_key, _provider_key

    now = 1_700_000_000.0
    key = "q:p:m:minute:requests"
    tx: dict = {key: "corrupt"}
    quota_observe(tx, "p", "m", RateSnapshot("minute", "requests", limit=10, remaining=8), now, "UTC")
    assert isinstance(tx[key], dict) and "tat" in tx[key]

    provider = Provider("p", "https://p.example/v1", api_key="k", models=["m"])
    store = MemoryStore()
    with store.transaction() as raw:
        raw[_learned_key(provider.state_id)] = "nope"
        raw[_provider_key(provider.state_id)] = "bad"
        raw[f"cd:m:{provider.state_id}|m"] = "bad"
    observe = Observe(
        lock=threading.Lock(),
        store=store,
        by_name={"p": provider},
        providers=[provider],
        clock=lambda: 50.0,
    )
    assert observe.status() == {"models": {}, "providers": {}}
    observe.observe("p", "m", [RateSnapshot("minute", "requests", limit=10, remaining=9)])
    with store.transaction() as raw:
        assert isinstance(raw[_learned_key(provider.state_id)], list)

    with store.transaction() as raw:
        raw[_learned_key(provider.state_id)] = [{"per": "minute"}]
    observe.observe("p", "m", [RateSnapshot("hour", "tokens", limit=100, remaining=50)])
    with store.transaction() as raw:
        rows = raw[_learned_key(provider.state_id)]
        assert all("unit" in row and "amount" in row for row in rows)


def test_unaccepted_reservations_are_shared_between_processes(tmp_path, clock):
    # Process A has sent a request the platform has not counted yet while process B, sharing the
    # store, gets two answers. B must add A's reservation, or it would wipe it from the shared count.
    outcomes: list[str] = []
    entered: list[bool] = []

    def meanwhile(accepted: list[dict]) -> None:
        if entered:
            return
        entered.append(True)
        for _ in range(3):
            try:
                other.chat("b")
                outcomes.append("sent")
            except ExhaustedError:
                outcomes.append("blocked")

    server = FakeServer()
    server.on("m1", _counting(3, meanwhile))
    path = tmp_path / "state.db"
    first = make(server, clock, make_provider(server, models=["m1"], limits=[rpm(3)]), store=SQLiteStore(path))
    other = make(server, clock, make_provider(server, models=["m1"], limits=[rpm(3)]), store=SQLiteStore(path))
    first.chat("a")
    assert outcomes == ["sent", "sent", "blocked"]
    assert _pending(first) == []


def test_observe_fixed_window_trusts_the_server_plus_pending():
    now = 1_700_000_000.0
    key = "q:p:m:day:requests"
    tx: dict = {key: {"period": "2023-11-14", "used": 900.0, "amount": 1000}}
    snapshot = RateSnapshot("day", "requests", limit=1000, remaining=995, reset_in=60.0)  # server reset early
    pending_add(tx, "elsewhere", [(key, 2.0)], expires_at=now + 300)
    quota_observe(tx, "p", "m", snapshot, now, "UTC")
    assert tx[key]["used"] == 7.0
    assert tx[key]["until"] - now == pytest.approx(60.0)


# -- presets and HTTP clients ---------------------------------------------------------------------


def test_gemini_declares_free_tier_limits_per_model(server, clock):
    gemini = Gemini(api_key="k", http_client=server.client())
    ai = make(server, clock, gemini)
    for _ in range(5):
        ai.chat("hi", model="gemini-3.8-flash")
    with pytest.raises(ExhaustedError) as info:
        ai.chat("hi", model="gemini-3.8-flash")  # Flash: 5 RPM
    assert info.value.next_available_in == pytest.approx(12.0 + EDGE_MARGIN)
    for _ in range(10):
        ai.chat("hi", model="gemini-3.5-flash-lite")  # Flash-Lite: 10 RPM
    with pytest.raises(ExhaustedError):
        ai.chat("hi", model="gemini-3.5-flash-lite")
    unlimited = make(server, clock, Gemini(api_key="k", http_client=server.client(), limits=[]))
    for _ in range(6):
        unlimited.chat("hi", model="gemini-3.8-flash")  # a paid project opts out


def test_provider_http_clients_allow_many_concurrent_streams():
    provider = Provider("p", "https://p.example/v1", api_key="k", models=["m"])
    for client in (provider._client(), provider._async_client()):
        pool = client._transport._pool  # type: ignore[attr-defined]
        assert (pool._max_connections, pool._max_keepalive_connections) == (1000, 100)


def test_sqlite_corrupt_json_row_raises_keyerror_and_is_deleted(tmp_path):
    import sqlite3

    path = tmp_path / "corrupt.db"
    store = SQLiteStore(path)
    store.close()
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE IF NOT EXISTS freeai_state (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute("INSERT INTO freeai_state (key, value) VALUES (?, ?)", ("bad", "{not-json"))
    conn.commit()
    conn.close()

    store = SQLiteStore(path)
    with store.transaction() as tx:
        with pytest.raises(KeyError):
            _ = tx["bad"]
        tx["ok"] = {"n": 1}
    with store.transaction() as tx:
        assert "bad" not in tx
        assert tx["ok"] == {"n": 1}
    store.close()


def test_sqlite_store_stays_closed_after_close(tmp_path):
    store = SQLiteStore(tmp_path / "closed.db")
    with store.transaction() as tx:
        tx["a"] = 1
    store.close()
    with pytest.raises(RuntimeError, match="closed"):
        store.version()
    with pytest.raises(RuntimeError, match="closed"):
        with store.transaction():
            pass
