"""``strategy="quota"``: spend what refills soonest first and keep daily allowances for the overflow."""

import asyncio
from collections import Counter
from typing import Any

import pytest

from freeai import (
    ZAI,
    AsyncFreeAI,
    BillingWarning,
    Cloudflare,
    ExhaustedError,
    FreeAI,
    Kilo,
    ModelInfo,
    Provider,
    rpd,
    rpm,
    tpm,
)
from freeai.limits import Cost, horizon_seconds, per_month, quota_charge, quota_refill, seconds_until

from .conftest import FakeClock, FakeServer

NOON = 1_700_000_000.0 - 1_700_000_000.0 % 86400 + 43200  # 12:00 UTC on some day
ONE = Cost(1.0)


def plain(server: FakeServer, name: str, models: Any = ("m1",), **kwargs: Any) -> Provider:
    return Provider(
        name, f"https://{name}.example/v1", api_key="k", models=list(models), http_client=server.client(), **kwargs
    )


def make(clock: FakeClock, *providers: Provider, **kwargs: Any) -> FreeAI:
    kwargs.setdefault("max_wait", 0.0)
    return FreeAI(list(providers), clock=clock, sleep=clock.sleep, **kwargs)


def served(ai: FreeAI, count: int, prompt: str = "hi") -> list[str]:
    return [ai.chat(prompt).provider for _ in range(count)]


def test_refill_is_a_requests_share_of_what_comes_back():
    tx: dict[str, Any] = {}

    def refill(limits, cost=ONE, now=NOON):
        return quota_refill(tx, limits, "p", "m", cost, now, "UTC")

    assert refill([]) == 0.0  # nothing declared: nothing to save
    assert refill([rpm(40)]) == pytest.approx(1.5)
    assert refill([tpm(8000)], Cost(1.0, 4000.0)) == pytest.approx(30.0)  # larger requests cost more
    assert refill([rpm(40), tpm(8000)], Cost(1.0, 4000.0)) == pytest.approx(30.0)  # the scarcest window decides
    assert refill([rpm(30, models=("other",))]) == 0.0  # a limit of another model

    daily = [rpd(50)]
    assert refill(daily) == pytest.approx(43200 / 50)  # a fresh day: its share of the time left
    quota_charge(tx, daily, "p", "m", Cost(49.0), NOON, "UTC")
    assert refill(daily) == pytest.approx(43200)  # the last one costs the rest of the day
    assert refill(daily, now=NOON + 43200 - 60) == pytest.approx(60)  # unspent quota gets cheap before the reset
    quota_charge(tx, daily, "p", "m", Cost(1.0), NOON, "UTC")
    assert refill(daily) == float("inf")  # nothing left
    assert refill(daily, now=NOON + 43200) == pytest.approx(86400 / 50)  # the next day

    monthly = [per_month(1000, scope="provider")]
    assert quota_refill({}, monthly, "p", "m", Cost(1.0), NOON, "UTC") == pytest.approx(
        seconds_until("month", NOON, "UTC") / 1000
    )


def test_an_allowance_of_unknown_size_costs_the_rest_of_its_window():
    assert horizon_seconds("minute", NOON, "UTC") == 60
    assert horizon_seconds("hour", NOON, "UTC") == 3600
    assert horizon_seconds("day", NOON, "UTC") == 43200
    assert horizon_seconds("day", NOON, "America/Los_Angeles") == seconds_until("day", NOON, "America/Los_Angeles")


def test_quota_strategy_spends_minute_quota_before_daily(clock):
    def providers():
        return plain(FakeServer(), "daily", limits=[rpd(50)]), plain(FakeServer(), "minute", limits=[rpm(4)])

    by_order = make(clock, *providers(), strategy="priority")
    assert served(by_order, 6) == ["daily"] * 6  # the order they were given in

    ai = make(clock, *providers())  # the default strategy
    assert ai._router.strategy == "quota"
    assert served(ai, 6) == ["minute"] * 4 + ["daily"] * 2  # the daily one only takes the overflow
    clock.now += 16.5  # a quarter of the minute (and the edge margin) later, one is back
    assert served(ai, 2) == ["minute", "daily"]


def test_quota_strategy_keeps_prefer_explicit_models_and_routers_last(clock):
    def providers():
        router = ModelInfo("minute", "auto", is_router=True)
        return (
            plain(FakeServer(), "daily", limits=[rpd(50)]),
            plain(FakeServer(), "minute", models=["m1", router], limits=[rpm(40)]),
        )

    ai = make(clock, *providers(), strategy="quota", prefer=["daily:*"])
    assert served(ai, 2) == ["daily"] * 2  # what the caller prefers stays first

    ai = make(clock, *providers(), strategy="quota")
    assert ai.chat("hi", model=["daily:m1", "minute:m1"]).provider == "daily"  # so does an explicit order
    assert ai.chat("hi", model="*:m1").provider == "minute"  # one pattern is one group: quota decides
    assert ai.chat("hi", model="fast").provider == "minute"  # an alias only prefers; nothing matches here

    ai = make(clock, plain(FakeServer(), "daily", limits=[rpd(50)]), providers()[1], strategy="quota", exclude=["m1"])
    assert ai.chat("hi").requested_model == "auto"  # the only candidate left
    ai = make(clock, *providers(), strategy="quota")
    for _ in range(45):  # the router shares the minute provider's window but is tried after every model
        assert ai.chat("hi").requested_model == "m1"


def test_an_uncounted_allowance_is_not_treated_as_unlimited(clock):
    clock.now = NOON

    def opaque():
        return plain(FakeServer(), "opaque", uncounted="day")

    # A known daily budget is spent before one whose size nobody knows. Its last request costs the
    # same (the rest of the day), so there the given order decides.
    ai = make(clock, opaque(), plain(FakeServer(), "known", limits=[rpd(3)]), strategy="quota")
    assert served(ai, 4) == ["known", "known", "opaque", "opaque"]
    ai = make(clock, plain(FakeServer(), "known", limits=[rpd(1)]), opaque(), strategy="quota")
    assert served(ai, 3) == ["known", "opaque", "opaque"]  # and when it is spent the other takes over

    ai = make(clock, opaque(), plain(FakeServer(), "open"), strategy="quota")
    assert served(ai, 3) == ["open"] * 3  # nothing declared at all: e.g. the caller's own server

    clock.now = NOON + 43200 - 30  # half a minute before the reset the unknown budget is cheaper than a minute's
    ai = make(clock, plain(FakeServer(), "minute", limits=[rpm(1)]), opaque(), strategy="quota")
    assert ai.chat("hi").provider == "opaque"


def test_presets_name_the_allowances_they_cannot_count():
    assert Cloudflare(api_key="k", account_id="a").uncounted == "day"  # daily neurons
    assert ZAI(api_key="k").uncounted == "day"
    assert Kilo(api_key="k").uncounted == "hour"
    assert Kilo().uncounted is None  # anonymous use declares its 200 requests per hour
    assert Kilo(api_key="k", uncounted="day").uncounted == "day"
    assert plain(FakeServer(), "p").uncounted is None


def test_quota_strategy_spends_free_allowances_before_credit_and_money(clock):
    def providers():
        return (
            plain(FakeServer(), "paid", tier="paid"),
            plain(FakeServer(), "trial", tier="trial", limits=[rpm(40)]),
            plain(FakeServer(), "free", tier="free", limits=[rpd(2)]),
        )

    with pytest.warns(BillingWarning):
        ai = make(clock, *providers(), strategy="quota")
    assert served(ai, 4) == ["free", "free", "trial", "trial"]  # the paid one has no limit and would cost nothing
    with pytest.warns(BillingWarning):
        assert make(clock, *providers(), strategy="priority").chat("hi").provider == "paid"  # the given order


def test_a_provider_that_names_no_tier_is_used_after_the_free_ones(clock):
    # A hand-made provider declares no limit and no tier. By quota alone it would look free of
    # cost and take every request, although it may be somebody's paid endpoint.
    def providers():
        return (
            plain(FakeServer(), "own"),  # tier "unknown"
            plain(FakeServer(), "free", tier="free", limits=[rpm(2)]),
        )

    ai = make(clock, *providers())
    assert served(ai, 4) == ["free", "free", "own", "own"]  # still used as soon as nothing free is ready
    mine = plain(FakeServer(), "own", tier="free")  # the caller says it costs nothing
    ai = make(clock, mine, providers()[1])
    assert served(ai, 3) == ["own"] * 3


def test_large_requests_go_where_tokens_are_plentiful(clock):
    small = plain(FakeServer(), "small", limits=[rpm(30), tpm(8000)])
    big = plain(FakeServer(), "big", limits=[rpm(10), tpm(250_000)])
    ai = make(clock, small, big, strategy="quota")
    assert ai.chat("hi").provider == "small"  # 512 tokens are 4 s of 8000 a minute; a request is 6 s of 10
    assert ai.chat("x" * 20000).provider == "big"  # 5512 tokens are 41 s of 8000 a minute


def test_unspent_daily_quota_is_used_before_it_is_lost(clock):
    def providers():
        return plain(FakeServer(), "minute", limits=[rpm(2)]), plain(FakeServer(), "daily", limits=[rpd(1000)])

    clock.now = NOON
    assert make(clock, *providers(), strategy="quota").chat("hi").provider == "minute"  # 30 s against 43 s
    clock.now = NOON + 43200 - 600
    assert make(clock, *providers(), strategy="quota").chat("hi").provider == "daily"  # 0.6 s: it resets anyway


def test_quota_strategy_never_waits_while_a_later_group_is_ready(clock):
    minute = plain(FakeServer(), "minute", limits=[rpm(1)])
    daily = plain(FakeServer(), "daily", limits=[rpd(50)])
    ai = make(clock, minute, daily, strategy="quota", prefer=["minute:*"], max_wait=30.0)
    assert served(ai, 3) == ["minute", "daily", "daily"]
    assert clock.sleeps == []


def test_async_client_uses_the_quota_strategy(clock):
    async def sleep(seconds: float) -> None:
        clock.sleep(seconds)

    def provider(name: str, **kwargs: Any) -> Provider:
        return Provider(
            name,
            f"https://{name}.example/v1",
            api_key="k",
            models=["m1"],
            async_http_client=FakeServer().aclient(),
            **kwargs,
        )

    async def main() -> list[str]:
        providers = [provider("daily", limits=[rpd(50)]), provider("minute", limits=[rpm(2)])]
        async with AsyncFreeAI(providers, clock=clock, sleep=sleep, max_wait=0.0, strategy="quota") as ai:
            return [(await ai.chat("hi")).provider for _ in range(3)]

    assert asyncio.run(main()) == ["minute", "minute", "daily"]


def _free_tiers() -> list[Provider]:
    """The shape of the free presets: two that only limit per minute, three with a daily allowance."""
    return [
        plain(FakeServer(), "gateway", limits=[rpm(20, scope="provider"), rpd(50, scope="provider")]),
        plain(FakeServer(), "fast", models=["a", "b", "c"], limits=[rpm(30), rpd(1000)]),
        plain(FakeServer(), "flash", models=["lite", "full"], limits=[rpm(10), rpd(500, models=("lite",))]),
        plain(FakeServer(), "trial-endpoints", limits=[rpm(40, scope="provider")]),
        plain(FakeServer(), "anonymous", models=[f"m{i}" for i in range(6)], limits=[rpm(2)]),
    ]


def _steady_load(clock: FakeClock, strategy: str, minutes: int = 10, per_minute: int = 60) -> Counter[str]:
    clock.now = NOON
    ai = make(clock, *_free_tiers(), strategy=strategy)
    counts: Counter[str] = Counter()
    for _ in range(minutes * per_minute):
        counts[ai.chat("hi", max_tokens=64).provider] += 1
        clock.now += 60 / per_minute
    return counts


def _a_day_with_a_late_peak(clock: FakeClock, **options: Any) -> tuple[int, int]:
    """Thirty requests a minute for half an hour, then ninety a minute for ten; (answered, turned away)."""
    clock.now = NOON
    daily = plain(FakeServer(), "daily", models=["a", "b"], tier="free", limits=[rpm(30), rpd(100)])
    minute = plain(FakeServer(), "minute", tier="free", limits=[rpm(40, scope="provider")])
    ai = make(clock, daily, minute, **options)
    answered = turned_away = 0
    for per_minute, minutes in ((30, 30), (90, 10)):
        for _ in range(per_minute * minutes):
            try:
                ai.chat("hi", max_tokens=16)
                answered += 1
            except ExhaustedError:
                turned_away += 1
            clock.now += 60 / per_minute
    return answered, turned_away


def test_the_default_answers_more_when_demand_peaks_late(clock):
    # Below the 40 a minute that refill, nothing is turned away either way. In the given order the
    # 200 a day are spent on that quiet stretch and are missing when demand exceeds 40 a minute.
    by_order = _a_day_with_a_late_peak(clock, strategy="priority")
    by_default = _a_day_with_a_late_peak(clock)
    assert by_default == _a_day_with_a_late_peak(clock, strategy="quota")
    assert sum(by_order) == sum(by_default) == 1800
    assert by_default[0] - by_order[0] >= 190  # the whole daily allowance, and no more than that
    assert by_default[0] - by_order[0] <= 200


def test_daily_allowances_last_longer_under_steady_load(clock):
    """Sixty requests a minute for ten minutes: all served either way, but from different allowances."""
    daily = ("gateway", "fast", "flash")
    by_priority = _steady_load(clock, "priority")
    by_quota = _steady_load(clock, "quota")
    assert sum(by_priority.values()) == sum(by_quota.values()) == 600

    # In the given order the gateway's 50 a day are gone at once and the daily allowances carry it all.
    assert by_priority["gateway"] == 50
    assert sum(by_priority[name] for name in daily) == 600
    # By quota the two that refill every minute (40 + 12) carry most of it and the gateway is untouched.
    assert by_quota["gateway"] == 0
    assert by_quota["trial-endpoints"] + by_quota["anonymous"] > 480
    assert sum(by_quota[name] for name in daily) < 120
