"""``adapt_limits``: daily limits declared as estimates are measured against what the platform does."""

import asyncio
import math
from typing import Any

import httpx
import pytest

from freeai import NVIDIA, AsyncFreeAI, ExhaustedError, FreeAI, Gemini, Groq, Provider, SQLiteStore, rpd, rpm, tpd
from freeai._adapt import adapts, backoff, grow, learned_amount, probe_ready_at
from freeai.limits import per_month

from .conftest import FakeClock, ok

DAY = 86400.0
MORNING = 20 * DAY + 8 * 3600  # 08:00 UTC


class Upstream:
    """A platform with its own idea of the limit: so many accepted per sliding minute and per UTC day."""

    def __init__(
        self,
        clock: FakeClock,
        *,
        per_minute: int | None = None,
        per_day: int | None = None,
        headers: bool = False,
        daily_message: str = "free tier limit of requests per day reached",
        daily_wait: float | None = None,
    ) -> None:
        self.clock = clock
        self.per_minute = per_minute
        self.per_day = per_day
        self.headers = headers
        self.daily_message = daily_message
        self.daily_wait = daily_wait  # the Retry-After of a daily refusal, if not the time until midnight
        self.accepted: list[float] = []
        self.refused: list[float] = []
        self.before: Any = None  # called with the number of requests seen so far, before answering
        self.broken = 0  # answer this many requests with a 500

    def __call__(self, request: httpx.Request) -> httpx.Response:
        now = self.clock.now
        if self.before is not None:
            self.before(len(self.accepted) + len(self.refused))
        if self.broken:
            self.broken -= 1
            return httpx.Response(500, json={"error": {"code": 500, "message": "boom"}})
        today = [t for t in self.accepted if t // DAY == now // DAY]
        if self.per_day is not None and len(today) >= self.per_day:
            return self._refuse(self.daily_message, self.daily_wait or DAY - now % DAY)
        recent = [t for t in self.accepted if t > now - 60]
        if self.per_minute is not None and len(recent) >= self.per_minute:
            return self._refuse("too many requests", recent[0] + 60 - now)
        self.accepted.append(now)
        headers = None
        if self.headers and self.per_day is not None:
            headers = {
                "x-ratelimit-limit-requests-day": str(self.per_day),
                "x-ratelimit-remaining-requests-day": str(self.per_day - len(today) - 1),
            }
        return ok(headers=headers)

    def _refuse(self, message: str, wait: float) -> httpx.Response:
        self.refused.append(self.clock.now)
        return httpx.Response(
            429, json={"error": {"code": 429, "message": message}}, headers={"Retry-After": str(math.ceil(wait))}
        )


def client(clock: FakeClock, upstream: Any, *limits: Any, models: Any = ("m",), **kwargs: Any) -> FreeAI:
    provider = Provider(
        "p",
        "https://p.example/v1",
        api_key="k",
        models=list(models),
        limits=list(limits),
        http_client=httpx.Client(transport=httpx.MockTransport(upstream)),
    )
    kwargs.setdefault("max_wait", 0.0)
    kwargs.setdefault("adapt_limits", True)
    return FreeAI(provider, clock=clock, sleep=clock.sleep, **kwargs)


def burst(ai: FreeAI, tries: int = 12) -> int:
    """How many of ``tries`` requests sent at one moment are answered."""
    answered = 0
    for _ in range(tries):
        try:
            ai.chat("hi")
        except ExhaustedError:
            continue
        answered += 1
    return answered


def key_of(ai: FreeAI, per: str = "day", target: str = "m") -> str:
    return f"q:{ai.providers[0].state_id}:{target}:{per}:requests"


def learned(ai: FreeAI, per: str = "day", target: str = "m") -> float | None:
    with ai._router.store.transaction() as tx:
        return learned_amount(tx, key_of(ai, per, target))


def test_presets_mark_the_numbers_that_are_guesses_and_only_daily_requests_are_learned():
    gemini = Gemini(api_key="k").limits
    assert all(limit.estimate for limit in gemini)  # a third-party table
    assert sorted((limit.per, limit.unit) for limit in gemini if adapts(limit)) == [("day", "requests")] * 2
    assert all(limit.estimate and not adapts(limit) for limit in NVIDIA(api_key="k").limits)  # a forum post, per minute
    assert not any(limit.estimate for limit in Groq(api_key="k").limits)  # documented and measured
    assert adapts(per_month(1000, estimate=True))
    assert not adapts(rpd(20)) and not adapts(tpd(1000, estimate=True)) and not adapts(rpm(5, estimate=True))


def test_adapting_is_the_default(clock):
    clock.now = MORNING
    upstream = Upstream(clock, per_day=4)
    provider = Provider(
        "p",
        "https://p.example/v1",
        api_key="k",
        models=["m"],
        limits=[rpd(2, estimate=True)],
        http_client=httpx.Client(transport=httpx.MockTransport(upstream)),
    )
    ai = FreeAI(provider, clock=clock, sleep=clock.sleep, max_wait=0.0)  # no adapt_limits given
    assert ai._router.adapt_limits is True
    assert burst(ai) == 4  # probed up to what the platform accepts


def test_an_estimate_is_a_hard_limit_unless_adapting(clock):
    clock.now = MORNING
    upstream = Upstream(clock, per_day=6)
    ai = client(clock, upstream, rpd(2, estimate=True), adapt_limits=False)
    assert burst(ai) == 2
    assert (len(upstream.accepted), len(upstream.refused)) == (2, 0)
    assert learned(ai) is None


def test_a_limit_that_is_no_estimate_is_never_probed(clock):
    clock.now = MORNING
    upstream = Upstream(clock, per_day=6)
    assert burst(client(clock, upstream, rpd(2))) == 2
    assert len(upstream.accepted) == 2


def test_estimates_per_minute_are_left_alone(clock):
    upstream = Upstream(clock, per_minute=3)
    ai = client(clock, upstream, rpm(6, estimate=True))
    assert burst(ai) == 3  # the fourth is refused; a minute's count is no measure of the platform's
    assert learned(ai, "minute") is None
    generous = Upstream(clock, per_minute=60)
    ai = client(clock, generous, rpm(2, estimate=True))
    assert burst(ai) == 2
    assert len(generous.accepted) == 2  # and nothing is sent beyond a full minute


def test_probes_raise_an_estimate_to_what_the_platform_accepts(clock):
    clock.now = MORNING
    upstream = Upstream(clock, per_day=6)
    ai = client(clock, upstream, rpd(2, estimate=True))
    # Two fit the estimate, then each request beyond the full day is a probe: 3, 4, 5 and 7 a day are
    # assumed in turn. The seventh is refused, which cuts the 7 back to the 6 accepted.
    assert burst(ai) == 6
    assert (learned(ai), len(upstream.refused)) == (6, 1)

    clock.now += DAY + 60
    assert burst(ai) == 6  # the learned size is used like a declared one
    assert len(upstream.refused) == 2  # a day after the cut, one probe: refused, 6 it is
    clock.now += DAY + 60
    assert burst(ai) == 6
    assert len(upstream.refused) == 3  # the first refused probe is repeated after a day
    clock.now += DAY + 60
    assert burst(ai) == 6
    assert len(upstream.refused) == 3  # the second one waits two days
    clock.now += DAY + 60
    assert burst(ai) == 6
    assert (learned(ai), len(upstream.refused)) == (6, 4)


def test_a_daily_estimate_follows_the_platform_both_ways(clock):
    clock.now = 20 * DAY + 23 * 3600  # 23:00 UTC
    upstream = Upstream(clock, per_day=4)
    ai = client(clock, upstream, rpd(2, estimate=True))
    assert burst(ai) == 4  # two by the estimate, two probes accepted, the third probe refused
    assert (learned(ai), len(upstream.refused)) == (4, 1)
    clock.now += 2 * 3600  # 01:00 the next day
    assert burst(ai) == 4
    assert len(upstream.refused) == 1  # the fifth is held back: a day has not passed since the refusal

    clock.now = 40 * DAY + 3600
    lower = Upstream(clock, per_day=3)
    ai = client(clock, lower, rpd(10, estimate=True))
    assert burst(ai) == 3
    assert (learned(ai), len(lower.refused)) == (3, 1)  # a day's count is exact: cut to what was accepted
    clock.now += DAY - 1800  # the next day, but less than a day after the refusal
    assert burst(ai) == 3
    assert len(lower.refused) == 1  # nothing is sent beyond the learned 3


def test_a_refused_probe_waits_whatever_reason_the_platform_gives(clock):
    # The day's allowance is spent, but this platform words the refusal like any rate limit and
    # asks to retry in half a minute. Only the probe's own backoff keeps it from being repeated.
    clock.now = MORNING
    upstream = Upstream(clock, per_day=2, daily_message="too many requests", daily_wait=30)
    ai = client(clock, upstream, rpd(2, estimate=True))
    assert burst(ai, 3) == 2
    assert len(upstream.refused) == 1  # the probe
    for _ in range(30):  # the rest of the day, long after every cooldown
        clock.now += 1800
        burst(ai, 2)
    assert len(upstream.refused) == 1  # it is not sent again and again


def test_a_probe_that_fails_for_another_reason_is_not_a_refused_probe(clock):
    clock.now = MORNING
    upstream = Upstream(clock, per_day=6)
    ai = client(clock, upstream, rpd(2, estimate=True))
    for day in (1, 2):  # on two days in a row the probe runs into a broken platform
        assert burst(ai, 2) == 2
        upstream.broken = 1
        assert burst(ai, 1) == 0  # the probe: sent, answered with a 500
        with ai._router.store.transaction() as tx:
            assert probe_ready_at(tx, key_of(ai)) == 0  # nothing refused it: no wait is set, let alone doubled
        assert learned(ai) is None
        clock.now += 3600
        assert burst(ai, 1) == 0
        assert len(upstream.accepted) == 2 * day  # no second probe on a day with a failure
        clock.now += DAY - 3600
    assert burst(ai) == 6  # a clean day: probed up to the platform's six
    assert (learned(ai), len(upstream.refused)) == (6, 1)


def test_exhaustion_before_anything_was_accepted_cuts_nothing(clock):
    # The platform's day is not ours, or something else spends the key: no count to learn from.
    clock.now = MORNING
    upstream = Upstream(clock, per_day=0)
    ai = client(clock, upstream, rpd(40, scope="provider", estimate=True), models=("a", "b", "c", "d"))
    with pytest.raises(ExhaustedError) as info:
        ai.chat("hi")
    assert len(info.value.attempts) == 4  # every model reported the day exhausted
    assert learned(ai, target="*") is None

    one = Upstream(clock, per_day=1)
    ai = client(clock, one, rpd(40, scope="provider", estimate=True), models=("a", "b", "c", "d"))
    assert burst(ai, 2) == 1
    assert learned(ai, target="*") == 1  # one accepted, then exhausted on every model: cut once, to 1


def test_a_failure_makes_the_count_unfit_for_a_probe_until_the_next_day(clock):
    clock.now = MORNING
    upstream = Upstream(clock, per_day=6)
    ai = client(clock, upstream, rpd(3, estimate=True))
    upstream.broken = 1
    assert burst(ai, 1) == 0  # a 500: charged here, perhaps not counted there
    clock.now += 600
    assert burst(ai) == 2  # the estimate's other two
    assert len(upstream.accepted) == 2  # no probe: an accepted one would prove nothing today
    assert learned(ai) is None
    with ai._router.store.transaction() as tx:
        assert probe_ready_at(tx, key_of(ai)) == 0  # and nothing was held against the window either

    clock.now += DAY
    assert burst(ai) == 6  # a clean day: probed up to the platform's six
    assert learned(ai) == 6


def test_a_window_about_to_reset_is_not_probed(clock):
    clock.now = 21 * DAY - 1.0  # a second before midnight
    upstream = Upstream(clock, per_day=6)
    ai = client(clock, upstream, rpd(2, estimate=True))
    assert burst(ai, 3) == 2
    assert len(upstream.accepted) == 2  # the third would land in the new day and prove nothing


def test_a_window_the_platform_reports_is_not_adapted(clock):
    clock.now = MORNING
    upstream = Upstream(clock, per_day=3, headers=True)
    ai = client(clock, upstream, rpd(5, estimate=True))
    assert burst(ai) == 3  # the header's 3 replaces the estimate after the first answer
    assert (len(upstream.accepted), len(upstream.refused)) == (3, 0)
    assert learned(ai) is None
    clock.now += DAY
    assert burst(ai) == 3
    assert len(upstream.accepted) == 6  # no probes either


def test_learned_sizes_live_in_the_store_and_only_count_while_adapting(tmp_path, clock):
    clock.now = MORNING
    path = tmp_path / "state.db"
    upstream = Upstream(clock, per_day=6)
    limit = rpd(2, estimate=True)
    assert burst(client(clock, upstream, limit, store=SQLiteStore(path))) == 6

    clock.now += DAY
    assert burst(client(clock, upstream, limit, store=SQLiteStore(path), adapt_limits=False)) == 2  # as declared
    clock.now += DAY - 3600
    again = client(clock, upstream, limit, store=SQLiteStore(path))  # another process, or a restart
    assert burst(again, 6) == 6  # six at once without a probe
    assert learned(again) == 6
    assert len(upstream.refused) == 1  # only the one that taught the size, on the first day

    with again._router.store.transaction() as tx:
        assert learned_amount(tx, key_of(again), 2) == 6  # learned against the declared 2
        assert learned_amount(tx, key_of(again), 4) is None  # a release that declares 4 starts from 4


def test_only_one_probe_of_a_window_is_in_flight(clock):
    clock.now = MORNING
    upstream = Upstream(clock, per_day=6)
    ai = client(clock, upstream, rpd(2, estimate=True))
    inner: list[str] = []

    def during_the_probe(seen: int) -> None:
        if seen == 2 and not inner:  # the third request is the probe, still unanswered
            inner.append("entered")
            try:
                ai.chat("hi")
                inner.append("sent")
            except ExhaustedError:
                inner.append("held")

    upstream.before = during_the_probe
    for _ in range(3):
        ai.chat("hi")
    assert inner == ["entered", "held"]
    assert learned(ai) == 3


def test_an_accepted_probe_raises_the_size_it_was_sent_beyond_only():
    key = "q:p:m:day:requests"
    tx: dict = {}
    assert grow(tx, rpd(20, estimate=True), key, probed=20) == 25
    # Another client's probe, sent when the size was still 20, is accepted a moment later.
    assert grow(tx, rpd(25, estimate=True), key, probed=20) is None
    assert learned_amount(tx, key) == 25


def test_backoff_stays_finite_however_often_a_probe_is_refused():
    assert backoff("day", 1) == DAY
    assert backoff("day", 3) == 4 * DAY
    assert backoff("day", 5000) == 7 * DAY
    assert backoff("month", 1) == backoff("month", 5000) == 30 * DAY


def test_async_client_adapts_too(clock):
    clock.now = MORNING
    upstream = Upstream(clock, per_day=4)

    async def sleep(seconds: float) -> None:
        clock.sleep(seconds)

    async def main() -> tuple[int, float | None]:
        provider = Provider(
            "p",
            "https://p.example/v1",
            api_key="k",
            models=["m"],
            limits=[rpd(2, estimate=True)],
            async_http_client=httpx.AsyncClient(transport=httpx.MockTransport(upstream)),
        )
        answered = 0
        async with AsyncFreeAI(provider, clock=clock, sleep=sleep, max_wait=0.0, adapt_limits=True) as ai:
            for _ in range(8):
                try:
                    await ai.chat("hi")
                except ExhaustedError:
                    continue
                answered += 1
            with ai._router.store.transaction() as tx:
                return answered, learned_amount(tx, f"q:{provider.state_id}:m:day:requests")

    assert asyncio.run(main()) == (4, 4)


def _ten_days(clock: FakeClock, upstream: Upstream, estimate: int, **kwargs: Any) -> tuple[int, int]:
    """Sixty requests spread over each of ten days; returns (answered, refused by the platform)."""
    clock.now = MORNING
    ai = client(clock, upstream, rpd(estimate, estimate=True), **kwargs)
    answered = 0
    for _ in range(10):
        start = clock.now
        for _ in range(60):
            answered += burst(ai, 1)
            clock.now += 600
        clock.now = start + DAY
    return answered, len(upstream.refused)


def test_ten_days_against_a_platform_that_allows_more_than_the_estimate(clock):
    assert _ten_days(clock, Upstream(clock, per_day=50), 20, adapt_limits=False) == (200, 0)  # 30 a day unused
    answered, refused = _ten_days(clock, Upstream(clock, per_day=50), 20)
    assert answered == 500  # all fifty, from the first day on
    assert refused <= 5  # the probe that found the top, repeated ever more rarely


def test_ten_days_against_a_platform_that_allows_less_than_the_estimate(clock):
    assert _ten_days(clock, Upstream(clock, per_day=12), 20, adapt_limits=False) == (120, 10)  # a 429 every day
    answered, refused = _ten_days(clock, Upstream(clock, per_day=12), 20)
    assert answered == 120
    assert refused <= 5
