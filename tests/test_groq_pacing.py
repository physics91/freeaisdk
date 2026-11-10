"""Groq's real per-minute behaviour, replayed on a virtual clock through the real preset."""

import contextlib
import json
import math
import random

import httpx
import pytest

from freeai import ExhaustedError, FailureKind, FreeAI, Groq, Limit, RateSnapshot, rpm
from freeai.limits import WHOLE_MARGIN, Cost, quota_adjust, quota_charge, quota_observe, quota_ready_at

from .conftest import FakeClock

GPT_OSS = "openai/gpt-oss-20b"
QWEN = "qwen/qwen3.8-27b"
ARRIVAL = (0.01, 0.05)  # sending a request until Groq's limiter sees it
ANSWER = (0.2, 0.4)  # sending a request until its answer is back


class GroqUpstream:
    """Groq's per-minute limits for one model, as measured on the real API on 2026-11-09.

    Requests: one comes back per whole 2 s since the last accepted request, and the remainder is
    dropped when the next one is accepted. A rejected request costs nothing and leaves that clock
    alone. Output tokens: refill continuously, and a request needs an estimate of its output free.
    """

    def __init__(self, clock: FakeClock, *, output_limit: int | None = None, output: int = 50) -> None:
        self.clock = clock
        self.tokens = 30
        self.last = clock.now
        self.output_limit = output_limit
        self.output = output
        self.output_used = 0.0
        self.output_at = clock.now
        self.accepted: list[float] = []
        self.rejected: list[float] = []
        self.rng = random.Random(1)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        arrival = self.clock.now + self.rng.uniform(*ARRIVAL)
        self.clock.now += self.rng.uniform(*ANSWER)
        back = math.floor((arrival - self.last) / 2.0)
        if min(30, self.tokens + back) < 1:
            return self._reject(arrival, "requests per minute (RPM): Limit 30, Used 30, Requested 1", 2.0)
        if self.output_limit is not None:
            per_second = self.output_limit / 60.0
            self.output_used = max(0.0, self.output_used - (arrival - self.output_at) * per_second)
            self.output_at = arrival
            over = self.output_used + 93 - self.output_limit
            if over > 0:
                used = int(self.output_used)
                reason = f"output tokens per minute (OTPM): Limit {self.output_limit}, Used {used}, Requested 93"
                return self._reject(arrival, reason, over / per_second)
            self.output_used += self.output
        self.tokens = min(30, self.tokens + back) - 1
        self.last = arrival
        self.accepted.append(arrival)
        return httpx.Response(
            200,
            json={
                "id": "1",
                "model": json.loads(request.content)["model"],
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "hi"}}],
                "usage": {"prompt_tokens": 107, "completion_tokens": self.output, "total_tokens": 107 + self.output},
            },
        )

    def _reject(self, arrival: float, reason: str, wait: float) -> httpx.Response:
        self.rejected.append(arrival)
        message = (
            "Rate limit reached for model `x` in organization `org_1` service tier `on_demand` "
            f"on {reason}. Please try again in {wait:.2f}s."
        )
        return httpx.Response(
            429,
            json={"error": {"message": message, "type": "requests", "code": "rate_limit_exceeded"}},
            headers={"retry-after": str(math.ceil(wait))},
        )


def per_minute(times: list[float], start: float) -> list[int]:
    counts = [0, 0, 0]
    for moment in times:
        counts[min(int((moment - start) // 60), 2)] += 1
    return counts


def run(upstream: GroqUpstream, clock: FakeClock, model: str, **kwargs) -> tuple[list[int], list[int]]:
    """Three minutes of more traffic than the limits allow, where nobody waits for a free slot."""
    start = clock.now
    provider = Groq(
        api_key="k", models=[model], http_client=httpx.Client(transport=httpx.MockTransport(upstream)), **kwargs
    )
    with FreeAI(provider, clock=clock, sleep=clock.sleep, max_wait=0.0) as ai:
        while clock.now < start + 180:
            asked = clock.now
            with contextlib.suppress(ExhaustedError):
                ai.chat("hi", max_tokens=256)
            clock.now = max(clock.now, asked + 0.25)
    return per_minute(upstream.accepted, start), per_minute(upstream.rejected, start)


def test_sustained_traffic_uses_groq_requests_per_minute():
    clock = FakeClock()
    accepted, rejected = run(GroqUpstream(clock), clock, GPT_OSS)
    assert accepted[1] >= 27 and accepted[2] >= 27, (accepted, rejected)  # one per 2.1 s, not 15 to 21
    assert rejected[1] + rejected[2] <= 2, (accepted, rejected)
    assert accepted[0] >= 30 + 20, (accepted, rejected)  # the burst out of a full window is not delayed


def test_sustained_traffic_stays_inside_qwen_output_tokens_per_minute():
    clock = FakeClock()
    accepted, rejected = run(GroqUpstream(clock, output_limit=1000), clock, QWEN)
    assert accepted[1] >= 18 and accepted[2] >= 18, (accepted, rejected)  # 1000 per minute / 50 per answer
    assert rejected[1] + rejected[2] <= 2, (accepted, rejected)


def test_output_tokens_limit_only_applies_to_qwen():
    limits = [limit for limit in Groq(api_key="k").limits if limit.unit == "output_tokens"]
    assert [(limit.amount, limit.per) for limit in limits] == [(1000, "minute")]
    assert limits[0].applies_to(QWEN) and not limits[0].applies_to(GPT_OSS)


def test_long_answer_holds_qwen_back_until_its_output_tokens_returned():
    clock = FakeClock()
    upstream = GroqUpstream(clock, output_limit=1000, output=950)  # no room for the next one's 93
    provider = Groq(api_key="k", models=[QWEN], http_client=httpx.Client(transport=httpx.MockTransport(upstream)))
    with FreeAI(provider, clock=clock, sleep=clock.sleep, max_wait=0.0) as ai:
        ai.chat("hi", max_tokens=1500)
        with pytest.raises(ExhaustedError) as info:
            ai.chat("hi", max_tokens=1500)
    assert upstream.rejected == []  # the SDK held it back; Groq never had to refuse it
    assert info.value.next_available_in is not None and info.value.next_available_in > 0


def test_rejected_attempt_gives_its_output_reserve_back():
    clock = FakeClock()
    upstream = GroqUpstream(clock, output_limit=1000)
    upstream.tokens = 0  # drained: the next attempt gets the RPM 429
    upstream.last = clock.now
    provider = Groq(api_key="k", models=[QWEN], http_client=httpx.Client(transport=httpx.MockTransport(upstream)))
    with FreeAI(provider, clock=clock, sleep=clock.sleep, max_wait=0.0) as ai:
        with pytest.raises(ExhaustedError):
            ai.chat("hi")
        with ai._router.store.transaction() as tx:
            output_row = next(tx[key] for key in tx if key.endswith(":minute:output_tokens"))
    assert output_row["tat"] <= clock.now + 1e-6  # nothing is held for an answer that never came


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        (
            "requests per minute (RPM): Limit 30, Used 30, Requested 1",
            RateSnapshot("minute", "requests", 30, 0, whole=True),
        ),
        (
            "output tokens per minute (OTPM): Limit 1000, Used 935, Requested 93",
            RateSnapshot("minute", "output_tokens", 1000, 65),
        ),
    ],
)
def test_groq_429_body_is_read_as_a_limit_reading(reason, expected):
    body = {"error": {"message": f"Rate limit reached for model `x` on {reason}. Please try again in 2s."}}
    failure = Groq(api_key="k").classify(429, {"retry-after": "2"}, body)
    assert failure.kind is FailureKind.RATE_LIMIT
    assert failure.retry_after == 2
    assert failure.snapshots == [expected]


def test_groq_request_too_large_is_not_a_limit_reading():
    body = {
        "error": {
            "message": "Request too large for model `x` on output tokens per minute (OTPM): "
            "Limit 1000, Requested 1500, please reduce your message size and try again."
        }
    }
    failure = Groq(api_key="k").classify(429, {}, body)
    assert failure.kind is FailureKind.BAD_REQUEST
    assert failure.snapshots == []


COST = Cost(1, 0)
WHOLE = [rpm(30, whole=True)]


def ready(tx, now):
    return quota_ready_at(tx, WHOLE, "p", "m", COST, now, "UTC")


def charge(tx, now):
    quota_charge(tx, WHOLE, "p", "m", COST, now, "UTC")


def test_whole_window_does_not_refill_during_a_burst():
    tx: dict = {}
    for index in range(30):  # 0.3 s apart is 9 s in all, which a continuous refill would pay for
        now = 1000.0 + index * 0.3
        assert ready(tx, now) == now
        charge(tx, now)
    assert ready(tx, now) == pytest.approx(now + 2.0 + WHOLE_MARGIN)


def test_whole_window_drops_the_remainder_of_an_interval():
    tx: dict = {}
    for _ in range(30):
        charge(tx, 1000.0)
    late = 1000.0 + 3.5  # one whole interval and a remainder: it pays for this request and no more
    assert ready(tx, late) == late
    charge(tx, late)
    assert ready(tx, late) == pytest.approx(late + 2.0 + WHOLE_MARGIN)


def test_whole_window_regains_what_slow_traffic_leaves_unused():
    tx: dict = {}
    for _ in range(30):
        charge(tx, 1000.0)
    now = 1000.0 + 4.0 + WHOLE_MARGIN + 0.01  # two whole intervals, one spent on this request
    charge(tx, now)
    assert ready(tx, now) == now
    charge(tx, now)
    assert ready(tx, now) > now


def test_whole_window_fills_up_to_its_capacity_when_idle():
    tx: dict = {}
    charge(tx, 1000.0)
    now = 1000.0 + 3600.0
    for _ in range(30):
        assert ready(tx, now) == now
        charge(tx, now)
    assert ready(tx, now) > now


def test_refunding_a_whole_window_gives_the_request_back():
    tx: dict = {}
    for _ in range(30):
        charge(tx, 1000.0)
    quota_adjust(tx, WHOLE, "p", "m", "requests", -1.0, 1000.5, "UTC")  # it never reached the platform
    assert ready(tx, 1000.5) == 1000.5


def test_whole_reading_replaces_a_row_of_the_other_shape():
    tx: dict = {}
    limits = [rpm(30)]  # Groq(limits=[...]) declared without whole: the platform's reading still decides
    quota_charge(tx, limits, "p", "m", COST, 1000.0, "UTC")
    quota_observe(tx, "p", "m", RateSnapshot("minute", "requests", 30, 0, whole=True), 1000.5, "UTC")
    due = 1000.5 + 2.0 + WHOLE_MARGIN
    assert quota_ready_at(tx, limits, "p", "m", COST, 1000.5, "UTC") == pytest.approx(due)
    assert quota_ready_at(tx, limits, "p", "m", COST, due, "UTC") == due


def test_whole_window_treats_a_row_from_an_older_version_as_full():
    tx: dict = {"q:p:m:minute:requests": {"tat": 1060.0, "amount": 30, "since": 1000.0}}
    assert ready(tx, 1000.0) == 1000.0  # at most a minute's count; the first 429 corrects it
    charge(tx, 1000.0)
    assert ready(tx, 1000.0) == 1000.0


def test_whole_window_needs_a_window_that_has_an_interval():
    with pytest.raises(ValueError, match="whole"):
        Limit(5, "month", whole=True)
