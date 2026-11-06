"""Mock free-tier upstream for load tests. NOT a real provider: every response is synthetic.

Serves ``/{platform}/v1/chat/completions`` for the platforms in :mod:`scenarios` and enforces their
free-tier limits the way the real services report them (status codes, error bodies, Retry-After,
Groq's ``x-ratelimit-*`` headers, OVHcloud's Kong headers, Gemini's QuotaFailure details,
Cloudflare and Z.ai error codes). Latency (time to first token, tokens per second) follows each
profile; 2% of requests get a 503.

Rolling windows are token buckets (``--window bucket``, like GCRA) or a strict sliding log
(``--window sliding``) where the real platform's method is unknown. What was measured on the real
platforms on 2026-11-06 is followed regardless of ``--window``:

- Groq: requests per day and tokens per minute are refilling buckets. Accepting a request
  pre-charges its prompt (chat template included) plus ``max_tokens``, and the unused output is
  refunded when the response ends. Without ``max_tokens`` gpt-oss reserves the length of one of the
  model's recent completions, and qwen one of the reserves Groq was seen to use.
- OVHcloud: a fixed minute aligned to the wall clock, reported with Kong-style headers whose
  ``ratelimit-reset`` is rounded up to whole seconds.

    uv run --with starlette==1.10.0 --with 'uvicorn[standard]==0.54.0' \
        python examples/chat_app/mock_upstream.py --scenario all --port 9100
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import resource
import time
from collections import Counter, defaultdict, deque
from collections.abc import AsyncIterator
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import uvicorn
from scenarios import NEURONS, NEURONS_DEFAULT, SCENARIOS, Platform, platforms
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

WORDS = ("무료", "모델", "응답", "토큰", "테스트", "부하", "측정", "결과", "샘플", "문장")


# -- limiters -------------------------------------------------------------------------------------


class Bucket:
    """Token bucket: capacity ``amount``, refilled continuously over ``seconds`` (GCRA-equivalent)."""

    def __init__(self, amount: float, seconds: float) -> None:
        self.capacity = amount
        self.rate = amount / seconds
        self.tokens = amount
        self.at = time.monotonic()

    def _refill(self, now: float) -> None:
        if now <= self.at:  # a bucket made after the request read the clock: nothing to refill yet
            return
        self.tokens = min(self.capacity, self.tokens + (now - self.at) * self.rate)
        self.at = now

    def wait(self, now: float, cost: float) -> float:
        self._refill(now)
        return 0.0 if self.tokens >= cost else (min(cost, self.capacity) - self.tokens) / self.rate

    def take(self, now: float, cost: float) -> None:
        self._refill(now)
        self.tokens -= cost

    def give(self, now: float, amount: float) -> None:
        """Settle a pre-charge: refund what was not used (or charge more when ``amount`` < 0)."""
        self._refill(now)
        self.tokens = min(self.capacity, self.tokens + amount)

    def until_full(self, now: float) -> float:
        self._refill(now)
        return max(0.0, (self.capacity - self.tokens) / self.rate)


class FixedMinute:
    """Fixed window that resets at every wall-clock minute (Kong's rate limiting, as on OVHcloud)."""

    def __init__(self, amount: float) -> None:
        self.amount = amount
        self.minute = -1
        self.used = 0.0

    def _roll(self) -> float:
        now = time.time()
        if (minute := int(now // 60)) != self.minute:
            self.minute, self.used = minute, 0.0
        return (minute + 1) * 60 - now

    def wait(self, now: float, cost: float) -> float:
        left = self._roll()
        return 0.0 if self.used + cost <= self.amount else left

    def take(self, now: float, cost: float) -> None:
        self._roll()
        self.used += cost

    def headers(self) -> dict[str, str]:
        left = self._roll()
        limit, remaining = str(int(self.amount)), str(max(0, int(self.amount - self.used)))
        return {
            "x-ratelimit-limit-minute": limit,
            "x-ratelimit-remaining-minute": remaining,
            "ratelimit-limit": limit,
            "ratelimit-remaining": remaining,
            "ratelimit-reset": str(math.ceil(left)),  # whole seconds, rounded up as OVHcloud does
        }


class Sliding:
    """Strict sliding log: at most ``amount`` in any ``seconds``-long window."""

    def __init__(self, amount: float, seconds: float) -> None:
        self.amount = amount
        self.seconds = seconds
        self.log: deque[tuple[float, float]] = deque()
        self.used = 0.0

    def _purge(self, now: float) -> None:
        while self.log and self.log[0][0] <= now - self.seconds:
            self.used -= self.log.popleft()[1]

    def wait(self, now: float, cost: float) -> float:
        self._purge(now)
        if self.used + cost <= self.amount:
            return 0.0
        excess = self.used + cost - self.amount
        for at, spent in self.log:
            excess -= spent
            if excess <= 0:
                return at + self.seconds - now
        return self.seconds

    def take(self, now: float, cost: float) -> None:
        self.log.append((now, cost))
        self.used += cost


class Daily:
    """Fixed calendar day in ``tz`` (UTC for most, Pacific time for Gemini)."""

    def __init__(self, amount: float, tz: str) -> None:
        self.amount = amount
        self.zone = ZoneInfo(tz)
        self.day = ""
        self.used = 0.0

    def _roll(self) -> float:
        now = datetime.now(self.zone)
        if (today := now.date().isoformat()) != self.day:
            self.day, self.used = today, 0.0
        midnight = datetime.combine(now.date() + timedelta(days=1), datetime.min.time(), self.zone)
        return (midnight - now).total_seconds()

    def wait(self, now: float, cost: float) -> float:
        left = self._roll()
        return 0.0 if self.used + cost <= self.amount else left

    def take(self, now: float, cost: float) -> None:
        self._roll()
        self.used += cost


# -- the mock -------------------------------------------------------------------------------------


class Upstream:
    def __init__(self, scenario: str, window: str, seed: int | None) -> None:
        self.platforms = {p.name: p for p in platforms(scenario)}
        self.rolling = Bucket if window == "bucket" else Sliding
        self.rng = random.Random(seed)
        self.limiters: dict[str, Any] = {}
        self.inflight: Counter[str] = Counter()
        self.peak: Counter[str] = Counter()
        self.stats: dict[str, Counter[str]] = defaultdict(Counter)
        # How far short of the limit each 429 was (ms until it would have passed), per platform:reason.
        self.shortfalls: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=500))
        # Completion lengths of each model's recent requests: Groq's gpt-oss reserves one of them.
        self.recent: dict[str, deque[int]] = defaultdict(lambda: deque(maxlen=20))
        self.started = time.monotonic()

    def _limiter(self, key: str, factory: Any) -> Any:
        limiter = self.limiters.get(key)
        if limiter is None:
            limiter = self.limiters[key] = factory()
        return limiter

    def checks(self, platform: Platform, model: str, tokens: float, neurons: float) -> list[tuple[str, Any, float]]:
        """(reason, limiter, cost) for every limit this request is subject to."""
        name, limits = platform.name, platform.models[model]
        found: list[tuple[str, Any, float]] = []
        rolling = self.rolling
        if platform.account_rpm:
            found.append(("account-rpm", self._limiter(f"{name}|rpm", lambda: rolling(platform.account_rpm, 60)), 1))
        if platform.account_rph:
            found.append(("account-rph", self._limiter(f"{name}|rph", lambda: rolling(platform.account_rph, 3600)), 1))
        if platform.account_rpd:
            found.append(
                ("account-rpd", self._limiter(f"{name}|rpd", lambda: Daily(platform.account_rpd, platform.day_tz)), 1)
            )
        if platform.neurons_per_day:
            found.append(
                ("neurons", self._limiter(f"{name}|neurons", lambda: Daily(platform.neurons_per_day, "UTC")), neurons)
            )
        groq, kong = platform.style == "groq", platform.style == "kong"
        if limits.rpm:
            per_minute: float = limits.rpm
            minute = (lambda: FixedMinute(per_minute)) if kong else (lambda: rolling(per_minute, 60))
            found.append(("rpm", self._limiter(f"{name}|{model}|rpm", minute), 1))
        if limits.rpd:
            per_day: float = limits.rpd
            day = (lambda: Bucket(per_day, 86400)) if groq else (lambda: Daily(per_day, platform.day_tz))
            found.append(("rpd", self._limiter(f"{name}|{model}|rpd", day), 1))
        if limits.tpm:
            tokens_per_minute: float = limits.tpm
            tpm = (lambda: Bucket(tokens_per_minute, 60)) if groq else (lambda: rolling(tokens_per_minute, 60))
            found.append(("tpm", self._limiter(f"{name}|{model}|tpm", tpm), tokens))
        return found

    async def chat(self, request: Request) -> Response:
        platform = self.platforms.get(request.path_params["platform"])
        if platform is None:
            return JSONResponse({"error": {"message": "unknown platform"}}, status_code=404)
        body = await request.json()
        model = body.get("model")
        stats = self.stats[platform.name]
        stats["requests"] += 1
        if model not in platform.models:
            stats["404"] += 1
            return JSONResponse({"error": {"message": f"model {model!r} not found", "code": "model_not_found"}}, 404)
        prompt_tokens = max(1, sum(len(str(m.get("content", ""))) for m in body.get("messages", [])) // 4)
        prompt_tokens += int(platform.extra.get("template", {}).get(model, 0))  # what the platform adds
        low, high = platform.output_tokens
        output = min(int(body.get("max_tokens") or high), self.rng.randint(low, high))
        rates = NEURONS.get(model, NEURONS_DEFAULT)
        neurons = (prompt_tokens * rates[0] + output * rates[1]) / 1e6
        charged = prompt_tokens + output
        recent = self.recent[f"{platform.name}|{model}"]
        if "reserve" in platform.extra:  # Groq: pre-charge max_tokens (or a varying reserve), refund later
            charged = prompt_tokens + int(body.get("max_tokens") or self._reserve(platform, model, recent))

        if platform.concurrency and self.inflight[platform.name] >= platform.concurrency:
            return self._limited(platform, model, "concurrency", 1.0)
        now = time.monotonic()
        checks = self.checks(platform, model, charged, neurons)
        for reason, limiter, cost in checks:
            wait = limiter.wait(now, cost)
            if wait > 0:
                return self._limited(platform, model, reason, wait)
        for _, limiter, cost in checks:
            limiter.take(now, cost)

        if self.rng.random() < platform.error_rate:  # after the quota was spent, like OpenRouter counts it
            stats["503"] += 1
            return JSONResponse({"error": {"message": "upstream overloaded, try again", "code": 503}}, 503)
        stats["ok"] += 1
        stats["completion_tokens"] += output
        recent.append(output)
        headers = self._headers(platform, model, now)
        usage = {"prompt_tokens": prompt_tokens, "completion_tokens": output, "total_tokens": prompt_tokens + output}
        refund = charged - (prompt_tokens + output)
        self._enter(platform.name)  # counted from acceptance, so a concurrency limit has no race window
        if body.get("stream"):
            events = self._events(platform, model, output, usage, refund)
            return StreamingResponse(events, media_type="text/event-stream", headers=headers)
        try:
            await asyncio.sleep(self._ttft(platform) + (output / self._tps(platform) if platform.tps else 0))
        finally:
            self._done(platform, model, refund)
        message = {"role": "assistant", "content": self._text(output)}
        completion = {
            "id": f"mock-{time.time_ns()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "usage": usage,
            "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        }
        return JSONResponse(completion, headers=headers)

    def _reserve(self, platform: Platform, model: str, recent: deque[int]) -> int:
        """Output reserve without max_tokens: on gpt-oss, the length of one of the model's recent
        completions (16 of 24 real reserves matched one); otherwise one of the measured reserves."""
        if recent and model in platform.extra.get("recent_reserve", ()):
            return self.rng.choice(recent)
        return int(self.rng.choice(platform.extra["reserve"][model]))

    def _enter(self, name: str) -> None:
        self.inflight[name] += 1
        self.peak[name] = max(self.peak[name], self.inflight[name])

    def _done(self, platform: Platform, model: str, refund: float) -> None:
        self.inflight[platform.name] -= 1
        tokens = self.limiters.get(f"{platform.name}|{model}|tpm")
        if refund and isinstance(tokens, Bucket) and "reserve" in platform.extra:
            tokens.give(time.monotonic(), refund)

    async def _events(
        self, platform: Platform, model: str, output: int, usage: dict[str, int], refund: float
    ) -> AsyncIterator[str]:
        try:
            await asyncio.sleep(self._ttft(platform))
            chunk_id, created = f"mock-{time.time_ns()}", int(time.time())
            per_chunk = 4
            delay = per_chunk / self._tps(platform) if platform.tps else 0.0
            sent = 0
            while sent < output:
                n = min(per_chunk, output - sent)
                delta = {"content": self._text(n)}
                if sent == 0:
                    delta["role"] = "assistant"
                yield self._sse(chunk_id, created, model, delta, None)
                sent += n
                if delay:
                    await asyncio.sleep(delay)
            yield self._sse(chunk_id, created, model, {}, "stop", usage)
            yield "data: [DONE]\n\n"
        finally:
            self._done(platform, model, refund)

    @staticmethod
    def _sse(
        chunk_id: str,
        created: int,
        model: str,
        delta: dict[str, Any],
        finish: str | None,
        usage: dict[str, int] | None = None,
    ) -> str:
        chunk: dict[str, Any] = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        if usage:
            chunk["usage"] = usage
        return "data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n"

    def _text(self, tokens: int) -> str:
        return "".join(self.rng.choice(WORDS) + " " for _ in range(tokens))

    def _ttft(self, platform: Platform) -> float:
        return self.rng.uniform(*platform.ttft)

    def _tps(self, platform: Platform) -> float:
        return self.rng.uniform(*platform.tps) if platform.tps else math.inf

    def _headers(self, platform: Platform, model: str, now: float) -> dict[str, str]:
        if platform.style == "groq":
            return self._groq_headers(platform, model, now)
        minute = self.limiters.get(f"{platform.name}|{model}|rpm")
        return minute.headers() if isinstance(minute, FixedMinute) else {}

    def _groq_headers(self, platform: Platform, model: str, now: float) -> dict[str, str]:
        limits = platform.models[model]
        day = self.limiters.get(f"{platform.name}|{model}|rpd")
        tokens = self.limiters.get(f"{platform.name}|{model}|tpm")
        headers: dict[str, str] = {}
        if isinstance(day, Bucket) and limits.rpd:  # reset: until the bucket is full again
            headers["x-ratelimit-limit-requests"] = str(int(limits.rpd))
            headers["x-ratelimit-remaining-requests"] = str(max(0, int(day.tokens)))
            headers["x-ratelimit-reset-requests"] = f"{day.until_full(now):.2f}s"
        if isinstance(tokens, Bucket) and limits.tpm:
            headers["x-ratelimit-limit-tokens"] = str(int(limits.tpm))
            headers["x-ratelimit-remaining-tokens"] = str(max(0, int(tokens.tokens)))
            headers["x-ratelimit-reset-tokens"] = f"{tokens.until_full(now):.2f}s"
        return headers

    def _limited(self, platform: Platform, model: str, reason: str, wait: float) -> Response:
        stats = self.stats[platform.name]
        stats["429"] += 1
        stats[f"429:{reason}"] += 1
        self.shortfalls[f"{platform.name}:{reason}"].append(round(wait * 1000, 1))
        retry = str(max(1, math.ceil(wait)))
        daily = reason in ("rpd", "account-rpd", "neurons")
        style = platform.style
        if style == "openrouter":
            if reason.startswith("account"):
                kind = "free-models-per-day" if daily else "free-models-per-min"
                message = f"Rate limit exceeded: {kind}. "
                reset = str(int((time.time() + wait) * 1000))
                headers = {"X-RateLimit-Limit": "20", "X-RateLimit-Remaining": "0", "X-RateLimit-Reset": reset}
                return JSONResponse({"error": {"message": message, "code": 429}}, 429, headers=headers)
            meta = {"raw": f"{model} is temporarily rate-limited upstream", "provider_name": "MockUpstream"}
            body: dict[str, Any] = {"error": {"message": "Provider returned error", "code": 429, "metadata": meta}}
            return JSONResponse(body, 429, headers={"Retry-After": retry})
        if style == "kong":
            minute = self.limiters.get(f"{platform.name}|{model}|rpm")
            headers = {"Retry-After": retry, **(minute.headers() if isinstance(minute, FixedMinute) else {})}
            return JSONResponse({"message": "API rate limit exceeded"}, 429, headers=headers)
        if style == "gemini":
            quota = (
                "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
                if daily
                else ("GenerateRequestsPerMinutePerProjectPerModel-FreeTier")
            )
            details = [
                {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [{"quotaId": quota}]},
                {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": f"{retry}s"},
            ]
            error = {
                "code": 429,
                "message": "You exceeded your current quota.",
                "status": "RESOURCE_EXHAUSTED",
                "details": details,
            }
            return JSONResponse([{"error": error}], 429)
        if style == "cloudflare":
            if reason == "neurons":
                message = "You have used up your daily free allocation of 10,000 neurons."
                return JSONResponse({"success": False, "errors": [{"code": 3036, "message": message}]}, 429)
            body = {
                "success": False,
                "errors": [{"code": 3040, "message": "Capacity temporarily exceeded"}],
            }
            return JSONResponse(body, 429, headers={"Retry-After": retry})
        if style == "zai":
            code = "1302" if reason == "concurrency" else "1308"
            return JSONResponse({"error": {"code": code, "message": "rate limit reached for requests"}}, 429)
        unit = {
            "tpm": "tokens per minute (tpm)",
            "rpd": "requests per day (rpd)",
            "account-rpd": "requests per day (rpd)",
            "account-rph": "requests per hour",
        }.get(reason, "requests per minute (rpm)")
        message = f"Rate limit reached for {unit} on {model}. Please try again in {retry}s."
        return JSONResponse(
            {"error": {"message": message, "type": "rate_limit_error", "code": "rate_limit_exceeded"}},
            429,
            headers={"Retry-After": retry},
        )

    async def models(self, request: Request) -> Response:
        platform = self.platforms.get(request.path_params["platform"])
        if platform is None:
            return JSONResponse({"error": {"message": "unknown platform"}}, status_code=404)
        return JSONResponse({"object": "list", "data": [{"id": m, "object": "model"} for m in platform.models]})

    async def key(self, request: Request) -> Response:  # OpenRouter GET /key
        platform = self.platforms.get(request.path_params["platform"])
        daily = self.limiters.get(f"{platform.name}|rpd") if platform else None
        used = int(daily.used) if daily else 0
        limit = int(platform.account_rpd or 0) if platform else 0
        return JSONResponse({"data": {"free_model_daily_requests": {"used": used, "limit": limit}}})

    async def snapshot(self, request: Request) -> Response:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return JSONResponse(
            {
                "uptime": time.monotonic() - self.started,
                "cpu_seconds": usage.ru_utime + usage.ru_stime,
                "platforms": {name: dict(counter) for name, counter in self.stats.items()},
                "peak_inflight": dict(self.peak),
                "shortfall_ms": {key: list(values) for key, values in self.shortfalls.items()},
            }
        )


def create_app(scenario: str = "all", window: str = "bucket", seed: int | None = None) -> Starlette:
    upstream = Upstream(scenario, window, seed)
    return Starlette(
        routes=[
            Route("/{platform}/v1/chat/completions", upstream.chat, methods=["POST"]),
            Route("/{platform}/v1/models", upstream.models),
            Route("/{platform}/v1/key", upstream.key),
            Route("/_stats", upstream.snapshot),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--scenario", choices=SCENARIOS, default="all")
    parser.add_argument("--window", choices=("bucket", "sliding"), default="bucket")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9100)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    app = create_app(args.scenario, args.window, args.seed)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
