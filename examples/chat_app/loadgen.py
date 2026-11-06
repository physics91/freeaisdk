"""Load generator for the sample chat app: virtual users over real HTTP and SSE.

Closed loop (``--users N``): N users each send a message, read the whole streamed reply, think,
and repeat; a 429 makes the user wait for Retry-After (capped). Open loop (``--rate R``): R new
messages per minute arrive as a Poisson process regardless of how fast replies come back.
Each user keeps a multi-turn session (``--turns``), so prompts grow with history.

    python examples/chat_app/loadgen.py http://127.0.0.1:8000 --users 50 --duration 30
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import httpx

PROMPTS = (
    "부하 테스트용 질문입니다. 파이썬 리스트와 튜플의 차이를 한 문단으로 설명해 줘.",
    "부하 테스트용 질문입니다. HTTP 429 응답이 무엇을 뜻하는지 짧게 알려 줘.",
    "부하 테스트용 질문입니다. 지수 백오프를 두 문장으로 설명해 줘.",
    "부하 테스트용 질문입니다. 앞의 답을 한 줄로 요약해 줘.",
)


async def one_request(client: httpx.AsyncClient, url: str, session: str | None, text: str) -> dict[str, Any]:
    started = time.monotonic()
    record: dict[str, Any] = {
        "t": time.time(),
        "status": None,
        "ttft": None,
        "total": None,
        "provider": None,
        "model": None,
        "chars": 0,
        "error": None,
        "retry_after": None,
        "session": session,
    }
    try:
        payload = {"message": text, "session_id": session}
        async with client.stream("POST", f"{url}/api/chat", json=payload) as response:
            record["status"] = response.status_code
            if response.status_code != 200:
                body = json.loads(await response.aread() or b"{}")
                record["error"] = (body.get("error") or {}).get("code")
                record["retry_after"] = (body.get("error") or {}).get("retry_after")
                record["total"] = time.monotonic() - started
                return record
            event = None
            async for line in response.aiter_lines():
                if line.startswith("event: "):
                    event = line[7:]
                elif line.startswith("data: "):
                    data = json.loads(line[6:])
                    if event == "meta":
                        record["session"] = data["session_id"]
                        record["provider"], record["model"] = data["provider"], data["model"]
                    elif event == "delta":
                        if record["ttft"] is None:
                            record["ttft"] = time.monotonic() - started
                        record["chars"] += len(data["text"])
                    elif event == "error":
                        record["error"] = data.get("code", "stream_error")
                    elif event == "done":
                        record["done"] = True
            if not record.get("done") and record["error"] is None:
                record["error"] = "incomplete"
    except httpx.HTTPError as exc:
        record["error"] = type(exc).__name__
    record["total"] = time.monotonic() - started
    return record


def _client(connections: int, timeout: float) -> httpx.AsyncClient:
    limits = httpx.Limits(max_connections=connections, max_keepalive_connections=connections)
    return httpx.AsyncClient(limits=limits, timeout=httpx.Timeout(timeout, connect=10))


async def closed_loop(
    url: str, users: int, duration: float, think: float, turns: int, retry_cap: float, timeout: float, seed: int
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    records: list[dict[str, Any]] = []
    deadline = time.monotonic() + duration

    async def user(client: httpx.AsyncClient, index: int) -> None:
        await asyncio.sleep(rng.uniform(0, min(1.0, duration / 10)))  # spread the start
        session, turn = None, 0
        while time.monotonic() < deadline:
            record = await one_request(client, url, session, PROMPTS[turn % len(PROMPTS)])
            record["user"] = index
            records.append(record)
            if record["status"] == 200 and not record["error"]:
                session, turn = record["session"], turn + 1
                if turn >= turns:
                    session, turn = None, 0
            pause = think * rng.uniform(0.5, 1.5) if think else 0.0
            if record["status"] == 429:
                pause = max(pause, min(retry_cap, float(record["retry_after"] or 1)))
            elif record["status"] != 200 or record["error"]:
                pause = max(pause, 0.5)  # do not spin on errors
            if pause:
                await asyncio.sleep(min(pause, max(0.0, deadline - time.monotonic())))

    async with _client(users + 4, timeout) as client:
        await asyncio.gather(*(user(client, i) for i in range(users)))
    return records


async def open_loop(
    url: str, rate: float, duration: float, turns: int, timeout: float, seed: int
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    records: list[dict[str, Any]] = []
    idle: list[tuple[str, int]] = []  # sessions waiting for their next turn
    tasks: set[asyncio.Task[None]] = set()

    async def arrival(client: httpx.AsyncClient) -> None:
        session, turn = idle.pop(rng.randrange(len(idle))) if idle and rng.random() < 0.8 else (None, 0)
        record = await one_request(client, url, session, PROMPTS[turn % len(PROMPTS)])
        records.append(record)
        if record["status"] == 200 and not record["error"] and turn + 1 < turns:
            idle.append((record["session"], turn + 1))

    async with _client(1000, timeout) as client:
        start = time.monotonic()
        next_at = start
        while next_at < start + duration:
            await asyncio.sleep(max(0.0, next_at - time.monotonic()))
            task = asyncio.create_task(arrival(client))
            tasks.add(task)
            task.add_done_callback(tasks.discard)
            next_at += rng.expovariate(rate / 60.0)
        if tasks:
            await asyncio.wait(tasks, timeout=timeout + 5)
    return records


def _worker(kind: str, args: dict[str, Any]) -> list[dict[str, Any]]:
    if kind == "closed":
        return asyncio.run(closed_loop(**args))
    return asyncio.run(open_loop(**args))


def run(
    url: str,
    *,
    users: int | None = None,
    rate: float | None = None,
    duration: float = 30.0,
    think: float = 0.0,
    turns: int = 4,
    retry_cap: float = 10.0,
    timeout: float = 120.0,
    procs: int = 1,
    seed: int = 1,
) -> list[dict[str, Any]]:
    """Run one load level and return per-request records (spread over ``procs`` processes)."""
    url = url.rstrip("/")
    jobs: list[tuple[str, dict[str, Any]]] = []
    for i in range(procs):
        common = {"url": url, "duration": duration, "turns": turns, "timeout": timeout, "seed": seed * 1000 + i}
        if users is not None:
            share = users // procs + (1 if i < users % procs else 0)
            if share:
                jobs.append(("closed", {**common, "users": share, "think": think, "retry_cap": retry_cap}))
        elif rate is not None:
            jobs.append(("open", {**common, "rate": rate / procs}))
    if len(jobs) == 1:
        return _worker(*jobs[0])
    with ProcessPoolExecutor(len(jobs)) as pool:
        futures = [pool.submit(_worker, kind, job) for kind, job in jobs]
        return [record for future in futures for record in future.result()]


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(q / 100 * len(ordered)) - 1))
    return ordered[index]


def summarize(records: list[dict[str, Any]], duration: float) -> dict[str, Any]:
    ok = [r for r in records if r["status"] == 200 and not r["error"]]
    statuses: dict[str, int] = {}
    for r in records:
        key = str(r["status"]) if r["status"] != 200 or not r["error"] else f"200:{r['error']}"
        if r["status"] is None:
            key = f"client:{r['error']}"
        statuses[key] = statuses.get(key, 0) + 1
    providers: dict[str, int] = {}
    for r in ok:
        providers[r["provider"]] = providers.get(r["provider"], 0) + 1
    ttft = [r["ttft"] for r in ok if r["ttft"] is not None]
    total = [r["total"] for r in ok]
    rejected = [r["total"] for r in records if r["status"] == 429]

    def ms(value: float | None) -> int | None:
        return None if value is None else round(value * 1000)

    return {
        "requests": len(records),
        "ok": len(ok),
        "success_ratio": round(len(ok) / len(records), 4) if records else None,
        "ok_per_min": round(len(ok) / duration * 60, 1),
        "ok_per_s": round(len(ok) / duration, 2),
        "statuses": dict(sorted(statuses.items())),
        "providers": dict(sorted(providers.items(), key=lambda kv: -kv[1])),
        "ttft_ms": {"p50": ms(percentile(ttft, 50)), "p95": ms(percentile(ttft, 95)), "p99": ms(percentile(ttft, 99))},
        "total_ms": {
            "p50": ms(percentile(total, 50)),
            "p95": ms(percentile(total, 95)),
            "p99": ms(percentile(total, 99)),
        },
        "rejected_ms_p50": ms(percentile(rejected, 50)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("url")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--users", type=int, help="closed loop: concurrent virtual users")
    mode.add_argument("--rate", type=float, help="open loop: new messages per minute")
    parser.add_argument("--duration", type=float, default=30)
    parser.add_argument("--think", type=float, default=0.0, help="mean think time between a user's messages")
    parser.add_argument("--turns", type=int, default=4)
    parser.add_argument("--procs", type=int, default=1)
    args = parser.parse_args()
    records = run(
        args.url,
        users=args.users,
        rate=args.rate,
        duration=args.duration,
        think=args.think,
        turns=args.turns,
        procs=args.procs,
    )
    print(json.dumps(summarize(records, args.duration), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
