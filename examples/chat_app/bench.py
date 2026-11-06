"""Benchmark driver: starts the mock upstream and the chat app, runs load levels, records results.

    uv run --with starlette==1.10.0 --with 'uvicorn[standard]==0.54.0' \
        python examples/chat_app/bench.py overhead --levels 1,10,50,100,200 --duration 20
    ... bench.py capacity --rates 15,30,60,120,240 --duration 120       # mock free tiers, fresh quota per level
    ... bench.py soak --rate 60 --duration 600                          # mock free tiers, one long run
    ... bench.py real --rates 6,12 --duration 120                       # REAL keyless providers (OVHcloud, Kilo)

Results are appended as JSON lines to ``--out`` (default ``reports/load-test/results.jsonl``).
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import httpx
from loadgen import run, summarize

HERE = Path(__file__).resolve().parent
TICK = os.sysconf("SC_CLK_TCK")


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def tree_cpu(pid: int) -> float:
    """CPU seconds of a process and its direct children (uvicorn workers), from /proc."""
    total = 0.0
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
        except OSError:
            continue
        if int(entry.name) == pid or int(fields[1]) == pid:
            total += (int(fields[11]) + int(fields[12])) / TICK
    return total


def wait_ready(url: str, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if httpx.get(url, timeout=1).status_code < 500:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.2)
    raise RuntimeError(f"{url} did not come up")


@contextmanager
def process(args: list[str], ready_url: str, env: dict[str, str] | None = None) -> Iterator[subprocess.Popen[bytes]]:
    proc = subprocess.Popen([sys.executable, *args], cwd=HERE, env={**os.environ, **(env or {})})
    try:
        wait_ready(ready_url)
        yield proc
    finally:
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()


@contextmanager
def stack(scenario: str | None, window: str, app_args: list[str], seed: int) -> Iterator[Any]:
    """Mock upstream (unless ``scenario`` is None, i.e. real providers) plus the app."""
    app_port = free_port()
    app_url = f"http://127.0.0.1:{app_port}"
    if scenario is None:
        with process(["app.py", "--port", str(app_port), *app_args], f"{app_url}/api/stats") as app:
            yield app_url, app, None, None
        return
    mock_port = free_port()
    mock_url = f"http://127.0.0.1:{mock_port}"
    mock_args = [
        "mock_upstream.py",
        "--scenario",
        scenario,
        "--window",
        window,
        "--port",
        str(mock_port),
        "--seed",
        str(seed),
    ]
    with process(mock_args, f"{mock_url}/_stats") as mock:
        app_cmd = ["app.py", "--port", str(app_port), "--mock", mock_url, "--scenario", scenario, *app_args]
        with process(app_cmd, f"{app_url}/api/stats") as app:
            yield app_url, app, mock_url, mock


def _get(url: str | None) -> dict[str, Any]:
    if url is None:
        return {}
    body: dict[str, Any] = httpx.get(url, timeout=10).json()
    return body


def _diff(after: dict[str, Any], before: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in after.items():
        old = before.get(key)
        if isinstance(value, dict):
            nested = _diff(value, old if isinstance(old, dict) else {})
            if nested:
                out[key] = nested
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            delta = value - (old if isinstance(old, (int, float)) else 0)
            if delta:
                out[key] = round(delta, 3) if isinstance(delta, float) else delta
    return out


def level(
    app_url: str,
    app: subprocess.Popen[bytes],
    mock_url: str | None,
    mock: subprocess.Popen[bytes] | None,
    load: dict[str, Any],
    label: dict[str, Any],
) -> dict[str, Any]:
    app_before, mock_before = _get(f"{app_url}/api/stats"), _get(mock_url and f"{mock_url}/_stats")
    cpu_app, cpu_mock = tree_cpu(app.pid), tree_cpu(mock.pid) if mock else 0.0
    started = time.monotonic()
    records = run(app_url, **load)
    wall = time.monotonic() - started
    app_after, mock_after = _get(f"{app_url}/api/stats"), _get(mock_url and f"{mock_url}/_stats")
    start_t = min((r["t"] for r in records), default=0.0)
    width = 60 if load["duration"] >= 300 else 30
    timeline: dict[int, dict[str, Any]] = {}
    for r in records:
        slot = timeline.setdefault(int((r["t"] - start_t) // width), {"ok": 0, "429": 0, "other": 0, "providers": {}})
        if r["status"] == 200 and not r["error"]:
            slot["ok"] += 1
            slot["providers"][r["provider"]] = slot["providers"].get(r["provider"], 0) + 1
        else:
            slot["429" if r["status"] == 429 else "other"] += 1
    result = {
        **label,
        "timeline": {"bin_s": width, "bins": [timeline[k] for k in sorted(timeline)]},
        "load": {k: v for k, v in load.items() if k != "procs"},
        "wall_s": round(wall, 1),
        "summary": summarize(records, load["duration"]),
        "app_cpu_pct": round((tree_cpu(app.pid) - cpu_app) / wall * 100, 1),
        "mock_cpu_pct": round((tree_cpu(mock.pid) - cpu_mock) / wall * 100, 1) if mock else None,
        "app": _diff(app_after, app_before)
        | {
            "peak_active": app_after.get("peak_active"),
            "max_rss_mb": app_after.get("max_rss_mb"),
            "cooldowns_after": app_after.get("cooldowns"),
            "recent_failures": app_after.get("recent_failures"),
        },
        "upstream": _diff(mock_after, mock_before).get("platforms") if mock else None,
        "upstream_peak_inflight": mock_after.get("peak_inflight") if mock else None,
        "upstream_429_shortfall_ms": mock_after.get("shortfall_ms") if mock else None,
    }
    return result


def emit(result: dict[str, Any], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as handle:
        handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    s = result["summary"]
    load = result["load"]
    what = f"users={load['users']}" if load.get("users") is not None else f"rate={load['rate']}/min"
    print(
        f"[{result['bench']}] {what:>14} ok/min={s['ok_per_min']:>8} ok/s={s['ok_per_s']:>7} "
        f"success={s['success_ratio']} ttft p50/p95={s['ttft_ms']['p50']}/{s['ttft_ms']['p95']}ms "
        f"total p95={s['total_ms']['p95']}ms app_cpu={result['app_cpu_pct']}% mock_cpu={result['mock_cpu_pct']}% "
        f"statuses={s['statuses']}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("bench", choices=("overhead", "capacity", "soak", "real"))
    parser.add_argument("--levels", default="1,10,50,100,200", help="closed-loop users per level")
    parser.add_argument("--rates", default="15,30,60,120,240", help="open-loop messages/minute per level")
    parser.add_argument("--rate", type=float, default=60.0, help="soak rate (messages/minute)")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--scenario", default=None, help="mock scenario (default: instant/all by bench)")
    parser.add_argument("--window", choices=("bucket", "sliding"), default="bucket")
    parser.add_argument("--store", default="memory", help="'memory' or 'sqlite'")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--deadline", type=float, default=0, help="app's per-request deadline (0: none)")
    parser.add_argument(
        "--pool", default="0", help="app's shared HTTP pool size (0: SDK default); overhead takes a list to alternate"
    )
    parser.add_argument("--repeat", type=int, default=1, help="overhead: run every level this many times")
    parser.add_argument("--procs", type=int, default=4, help="load generator processes")
    parser.add_argument("--max-wait", type=float, default=2.0)
    parser.add_argument("--max-tokens", type=int, default=256, help="app's max_tokens per request (0: not set)")
    parser.add_argument("--turns", type=int, default=4)
    parser.add_argument("--think", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--state-dir", help="directory for SQLite state files (default: a new temp dir)")
    parser.add_argument("--no-training", action="store_true", help="real bench: skip models that train on prompts")
    parser.add_argument("--out", type=Path, default=HERE.parent.parent / "reports" / "load-test" / "results.jsonl")
    args = parser.parse_args()

    state_dir = Path(args.state_dir or tempfile.mkdtemp(prefix="freeai-bench-"))
    pools = [int(x) for x in args.pool.split(",")]
    if len(pools) > 1 and args.bench != "overhead":
        parser.error("only overhead alternates pool sizes")

    def app_args(tag: str, pool: int = pools[0]) -> list[str]:
        extra = ["--max-wait", str(args.max_wait), "--workers", str(args.workers), "--pool", str(pool)]
        extra += ["--deadline", str(args.deadline), "--max-tokens", str(args.max_tokens)]
        if args.store == "sqlite":  # a new database per level, so no quota or cooldown carries over
            state_dir.mkdir(parents=True, exist_ok=True)
            extra += ["--store", str(state_dir / f"state-{os.getpid()}-{tag}.db")]
        return extra

    label = {
        "bench": args.bench,
        "store": args.store,
        "workers": args.workers,
        "window": args.window,
        "pool": pools[0],
        "deadline": args.deadline,
        "max_wait": args.max_wait,
        "max_tokens": args.max_tokens,
        "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }

    if args.bench == "overhead":
        scenario = args.scenario or "instant"
        label["scenario"] = scenario
        # Repeats alternate pool sizes on fresh processes, so slow drift of the machine hits both alike.
        for repeat in range(args.repeat):
            for pool in pools:
                tag = f"overhead-{repeat}-{pool}"
                with stack(scenario, args.window, app_args(tag, pool), args.seed) as (app_url, app, mock_url, mock):
                    run(app_url, users=4, duration=3, turns=args.turns, procs=1)  # warm-up
                    for users in (int(x) for x in args.levels.split(",")):
                        load = {
                            "users": users,
                            "duration": args.duration,
                            "think": args.think,
                            "turns": args.turns,
                            "procs": min(args.procs, users),
                        }
                        run_label = {**label, "pool": pool, "repeat": repeat}
                        emit(level(app_url, app, mock_url, mock, load, run_label), args.out)
    elif args.bench in ("capacity", "soak"):
        scenario = args.scenario or "all"
        label["scenario"] = scenario
        rates = [args.rate] if args.bench == "soak" else [float(x) for x in args.rates.split(",")]
        for rate in rates:  # fresh processes: every level starts with full daily quotas
            with stack(scenario, args.window, app_args(f"r{rate:g}"), args.seed) as (app_url, app, mock_url, mock):
                load = {"rate": rate, "duration": args.duration, "turns": args.turns, "procs": 1}
                emit(level(app_url, app, mock_url, mock, load, label), args.out)
    else:  # real keyless providers: synthetic prompts only, the SDK paces within published limits
        label["scenario"] = "real-keyless"
        real_args = [*app_args("real"), "--keyless", *(["--no-training"] if args.no_training else [])]
        with stack(None, args.window, real_args, args.seed) as (app_url, app, _, _):
            for rate in (float(x) for x in args.rates.split(",")):
                load = {"rate": rate, "duration": args.duration, "turns": args.turns, "procs": 1}
                emit(level(app_url, app, None, None, load, label), args.out)


if __name__ == "__main__":
    main()
