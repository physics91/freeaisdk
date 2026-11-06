"""Sample AI chat app on freeai: streaming web chat with per-session history.

Real providers (keys from the environment, ``--keyless`` adds anonymous OVHcloud and Kilo)::

    uv run --with starlette==1.10.0 --with 'uvicorn[standard]==0.54.0' \
        python examples/chat_app/app.py --keyless

Load tests point the same SDK presets at the mock upstream instead (``--mock URL --scenario all``).
Open http://127.0.0.1:8000 in a browser.

Threat model: the app spends the quota of every key it was started with, so it only binds to
loopback. Conversation history lives in process memory (LRU, capped) and is never logged.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import resource
import sys
import time
import uuid
from collections import Counter, OrderedDict, defaultdict, deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from freeai import AsyncFreeAI, ExhaustedError, FreeAIError, MemoryStore, SQLiteStore, StreamInterrupted
from freeai.compat import parse_model
from freeai.types import Attempt, StreamChunk

HERE = Path(__file__).resolve().parent
MAX_BODY = 64 * 1024
MAX_MESSAGE_CHARS = 8000
SESSION_RE = re.compile(r"^[0-9a-f]{32}$")


@dataclass(frozen=True)
class Settings:
    mock_url: str | None
    scenario: str
    keyless: bool
    store: str  # "memory" or a SQLite path
    max_wait: float
    max_tokens: int
    history_turns: int
    max_sessions: int
    allow_training: bool
    system: str | None
    pool: int | None  # shared httpx connection pool size; None keeps the SDK default per provider
    deadline: float | None  # seconds for the whole request including failover; None waits as long as it takes

    @classmethod
    def from_env(cls) -> Settings:
        env = os.environ.get
        return cls(
            mock_url=env("CHATAPP_MOCK_URL") or None,
            scenario=env("CHATAPP_SCENARIO", "all"),
            keyless=env("CHATAPP_KEYLESS", "0") == "1",
            store=env("CHATAPP_STORE", "memory"),
            max_wait=float(env("CHATAPP_MAX_WAIT", "2")),
            max_tokens=int(env("CHATAPP_MAX_TOKENS", "256")),
            history_turns=int(env("CHATAPP_HISTORY_TURNS", "10")),
            max_sessions=int(env("CHATAPP_MAX_SESSIONS", "5000")),
            allow_training=env("CHATAPP_NO_TRAINING", "0") != "1",
            system=env("CHATAPP_SYSTEM") or "You are a helpful assistant. Answer briefly.",
            pool=int(env("CHATAPP_POOL") or 0) or None,
            deadline=float(env("CHATAPP_DEADLINE") or 0) or None,
        )


class Stats:
    """Counters for ``GET /api/stats``; the load generator diffs two snapshots."""

    def __init__(self) -> None:
        self.started = time.monotonic()
        self.requests = 0
        self.active = 0
        self.peak_active = 0
        self.responses: Counter[str] = Counter()  # "ok", "429", "503", "stream_error", ...
        self.served: Counter[str] = Counter()  # provider:model that answered
        self.attempts: Counter[str] = Counter()  # "ok" or failure kind, per upstream attempt
        self.attempts_by_provider: Counter[str] = Counter()
        self.attempt_seconds: defaultdict[str, float] = defaultdict(float)  # summed latency per provider:outcome
        self.failures: deque[dict[str, Any]] = deque(maxlen=50)  # recent failed attempts, for diagnosis

    def attempt(self, attempt: Attempt) -> None:
        outcome = "ok" if attempt.ok else str(attempt.kind)
        self.attempts[outcome] += 1
        self.attempts_by_provider[f"{attempt.provider}:{outcome}"] += 1
        self.attempt_seconds[f"{attempt.provider}:{outcome}"] += attempt.latency
        if not attempt.ok:
            self.failures.append(
                {
                    "provider": attempt.provider,
                    "model": attempt.model,
                    "kind": outcome,
                    "status": attempt.status,
                    "latency": round(attempt.latency, 2),
                    "cooldown": round(attempt.cooldown, 1),
                    "message": (attempt.message or "")[:300],
                }
            )

    def snapshot(self) -> dict[str, Any]:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return {
            "pid": os.getpid(),
            "uptime": time.monotonic() - self.started,
            "cpu_seconds": usage.ru_utime + usage.ru_stime,
            "max_rss_mb": usage.ru_maxrss / 1024,
            "requests": self.requests,
            "active": self.active,
            "peak_active": self.peak_active,
            "responses": dict(self.responses),
            "served": dict(self.served),
            "attempts": dict(self.attempts),
            "attempts_by_provider": dict(self.attempts_by_provider),
            "attempt_seconds": {k: round(v, 3) for k, v in self.attempt_seconds.items()},
        }


class Sessions:
    """Bounded in-memory conversation history (LRU over sessions, last N turns per session)."""

    def __init__(self, max_sessions: int, turns: int) -> None:
        self._data: OrderedDict[str, list[dict[str, str]]] = OrderedDict()
        self.max_sessions = max_sessions
        self.max_messages = turns * 2

    def history(self, session_id: str) -> list[dict[str, str]]:
        messages = self._data.get(session_id)
        if messages is None:
            return []
        self._data.move_to_end(session_id)
        return list(messages)

    def append(self, session_id: str, user: str, assistant: str) -> None:
        messages = self._data.setdefault(session_id, [])
        messages += [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
        del messages[: max(0, len(messages) - self.max_messages)]
        self._data.move_to_end(session_id)
        while len(self._data) > self.max_sessions:
            self._data.popitem(last=False)

    def __len__(self) -> int:
        return len(self._data)


def build_ai(settings: Settings, stats: Stats, http: httpx.AsyncClient | None) -> AsyncFreeAI:
    store = MemoryStore() if settings.store == "memory" else SQLiteStore(settings.store)
    provider_options: dict[str, Any] = {"async_http_client": http} if http is not None else {}
    options: dict[str, Any] = {
        "store": store,
        "max_wait": settings.max_wait,
        "allow_training": settings.allow_training,
        "first_token_timeout": 20.0,
        "on_attempt": stats.attempt,
    }
    if settings.mock_url:
        sys.path.insert(0, str(HERE))
        from scenarios import build_providers

        return AsyncFreeAI(build_providers(settings.mock_url, settings.scenario, **provider_options), **options)
    # Real free tiers allow far fewer than the default 1000 connections per provider, so --pool only
    # applies to the mock (from_env builds presets without a client argument).
    return AsyncFreeAI.from_env(include_keyless=settings.keyless, **options)


def sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def create_app(settings: Settings | None = None) -> Starlette:
    settings = settings or Settings.from_env()
    stats = Stats()
    sessions = Sessions(settings.max_sessions, settings.history_turns)
    http = None
    if settings.pool:
        limits = httpx.Limits(max_connections=settings.pool, max_keepalive_connections=settings.pool)
        http = httpx.AsyncClient(limits=limits, timeout=httpx.Timeout(60.0, connect=5.0))
    ai = build_ai(settings, stats, http)
    secrets = [p.api_key for p in ai.providers if p.api_key and len(p.api_key) >= 8]

    def redact(text: str) -> str:
        for secret in secrets:
            text = text.replace(secret, "[redacted]")
        return text

    def error(status: int, code: str, message: str, retry_after: float | None = None) -> JSONResponse:
        stats.responses[str(status)] += 1
        headers = {"Retry-After": str(max(1, math.ceil(retry_after)))} if retry_after is not None else None
        body = {"error": {"code": code, "message": redact(message), "retry_after": retry_after}}
        return JSONResponse(body, status_code=status, headers=headers)

    async def index(request: Request) -> Response:
        return FileResponse(HERE / "static" / "index.html")

    async def models(request: Request) -> Response:
        listed = await ai.models()
        return JSONResponse(
            {
                "aliases": ["auto", "fast", "smart", "coding"],
                "models": [{"key": m.key, "provider": m.provider, "id": m.id} for m in listed],
            }
        )

    async def chat(request: Request) -> Response:
        stats.requests += 1
        length = request.headers.get("content-length")
        if length is None or not length.isdigit():
            return error(411, "length_required", "Content-Length is required")
        if int(length) > MAX_BODY:
            return error(413, "body_too_large", f"request body must be at most {MAX_BODY} bytes")
        try:
            body = json.loads(await request.body())
        except (json.JSONDecodeError, UnicodeDecodeError):
            return error(400, "invalid_json", "body is not valid JSON")
        message = body.get("message") if isinstance(body, dict) else None
        if not isinstance(message, str) or not message.strip() or len(message) > MAX_MESSAGE_CHARS:
            return error(400, "invalid_message", f"message must be 1-{MAX_MESSAGE_CHARS} characters")
        session_id = body.get("session_id") or uuid.uuid4().hex
        if not isinstance(session_id, str) or not SESSION_RE.match(session_id):
            return error(400, "invalid_session", "session_id must be 32 lowercase hex characters")
        model = body.get("model") or None
        if model is not None and (not isinstance(model, str) or len(model) > 200):
            return error(400, "invalid_model", "model must be a short string")

        messages = [*sessions.history(session_id), {"role": "user", "content": message}]
        params: dict[str, Any] = {"max_tokens": settings.max_tokens} if settings.max_tokens > 0 else {}
        started = time.monotonic()
        try:
            chunks = ai.stream(
                messages, system=settings.system, model=parse_model(model), deadline=settings.deadline, **params
            )
            first = await anext(chunks, None)  # failover happens here, so its errors are still plain HTTP errors
        except ExhaustedError as exc:
            if exc.reason.startswith("no free model matches"):
                return error(400, "model_not_found", "선택한 모델을 쓸 수 없습니다. 다른 모델을 고르세요.")
            if exc.next_available_in is not None:
                return error(429, "all_models_busy", "모든 무료 모델이 한도에 도달했습니다.", exc.next_available_in)
            return error(503, "all_models_failed", str(exc))
        except (TypeError, ValueError) as exc:
            return error(400, "invalid_parameter", str(exc))
        except FreeAIError as exc:
            return error(502, "upstream_error", str(exc))
        stats.active += 1
        stats.peak_active = max(stats.peak_active, stats.active)
        return StreamingResponse(
            relay(session_id, message, first, chunks, started),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    async def relay(
        session_id: str, message: str, first: StreamChunk | None, chunks: AsyncIterator[StreamChunk], started: float
    ) -> AsyncIterator[str]:
        parts: list[str] = []
        outcome = "stream_error"
        try:
            ttft = time.monotonic() - started
            provider, model = (first.provider, first.model) if first else ("", "")
            yield sse("meta", {"session_id": session_id, "provider": provider, "model": model})
            if first is not None:
                parts.append(first.text)
                yield sse("delta", {"text": first.text})
            try:
                async for chunk in chunks:
                    if chunk.text:
                        parts.append(chunk.text)
                        yield sse("delta", {"text": chunk.text})
            except StreamInterrupted as exc:
                yield sse("error", {"code": "stream_interrupted", "message": redact(str(exc))})
                return
            reply = "".join(parts)
            sessions.append(session_id, message, reply)
            outcome = "ok"
            stats.served[f"{provider}:{model}"] += 1
            total = time.monotonic() - started
            yield sse(
                "done",
                {
                    "provider": provider,
                    "model": model,
                    "ttft_ms": round(ttft * 1000),
                    "total_ms": round(total * 1000),
                    "chars": len(reply),
                },
            )
        finally:
            stats.active -= 1
            stats.responses[outcome] += 1
            await chunks.aclose()  # type: ignore[attr-defined]  # releases the model slot on disconnect

    async def stats_view(request: Request) -> Response:
        snapshot = stats.snapshot()
        snapshot["sessions"] = len(sessions)
        snapshot["cooldowns"] = ai.status()
        snapshot["recent_failures"] = [{**f, "message": redact(f["message"])} for f in stats.failures]
        return JSONResponse(snapshot)

    @asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        yield
        await ai.aclose()
        if http is not None:
            await http.aclose()

    return Starlette(
        routes=[
            Route("/", index),
            Route("/api/models", models),
            Route("/api/chat", chat, methods=["POST"]),
            Route("/api/stats", stats_view),
        ],
        lifespan=lifespan,
    )


def main() -> None:
    import ipaddress

    import uvicorn

    parser = argparse.ArgumentParser(description="freeai sample chat app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--workers", type=int, default=1, help="processes; >1 needs --store PATH to share quotas")
    parser.add_argument("--mock", metavar="URL", help="use the mock upstream at URL instead of real providers")
    parser.add_argument("--scenario", default="all", help="mock scenario (all, keyless, instant)")
    parser.add_argument("--keyless", action="store_true", help="also use anonymous OVHcloud and Kilo")
    parser.add_argument("--store", default="memory", help="'memory' or a SQLite path shared by workers")
    parser.add_argument("--max-wait", type=float, default=2.0, help="seconds to wait for a free model")
    parser.add_argument("--max-tokens", type=int, default=256, help="per request; 0 leaves it to the platform")
    parser.add_argument("--no-training", action="store_true", help="skip models that train on prompts")
    parser.add_argument("--pool", type=int, default=0, help="shared HTTP connection pool size (0: SDK default)")
    parser.add_argument("--deadline", type=float, default=0, help="seconds per request incl. failover (0: none)")
    parser.add_argument("--access-log", action="store_true")
    args = parser.parse_args()
    if args.host != "localhost" and not ipaddress.ip_address(args.host).is_loopback:
        parser.error("the sample app binds to loopback only: anyone who reaches it spends your quota")
    if args.workers > 1 and args.store == "memory":
        parser.error("--workers > 1 needs --store PATH, or each worker would spend the full quota on its own")
    os.environ.update(
        {
            "CHATAPP_MOCK_URL": args.mock or "",
            "CHATAPP_SCENARIO": args.scenario,
            "CHATAPP_KEYLESS": "1" if args.keyless else "0",
            "CHATAPP_STORE": args.store,
            "CHATAPP_MAX_WAIT": str(args.max_wait),
            "CHATAPP_MAX_TOKENS": str(args.max_tokens),
            "CHATAPP_NO_TRAINING": "1" if args.no_training else "0",
            "CHATAPP_POOL": str(args.pool),
            "CHATAPP_DEADLINE": str(args.deadline),
        }
    )
    uvicorn.run(
        "app:create_app",
        factory=True,
        app_dir=str(HERE),
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="warning",
        access_log=args.access_log,
    )


if __name__ == "__main__":
    main()
