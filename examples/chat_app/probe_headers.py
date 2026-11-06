"""Probe when a platform charges a request against its rate limits, using a handful of real requests.

The SDK assumes a platform counts a streamed request when it accepts it, i.e. by the time the
response headers arrive (see ``Plan.admitted``). This script checks that on a real platform:

1. open stream A and read its rate-limit headers, without reading the body;
2. open stream B while A is still open and read its headers;
3. finish both streams, then send request C and read its headers.

If B's ``remaining`` already excludes A, the platform counted A on admission. Comparing token
counters across the steps also shows whether ``max_tokens`` is reserved up front and whether
unused output is handed back later. Only header names containing ``ratelimit`` (plus
``retry-after``) are printed; keys and other headers never are.

    python examples/chat_app/probe_headers.py ovh --model gpt-oss-120b
    GROQ_API_KEY=... python examples/chat_app/probe_headers.py groq --model openai/gpt-oss-20b

Each run sends three chat requests, which count against the platform's free quota.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import httpx

import freeai
from freeai import Provider


def _rate_headers(headers: httpx.Headers) -> dict[str, str]:
    return {k: v for k, v in headers.items() if "ratelimit" in k.lower() or k.lower() == "retry-after"}


def _snapshots(provider: Provider, headers: httpx.Headers) -> list[dict[str, Any]]:
    return [
        {"window": s.per, "unit": s.unit, "limit": s.limit, "remaining": s.remaining, "reset_in": s.reset_in}
        for s in provider.rate_snapshots(headers)
    ]


def _body(model: str, max_tokens: int, stream: bool) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": "Count from 1 to 5, separated by spaces."}],
        "max_tokens": max_tokens,
        "stream": stream,
    }


def _record(step: str, provider: Provider, response: httpx.Response, started: float) -> dict[str, Any]:
    return {
        "step": step,
        "status": response.status_code,
        "at_s": round(time.monotonic() - started, 3),
        "rate_headers": _rate_headers(response.headers),
        "parsed": _snapshots(provider, response.headers),
    }


def probe(provider: Provider, model: str, max_tokens_a: int, max_tokens_b: int) -> list[dict[str, Any]]:
    url, _, headers = provider._request(model, {}, stream=True)  # same URL and auth headers as the SDK
    started = time.monotonic()
    steps: list[dict[str, Any]] = []
    with httpx.Client(timeout=httpx.Timeout(60.0, connect=10.0)) as client:
        with client.stream("POST", url, json=_body(model, max_tokens_a, True), headers=headers) as a:
            steps.append(_record("A headers (A open)", provider, a, started))
            with client.stream("POST", url, json=_body(model, max_tokens_b, True), headers=headers) as b:
                steps.append(_record("B headers (A still open)", provider, b, started))
                b.read()
            a.read()
        steps.append({"step": "A and B finished", "at_s": round(time.monotonic() - started, 3)})
        c = client.post(url, json=_body(model, max_tokens_b, False), headers=headers)
        steps.append(_record("C headers (after A and B)", provider, c, started))
    return steps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("preset", help="preset name, e.g. ovh, groq, gemini (lower case)")
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-tokens-a", type=int, default=512, help="large, to see whether it is reserved")
    parser.add_argument("--max-tokens-b", type=int, default=16)
    args = parser.parse_args()
    classes = {name.lower(): cls for name, cls in vars(freeai).items() if isinstance(cls, type)}
    preset = classes.get(args.preset.lower())
    if preset is None or not issubclass(preset, Provider) or preset is Provider:
        parser.error(f"unknown preset {args.preset!r}")
    provider: Provider = preset()  # type: ignore[call-arg]  # presets read their key from the environment
    for step in probe(provider, args.model, args.max_tokens_a, args.max_tokens_b):
        print(json.dumps(step, ensure_ascii=False))


if __name__ == "__main__":
    main()
