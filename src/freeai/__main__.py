"""Command line: ``freeai serve``, ``freeai models``, ``freeai audit``, ``freeai chat``."""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import sys
from collections.abc import Sequence
from typing import Any

from ._core import STRATEGIES
from .client import FreeAI
from .errors import FreeAIError
from .state import SQLiteStore


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="freeai", description="Free LLMs with automatic failover.")
    parser.add_argument("--tiers", default="free", help="comma-separated preset tiers (free,trial,paid)")
    parser.add_argument(
        "--keyless",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="anonymous endpoints: OVHcloud by default (it does not train on prompts); "
        "--keyless adds Kilo, --no-keyless uses none",
    )
    parser.add_argument("--store", help="SQLite file to share cooldowns and quotas between processes")
    parser.add_argument("--no-training", action="store_true", help="only models that do not train on prompts")
    parser.add_argument(
        "--strategy",
        choices=STRATEGIES,
        help="how a model is chosen (default: quota, what refills soonest first; priority: the listed order)",
    )
    parser.add_argument("--max-wait", type=float, help="seconds to wait for quota or cooldowns (default: 30)")
    parser.add_argument(
        "--adapt-limits",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="learn daily and monthly request limits that are declared as estimates (default: on)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    commands = parser.add_subparsers(dest="command", required=True)

    serve = commands.add_parser("serve", help="run a local OpenAI-compatible endpoint")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8787)
    serve.add_argument("--token", default=os.environ.get("FREEAI_PROXY_TOKEN"), help="bearer token clients must send")

    models = commands.add_parser("models", help="list candidate models in try order")
    models.add_argument("--require", action="append", default=[], help="required parameter, e.g. tools")
    models.add_argument("--json", action="store_true")

    audit = commands.add_parser("audit", help="compare pinned model lists with each platform's catalog")
    audit.add_argument("--json", action="store_true")

    chat = commands.add_parser("chat", help="send one prompt")
    chat.add_argument("prompt")
    chat.add_argument("--model", help="model id, glob, provider:id, or alias (fast, smart, coding)")
    chat.add_argument("--system")
    chat.add_argument("--max-tokens", type=int)
    chat.add_argument("--stream", action="store_true")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    options: dict[str, Any] = {"allow_training": not args.no_training}
    if args.strategy is not None:
        options["strategy"] = args.strategy
    if args.max_wait is not None:
        options["max_wait"] = max(0.0, args.max_wait)
    if args.adapt_limits is not None:
        options["adapt_limits"] = args.adapt_limits
    store: SQLiteStore | None = None
    if args.store:
        store = SQLiteStore(args.store)
        options["store"] = store
    try:
        try:
            ai = FreeAI.from_env(tiers=tuple(args.tiers.split(",")), include_keyless=args.keyless, **options)
        except FreeAIError as exc:
            print(f"freeai: {exc}", file=sys.stderr)
            return 2

        if args.command == "serve":
            from .server import serve as run_server

            try:
                run_server(ai, host=args.host, port=args.port, token=args.token)
            except ValueError as exc:
                print(f"freeai: {exc}", file=sys.stderr)
                return 2
            return 0

        with ai:
            try:
                if args.command == "models":
                    found = ai.models(require=args.require)
                    if args.json:
                        print(json.dumps([{"key": m.key, "context": m.context_length} for m in found], indent=2))
                    else:
                        for m in found:
                            print(m.key)
                    return 0
                if args.command == "audit":
                    return _audit(ai, as_json=args.json)
                params: dict[str, Any] = {}
                if args.max_tokens is not None:
                    params["max_tokens"] = args.max_tokens
                if args.stream:
                    for chunk in ai.stream(args.prompt, system=args.system, model=args.model, **params):
                        print(chunk.text, end="", flush=True)
                    print()
                else:
                    response = ai.chat(args.prompt, system=args.system, model=args.model, **params)
                    print(response.text)
                    print(
                        f"[{response.provider}:{response.model}, {len(response.attempts)} attempt(s)]",
                        file=sys.stderr,
                    )
            except FreeAIError as exc:
                print(f"freeai: {exc}", file=sys.stderr)
                return 1
        return 0
    finally:
        if store is not None:
            store.close()


def _audit(ai: FreeAI, *, as_json: bool) -> int:
    """Print where pinned model lists went stale. Exit 1 when a pinned model is gone."""
    drifts = ai.audit()
    if as_json:
        print(json.dumps([dataclasses.asdict(drift) for drift in drifts], indent=2))
    else:
        if not drifts:
            print("no provider pins its models; nothing to compare")
        for drift in drifts:
            if drift.error is not None:
                print(f"{drift.provider}: could not read the catalog: {drift.error}")
                continue
            state = "differs from the catalog" if drift.changed else "matches the catalog"
            print(f"{drift.provider}: {len(drift.pinned)} pinned, {state}")
            if drift.missing:
                print(f"  no longer listed: {', '.join(drift.missing)}")
            if drift.unpinned:
                print(f"  listed but not pinned (not shown to be free): {', '.join(drift.unpinned)}")
    return 1 if any(drift.missing for drift in drifts) else 0


if __name__ == "__main__":
    sys.exit(main())
