"""Failure cooldowns shared through a store transaction (see ``state.py``)."""

from __future__ import annotations

import math
from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Any

from .errors import FailureKind

State = MutableMapping[str, Any]

_MIN_HINT = 1.0  # never hammer a model again within a second, even on "Retry-After: 0"
_NO_COOLDOWN = frozenset(
    {
        FailureKind.BAD_REQUEST,
        FailureKind.CONTEXT_EXCEEDED,
        FailureKind.REFUSED,
        FailureKind.OUTPUT_LIMIT,
        FailureKind.REJECTED,
    }
)


@dataclass(frozen=True, slots=True)
class CooldownPolicy:
    """How long a model (or a whole provider) is benched after a failure, in seconds.

    Server hints (Retry-After, reset headers, RetryInfo) win over the defaults for rate limits,
    quota resets and 503s; without one, the default doubles on every consecutive failure up to
    ``max_backoff``. Quota exhaustion without a hint waits for the provider's daily reset.
    """

    rate_limit: float = 30.0
    provider_rate_limit: float = 60.0
    quota_exhausted: float = 3600.0
    unavailable: float = 15.0
    empty: float = 5.0
    model_gone: float = 3600.0
    not_entitled: float = 86400.0
    quota: float = 3600.0
    max_backoff: float = 900.0
    max_hint: float = 32 * 86400.0  # monthly allowances (Cohere) reset weeks away

    def duration(self, kind: FailureKind, failures: int, hint: float | None) -> float:
        if kind is FailureKind.AUTH:
            return math.inf
        if kind in _NO_COOLDOWN:
            return 0.0
        fixed = {
            FailureKind.MODEL_GONE: self.model_gone,
            FailureKind.NOT_ENTITLED: self.not_entitled,
            FailureKind.QUOTA: self.quota,
            FailureKind.EMPTY: self.empty,
        }
        if kind in fixed:
            return fixed[kind]
        if hint is not None:
            return min(max(hint, _MIN_HINT), self.max_hint)
        if kind is FailureKind.QUOTA_EXHAUSTED:
            return self.quota_exhausted
        base = {
            FailureKind.RATE_LIMIT: self.rate_limit,
            FailureKind.PROVIDER_RATE_LIMIT: self.provider_rate_limit,
        }.get(kind, self.unavailable)
        return min(base * 2.0 ** (failures - 1), self.max_backoff)


def cooldown_ready_at(tx: State, key: str) -> float:
    entry = tx.get(key)
    return float(entry["until"]) if entry else 0.0


def cooldown_failures(tx: State, key: str) -> int:
    entry = tx.get(key)
    return int(entry["failures"]) if entry else 0


def cooldown_fail(
    tx: State, key: str, kind: FailureKind, hint: float | None, now: float, policy: CooldownPolicy
) -> float:
    entry = tx.get(key) or {}
    failures = int(entry.get("failures", 0)) + 1
    cooldown = policy.duration(kind, failures, hint)
    until = float(entry.get("until", 0.0))
    if cooldown > 0:
        until = max(until, now + cooldown)
    tx[key] = {"until": until, "failures": failures}
    return cooldown


def cooldown_clear(tx: State, key: str) -> None:
    if key in tx:
        del tx[key]


def cooldown_snapshot(tx: State, prefix: str, now: float) -> dict[str, float]:
    items_with_prefix = getattr(tx, "items_with_prefix", None)
    if callable(items_with_prefix):
        items = items_with_prefix(prefix)
    else:
        keys_with_prefix = getattr(tx, "keys_with_prefix", None)
        if callable(keys_with_prefix):
            items = ((key, tx[key]) for key in keys_with_prefix(prefix))
        else:
            items = ((key, value) for key, value in tx.items() if key.startswith(prefix))
    return {
        key[len(prefix) :]: float(entry["until"]) - now
        for key, entry in items
        if float(entry["until"]) > now
    }
