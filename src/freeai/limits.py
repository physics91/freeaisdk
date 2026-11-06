"""Quota windows, cooldowns, health statistics and the retry budget.

Window and cooldown state is read and written through a store transaction (see ``state.py``) so
several processes can share it; health statistics and the retry budget stay in-process.
"""

from __future__ import annotations

import fnmatch
import math
import re
from collections.abc import Iterable, MutableMapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any, Literal

from ._http import _zone
from .errors import FailureKind

Window = Literal["minute", "hour", "day", "month"]
Unit = Literal["requests", "tokens"]
LimitScope = Literal["model", "provider"]
State = MutableMapping[str, Any]

_WINDOW_SECONDS = {"minute": 60.0, "hour": 3600.0, "day": 86400.0}
# Minute and hour windows refill continuously (GCRA) and day and month windows reset at a boundary,
# unless the platform's headers say otherwise (Groq's daily window refills, OVHcloud's minute resets).
_ROLLING_BY_DEFAULT = frozenset({"minute", "hour"})
# A request reaches the platform a little after we reserve it, and that delay varies (connection
# reuse, TLS handshakes, scheduling). Quota a rolling window regains over time is therefore used
# this many seconds late, or a request sent exactly at the edge can land just before it. What was
# left in the window when it started refilling is not affected, so bursts are unchanged.
EDGE_MARGIN = 1.0
_PENDING_PREFIX = "p:"  # reservations the platform has not accepted yet, keyed by quota key
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


# -- cooldowns ---------------------------------------------------------------------------------


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
            keys = keys_with_prefix(prefix)
        else:
            keys = (key for key in list(tx) if key.startswith(prefix))
        items = ((key, tx[key]) for key in keys)
    return {
        key[len(prefix) :]: float(entry["until"]) - now
        for key, entry in items
        if float(entry["until"]) > now
    }


# -- quota windows -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Limit:
    """One allowance window, e.g. 30 requests per minute per model.

    Minute and hour windows are rolling (GCRA); day and month windows are fixed and reset at the
    provider's ``reset_tz`` boundary. Response headers can override this per window (see
    :class:`RateSnapshot`). ``models`` restricts the limit to matching model ids (globs).
    ``source`` and ``last_verified`` record where the number came from, since free limits change.
    """

    amount: float
    per: Window
    unit: Unit = "requests"
    scope: LimitScope = "model"
    models: tuple[str, ...] | None = None
    source: str | None = None
    last_verified: str | None = None

    def applies_to(self, model_id: str) -> bool:
        return self.models is None or any(rx.match(model_id) is not None for rx in _compiled_globs(self.models))

    def key(self, provider: str, model_id: str) -> str:
        target = model_id if self.scope == "model" else "*"
        return f"q:{provider}:{target}:{self.per}:{self.unit}"


def rpm(amount: float, **kwargs: Any) -> Limit:
    return Limit(amount, "minute", "requests", **kwargs)


def rph(amount: float, **kwargs: Any) -> Limit:
    return Limit(amount, "hour", "requests", **kwargs)


def rpd(amount: float, **kwargs: Any) -> Limit:
    return Limit(amount, "day", "requests", **kwargs)


def tpm(amount: float, **kwargs: Any) -> Limit:
    return Limit(amount, "minute", "tokens", **kwargs)


def tph(amount: float, **kwargs: Any) -> Limit:
    return Limit(amount, "hour", "tokens", **kwargs)


def tpd(amount: float, **kwargs: Any) -> Limit:
    return Limit(amount, "day", "tokens", **kwargs)


def per_month(amount: float, unit: Unit = "requests", **kwargs: Any) -> Limit:
    return Limit(amount, "month", unit, **kwargs)


def _compiled_globs(patterns: Sequence[str]) -> tuple[re.Pattern[str], ...]:
    """Compile shell-style globs once; ``fnmatchcase``-equivalent via ``translate`` + ``match``."""
    return _compiled_globs_cached(tuple(patterns))


@lru_cache(maxsize=256)
def _compiled_globs_cached(patterns: tuple[str, ...]) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(fnmatch.translate(pattern)) for pattern in patterns)


@dataclass(frozen=True, slots=True)
class RateSnapshot:
    """Limit state reported by a platform (response headers or a quota endpoint).

    ``reset_in`` is the server's "seconds until this window resets" when it sends one; for fixed
    windows it replaces the calendar boundary. ``rolling`` says how the platform counts: True for a
    window that refills over time (Groq's daily requests), False for one that resets all at once
    when ``reset_in`` runs out (OVHcloud's minute), None for the default of ``per`` (minute and hour
    roll, day and month are fixed).
    """

    per: Window
    unit: Unit
    limit: float
    remaining: float
    scope: LimitScope = "model"
    reset_in: float | None = None
    rolling: bool | None = None


@dataclass(frozen=True, slots=True)
class Cost:
    """What one request is expected to consume. Tokens are an estimate until usage arrives."""

    requests: float = 1.0
    tokens: float = 0.0

    def of(self, unit: Unit) -> float:
        return self.requests if unit == "requests" else self.tokens


def quota_ready_at(
    tx: State, limits: list[Limit], provider: str, model_id: str, cost: Cost, now: float, tz: str
) -> float:
    ready = now
    for limit in limits:
        amount = cost.of(limit.unit)
        if amount > 0 and limit.applies_to(model_id):
            ready = max(ready, _window_ready(limit, tx.get(limit.key(provider, model_id)), amount, now, tz))
    return ready


def quota_charge(tx: State, limits: list[Limit], provider: str, model_id: str, cost: Cost, now: float, tz: str) -> None:
    for limit in limits:
        amount = cost.of(limit.unit)
        if amount > 0 and limit.applies_to(model_id):
            key = limit.key(provider, model_id)
            tx[key] = _window_add(limit, tx.get(key), amount, now, tz)


def quota_adjust(
    tx: State, limits: list[Limit], provider: str, model_id: str, unit: Unit, delta: float, now: float, tz: str
) -> None:
    """Settle an estimate (``delta`` > 0 charges more, < 0 refunds)."""
    for limit in limits:
        if limit.unit == unit and delta and limit.applies_to(model_id):
            key = limit.key(provider, model_id)
            tx[key] = _window_add(limit, tx.get(key), delta, now, tz)


def quota_observe(tx: State, provider: str, model_id: str, snapshot: RateSnapshot, now: float, tz: str) -> Limit:
    """Replace local counters with what the platform reported; returns the learned limit.

    A snapshot counts the requests the platform had accepted when it answered. Requests sent but not
    accepted yet (no response headers back, see :func:`pending_add`) are added on top so their
    reservations are not wiped out; these live in the store, so every process sharing it sees them.
    Otherwise the server's count wins, which also settles over-estimated token reservations.

    Concurrent responses can be handled in another order than the platform counted them. Platforms
    never hand requests back, so a request count read lower than one already applied is a late
    reading and is ignored. Token counts can drop for real (unused output handed back), so for
    them only readings of a window that has already ended are ignored.
    """
    limit = Limit(snapshot.limit, snapshot.per, snapshot.unit, snapshot.scope, source="response headers")
    key = limit.key(provider, model_id)
    seen = max(0.0, snapshot.limit - snapshot.remaining)
    requests = limit.unit == "requests"
    old: Any = tx.get(key)  # what the last reading (and later charges) left
    rolling = limit.per in _ROLLING_BY_DEFAULT if snapshot.rolling is None else snapshot.rolling
    if limit.per in _ROLLING_BY_DEFAULT and snapshot.reset_in is None:
        rolling = True  # a fixed minute or hour window is no use without knowing when it ends
    if rolling and limit.per in _WINDOW_SECONDS:
        interval = _WINDOW_SECONDS[limit.per] / max(snapshot.limit, 1e-9)
        served = now + seen * interval  # when the platform's own count is whole again
        if requests and old and "served" in old and float(old.get("amount", 0.0)) == snapshot.limit:
            served = max(served, float(old["served"]))  # only ever moves forward for requests
        tat = served + _pending_total(tx, key, now) * interval
        state: dict[str, Any] = {"tat": tat, "amount": snapshot.limit, "since": now}
        if requests:
            state["served"] = served
    else:
        until = now + snapshot.reset_in if snapshot.reset_in is not None else None
        period = _period(limit.per, now, tz) if limit.per not in _ROLLING_BY_DEFAULT else None
        order = _reading_order(limit.per, old, until, period)
        if order < 0:  # the window this reading counted has already ended
            seen, until = float(old["seen"]), float(old["until"])
        elif order == 0 and requests:
            seen = max(seen, float(old["seen"]))
            if until is not None:
                until = max(until, float(old["until"]))
        state = {"used": seen + _pending_total(tx, key, now), "seen": seen, "amount": snapshot.limit}
        if period is not None:
            state["period"] = period
        if until is not None:
            state["until"] = until
    tx[key] = state
    return limit


def _reading_order(per: Window, old: Any, until: float | None, period: str | None) -> int:
    """Whether a fixed-window reading is of an earlier (-1), the same (0) or a later (1) window.

    ``old`` is the state the last reading left. Readings of one window report resets a rounding
    error apart, and a window's successor ends a whole window later. Unknown counts as later.
    """
    if not old or "seen" not in old or old.get("period") != period:
        return 1
    old_until = old.get("until")
    if until is None or old_until is None:
        return 0 if until is None and old_until is None else 1
    half = _WINDOW_SECONDS.get(per, _WINDOW_SECONDS["day"]) / 2
    delta = until - float(old_until)
    return -1 if delta <= -half else 0 if delta < half else 1


def pending_add(tx: State, reservation: str, charges: Iterable[tuple[str, float]], expires_at: float) -> None:
    """Record charges of a request the platform has not accepted yet (until ``expires_at``)."""
    for key, amount in charges:
        entries = dict(tx.get(_PENDING_PREFIX + key) or {})
        entries[reservation] = [amount, expires_at]
        tx[_PENDING_PREFIX + key] = entries


def pending_drop(tx: State, reservation: str, keys: Iterable[str]) -> None:
    """Forget a reservation: the platform accepted the request, or it is over."""
    for key in keys:
        entries = dict(tx.get(_PENDING_PREFIX + key) or {})
        if entries.pop(reservation, None) is None:
            continue
        if entries:
            tx[_PENDING_PREFIX + key] = entries
        else:
            del tx[_PENDING_PREFIX + key]


def _pending_total(tx: State, key: str, now: float) -> float:
    entries = tx.get(_PENDING_PREFIX + key) or {}
    live = {rid: entry for rid, entry in entries.items() if float(entry[1]) > now}
    if len(live) != len(entries):  # a process died before handing these back
        if live:
            tx[_PENDING_PREFIX + key] = live
        else:
            del tx[_PENDING_PREFIX + key]
    return sum(max(0.0, float(entry[0])) for entry in live.values())


def _window_ready(limit: Limit, state: Any, amount: float, now: float, tz: str) -> float:
    capacity = float(state.get("amount", limit.amount)) if state else limit.amount
    if _rolling(limit.per, state):
        period = _WINDOW_SECONDS[limit.per]
        interval = period / capacity
        tat = float(state["tat"]) if state else now  # GCRA: when the window is whole again
        if tat <= now:
            return now
        # A single request larger than the window waits until it is whole again.
        due = tat if amount >= capacity else tat + amount * interval - period
        if due <= float(state.get("since", -math.inf)):
            return now  # covered by what was left when the window started refilling
        return max(now, due + EDGE_MARGIN)
    used = _used_in_period(limit, state, now, tz)
    if used + amount <= capacity or used == 0:
        return now
    until = _window_end(limit.per, state, now)
    return until + EDGE_MARGIN if until is not None else _next_boundary(limit.per, now, tz)


def _window_add(limit: Limit, state: Any, amount: float, now: float, tz: str) -> dict[str, Any]:
    capacity = float(state.get("amount", limit.amount)) if state else limit.amount
    if _rolling(limit.per, state):
        interval = _WINDOW_SECONDS[limit.per] / capacity
        tat = float(state["tat"]) if state else now
        if tat <= now:  # whole: refilling starts with this charge
            return {"tat": max(now, now + amount * interval), "amount": capacity, "since": now}
        since = float(state.get("since", now))
        added_tat: dict[str, Any] = {"tat": max(now, tat + amount * interval), "amount": capacity, "since": since}
        if "served" in state:
            added_tat["served"] = state["served"]  # the platform's count, if it reported one
        return added_tat
    used = _used_in_period(limit, state, now, tz)
    added: dict[str, Any] = {"used": max(0.0, used + amount), "amount": capacity}
    if state and "seen" in state and _in_window(limit, state, now, tz):
        added["seen"] = state["seen"]  # the platform's count for this window, if it reported one
    if limit.per not in _ROLLING_BY_DEFAULT:
        added["period"] = _period(limit.per, now, tz)
    until = _window_end(limit.per, state, now)
    if until is not None:
        added["until"] = until
    return added


def _rolling(per: Window, state: Any) -> bool:
    """Whether a window refills continuously (GCRA) or counts up to a reset; headers may decide."""
    return "tat" in state if state else per in _ROLLING_BY_DEFAULT


def _used_in_period(limit: Limit, state: Any, now: float, tz: str) -> float:
    return float(state.get("used", 0.0)) if state and _in_window(limit, state, now, tz) else 0.0


def _in_window(limit: Limit, state: Any, now: float, tz: str) -> bool:
    """Whether ``now`` is still in the fixed window ``state`` counts."""
    if state.get("until") is not None:  # the server said when this window resets
        return now < float(state["until"]) + EDGE_MARGIN
    return bool(state.get("period") == _period(limit.per, now, tz))


def _window_end(per: Window, state: Any, now: float) -> float | None:
    """When the fixed window the server reported ends, while ``now`` is still in it.

    The server's reset is honored :data:`EDGE_MARGIN` late, like refilled quota, because it is
    rounded and our clock is not the platform's. Minute and hour windows repeat, so once one is
    over this is the end of the window ``now`` falls in.
    """
    until = state.get("until") if state else None
    if until is None:
        return None
    end = float(until)
    if now < end + EDGE_MARGIN:
        return end
    if per in _ROLLING_BY_DEFAULT:
        period = _WINDOW_SECONDS[per]
        return end + (math.floor((now - EDGE_MARGIN - end) / period) + 1) * period
    return None


def _period(per: Window, now: float, tz: str) -> str:
    moment = datetime.fromtimestamp(now, _zone(tz))
    return moment.date().isoformat() if per == "day" else f"{moment.year:04d}-{moment.month:02d}"


def seconds_until(per: Window, now: float, tz: str) -> float:
    """Seconds from ``now`` (epoch) until the next fixed-window boundary in ``tz``."""
    return max(0.0, _next_boundary(per, now, tz) - now)


def _next_boundary(per: Window, now: float, tz: str) -> float:
    zone = _zone(tz)
    moment = datetime.fromtimestamp(now, zone)
    if per == "day":
        start = datetime.combine(moment.date() + timedelta(days=1), datetime.min.time(), tzinfo=zone)
    else:
        year, month = (moment.year + 1, 1) if moment.month == 12 else (moment.year, moment.month + 1)
        start = datetime(year, month, 1, tzinfo=zone)
    return start.timestamp()


# -- health and retry budget (in-process) ------------------------------------------------------


@dataclass(slots=True)
class _Ewma:
    success: float = 1.0
    latency: float = 1.0
    samples: int = 0


class HealthStats:
    """Exponentially weighted success rate and latency per model, for ``strategy="health"``.

    Unknown models start optimistic so that new free models get tried.
    """

    def __init__(self, alpha: float = 0.3) -> None:
        self.alpha = alpha
        self._stats: dict[str, _Ewma] = {}

    def record(self, key: str, ok: bool, latency: float) -> None:
        stat = self._stats.setdefault(key, _Ewma())
        a = self.alpha if stat.samples else 1.0
        stat.success = (1 - a) * stat.success + a * (1.0 if ok else 0.0)
        if ok:
            stat.latency = (1 - a) * stat.latency + a * max(latency, 0.0) if stat.samples else max(latency, 0.0)
        stat.samples += 1

    def score(self, key: str) -> float:
        stat = self._stats.get(key)
        if stat is None:
            return 1.0
        return stat.success / (1.0 + stat.latency)

    def snapshot(self) -> dict[str, dict[str, float]]:
        return {k: {"success": v.success, "latency": v.latency, "samples": v.samples} for k, v in self._stats.items()}


@dataclass(slots=True)
class RetryBudget:
    """gRPC-style retry throttle: server failures spend a token, successes earn ``token_ratio``.

    While fewer than half the tokens remain, a request stops failing over after a server failure,
    so an upstream outage does not turn every call into a sweep over every free endpoint.
    """

    max_tokens: float = 10.0
    token_ratio: float = 0.1
    tokens: float = field(default=-1.0)

    def __post_init__(self) -> None:
        if self.tokens < 0:
            self.tokens = self.max_tokens

    def allows_retry(self) -> bool:
        return self.tokens > self.max_tokens / 2

    def on_failure(self) -> None:
        self.tokens = max(0.0, self.tokens - 1.0)

    def on_success(self) -> None:
        self.tokens = min(self.max_tokens, self.tokens + self.token_ratio)
