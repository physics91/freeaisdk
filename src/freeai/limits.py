"""Quota windows and pending reservations shared through a store transaction (see ``state.py``).

Failure cooldowns live in :mod:`freeai.cooldown` (re-exported here for compatibility).
Routing health scores live in :mod:`freeai.health`. The failover retry throttle lives in
:mod:`freeai.retry` (re-exported here as :class:`RetryBudget` for compatibility).
"""

from __future__ import annotations

import fnmatch
import math
import re
from collections.abc import Iterable, MutableMapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any, Literal

from ._http import _zone
from .cooldown import CooldownPolicy as CooldownPolicy
from .cooldown import (
    cooldown_clear as cooldown_clear,
)
from .cooldown import (
    cooldown_fail as cooldown_fail,
)
from .cooldown import (
    cooldown_failures as cooldown_failures,
)
from .cooldown import (
    cooldown_ready_at as cooldown_ready_at,
)
from .cooldown import (
    cooldown_snapshot as cooldown_snapshot,
)
from .retry import RetryBudget as RetryBudget

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
    requests, tokens = cost.requests, cost.tokens
    for limit in limits:
        amount = requests if limit.unit == "requests" else tokens
        if amount > 0 and math.isfinite(amount) and limit.applies_to(model_id):
            ready = max(ready, _window_ready(limit, tx.get(limit.key(provider, model_id)), amount, now, tz))
    return ready


def quota_charge(
    tx: State, limits: list[Limit], provider: str, model_id: str, cost: Cost, now: float, tz: str
) -> list[tuple[str, float]]:
    """Charge windows for ``cost``; return ``(quota_key, amount)`` pairs actually charged."""
    charges: list[tuple[str, float]] = []
    requests, tokens = cost.requests, cost.tokens
    for limit in limits:
        amount = requests if limit.unit == "requests" else tokens
        if amount > 0 and math.isfinite(amount) and limit.applies_to(model_id):
            key = limit.key(provider, model_id)
            tx[key] = _window_add(limit, tx.get(key), amount, now, tz)
            charges.append((key, amount))
    return charges


def quota_adjust(
    tx: State, limits: list[Limit], provider: str, model_id: str, unit: Unit, delta: float, now: float, tz: str
) -> None:
    """Settle an estimate (``delta`` > 0 charges more, < 0 refunds)."""
    for limit in limits:
        if limit.unit == unit and delta and math.isfinite(delta) and limit.applies_to(model_id):
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
    old = _quota_state(tx.get(key))  # what the last reading (and later charges) left
    rolling = limit.per in _ROLLING_BY_DEFAULT if snapshot.rolling is None else snapshot.rolling
    if limit.per in _ROLLING_BY_DEFAULT and snapshot.reset_in is None:
        rolling = True  # a fixed minute or hour window is no use without knowing when it ends
    if rolling and limit.per in _WINDOW_SECONDS:
        interval = _WINDOW_SECONDS[limit.per] / max(snapshot.limit, 1e-9)
        served = now + seen * interval  # when the platform's own count is whole again
        old_amount = _quota_number(old.get("amount")) if old else None
        old_served = _quota_number(old.get("served")) if old else None
        if requests and old_amount == snapshot.limit and old_served is not None:
            served = max(served, old_served)  # only ever moves forward for requests
        tat = served + _pending_total(tx, key, now) * interval
        state: dict[str, Any] = {"tat": tat, "amount": snapshot.limit, "since": now}
        if requests:
            state["served"] = served
    else:
        until = now + snapshot.reset_in if snapshot.reset_in is not None else None
        period = _period(limit.per, now, tz) if limit.per not in _ROLLING_BY_DEFAULT else None
        order = _reading_order(limit.per, old, until, period)
        if order < 0:  # the window this reading counted has already ended
            seen = _quota_number(old.get("seen")) or 0.0
            until = _quota_number(old.get("until"))
        elif order == 0 and requests and old is not None:
            old_seen = _quota_number(old.get("seen"))
            if old_seen is not None:
                seen = max(seen, old_seen)
            if until is not None:
                old_until = _quota_number(old.get("until"))
                if old_until is not None:
                    until = max(until, old_until)
        state = {"used": seen + _pending_total(tx, key, now), "seen": seen, "amount": snapshot.limit}
        if period is not None:
            state["period"] = period
        if until is not None:
            state["until"] = until
    tx[key] = state
    return limit


def _quota_state(value: Any) -> dict[str, Any] | None:
    """Quota counter row from the store; non-dict values are treated as missing."""
    return value if isinstance(value, dict) else None


def _finite_float(value: Any) -> float | None:
    """Parse a finite float; bools, non-numeric values, and inf/nan are rejected."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            number = float(text)
        except ValueError:
            return None
        return number if math.isfinite(number) else None
    return None


def _quota_number(value: Any) -> float | None:
    """Numeric quota field from the store."""
    return _finite_float(value)


def _reading_order(per: Window, old: dict[str, Any] | None, until: float | None, period: str | None) -> int:
    """Whether a fixed-window reading is of an earlier (-1), the same (0) or a later (1) window.

    ``old`` is the state the last reading left. Readings of one window report resets a rounding
    error apart, and a window's successor ends a whole window later. Unknown counts as later.
    """
    if not old or _quota_number(old.get("seen")) is None or old.get("period") != period:
        return 1
    old_until = _quota_number(old.get("until")) if "until" in old else None
    if until is None or old_until is None:
        return 0 if until is None and old_until is None else 1
    half = _WINDOW_SECONDS.get(per, _WINDOW_SECONDS["day"]) / 2
    delta = until - old_until
    return -1 if delta <= -half else 0 if delta < half else 1


def _pending_entries(tx: State, key: str) -> dict[str, Any]:
    """Reservation map for one quota key; replace a corrupt non-dict value with a fresh map."""
    raw = tx.get(_PENDING_PREFIX + key)
    return dict(raw) if isinstance(raw, dict) else {}


def pending_add(tx: State, reservation: str, charges: Iterable[tuple[str, float]], expires_at: float) -> None:
    """Record charges of a request the platform has not accepted yet (until ``expires_at``)."""
    for key, amount in charges:
        entries = _pending_entries(tx, key)
        entries[reservation] = [amount, expires_at]
        tx[_PENDING_PREFIX + key] = entries


def pending_drop(tx: State, reservation: str, keys: Iterable[str]) -> None:
    """Forget a reservation: the platform accepted the request, or it is over."""
    for key in keys:
        entries = _pending_entries(tx, key)
        if entries.pop(reservation, None) is None:
            continue
        if entries:
            tx[_PENDING_PREFIX + key] = entries
        else:
            del tx[_PENDING_PREFIX + key]


def _pending_row(entry: Any) -> list[float] | None:
    """``[amount, expires_at]`` from the store; skip truncated or non-numeric rows."""
    if not isinstance(entry, (list, tuple)) or len(entry) < 2:
        return None
    try:
        amount, expires = float(entry[0]), float(entry[1])
    except (TypeError, ValueError):
        return None
    if not math.isfinite(amount) or not math.isfinite(expires):
        return None
    return [amount, expires]


def _pending_total(tx: State, key: str, now: float) -> float:
    store_key = _PENDING_PREFIX + key
    raw = tx.get(store_key)
    entries = dict(raw) if isinstance(raw, dict) else {}
    live: dict[str, list[float]] = {}
    for rid, entry in entries.items():
        row = _pending_row(entry)
        if row is not None and row[1] > now:
            live[rid] = row
    # Drop expired/corrupt rows, and replace a corrupt non-dict value left in the store.
    if not isinstance(raw, dict) or len(live) != len(entries):
        if live:
            tx[store_key] = live
        elif store_key in tx:
            del tx[store_key]
    return sum(max(0.0, amount) for amount, _expires in live.values())


def _window_ready(limit: Limit, state: Any, amount: float, now: float, tz: str) -> float:
    state = _usable_quota_state(state)
    capacity_n = _quota_number(state.get("amount")) if state else None
    capacity = limit.amount if capacity_n is None else capacity_n
    if _rolling(limit.per, state):
        period = _WINDOW_SECONDS[limit.per]
        interval = period / capacity
        tat = _quota_number(state.get("tat")) if state else now
        if tat <= now:
            return now
        # A single request larger than the window waits until it is whole again.
        due = tat if amount >= capacity else tat + amount * interval - period
        since = _quota_number(state.get("since")) if state else None
        if due <= (since if since is not None else -math.inf):
            return now  # covered by what was left when the window started refilling
        return max(now, due + EDGE_MARGIN)
    used = _used_in_period(limit, state, now, tz)
    if used + amount <= capacity or used == 0:
        return now
    until = _window_end(limit.per, state, now)
    return until + EDGE_MARGIN if until is not None else _next_boundary(limit.per, now, tz)


def _window_add(limit: Limit, state: Any, amount: float, now: float, tz: str) -> dict[str, Any]:
    state = _usable_quota_state(state)
    capacity_n = _quota_number(state.get("amount")) if state else None
    capacity = limit.amount if capacity_n is None else capacity_n
    if _rolling(limit.per, state):
        interval = _WINDOW_SECONDS[limit.per] / capacity
        tat = _quota_number(state.get("tat")) if state else now
        if tat <= now:  # whole: refilling starts with this charge
            return {"tat": max(now, now + amount * interval), "amount": capacity, "since": now}
        since = _quota_number(state.get("since"))
        if since is None:
            since = now
        added_tat: dict[str, Any] = {"tat": max(now, tat + amount * interval), "amount": capacity, "since": since}
        served = _quota_number(state.get("served"))
        if served is not None:
            added_tat["served"] = served  # the platform's count, if it reported one
        return added_tat
    used = _used_in_period(limit, state, now, tz)
    added: dict[str, Any] = {"used": max(0.0, used + amount), "amount": capacity}
    if state and _in_window(limit, state, now, tz):
        seen = _quota_number(state.get("seen"))
        if seen is not None:
            added["seen"] = seen  # the platform's count for this window, if it reported one
    if limit.per not in _ROLLING_BY_DEFAULT:
        added["period"] = _period(limit.per, now, tz)
    until = _window_end(limit.per, state, now)
    if until is not None:
        added["until"] = until
    return added


def _usable_quota_state(value: Any) -> dict[str, Any] | None:
    """Drop non-dict rows and rolling rows whose ``tat`` is not numeric."""
    state = _quota_state(value)
    if state and "tat" in state and _quota_number(state.get("tat")) is None:
        return None
    return state


def _rolling(per: Window, state: dict[str, Any] | None) -> bool:
    """Whether a window refills continuously (GCRA) or counts up to a reset; headers may decide."""
    return "tat" in state if state else per in _ROLLING_BY_DEFAULT


def _used_in_period(limit: Limit, state: dict[str, Any] | None, now: float, tz: str) -> float:
    if not state or not _in_window(limit, state, now, tz):
        return 0.0
    return _quota_number(state.get("used")) or 0.0


def _in_window(limit: Limit, state: dict[str, Any], now: float, tz: str) -> bool:
    """Whether ``now`` is still in the fixed window ``state`` counts."""
    until = _quota_number(state.get("until")) if "until" in state else None
    if until is not None:  # the server said when this window resets
        return now < until + EDGE_MARGIN
    if "until" in state:
        return False  # corrupt until: do not trust this row
    return bool(state.get("period") == _period(limit.per, now, tz))


def _window_end(per: Window, state: dict[str, Any] | None, now: float) -> float | None:
    """When the fixed window the server reported ends, while ``now`` is still in it.

    The server's reset is honored :data:`EDGE_MARGIN` late, like refilled quota, because it is
    rounded and our clock is not the platform's. Minute and hour windows repeat, so once one is
    over this is the end of the window ``now`` falls in.
    """
    until = _quota_number(state.get("until")) if state else None
    if until is None:
        return None
    end = until
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
