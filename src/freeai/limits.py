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
Unit = Literal["requests", "tokens", "output_tokens"]
LimitScope = Literal["model", "provider"]
State = MutableMapping[str, Any]

_WINDOW_SECONDS = {"minute": 60.0, "hour": 3600.0, "day": 86400.0}
# Half-window used only by :func:`_reading_order` to tell same-window drift from a successor.
# Includes ``month`` (absent from ``_WINDOW_SECONDS``, which is for GCRA intervals only).
_READING_HALF = {"minute": 30.0, "hour": 1800.0, "day": 43200.0, "month": 15 * 86400.0}
# Minute and hour windows refill continuously (GCRA) and day and month windows reset at a boundary,
# unless the platform's headers say otherwise (Groq's daily window refills, OVHcloud's minute resets).
_ROLLING_BY_DEFAULT = frozenset({"minute", "hour"})
# A request reaches the platform a little after we reserve it, and that delay varies (connection
# reuse, TLS handshakes, scheduling). Quota a rolling window regains over time is therefore used
# this many seconds late, or a request sent exactly at the edge can land just before it. What was
# left in the window when it started refilling is not affected, so bursts are unchanged.
EDGE_MARGIN = 1.0
# A window that gives quota back by whole intervals (see ``Limit.whole``) measures them between the
# moments requests arrive, and arrival times vary too. Each request waits this much past an interval
# or it can land just short of one. Unlike EDGE_MARGIN this is lost on every request at the limit, so
# it stays small: on Groq, requests 2.1 s apart got no 429s through the SDK (measured 2026-11-09).
WHOLE_MARGIN = 0.1
# A reading that came back with request R counts everything the platform accepted before R. A
# request reserved this many seconds before R was has reached the platform by then, so its
# reservation is not added on top of that reading a second time. The bound is how long a request
# can take to arrive: connecting (the provider clients give up after 5 s) plus EDGE_MARGIN for the
# rest. A client of your own with a longer connect timeout, or a pool so small that requests queue
# for a connection, can exceed it; the reading is then one request short until that request answers.
ACCEPT_MARGIN = 6.0
_PENDING_PREFIX = "p:"  # reservations the platform has not accepted yet, keyed by quota key


# -- quota windows -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Limit:
    """One allowance window, e.g. 30 requests per minute per model.

    Minute and hour windows are rolling (GCRA); day and month windows are fixed and reset at the
    provider's ``reset_tz`` boundary. Response headers can override this per window (see
    :class:`RateSnapshot`). ``models`` restricts the limit to matching model ids (globs).
    ``source`` and ``last_verified`` record where the number came from, since free limits change.

    ``whole`` is for a platform that gives one back per whole interval (``per`` / ``amount``) since
    the last request it accepted, and drops the remainder when it accepts the next one. Such a window
    stays empty under steady traffic instead of catching up after a late request: Groq's 30 per
    minute lets requests through only when they are 2 s apart, and regains none during a burst.

    ``estimate`` marks a number that is a guess (a third-party table, a forum post) for a window the
    platform never reports. With ``adapt_limits=True`` the SDK finds the real size of estimated
    requests per day or month by itself: it sends one request more when the window is full and
    raises the number when that is accepted, and lowers it when the platform reports the allowance
    exhausted although the window had room (see :mod:`freeai._adapt`).
    """

    amount: float
    per: Window
    unit: Unit = "requests"
    scope: LimitScope = "model"
    models: tuple[str, ...] | None = None
    source: str | None = None
    last_verified: str | None = None
    whole: bool = False
    estimate: bool = False

    def __post_init__(self) -> None:
        if self.whole and self.per not in _WINDOW_SECONDS:
            raise ValueError(f"whole needs a window with a fixed length, not per {self.per}")

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


def otpm(amount: float, **kwargs: Any) -> Limit:
    """Output tokens per minute, for platforms that limit them apart from total tokens."""
    return Limit(amount, "minute", "output_tokens", **kwargs)


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
    roll, day and month are fixed). ``whole`` says the window gives quota back by whole intervals
    (see :class:`Limit`), which also decides for a limit that was declared without saying so.
    """

    per: Window
    unit: Unit
    limit: float
    remaining: float
    scope: LimitScope = "model"
    reset_in: float | None = None
    rolling: bool | None = None
    whole: bool = False


@dataclass(frozen=True, slots=True)
class Cost:
    """What one request is expected to consume. Tokens are an estimate until usage arrives."""

    requests: float = 1.0
    tokens: float = 0.0
    output_tokens: float = 0.0

    def of(self, unit: Unit) -> float:
        if unit == "requests":
            return self.requests
        return self.tokens if unit == "tokens" else self.output_tokens


def quota_ready_at(
    tx: State, limits: list[Limit], provider: str, model_id: str, cost: Cost, now: float, tz: str
) -> float:
    ready = now
    for limit in limits:
        amount = cost.of(limit.unit)
        if amount > 0 and math.isfinite(amount) and limit.applies_to(model_id):
            ready = max(ready, _window_ready(limit, tx.get(limit.key(provider, model_id)), amount, now, tz))
    return ready


def quota_refill(
    tx: State, limits: list[Limit], provider: str, model_id: str, cost: Cost, now: float, tz: str
) -> float:
    """Seconds the platform needs to give back what ``cost`` uses; the scarcest window decides.

    This is what a request costs in quota terms, and ``strategy="quota"`` spends the cheapest first.
    A rolling window refills at ``amount / period``, so a request costs its share of the period:
    1.5 s at 40 per minute, 86.4 s at 1000 per day. A fixed window gives nothing back before it
    resets, so a request costs its share of the time left, measured against what is left: 1728 s
    of a fresh 50 per day, the whole remaining time for the last one, and next to nothing for what
    is still unspent shortly before the reset, because that would be lost anyway.
    """
    refill = 0.0
    for limit in limits:
        amount = cost.of(limit.unit)
        if amount > 0 and math.isfinite(amount) and limit.applies_to(model_id):
            refill = max(refill, _window_refill(limit, tx.get(limit.key(provider, model_id)), amount, now, tz))
    return refill


def quota_used(tx: State, limit: Limit, provider: str, model_id: str, now: float, tz: str) -> float:
    """How much of the window is taken by the local count, in the limit's unit."""
    state = _usable_quota_state(tx.get(limit.key(provider, model_id)))
    capacity = _capacity(limit, state)
    if capacity <= 0:
        return 0.0
    if _whole(limit, state):
        row = _whole_row(state)
        return 0.0 if row is None else max(0.0, capacity - row[0])
    if _rolling(limit.per, state):
        tat = _quota_number(state.get("tat")) if state else None
        if tat is None or tat <= now:
            return 0.0
        return min(capacity, (tat - now) * capacity / _WINDOW_SECONDS[limit.per])
    return _used_in_period(limit, state, now, tz)


def quota_started(tx: State, limit: Limit, provider: str, model_id: str, now: float, tz: str) -> float:
    """Since when the window's local count has been running without a break.

    A rolling window starts over when it is whole again, a fixed one when its period begins. What
    was charged before that no longer weighs on the count, so a mistake made before it (a request
    the platform refused but that stays charged here) is gone.
    """
    state = _usable_quota_state(tx.get(limit.key(provider, model_id)))
    if _whole(limit, state):
        row = _whole_row(state)
        return now if row is None or row[0] >= _capacity(limit, state) else 0.0
    if _rolling(limit.per, state):
        tat = _quota_number(state.get("tat")) if state else None
        if tat is None or tat <= now:
            return now
        return _quota_number(state.get("since")) or 0.0 if state else 0.0
    if limit.per in _ROLLING_BY_DEFAULT:
        return 0.0  # a fixed minute or hour is a platform's reading, which is not adapted
    moment = datetime.fromtimestamp(now, _zone(tz))
    first = moment.date() if limit.per == "day" else moment.date().replace(day=1)
    return datetime.combine(first, datetime.min.time(), tzinfo=_zone(tz)).timestamp()


def horizon_seconds(per: Window, now: float, tz: str) -> float:
    """What a request costs against an allowance of unknown size: at worst the rest of its window."""
    return _WINDOW_SECONDS[per] if per in _ROLLING_BY_DEFAULT else seconds_until(per, now, tz)


def quota_charge(
    tx: State, limits: list[Limit], provider: str, model_id: str, cost: Cost, now: float, tz: str
) -> list[tuple[str, float]]:
    """Charge windows for ``cost``; return ``(quota_key, amount)`` pairs actually charged."""
    charges: list[tuple[str, float]] = []
    for limit in limits:
        amount = cost.of(limit.unit)
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


def quota_observe(
    tx: State,
    provider: str,
    model_id: str,
    snapshot: RateSnapshot,
    now: float,
    tz: str,
    sent_at: float | None = None,
) -> Limit:
    """Replace local counters with what the platform reported; returns the learned limit.

    A snapshot counts the requests the platform had accepted when it answered. Requests sent but not
    accepted yet (no response headers back, see :func:`pending_add`) are added on top so their
    reservations are not wiped out; these live in the store, so every process sharing it sees them.
    Otherwise the server's count wins, which also settles over-estimated token reservations.

    ``sent_at`` is when the request this reading came back with was reserved. The platform accepted
    it after everything reserved :data:`ACCEPT_MARGIN` earlier, so the reading already counts those
    and their reservations are left out: a request that is still being answered is then counted
    once, not twice. Without ``sent_at`` (a quota endpoint) every reservation is added.

    Readings can also arrive in another order than they were taken: the headers of an answer that
    is not streamed come with its body but hold the count from when the request was accepted
    (measured on Groq, 2026-11-10). For tokens, a reading whose request was sent before the request
    of the reading already applied (``asof``) is therefore the older one and is ignored: taking it
    would forget what later requests used. Its own reservation, if the applied reading had added
    it, stays counted until the next reading, which errs on the safe side. Request counts never go
    down, so they are told apart by the count itself (below).

    Concurrent responses can be handled in another order than the platform counted them. Platforms
    never hand requests back, so a request count read lower than one already applied is a late
    reading and is ignored. Token counts can drop for real (unused output handed back), so for
    them only readings of a window that has already ended are ignored.

    Non-positive or non-finite ``snapshot.limit`` values are ignored (store unchanged).
    """
    limit = Limit(snapshot.limit, snapshot.per, snapshot.unit, snapshot.scope, source="response headers")
    if not math.isfinite(snapshot.limit) or snapshot.limit <= 0:
        return limit
    key = limit.key(provider, model_id)
    old = _quota_state(tx.get(key))  # what the last reading (and later charges) left
    stamp: dict[str, Any] = {}
    if sent_at is not None:
        asof = _quota_number(old.get("asof")) if old else None
        if asof is not None and asof > now:
            asof = None  # the wall clock was set back: that reading is not from the future
        if asof is not None and sent_at < asof:
            if limit.unit != "requests":
                return limit  # taken before the reading that is already applied
            stamp["asof"] = asof  # keep the later time; reservations are still judged by this reading's own
        else:
            stamp["asof"] = sent_at
    if snapshot.whole:
        tx[key] = {"tokens": max(0.0, snapshot.remaining), "last": now, "amount": snapshot.limit, "srv": 1, **stamp}
        return limit
    seen = max(0.0, snapshot.limit - snapshot.remaining)
    requests = limit.unit == "requests"
    rolling = limit.per in _ROLLING_BY_DEFAULT if snapshot.rolling is None else snapshot.rolling
    if limit.per in _ROLLING_BY_DEFAULT and snapshot.reset_in is None:
        rolling = True  # a fixed minute or hour window is no use without knowing when it ends
    if rolling and limit.per in _WINDOW_SECONDS:
        interval = _WINDOW_SECONDS[limit.per] / max(snapshot.limit, 1e-9)
        served = now + seen * interval  # when the platform's own count is whole again
        old_served = _quota_number(old.get("served")) if old else None
        if requests and old_served is not None:
            # Platforms never hand requests back, even when the reported limit amount changes.
            served = max(served, old_served)
        tat = served + _pending_total(tx, key, now, sent_at) * interval
        state: dict[str, Any] = {"tat": tat, "amount": snapshot.limit, "srv": 1, "since": now, **stamp}
        if requests:
            state["served"] = served
    else:
        until = now + snapshot.reset_in if snapshot.reset_in is not None else None
        period = _period(limit.per, now, tz) if limit.per not in _ROLLING_BY_DEFAULT else None
        order = _reading_order(limit.per, old, until, period)
        if order < 0:  # the window this reading counted has already ended
            if old is None:
                raise RuntimeError("fixed-window reading order < 0 requires prior state")
            seen = _quota_number(old.get("seen")) or 0.0
            until = _quota_number(old.get("until"))
        elif order == 0 and old is not None:
            # Requests never go backwards. Tokens may drop (unused output handed back), but the
            # window end must not: a past until would make the window look empty.
            if requests:
                old_seen = _quota_number(old.get("seen"))
                if old_seen is not None:
                    seen = max(seen, old_seen)
            if until is not None:
                old_until = _quota_number(old.get("until"))
                if old_until is not None:
                    until = max(until, old_until)
        state = {"used": seen + _pending_total(tx, key, now, sent_at), "seen": seen, "amount": snapshot.limit, "srv": 1}
        if period is not None:
            state["period"] = period
        if until is not None:
            state["until"] = until
        state.update(stamp)
    tx[key] = state
    return limit


def _quota_state(value: Any) -> dict[str, Any] | None:
    """Quota counter row from the store; non-dict values are treated as missing."""
    return value if isinstance(value, dict) else None


def _reported(state: dict[str, Any] | None) -> float | None:
    """The window's size as the platform last reported it (:func:`quota_observe`), if it did.

    Only a platform's reading is kept in a row (``amount``, marked ``srv``). A size copied from the
    declared limit would outlive a change of that limit, because rows of rolling windows never
    expire; rows written that way by older versions are read as not reported.
    """
    if not state or not state.get("srv"):
        return None
    return _quota_number(state.get("amount"))


def _capacity(limit: Limit, state: dict[str, Any] | None) -> float:
    """The window's size: what the platform reported, else what the limit declares."""
    reported = _reported(state)
    return limit.amount if reported is None else reported


def _size(state: dict[str, Any] | None) -> dict[str, Any]:
    """What a reading left that the row a charge rewrites must keep: the reported size and its time.

    Nothing for a window the platform has not reported.
    """
    reported = _reported(state)
    if reported is None or state is None:
        return {}
    asof = _quota_number(state.get("asof"))
    return {"amount": reported, "srv": 1} if asof is None else {"amount": reported, "srv": 1, "asof": asof}


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
    half = _READING_HALF[per]
    delta = until - old_until
    return -1 if delta <= -half else 0 if delta < half else 1


def _pending_entries(tx: State, key: str) -> dict[str, Any]:
    """Reservation map for one quota key; replace a corrupt non-dict value with a fresh map."""
    raw = tx.get(_PENDING_PREFIX + key)
    return dict(raw) if isinstance(raw, dict) else {}


def pending_add(
    tx: State,
    reservation: str,
    charges: Iterable[tuple[str, float]],
    expires_at: float,
    sent_at: float | None = None,
) -> None:
    """Record charges of a request the platform has not accepted yet (until ``expires_at``).

    ``sent_at`` is when the request was reserved, which lets a later reading tell that it already
    counts this request (see :func:`quota_observe`).
    """
    for key, amount in charges:
        entries = _pending_entries(tx, key)
        entries[reservation] = [amount, expires_at] if sent_at is None else [amount, expires_at, sent_at]
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
    """``[amount, expires_at]`` or ``[amount, expires_at, sent_at]`` from the store.

    Truncated or non-numeric rows are skipped; a ``sent_at`` that is not a number is dropped, which
    reads the row like one of an older version (always added to a reading).
    """
    if not isinstance(entry, (list, tuple)) or len(entry) < 2:
        return None
    try:
        amount, expires = float(entry[0]), float(entry[1])
    except (TypeError, ValueError):
        return None
    if not math.isfinite(amount) or not math.isfinite(expires):
        return None
    sent = _finite_float(entry[2]) if len(entry) > 2 else None
    return [amount, expires] if sent is None else [amount, expires, sent]


def _pending_total(tx: State, key: str, now: float, reading_sent_at: float | None = None) -> float:
    """Reservations to add to a reading: all that are alive, or those the reading cannot contain yet."""
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
    counted = None if reading_sent_at is None else reading_sent_at - ACCEPT_MARGIN
    return sum(
        max(0.0, row[0])
        for row in live.values()
        if counted is None or len(row) < 3 or row[2] > counted  # reserved too late to be in the reading
    )


def _window_ready(limit: Limit, state: Any, amount: float, now: float, tz: str) -> float:
    state = _usable_quota_state(state)
    capacity = _capacity(limit, state)
    if capacity <= 0:
        return math.inf  # no allowance; never ready for a positive charge
    if _whole(limit, state):
        return _whole_ready(state, capacity, _WINDOW_SECONDS[limit.per] / capacity, amount, now)
    if _rolling(limit.per, state):
        period = _WINDOW_SECONDS[limit.per]
        interval = period / capacity
        tat_n = _quota_number(state.get("tat")) if state else None
        tat = now if tat_n is None else tat_n
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


def _window_refill(limit: Limit, state: Any, amount: float, now: float, tz: str) -> float:
    state = _usable_quota_state(state)
    capacity = _capacity(limit, state)
    if capacity <= 0:
        return math.inf
    if _whole(limit, state) or _rolling(limit.per, state):
        return amount * _WINDOW_SECONDS[limit.per] / capacity
    left = capacity - _used_in_period(limit, state, now, tz)
    if left <= 0:
        return math.inf
    until = _window_end(limit.per, state, now)
    if until is None:
        calendar = limit.per not in _ROLLING_BY_DEFAULT
        until = _next_boundary(limit.per, now, tz) if calendar else now + _WINDOW_SECONDS[limit.per]
    return amount * max(0.0, until - now) / left


def _window_add(limit: Limit, state: Any, amount: float, now: float, tz: str) -> dict[str, Any]:
    state = _usable_quota_state(state)
    capacity = _capacity(limit, state)
    size = _size(state)
    if capacity <= 0:
        if _rolling(limit.per, state):
            return {"tat": math.inf, **size, "since": now}
        added_zero: dict[str, Any] = {"used": max(0.0, amount), **size}
        if limit.per not in _ROLLING_BY_DEFAULT:
            added_zero["period"] = _period(limit.per, now, tz)
        return added_zero
    if _whole(limit, state):
        return {**_whole_add(state, capacity, _WINDOW_SECONDS[limit.per] / capacity, amount, now), **size}
    if _rolling(limit.per, state):
        interval = _WINDOW_SECONDS[limit.per] / capacity
        tat_n = _quota_number(state.get("tat")) if state else None
        tat = now if tat_n is None else tat_n
        if tat <= now:  # whole: refilling starts with this charge
            return {"tat": max(now, now + amount * interval), **size, "since": now}
        since = _quota_number(state.get("since")) if state else None
        if since is None:
            since = now
        added_tat: dict[str, Any] = {"tat": max(now, tat + amount * interval), **size, "since": since}
        served = _quota_number(state.get("served"))
        if served is not None:
            added_tat["served"] = served  # the platform's count, if it reported one
        return added_tat
    used = _used_in_period(limit, state, now, tz)
    added: dict[str, Any] = {"used": max(0.0, used + amount), **size}
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


def _whole(limit: Limit, state: dict[str, Any] | None) -> bool:
    """Whether a window gives quota back by whole intervals; the platform's reading also decides."""
    return limit.whole or (state is not None and "tokens" in state)


def _whole_row(state: dict[str, Any] | None) -> tuple[float, float] | None:
    """``(tokens, last)`` of a whole-interval window; any other row counts as a full window.

    An older version's row holds at most a minute's count, and the first 429 corrects it.
    """
    tokens = _quota_number(state.get("tokens")) if state else None
    last = _quota_number(state.get("last")) if state else None
    return None if tokens is None or last is None else (tokens, last)


def _whole_ready(state: dict[str, Any] | None, capacity: float, interval: float, amount: float, now: float) -> float:
    row = _whole_row(state)
    if row is None:
        return now
    tokens, last = row
    missing = math.ceil(min(amount, capacity) - tokens)
    return now if missing <= 0 else max(now, last + WHOLE_MARGIN + missing * interval)


def _whole_add(
    state: dict[str, Any] | None, capacity: float, interval: float, amount: float, now: float
) -> dict[str, Any]:
    tokens, last = _whole_row(state) or (capacity, now)
    if amount < 0:  # a refund: the platform never saw this request, so its clock did not move
        return {"tokens": min(capacity, tokens - amount), "last": last}
    back = max(0, math.floor((now - last - WHOLE_MARGIN) / interval))
    return {"tokens": max(0.0, min(capacity, tokens + back) - amount), "last": now}


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
