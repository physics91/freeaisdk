"""Daily and monthly limits declared as estimates: find the real size (``adapt_limits``).

Some free limits are not published and no header reports them (Gemini's come from a third-party
table). A wrong guess costs either way: too low and quota is left unused every day, too high and
every day ends in a 429. For request limits per day or month that are marked ``estimate`` the
router therefore learns the size from what the platform does:

- **Probe.** When such a window is the only thing holding a request back, one request is sent
  beyond it. If the platform accepts it, the window is larger than assumed and the size grows
  (:data:`GROWTH`, at least by one); the next request beyond the new size is a probe again, so one
  day is enough to find a much larger allowance. If the platform refuses the probe (a rate limit
  or an exhausted quota, however it words it), the size stays and the next probe waits
  (:func:`backoff`: a window, then twice as long with every refused probe in a row, up to a week).
  A probe that fails in another way, such as a 500, was not refused and is not held against the
  window.
- **Cut.** When the platform reports the allowance exhausted for a request that was no probe,
  although the window had room by the local count, the window is smaller than assumed and is cut
  to what had been accepted in it.

An accepted probe only proves something while the local count is no higher than the platform's. A
request that failed stays charged here although the platform may not have counted it, so after any
failure the window is not probed until its next period has begun.

Windows per minute and per hour are deliberately left alone, estimate or not. They are counted as
rolling windows, and whether a request beyond one is accepted depends on how the platform counts
(a bucket that allows bursts, or a sliding window that does not), which is exactly what is unknown
about such a platform. Simulated against both kinds, every rule tried either raised a correct
number or lowered one, at the price of hundreds of 429s in the worst case.

Learned sizes live in the store under their own keys (``qa:``), next to the window rows and never in
them: turning the option off restores the declared numbers, a learned size is dropped when the
declared number changes, and a window the platform starts reporting (headers, a quota endpoint, a
429 body) is no longer adapted at all. Only one probe per window is in flight in a process; two
processes sharing a store may each send one.
"""

from __future__ import annotations

import math
from typing import Any

from ._http import logger
from .errors import FailureKind
from .limits import _WINDOW_SECONDS, EDGE_MARGIN, Limit, State, Window, _finite_float

_PREFIX = "qa:"
GROWTH = 1.25  # an accepted probe raises the size by a quarter (at least by one)
_MAX_BACKOFF = 7 * 86400.0
_MAX_DOUBLINGS = 16  # beyond this the backoff is capped anyway; keeps 2**n finite
_MONTH = 30 * 86400.0
PROBE_GUARD = 1.5 * EDGE_MARGIN  # a window this close to its reset is not probed: it is about to open anyway
ADAPTED: frozenset[Window] = frozenset({"day", "month"})
# The failures that are a platform refusing for lack of allowance. A probe that fails any other way
# (a 500, an empty answer) was not refused and says nothing about the window's size.
REFUSALS = frozenset({FailureKind.RATE_LIMIT, FailureKind.PROVIDER_RATE_LIMIT, FailureKind.QUOTA_EXHAUSTED})


def adapts(limit: Limit) -> bool:
    """Whether the size of ``limit`` is learned: an estimated number of requests per day or month."""
    return limit.estimate and limit.unit == "requests" and limit.per in ADAPTED


def _row(tx: State, key: str) -> dict[str, Any]:
    row = tx.get(_PREFIX + key)
    return row if isinstance(row, dict) else {}


def _number(row: dict[str, Any], name: str) -> float:
    return _finite_float(row.get(name)) or 0.0


def _period(per: Window) -> float:
    return _WINDOW_SECONDS.get(per, _MONTH)


def learned_amount(tx: State, key: str, declared: float | None = None) -> float | None:
    """The size learned for the window ``key``, if any.

    With ``declared`` (what the limit declares now), a size learned against another declared number
    does not count: the declaration was changed, and the new number is the better guess.
    """
    row = _row(tx, key)
    amount = _finite_float(row.get("amount"))
    if amount is None or amount <= 0:
        return None
    if declared is not None and _finite_float(row.get("base")) != declared:
        return None
    return amount


def probe_ready_at(tx: State, key: str) -> float:
    """When the window ``key`` may be probed again (0 if never refused)."""
    return _number(_row(tx, key), "until")


def unsure_since(tx: State, key: str) -> float:
    """When a request that may not have been counted by the platform was last charged to ``key``.

    -1 if never. The local count is trusted again once the window's next period has begun.
    """
    marked = _finite_float(_row(tx, key).get("dirty"))
    return -1.0 if marked is None else marked


def backoff(per: Window, refusals: int) -> float:
    """Seconds before the next probe after ``refusals`` refused probes in a row."""
    period = _period(per)
    doublings = min(max(refusals, 1) - 1, _MAX_DOUBLINGS)
    return min(period * 2.0**doublings, max(_MAX_BACKOFF, period))


def _store(tx: State, key: str, row: dict[str, Any], limit: Limit, amount: float | None, **changes: float) -> None:
    """Write the row for ``key``; ``amount`` None keeps what was learned (or nothing) as it is."""
    new: dict[str, Any] = {name: _number(row, name) for name in ("until", "fails")}
    if _finite_float(row.get("dirty")) is not None:
        new["dirty"] = _number(row, "dirty")
    learned, base = _finite_float(row.get("amount")), _finite_float(row.get("base"))
    if amount is not None:
        # The declared number this size was learned against: ``limit`` carries it unless a learned
        # size has already replaced it, in which case the row does.
        learned, base = amount, base if learned is not None and base is not None else limit.amount
    if learned is not None and base is not None:
        new["amount"], new["base"] = learned, base
    new.update(changes)
    tx[_PREFIX + key] = new


def distrust(tx: State, limit: Limit, key: str, now: float) -> None:
    """A request charged to the window failed: the platform may not have counted it."""
    _store(tx, key, _row(tx, key), limit, None, dirty=now)


def grow(tx: State, limit: Limit, key: str, probed: float) -> float | None:
    """A probe of the full window was accepted: the window is larger than ``limit.amount``.

    ``probed`` is the size the window had when the probe was sent. If it has changed since (another
    client's probe was accepted first), this acceptance says nothing new about the current size.
    """
    if limit.amount != probed:
        return None
    amount = max(limit.amount + 1.0, float(math.ceil(limit.amount * GROWTH)))
    _store(tx, key, _row(tx, key), limit, amount, until=0.0, fails=0.0)
    logger.info("%s: a request beyond the estimated %g was accepted; now assuming %g", key, limit.amount, amount)
    return amount


def refuse_probe(tx: State, limit: Limit, key: str, now: float) -> None:
    """A probe was refused: the size stays, and the next probe waits longer each time in a row."""
    row = _row(tx, key)
    fails = _number(row, "fails") + 1.0
    _store(tx, key, row, limit, None, until=now + backoff(limit.per, int(fails)), fails=fails, dirty=now)


def cut(tx: State, limit: Limit, key: str, now: float, accepted: float) -> float | None:
    """The allowance is exhausted with only ``accepted`` of the window used: that is its size.

    Returns the new size, or None when the refusal says nothing about it: it does not point below
    the current size, or nothing was accepted in this window at all (then the window was never
    open by the platform's count: its day may begin at another hour, or something else spends the
    same key). The window is not probed again before a full period has passed.
    """
    amount = float(math.floor(accepted))
    if amount < 1 or amount >= limit.amount:
        return None
    row = _row(tx, key)
    until = max(_number(row, "until"), now + _period(limit.per))
    _store(tx, key, row, limit, amount, until=until, dirty=now)
    logger.info("%s: exhausted with %g of an estimated %g used; now assuming %g", key, accepted, limit.amount, amount)
    return amount
