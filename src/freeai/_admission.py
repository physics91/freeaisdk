"""Quota admission: reserve a ready candidate and release concurrency/pending state."""

from __future__ import annotations

import dataclasses
import math
import secrets
import threading
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from typing import Any, NamedTuple

from . import _adapt
from ._routing_types import (
    _LEARNED_SCOPES,
    _LEARNED_UNITS,
    _LEARNED_WINDOWS,
    _NO_RESERVATION,
    Request,
    Reservation,
    _cooldown_bundle,
    _learned_key,
    _model_key,
    _provider_key,
    _usage_number,
)
from .cooldown import _cooldown_failure_count, _cooldown_until, cooldown_failures, cooldown_ready_at
from .limits import (
    Cost,
    Limit,
    State,
    horizon_seconds,
    pending_add,
    pending_drop,
    quota_charge,
    quota_ready_at,
    quota_refill,
    quota_started,
)
from .providers import Provider
from .types import ModelInfo

# A reservation counts on top of header readings until the platform accepts the request. It must
# outlive the wait for response headers (bounded by the read timeout) or a live request's charge
# would be wiped; if a process dies, its reservations over-count for at most this long.
_PENDING_TTL = 300.0
_SLOT_POLL = 0.25  # re-check interval while a provider's concurrency slots or a probe are busy
# ``strategy="quota"`` spends free allowances before trial credit, and that before money. A provider
# that does not say what it is (``tier="unknown"``, the default of a hand-made ``Provider``) comes
# after the free ones: with no limit declared it would otherwise look like the cheapest candidate and
# take every request, and it may well be somebody's paid endpoint. It is still used whenever no free
# candidate is ready, so this costs no throughput.
_BILLING_RANK = {"free": 0, "unknown": 1, "trial": 2, "paid": 3}


class _Ready(NamedTuple):
    """A candidate that can be sent now, with what admitting it needs."""

    model: ModelInfo
    provider: Provider
    limits: list[Limit]
    cost: Cost
    probes: list[str]
    probed: tuple[tuple[str, float], ...] = ()  # estimated windows this request goes beyond, and their size


class Admission:
    """Process-local slots/probes plus store-backed quota charge for the next ready model."""

    __slots__ = (
        "_by_name",
        "_cost",
        "_inflight",
        "_limit_probes",
        "_lock",
        "_probe_claims",
        "_probing",
        "adapt",
        "clock",
        "store",
    )

    def __init__(
        self,
        *,
        lock: threading.Lock,
        store: Any,
        by_name: Mapping[str, Provider],
        inflight: MutableMapping[str, int],
        probing: set[str],
        clock: Callable[[], float],
        cost: Callable[[Provider, Request], Cost],
        adapt: bool = False,
    ) -> None:
        self.adapt = adapt  # learn the size of limits declared as estimates
        self._limit_probes: set[str] = set()  # estimated windows with a probe in flight
        self._lock = lock
        self.store = store
        self._by_name = by_name
        self._inflight = inflight
        self._probing = probing
        self._probe_claims: dict[str, frozenset[str]] = {}
        self.clock = clock
        self._cost = cost

    def reserve(
        self, candidates: list[ModelInfo], request: Request, tiers: Mapping[str, int] | None = None
    ) -> tuple[ModelInfo | None, float, Reservation]:
        """Pick a candidate that is ready now and charge its quota windows.

        Without ``tiers`` that is the first ready candidate. With ``tiers`` (``strategy="quota"``,
        the static group of each candidate) it is the ready candidate of the first group that has
        one whose quota comes back soonest, so allowances that refill every minute are spent before
        daily ones. Groups keep their order: ``prefer``, an explicit ``model`` and routers-last mean
        what they mean for every other strategy.
        """
        with self._lock:
            with self.store.transaction() as tx:
                now = self.clock()
                found, earliest = self._choose(tx, candidates, request, tiers, now)
                reservation = _NO_RESERVATION if found is None else self._charge(tx, found, now)
            if found is None:
                return None, earliest, _NO_RESERVATION
            # Only now that the charge is committed: a commit that fails must not leave a slot or a
            # probe taken for a request that is never sent.
            self._claim(found, reservation)
            return found.model, now, reservation

    def _choose(
        self, tx: State, candidates: list[ModelInfo], request: Request, tiers: Mapping[str, int] | None, now: float
    ) -> tuple[_Ready | None, float]:
        """The candidate to send now, or None and when the first one will be ready."""
        earliest = math.inf
        learned_by_sid: dict[str, list[Any]] = {}
        cooldown_by_sid: dict[str, tuple[Any, dict[str, Any], bool]] = {}
        best: _Ready | None = None
        best_score = (0, 0.0)
        best_tier = -1
        for model in candidates:
            if best is not None and tiers is not None and tiers.get(model.key, -1) != best_tier:
                break  # a later group never beats a ready candidate of an earlier one
            provider = self._by_name[model.provider]
            sid = provider.state_id
            model_key, provider_key = _model_key(sid, model.id), _provider_key(sid)
            if sid not in cooldown_by_sid:
                cooldown_by_sid[sid] = _cooldown_bundle(tx, sid)
            provider_entry, model_entries, authoritative = cooldown_by_sid[sid]
            if authoritative:
                model_entry = model_entries.get(model.id)
                model_ready = _cooldown_until(model_entry)
                model_failures = _cooldown_failure_count(model_entry)
            else:
                model_ready = cooldown_ready_at(tx, model_key)
                model_failures = cooldown_failures(tx, model_key)
            provider_ready = _cooldown_until(provider_entry)
            provider_failures = _cooldown_failure_count(provider_entry)
            ready = max(model_ready, provider_ready)
            # Skip learned/limits/cost work while a cooldown already keeps the candidate busy.
            if ready <= now:
                if sid not in learned_by_sid:
                    raw_learned = tx.get(_learned_key(sid), [])
                    learned_by_sid[sid] = (
                        [row for row in raw_learned if isinstance(row, Mapping)]
                        if isinstance(raw_learned, list)
                        else []
                    )
                limits = self.limits(tx, provider, model.id, learned=learned_by_sid[sid])
                cost = self._cost(provider, request)
                ready = quota_ready_at(tx, limits, sid, model.id, cost, now, provider.reset_tz)
                probed: tuple[tuple[str, float], ...] = ()
                if ready > now and self.adapt:
                    probed = self._limit_probe(tx, limits, provider, model.id, cost, now)
                    if probed:
                        ready = now  # only estimates are in the way, and they are due for a probe
                probes = [
                    key for key, fails in ((model_key, model_failures), (provider_key, provider_failures)) if fails > 0
                ]
                if ready <= now and (self._slots_full(provider) or self._probing.intersection(probes)):
                    ready = now + _SLOT_POLL
                if ready <= now:
                    found = _Ready(model, provider, limits, cost, probes, probed)
                    if tiers is None:
                        return found, now
                    score = (_BILLING_RANK.get(provider.tier, 0), self._refill(tx, found, now))
                    if best is None or score < best_score:
                        best, best_score, best_tier = found, score, tiers.get(model.key, -1)
                    continue
            earliest = min(earliest, ready)
        return best, earliest

    def _refill(self, tx: State, found: _Ready, now: float) -> float:
        """Seconds until the platform has given back what this request would use (see ``quota_refill``)."""
        provider = found.provider
        refill = quota_refill(tx, found.limits, provider.state_id, found.model.id, found.cost, now, provider.reset_tz)
        if provider.uncounted is not None:
            refill = max(refill, horizon_seconds(provider.uncounted, now, provider.reset_tz))
        return refill

    def _charge(self, tx: State, found: _Ready, now: float) -> Reservation:
        """Charge the candidate's quota windows and record that the platform has not accepted it yet."""
        provider = found.provider
        sid = provider.state_id
        charges = quota_charge(tx, found.limits, sid, found.model.id, found.cost, now, provider.reset_tz)
        if not charges:
            return _NO_RESERVATION
        reservation = Reservation(secrets.token_hex(8), tuple(key for key, _ in charges), found.probed)
        pending_add(tx, reservation.id, charges, now + max(_PENDING_TTL, 2 * provider.timeout), now)
        return reservation

    def _claim(self, found: _Ready, reservation: Reservation) -> None:
        """Take the candidate's concurrency slot and its probes. The caller holds the lock."""
        provider = found.provider
        self._limit_probes.update(key for key, _ in reservation.probed)
        self._inflight[provider.name] = self._inflight.get(provider.name, 0) + 1
        claimed = frozenset(found.probes)
        self._probe_claims[_model_key(provider.state_id, found.model.id)] = claimed
        self._probing.update(claimed)  # half-open: one trial request after a cooldown

    def _limit_probe(
        self, tx: State, limits: list[Limit], provider: Provider, model_id: str, cost: Cost, now: float
    ) -> tuple[tuple[str, float], ...]:
        """The estimated windows to send one request beyond, each with its size, or () for no probe.

        A probe needs every window that holds the request back to be one whose size is learned (an
        estimated number of requests per day or month) and whose count can be trusted (nothing that
        failed is still charged to it), none of them about to reset, none probed right now, and
        none refused recently.
        """
        sid, tz = provider.state_id, provider.reset_tz
        soft = [limit for limit in limits if _adapt.adapts(limit)]
        firm = [limit for limit in limits if not _adapt.adapts(limit)]
        if not soft or quota_ready_at(tx, firm, sid, model_id, cost, now, tz) > now:
            return ()
        full: list[tuple[str, float]] = []
        for limit in soft:
            ready = quota_ready_at(tx, [limit], sid, model_id, cost, now, tz)
            if ready <= now:
                continue
            key = limit.key(sid, model_id)
            if (
                ready - now <= _adapt.PROBE_GUARD
                or key in self._limit_probes
                or _adapt.probe_ready_at(tx, key) > now
                or _adapt.unsure_since(tx, key) >= quota_started(tx, limit, sid, model_id, now, tz)
            ):
                return ()
            full.append((key, limit.amount))
        return tuple(full)

    def _slots_full(self, provider: Provider) -> bool:
        return provider.max_concurrency is not None and self._inflight.get(provider.name, 0) >= provider.max_concurrency

    def limits(
        self,
        tx: State,
        provider: Provider,
        model_id: str,
        *,
        learned: Sequence[Mapping[str, Any]] | None = None,
    ) -> list[Limit]:
        """Declared limits plus those learned from response headers or quota endpoints.

        With ``adapt``, an estimated number of requests per day or month carries the size learned
        for it, and an estimate whose window the platform reports is no estimate any more.
        """
        sid = provider.state_id
        by_key: dict[str, Limit] = {}
        for limit in provider.limits:
            if limit.applies_to(model_id):
                # One window declared twice (a preset's default and the caller's ``rpm=``): the later
                # one counts. Charging both would count every request twice in the same row.
                by_key[limit.key(sid, model_id)] = limit
        limits = list(by_key.values())
        keys = set(by_key)
        declared = set(keys)
        reported: set[str] = set()
        if learned is not None:
            entries = learned
        else:
            raw = tx.get(_learned_key(sid), [])
            entries = raw if isinstance(raw, list) else []
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            if entry.get("model") not in (None, model_id):
                continue
            amount = _usage_number(entry.get("amount"))
            per, unit, scope = entry.get("per"), entry.get("unit"), entry.get("scope")
            if (
                amount is None
                or amount <= 0
                or per not in _LEARNED_WINDOWS
                or unit not in _LEARNED_UNITS
                or scope not in _LEARNED_SCOPES
            ):
                continue
            observed = Limit(amount, per, unit, scope, source="observed")
            key = observed.key(sid, model_id)
            if key in declared:
                reported.add(key)
            elif key not in keys:
                limits.append(observed)
                keys.add(key)
        if self.adapt:
            limits = [self._effective(tx, limit, sid, model_id, reported) for limit in limits]
        return limits

    @staticmethod
    def _effective(tx: State, limit: Limit, sid: str, model_id: str, reported: set[str]) -> Limit:
        if not limit.estimate:
            return limit
        key = limit.key(sid, model_id)
        if key in reported:
            return dataclasses.replace(limit, estimate=False)  # the platform's own number is used
        if not _adapt.adapts(limit):
            return limit
        learned = _adapt.learned_amount(tx, key, limit.amount)
        return limit if learned is None else dataclasses.replace(limit, amount=learned)

    def release_probes(self, probed: tuple[tuple[str, float], ...]) -> None:
        if probed:
            with self._lock:
                self._limit_probes.difference_update(key for key, _ in probed)

    def release_slot(self, model: ModelInfo, probed: tuple[tuple[str, float], ...] = ()) -> None:
        """Free concurrency/probe bookkeeping only. Caller owns any ``pending_drop``."""
        provider = self._by_name[model.provider]
        sid = provider.state_id
        with self._lock:
            self._limit_probes.difference_update(key for key, _ in probed)
            count = self._inflight.get(provider.name, 0)
            if count > 0:
                self._inflight[provider.name] = count - 1
            # Only drop probe keys this attempt claimed (do not clear a peer's provider probe).
            claimed = self._probe_claims.pop(_model_key(sid, model.id), frozenset())
            self._probing.difference_update(claimed)

    def drop_pending(self, reservation: Reservation) -> None:
        if reservation.keys:
            with self._lock, self.store.transaction() as tx:
                pending_drop(tx, reservation.id, reservation.keys)

    def release(self, model: ModelInfo, reservation: Reservation = _NO_RESERVATION) -> None:
        """Free the concurrency slot, probe flags and any reservation Plan.admitted did not take."""
        self.release_slot(model, reservation.probed)
        self.drop_pending(reservation)
