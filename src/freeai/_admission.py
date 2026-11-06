"""Quota admission: reserve a ready candidate and release concurrency/pending state."""

from __future__ import annotations

import math
import secrets
import threading
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from typing import Any

from ._routing_types import (
    _NO_RESERVATION,
    Request,
    Reservation,
    _cooldown_bundle,
    _learned_key,
    _model_key,
    _provider_key,
)
from .cooldown import cooldown_failures, cooldown_ready_at
from .limits import Cost, Limit, State, pending_add, pending_drop, quota_charge, quota_ready_at
from .providers import Provider
from .types import ModelInfo

# A reservation counts on top of header readings until the platform accepts the request. It must
# outlive the wait for response headers (bounded by the read timeout) or a live request's charge
# would be wiped; if a process dies, its reservations over-count for at most this long.
_PENDING_TTL = 300.0
_SLOT_POLL = 0.25  # re-check interval while a provider's concurrency slots or a probe are busy


class Admission:
    """Process-local slots/probes plus store-backed quota charge for the next ready model."""

    __slots__ = ("_by_name", "_cost", "_inflight", "_lock", "_probing", "clock", "store")

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
    ) -> None:
        self._lock = lock
        self.store = store
        self._by_name = by_name
        self._inflight = inflight
        self._probing = probing
        self.clock = clock
        self._cost = cost

    def reserve(self, candidates: list[ModelInfo], request: Request) -> tuple[ModelInfo | None, float, Reservation]:
        """Pick the first candidate that is ready now and charge its quota windows."""
        with self._lock, self.store.transaction() as tx:
            now = self.clock()
            earliest = math.inf
            learned_by_sid: dict[str, list[Any]] = {}
            cooldown_by_sid: dict[str, tuple[Any, dict[str, Any], bool]] = {}
            for model in candidates:
                provider = self._by_name[model.provider]
                sid = provider.state_id
                model_key, provider_key = _model_key(sid, model.id), _provider_key(sid)
                if sid not in cooldown_by_sid:
                    cooldown_by_sid[sid] = _cooldown_bundle(tx, sid)
                provider_entry, model_entries, authoritative = cooldown_by_sid[sid]
                if authoritative:
                    model_entry = model_entries.get(model.id)
                    model_ready = float(model_entry["until"]) if model_entry else 0.0
                    model_failures = int(model_entry["failures"]) if model_entry else 0
                else:
                    model_ready = cooldown_ready_at(tx, model_key)
                    model_failures = cooldown_failures(tx, model_key)
                provider_ready = float(provider_entry["until"]) if provider_entry else 0.0
                provider_failures = int(provider_entry["failures"]) if provider_entry else 0
                ready = max(model_ready, provider_ready)
                if sid not in learned_by_sid:
                    learned_by_sid[sid] = list(tx.get(_learned_key(sid), []))
                limits = self.limits(tx, provider, model.id, learned=learned_by_sid[sid])
                cost = self._cost(provider, request)
                if ready <= now:
                    ready = quota_ready_at(tx, limits, sid, model.id, cost, now, provider.reset_tz)
                probes = [
                    key
                    for key, fails in ((model_key, model_failures), (provider_key, provider_failures))
                    if fails > 0
                ]
                if ready <= now and (self._slots_full(provider) or self._probing.intersection(probes)):
                    ready = now + _SLOT_POLL
                if ready <= now:
                    quota_charge(tx, limits, sid, model.id, cost, now, provider.reset_tz)
                    charges = [(lim.key(sid, model.id), cost.of(lim.unit)) for lim in limits if cost.of(lim.unit) > 0]
                    reservation = _NO_RESERVATION
                    if charges:
                        reservation = Reservation(secrets.token_hex(8), tuple(key for key, _ in charges))
                        ttl = max(_PENDING_TTL, 2 * provider.timeout)
                        pending_add(tx, reservation.id, charges, now + ttl)
                    self._inflight[provider.name] = self._inflight.get(provider.name, 0) + 1
                    self._probing.update(probes)  # half-open: one trial request after a cooldown
                    return model, now, reservation
                earliest = min(earliest, ready)
            return None, earliest, _NO_RESERVATION

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
        """Declared limits plus those learned from response headers or quota endpoints."""
        sid = provider.state_id
        limits = [limit for limit in provider.limits if limit.applies_to(model_id)]
        keys = {limit.key(sid, model_id) for limit in limits}
        entries = learned if learned is not None else tx.get(_learned_key(sid), [])
        for entry in entries:
            if entry["model"] not in (None, model_id):
                continue
            observed = Limit(float(entry["amount"]), entry["per"], entry["unit"], entry["scope"], source="observed")
            key = observed.key(sid, model_id)
            if key not in keys:
                limits.append(observed)
                keys.add(key)
        return limits

    def release_slot(self, model: ModelInfo) -> None:
        """Free concurrency/probe bookkeeping only. Caller owns any ``pending_drop``."""
        provider = self._by_name[model.provider]
        sid = provider.state_id
        with self._lock:
            count = self._inflight.get(provider.name, 0)
            if count > 0:
                self._inflight[provider.name] = count - 1
            self._probing.difference_update({_model_key(sid, model.id), _provider_key(sid)})

    def drop_pending(self, reservation: Reservation) -> None:
        if reservation.keys:
            with self._lock, self.store.transaction() as tx:
                pending_drop(tx, reservation.id, reservation.keys)

    def release(self, model: ModelInfo, reservation: Reservation = _NO_RESERVATION) -> None:
        """Free the concurrency slot, probe flags and any reservation Plan.admitted did not take."""
        self.release_slot(model)
        self.drop_pending(reservation)
