"""Store-backed quota observation and cooldown status snapshots."""

from __future__ import annotations

import math
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from ._routing_types import _learned_identity, _learned_key, _provider_key
from .cooldown import cooldown_ready_at
from .limits import RateSnapshot, State, quota_observe
from .providers import Provider


class Observe:
    """Apply platform rate snapshots and report remaining cooldowns."""

    __slots__ = ("_by_name", "_lock", "clock", "providers", "store")

    def __init__(
        self,
        *,
        lock: threading.Lock,
        store: Any,
        by_name: Mapping[str, Provider],
        providers: Sequence[Provider],
        clock: Callable[[], float],
    ) -> None:
        self._lock = lock
        self.store = store
        self._by_name = by_name
        self.providers = providers
        self.clock = clock

    def observe(self, provider_name: str, model_id: str, snapshots: Iterable[RateSnapshot]) -> None:
        """Update local quota counters with limit state the platform reported."""
        self.observe_many(((provider_name, model_id, snapshots),))

    def observe_many(self, items: Iterable[tuple[str, str, Iterable[RateSnapshot]]]) -> None:
        """Apply several snapshot batches in one store transaction (empty batches are skipped)."""
        batches: list[tuple[str, str, list[RateSnapshot]]] = []
        for provider_name, model_id, snapshots in items:
            snaps = list(snapshots)
            if snaps:
                batches.append((provider_name, model_id, snaps))
        if not batches:
            return
        with self._lock, self.store.transaction() as tx:
            for provider_name, model_id, snaps in batches:
                self.apply_tx(tx, provider_name, model_id, snaps)

    def apply_tx(
        self,
        tx: State,
        provider_name: str,
        model_id: str,
        snapshots: Iterable[RateSnapshot],
        sent_at: float | None = None,
    ) -> None:
        """Apply ``snapshots`` into an open transaction. Caller must hold ``_lock``; no nested lock.

        ``sent_at`` is when the request the snapshots came back with was reserved (see
        :func:`~freeai.limits.quota_observe`); None for readings of a quota endpoint.
        """
        snapshots = list(snapshots)
        if not snapshots:
            return
        provider = self._by_name[provider_name]
        sid = provider.state_id
        now = self.clock()
        raw_learned = tx.get(_learned_key(sid), [])
        learned_rows = raw_learned if isinstance(raw_learned, list) else []
        by_id: dict[tuple[Any, Any, Any, Any], dict[str, Any]] = {}
        for row in learned_rows:
            if not isinstance(row, Mapping) or "amount" not in row:
                continue
            identity = _learned_identity(row)
            if identity is not None:
                by_id[identity] = dict(row)
        for snapshot in snapshots:
            if isinstance(snapshot.limit, bool) or not math.isfinite(snapshot.limit) or snapshot.limit <= 0:
                continue
            quota_observe(tx, sid, model_id, snapshot, now, provider.reset_tz, sent_at)
            entry = {
                "per": snapshot.per,
                "unit": snapshot.unit,
                "scope": snapshot.scope,
                "model": model_id if snapshot.scope == "model" else None,
                "amount": snapshot.limit,
            }
            identity = _learned_identity(entry)
            if identity is not None:
                by_id[identity] = entry
        # Sorted only for stable store bytes; consumers treat the registry as a bag.
        tx[_learned_key(sid)] = [
            by_id[identity] for identity in sorted(by_id, key=lambda item: (*item[:3], item[3] or ""))
        ]

    def status(self) -> dict[str, dict[str, float]]:
        """Remaining cooldown seconds per model key and per provider."""
        models: dict[str, float] = {}
        providers: dict[str, float] = {}
        with self.store.transaction() as tx:
            now = self.clock()
            by_sid = {provider.state_id: provider.name for provider in self.providers}
            for sid, name in by_sid.items():
                remaining = cooldown_ready_at(tx, _provider_key(sid)) - now
                if remaining > 0:
                    providers[name] = remaining
            # One ``cd:m:`` scan for every provider instead of N prefix snapshots.
            items_with_prefix = getattr(tx, "items_with_prefix", None)
            if callable(items_with_prefix):
                model_entries = items_with_prefix("cd:m:")
            else:
                model_entries = ((key, tx[key]) for key in list(tx) if key.startswith("cd:m:"))
            for key, entry in model_entries:
                if not isinstance(entry, dict):
                    continue
                try:
                    until = entry.get("until", 0.0)
                    if isinstance(until, bool):
                        continue
                    left = float(until) - now
                except (TypeError, ValueError):
                    continue
                if left <= 0:
                    continue
                sid, sep, model_id = key.removeprefix("cd:m:").partition("|")
                if not sep:
                    continue
                provider_name = by_sid.get(sid)
                if provider_name is not None:
                    models[f"{provider_name}:{model_id}"] = left
        return {"models": models, "providers": providers}
