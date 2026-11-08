"""Candidate ordering for :class:`~freeai._core.Router` (prefer/exclude/strategy/health)."""

from __future__ import annotations

import random
import threading
from collections.abc import Mapping, Sequence
from typing import Protocol

from .health import HealthStats
from .limits import _compiled_globs
from .providers import Provider
from .types import ModelInfo, Requirements


class _OrderRequest(Protocol):
    @property
    def models(self) -> tuple[str, ...]: ...
    @property
    def prefer(self) -> tuple[str, ...]: ...
    @property
    def requirements(self) -> Requirements: ...


class OrderPolicy:
    """Owns prefer/exclude/strategy filtering; shares the router's health, rng, and lock."""

    __slots__ = (
        "_by_name",
        "_lock",
        "_rng",
        "allow_training",
        "exclude",
        "health",
        "prefer",
        "providers",
        "strategy",
    )

    def __init__(
        self,
        *,
        providers: Sequence[Provider],
        prefer: Sequence[str],
        exclude: Sequence[str],
        strategy: str,
        allow_training: bool,
        by_name: Mapping[str, Provider],
        health: HealthStats,
        rng: random.Random,
        lock: threading.Lock,
    ) -> None:
        self.providers = providers
        self.prefer = prefer
        self.exclude = exclude
        self.strategy = strategy
        self.allow_training = allow_training
        self._by_name = by_name
        self.health = health
        self._rng = rng
        self._lock = lock

    def order(self, request: _OrderRequest, pools: Mapping[str, Sequence[ModelInfo]], *, turn: int) -> list[ModelInfo]:
        """Candidates for ``request`` in the order they will be tried (ignoring cooldowns)."""
        pool = [m for provider in self.providers for m in pools.get(provider.name, ())]
        ranks: dict[str, int] | None = None
        if request.models:  # explicit choice overrides the default exclusions
            # Same patterns drive filter and prefer-order; compute rank once per model.
            ranked = [(m, _rank(m, request.models)) for m in pool]
            pool = [m for m, rank in ranked if rank < len(request.models)]
            ranks = {m.key: rank for m, rank in ranked if rank < len(request.models)}
        else:
            pool = [m for m in pool if _rank(m, self.exclude) == len(self.exclude)]
        pool = [m for m in pool if m.satisfies(request.requirements) and self._privacy_ok(m)]

        patterns = request.models or request.prefer or self.prefer
        groups: dict[tuple[bool, int], list[ModelInfo]] = {}
        for m in pool:
            rank = ranks[m.key] if ranks is not None else _rank(m, patterns)
            groups.setdefault((m.is_router, rank), []).append(m)
        ordered: list[ModelInfo] = []
        for key in sorted(groups):
            ordered.extend(self._arrange(groups[key], turn))
        return ordered

    def _privacy_ok(self, model: ModelInfo) -> bool:
        if self.allow_training:
            return True
        return model.trains_on_prompts is False or self._by_name[model.provider].can_deny_training

    def _arrange(self, group: list[ModelInfo], turn: int) -> list[ModelInfo]:
        if self.strategy == "random":
            group = group[:]
            self._rng.shuffle(group)
        elif self.strategy == "round_robin" and group:
            offset = turn % len(group)
            group = group[offset:] + group[:offset]
        elif self.strategy == "health":
            group = self._power_of_two(group)
        return group

    def _power_of_two(self, group: list[ModelInfo]) -> list[ModelInfo]:
        """Order by repeated power-of-two-choices on the health score (load spreads, bad models sink)."""
        with self._lock:
            scores = {m.key: self.health.score(m.key) for m in group}
        remaining = group[:]
        ordered: list[ModelInfo] = []
        while len(remaining) > 1:
            a, b = self._rng.sample(range(len(remaining)), 2)
            pick = a if scores[remaining[a].key] >= scores[remaining[b].key] else b
            ordered.append(remaining.pop(pick))
        return ordered + remaining


def _rank(model: ModelInfo, patterns: Sequence[str]) -> int:
    """Index of the first pattern matching the model's id or provider:id key, else len(patterns)."""
    for index, matcher in enumerate(_compiled_globs(patterns)):
        if matcher.match(model.key) is not None or matcher.match(model.id) is not None:
            return index
    return len(patterns)
