"""In-process health scores for ``strategy="health"`` routing."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class _Ewma:
    success: float = 1.0
    latency: float | None = None  # set only after a successful sample
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
            lat = max(latency, 0.0)
            if stat.latency is None:
                stat.latency = lat
            else:
                a = self.alpha
                stat.latency = (1 - a) * stat.latency + a * lat
        stat.samples += 1

    def score(self, key: str) -> float:
        stat = self._stats.get(key)
        if stat is None:
            return 1.0
        return stat.success / (1.0 + (stat.latency or 0.0))

    def snapshot(self) -> dict[str, dict[str, float]]:
        return {
            k: {"success": v.success, "latency": v.latency or 0.0, "samples": float(v.samples)}
            for k, v in self._stats.items()
        }
