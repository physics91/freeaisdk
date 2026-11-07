"""Shared routing types and store-key helpers used by Router and Plan."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, NamedTuple

from .limits import Cost, RateSnapshot, State
from .types import Requirements


class Reservation(NamedTuple):
    """Quota charges of one request the platform has not accepted yet, as recorded in the store."""

    id: str
    keys: tuple[str, ...]


_NO_RESERVATION = Reservation("", ())
_MAX_BAD_REQUESTS_PER_PROVIDER = 2  # the same rejection twice means this provider dislikes the request


@dataclass(frozen=True, slots=True)
class Request:
    payload: dict[str, Any]
    requirements: Requirements
    models: tuple[str, ...]
    cost: Cost
    prefer: tuple[str, ...] = ()  # alias patterns: ordering only


@dataclass(frozen=True, slots=True)
class Wait:
    """Every candidate is busy; sleep this long, then ask the plan again.

    ``group`` names the models the request can still use. Requests waiting with the same group are
    interchangeable, so one of them is enough to try when quota frees up.
    """

    delay: float
    group: frozenset[str] = field(default=frozenset(), compare=False)


def _cooldown_bundle(tx: State, sid: str) -> tuple[Any, dict[str, Any], bool]:
    """Provider cooldown entry plus optional authoritative model-id map for one ``state_id``.

    When ``tx`` supports ``items_with_prefix``, the model map is complete for this transaction
    (missing id means ready). Otherwise the map is empty and callers must use point lookups.
    """
    provider_entry = tx.get(_provider_key(sid))
    items_with_prefix = getattr(tx, "items_with_prefix", None)
    if not callable(items_with_prefix):
        return provider_entry, {}, False
    prefix = f"cd:m:{sid}|"
    models = {key.removeprefix(prefix): entry for key, entry in items_with_prefix(prefix)}
    return provider_entry, models, True


def _model_key(sid: str, model_id: str) -> str:
    """Cooldown key for one model. ``sid`` must not contain ``|`` (names use ``#`` for key digests)."""
    return f"cd:m:{sid}|{model_id}"


def _provider_key(sid: str) -> str:
    return f"cd:p:{sid}"


def _learned_key(sid: str) -> str:
    return f"ql:{sid}"


def _learned_identity(entry: Mapping[str, Any]) -> tuple[Any, Any, Any, Any]:
    """Identity of a learned limit row; ``model`` stays ``None`` for provider scope."""
    return entry["per"], entry["unit"], entry["scope"], entry["model"]


def _covered(*groups: Iterable[RateSnapshot]) -> set[tuple[str, str]]:
    """Windows whose counters the platform just reported, so local estimates must not touch them."""
    return {(s.per, s.unit) for group in groups for s in group}


def _usage_number(value: Any) -> float | None:
    """Token counts from usage objects; JSON may send floats or numeric strings."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _total_tokens(usage: Mapping[str, Any] | None) -> float | None:
    if not isinstance(usage, Mapping):
        return None
    if (total := _usage_number(usage.get("total_tokens"))) is not None:
        return total
    parts = [
        number
        for key in ("prompt_tokens", "completion_tokens")
        if (number := _usage_number(usage.get(key))) is not None
    ]
    return float(sum(parts)) if parts else None
