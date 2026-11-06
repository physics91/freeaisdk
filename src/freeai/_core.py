"""Routing core without I/O: candidate order, reservations, and failure bookkeeping.

Message/response parsing lives in :mod:`freeai._messages`. The sync
:class:`~freeai.FreeAI` and async :class:`~freeai.AsyncFreeAI` clients are thin shells that
perform the network calls this module asks for.
"""

from __future__ import annotations

import math
import random
import secrets
import threading
import time
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, NamedTuple, TypedDict

from ._http import logger
from ._messages import (
    Messages,
    _estimate_tokens,
    _input_modalities,
    _normalize_messages,
    empty_stream,
    to_chunk,
    to_response,
)
from .errors import (
    SKIP_FOR_REQUEST,
    BillingWarning,
    ExhaustedError,
    FailureKind,
    ProviderFailure,
    Scope,
)
from .limits import (
    CooldownPolicy,
    Cost,
    HealthStats,
    Limit,
    RateSnapshot,
    RetryBudget,
    State,
    _compiled_globs,
    cooldown_fail,
    cooldown_failures,
    cooldown_ready_at,
    pending_add,
    pending_drop,
    quota_adjust,
    quota_charge,
    quota_observe,
    quota_ready_at,
)
from .providers import OpenRouter, Provider
from .state import MemoryStore, StateStore
from .types import Attempt, ChatResponse, ModelInfo, Requirements

Strategy = Literal["priority", "round_robin", "random", "health"]

# Re-export for existing ``from freeai._core import …`` call sites.
__all__ = [
    "DEFAULT_EXCLUDE",
    "Messages",
    "Plan",
    "Request",
    "Reservation",
    "Router",
    "RouterOptions",
    "Wait",
    "empty_stream",
    "to_chunk",
    "to_response",
]


class Reservation(NamedTuple):
    """Quota charges of one request the platform has not accepted yet, as recorded in the store."""

    id: str
    keys: tuple[str, ...]


_NO_RESERVATION = Reservation("", ())
# A reservation counts on top of header readings until the platform accepts the request. It must
# outlive the wait for response headers (bounded by the read timeout) or a live request's charge
# would be wiped; if a process dies, its reservations over-count for at most this long.
_PENDING_TTL = 300.0
ResponseCheck = Callable[[ChatResponse], bool | str | None]

# Model families that are listed by providers but cannot hold a conversation.
DEFAULT_EXCLUDE: tuple[str, ...] = ("*content-safety*", "*guard*", "*whisper*", "*tts*", "*embed*")
STRATEGIES: tuple[str, ...] = ("priority", "round_robin", "random", "health")

# ``model="fast"`` etc.: preference patterns (by model-id naming convention), not a hard filter.
# Unmatched models stay as a fallback, so an alias never makes a request unservable.
ALIASES: dict[str, tuple[str, ...]] = {
    "fast": ("*flash-lite*", "*lightning*", "*nano*", "*-mini*", "*-20b*", "*8b*", "*30b-a3b*", "*flash*"),
    "smart": (
        "*ultra*",
        "*kimi*",
        "*deepseek*",
        "*397b*",
        "*120b*",
        "*super*",
        "*70b*",
        "*command-a-plus*",
        "gemini-3.8*",
    ),
    "coding": ("*coder*", "*code*", "*kimi*", "*deepseek*", "*glm*", "*qwen3*", "*gpt-oss-120b*"),
}

# Request parameters that only some models support; their presence narrows the candidates.
_CAPABILITY_PARAMS = ("tools", "tool_choice", "response_format", "reasoning")
_RESERVED_PARAMS = {"messages", "stream", "models"}
# Only choices[0] is read, so extra choices would just burn quota (and Groq/Cerebras reject n != 1).
_STRIPPED_PARAMS = {"n"}
_MAX_BAD_REQUESTS_PER_PROVIDER = 2  # the same rejection twice means this provider dislikes the request
_SLOT_POLL = 0.25  # re-check interval while a provider's concurrency slots or a probe are busy


class RouterOptions(TypedDict, total=False):
    """Keyword options shared by :class:`~freeai.FreeAI` and :class:`~freeai.AsyncFreeAI`."""

    prefer: Sequence[str]
    exclude: Sequence[str]
    aliases: Mapping[str, Sequence[str]] | None
    strategy: Strategy
    max_attempts: int
    max_wait: float
    deadline: float | None
    first_token_timeout: float | None
    retry_on_empty: bool
    allow_training: bool
    reject_response: ResponseCheck | None
    cooldowns: CooldownPolicy | None
    retry_budget: RetryBudget | bool
    store: StateStore | None
    on_attempt: Callable[[Attempt], None] | None
    clock: Callable[[], float]
    rng: random.Random | None


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


class Router:
    """Shared state and policy for one client: which model to try, and what each outcome means.

    Cooldowns and quota counters live in the :class:`StateStore` (epoch seconds), so several
    processes can share them; health statistics, the retry budget, probes and concurrency slots
    are process-local.
    """

    def __init__(
        self,
        providers: Provider | Sequence[Provider] | None = None,
        *,
        prefer: Sequence[str] = (),
        exclude: Sequence[str] = DEFAULT_EXCLUDE,
        aliases: Mapping[str, Sequence[str]] | None = None,
        strategy: Strategy = "priority",
        max_attempts: int = 8,
        max_wait: float = 30.0,
        deadline: float | None = None,
        first_token_timeout: float | None = None,
        retry_on_empty: bool = True,
        allow_training: bool = True,
        reject_response: ResponseCheck | None = None,
        cooldowns: CooldownPolicy | None = None,
        retry_budget: RetryBudget | bool = True,
        store: StateStore | None = None,
        on_attempt: Callable[[Attempt], None] | None = None,
        clock: Callable[[], float] = time.time,
        rng: random.Random | None = None,
    ) -> None:
        if providers is None:
            providers = [OpenRouter()]
        elif isinstance(providers, Provider):
            providers = [providers]
        self.providers = list(providers)
        if not self.providers:
            raise ValueError("at least one provider is required")
        names = [p.name for p in self.providers]
        if len(set(names)) != len(names):
            raise ValueError(f"provider names must be unique: {names}")
        if strategy not in STRATEGIES:
            raise ValueError(f"unknown strategy: {strategy!r}")
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self.prefer = tuple(prefer)
        self.aliases = {name: tuple(patterns) for name, patterns in {**ALIASES, **(aliases or {})}.items()}
        self.exclude = tuple(exclude)
        self.strategy = strategy
        self.max_attempts = max_attempts
        self.max_wait = max_wait
        # Called when quota, a slot or a reservation may have freed up, so a client can wake one of
        # its waiting requests instead of letting it sleep out a delay computed before the change.
        self.on_change: Callable[[], None] | None = None
        self.deadline = deadline
        self.first_token_timeout = first_token_timeout
        self.retry_on_empty = retry_on_empty
        self.allow_training = allow_training
        self.reject_response = reject_response
        self.policy = cooldowns or CooldownPolicy()
        if isinstance(retry_budget, RetryBudget):
            self.retry_budget: RetryBudget | None = retry_budget
        else:
            self.retry_budget = RetryBudget() if retry_budget else None
        self.store: StateStore = store or MemoryStore()
        self.on_attempt = on_attempt
        self.clock = clock
        self.health = HealthStats()
        self._by_name = {p.name: p for p in self.providers}
        self._rng = rng or random.Random()
        self._lock = threading.Lock()
        self._turn = 0
        self._warmed: set[str] = set()
        self._probing: set[str] = set()
        self._inflight: dict[str, int] = {}
        for provider in self.providers:
            if provider.tier in ("trial", "paid"):
                warnings.warn(
                    f"provider {provider.name!r} is tier {provider.tier!r}: requests may bill or use trial credit",
                    BillingWarning,
                    stacklevel=4,
                )

    # -- request preparation ---------------------------------------------------------------

    def prepare(
        self,
        messages: Messages,
        system: str | None,
        model: str | Sequence[str] | None,
        require: Iterable[str],
        min_context: int | None,
        params: dict[str, Any],
    ) -> Request:
        reserved = _RESERVED_PARAMS.intersection(params)
        if reserved:
            raise TypeError(f"unsupported keyword(s) {sorted(reserved)}: use model=[...] or stream()")
        chat_messages = _normalize_messages(messages, system)
        params = {k: v for k, v in params.items() if k not in _STRIPPED_PARAMS}
        parameters = set(require) | {name for name in _CAPABILITY_PARAMS if params.get(name) is not None}
        requirements = Requirements(frozenset(parameters), _input_modalities(chat_messages), min_context)
        models: tuple[str, ...] = (model,) if isinstance(model, str) else tuple(model or ())
        prefer: tuple[str, ...] = ()
        if len(models) == 1 and models[0] in self.aliases:
            prefer = self.aliases[models[0]]
            models = ()
        elif len(models) == 1 and models[0] in ("auto", "freeai", "freeai/auto"):
            models = ()
        payload = {**params, "messages": chat_messages}
        return Request(payload, requirements, models, Cost(1.0, _estimate_tokens(payload)), prefer)

    def payload_for(self, model: ModelInfo, request: Request) -> dict[str, Any]:
        provider = self._by_name[model.provider]
        if not self.allow_training and provider.can_deny_training:
            return provider.deny_training(request.payload)
        return request.payload

    # -- candidate selection ---------------------------------------------------------------

    def next_turn(self) -> int:
        with self._lock:
            turn = self._turn
            self._turn += 1
            return turn

    def order(self, request: Request, pools: Mapping[str, Sequence[ModelInfo]], *, turn: int) -> list[ModelInfo]:
        """Candidates for ``request`` in the order they will be tried (ignoring cooldowns)."""
        pool = [m for provider in self.providers for m in pools.get(provider.name, ())]
        if request.models:  # explicit choice overrides the default exclusions
            pool = [m for m in pool if _rank(m, request.models) < len(request.models)]
        else:
            pool = [m for m in pool if _rank(m, self.exclude) == len(self.exclude)]
        pool = [m for m in pool if m.satisfies(request.requirements) and self._privacy_ok(m)]

        patterns = request.models or request.prefer or self.prefer
        groups: dict[tuple[bool, int], list[ModelInfo]] = {}
        for m in pool:
            groups.setdefault((m.is_router, _rank(m, patterns)), []).append(m)
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

    def plan(
        self, request: Request, ordered: list[ModelInfo], *, max_attempts: int | None, deadline: float | None
    ) -> Plan:
        if not ordered:
            privacy = "" if self.allow_training else "; allow_training=False"
            raise ExhaustedError(
                f"no free model matches the request (requirements: {request.requirements.describe()}{privacy})", []
            )
        budget = deadline if deadline is not None else self.deadline
        deadline_at = None if budget is None else self.clock() + budget
        return Plan(self, request, ordered, max_attempts or self.max_attempts, deadline_at)

    def claim_warmups(self, names: Iterable[str]) -> list[Provider]:
        """Providers whose one-time quota sync has not run yet (each is returned once)."""
        with self._lock:
            pending = sorted(set(names) - self._warmed)
            self._warmed.update(pending)
        return [self._by_name[name] for name in pending]

    def provider(self, model: ModelInfo) -> Provider:
        return self._by_name[model.provider]

    # -- shared state ----------------------------------------------------------------------

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
                self._observe_tx(tx, provider_name, model_id, snaps)

    def _observe_tx(
        self, tx: State, provider_name: str, model_id: str, snapshots: Iterable[RateSnapshot]
    ) -> None:
        """Apply ``snapshots`` into an open transaction. Caller must hold ``_lock``; no nested lock."""
        snapshots = list(snapshots)
        if not snapshots:
            return
        provider = self._by_name[provider_name]
        sid = provider.state_id
        now = self.clock()
        by_id = {_learned_identity(entry): dict(entry) for entry in tx.get(_learned_key(sid), [])}
        for snapshot in snapshots:
            quota_observe(tx, sid, model_id, snapshot, now, provider.reset_tz)
            entry = {
                "per": snapshot.per,
                "unit": snapshot.unit,
                "scope": snapshot.scope,
                "model": model_id if snapshot.scope == "model" else None,
                "amount": snapshot.limit,
            }
            by_id[_learned_identity(entry)] = entry
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
                left = float(entry["until"]) - now
                if left <= 0:
                    continue
                sid, sep, model_id = key.removeprefix("cd:m:").partition("|")
                if not sep:
                    continue
                name = by_sid.get(sid)
                if name is not None:
                    models[f"{name}:{model_id}"] = left
        return {"models": models, "providers": providers}

    def health_snapshot(self) -> dict[str, dict[str, float]]:
        with self._lock:
            return self.health.snapshot()

    def cost(self, provider: Provider, request: Request) -> Cost:
        """What a request is charged on this provider's windows until the platform reports usage."""
        cost = request.cost
        if cost.tokens <= 0 or not provider.prompt_overhead_tokens:
            return cost
        return Cost(cost.requests, cost.tokens + provider.prompt_overhead_tokens)

    def reserve(self, candidates: list[ModelInfo], request: Request) -> tuple[ModelInfo | None, float, Reservation]:
        """Pick the first candidate that is ready now and charge its quota windows.

        Returns the model (or None), when it was or will be ready, and the reservation to hand to
        :meth:`Plan.admitted` or :meth:`release`.
        """
        with self._lock, self.store.transaction() as tx:
            now = self.clock()
            earliest = math.inf
            for model in candidates:
                provider = self._by_name[model.provider]
                sid = provider.state_id
                model_key, provider_key = _model_key(sid, model.id), _provider_key(sid)
                ready = max(cooldown_ready_at(tx, model_key), cooldown_ready_at(tx, provider_key))
                limits = self._limits(tx, provider, model.id)
                cost = self.cost(provider, request)
                if ready <= now:
                    ready = quota_ready_at(tx, limits, sid, model.id, cost, now, provider.reset_tz)
                probes = [k for k in (model_key, provider_key) if cooldown_failures(tx, k) > 0]
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

    def _limits(self, tx: State, provider: Provider, model_id: str) -> list[Limit]:
        """Declared limits plus those learned from response headers or quota endpoints."""
        sid = provider.state_id
        limits = [limit for limit in provider.limits if limit.applies_to(model_id)]
        keys = {limit.key(sid, model_id) for limit in limits}
        for entry in tx.get(_learned_key(sid), []):
            if entry["model"] not in (None, model_id):
                continue
            learned = Limit(float(entry["amount"]), entry["per"], entry["unit"], entry["scope"], source="observed")
            key = learned.key(sid, model_id)
            if key not in keys:
                limits.append(learned)
                keys.add(key)
        return limits

    def changed(self) -> None:
        if self.on_change is not None:
            self.on_change()

    def _release_slot(self, model: ModelInfo) -> None:
        """Free concurrency/probe bookkeeping only. Caller owns any ``pending_drop``."""
        provider = self._by_name[model.provider]
        sid = provider.state_id
        with self._lock:
            count = self._inflight.get(provider.name, 0)
            if count > 0:
                self._inflight[provider.name] = count - 1
            self._probing.difference_update({_model_key(sid, model.id), _provider_key(sid)})

    def _drop_pending(self, reservation: Reservation) -> None:
        if reservation.keys:
            with self._lock, self.store.transaction() as tx:
                pending_drop(tx, reservation.id, reservation.keys)

    def release(self, model: ModelInfo, reservation: Reservation = _NO_RESERVATION) -> None:
        """Free the concurrency slot, probe flags and any reservation :meth:`Plan.admitted` did not take."""
        self._release_slot(model)
        self._drop_pending(reservation)

    # -- response parsing ------------------------------------------------------------------

    def to_response(self, model: ModelInfo, data: dict[str, Any]) -> ChatResponse:
        response = to_response(model, data, retry_on_empty=self.retry_on_empty)
        if self.reject_response is not None:
            verdict = self.reject_response(response)
            if verdict:
                reason = verdict if isinstance(verdict, str) else "rejected by reject_response"
                raise ProviderFailure(FailureKind.REJECTED, f"{model.provider}: {model.id}: {reason}")
        return response


class Plan:
    """Attempt bookkeeping for one request: which candidate next, when to wait, when to give up."""

    def __init__(
        self, router: Router, request: Request, ordered: list[ModelInfo], limit: int, deadline_at: float | None
    ) -> None:
        self.router = router
        self.request = request
        self.ordered = ordered
        self.limit = limit
        self.deadline_at = deadline_at
        self.attempts: list[Attempt] = []
        self._skipped: set[str] = set()
        self._rejected_by: dict[str, int] = {}
        self._min_context = 0
        self._waited = 0.0  # time actually spent waiting; a wake-up can cut a wait short
        self._waiting_since: float | None = None
        self._budget_spent = False
        self._active: ModelInfo | None = None
        self._reservation = _NO_RESERVATION
        self._admitted: list[RateSnapshot] = []  # headers of the active attempt, applied on arrival

    @property
    def providers(self) -> set[str]:
        return {m.provider for m in self.ordered}

    def next(self) -> ModelInfo | Wait:
        """The next model to call (already reserved), or how long to wait. Raises when exhausted."""
        if len(self.attempts) >= self.limit:
            raise ExhaustedError(f"gave up after {self.limit} attempts", self.attempts)
        router = self.router
        now = router.clock()
        if self._waiting_since is not None:
            self._waited += max(0.0, now - self._waiting_since)
            self._waiting_since = None
        if self.deadline_at is not None and now >= self.deadline_at:
            raise ExhaustedError("deadline exceeded", self.attempts)
        if self._budget_spent:
            raise ExhaustedError("upstream failures are widespread; the retry budget is spent", self.attempts)
        remaining = [
            m
            for m in self.ordered
            if m.key not in self._skipped
            and self._rejected_by.get(m.provider, 0) < _MAX_BAD_REQUESTS_PER_PROVIDER
            and (m.context_length is None or m.context_length >= self._min_context)
        ]
        if not remaining:
            raise ExhaustedError("every candidate model failed", self.attempts)
        model, ready_at, reservation = router.reserve(remaining, self.request)
        if model is not None:
            self._active, self._reservation, self._admitted = model, reservation, []
            return model
        delay = max(0.0, ready_at - now)
        if math.isinf(delay):
            raise ExhaustedError("every candidate provider is disabled; check the API keys", self.attempts)
        allowance = self.router.max_wait - self._waited
        if self.deadline_at is not None:
            allowance = min(allowance, self.deadline_at - now)
        if delay > allowance:
            raise ExhaustedError(
                "all candidate models are cooling down or rate limited", self.attempts, next_available_in=delay
            )
        logger.info("all candidates busy; waiting %.2fs", delay)
        self._waiting_since = now
        return Wait(delay, frozenset(m.key for m in remaining))

    def time_left(self) -> float | None:
        """Seconds until the deadline, used as the per-attempt HTTP timeout."""
        if self.deadline_at is None:
            return None
        return max(0.1, self.deadline_at - self.router.clock())

    def admitted(self, snapshots: Iterable[RateSnapshot]) -> None:
        """The platform accepted the active attempt: its stream headers arrived.

        Its counters now include this request, so the reservation stops being added to header
        readings, and the headers are applied now. By the time the stream ends they would no longer
        count requests the platform accepted meanwhile.
        """
        model = self._active
        if model is None:
            return
        reservation, self._reservation = self._reservation, _NO_RESERVATION
        snapshots = list(snapshots)
        self._admitted.extend(snapshots)
        router = self.router
        with router._lock, router.store.transaction() as tx:
            # Drop pending before observe so header readings are not double-counted with the reserve.
            if reservation.keys:
                pending_drop(tx, reservation.id, reservation.keys)
            router._observe_tx(tx, model.provider, model.id, snapshots)
        router.changed()

    def release(self, model: ModelInfo) -> None:
        """The attempt ended without an outcome (e.g. the caller stopped reading or was cancelled)."""
        detached, reservation = self._detach(model)
        if not detached:
            return
        self.router._drop_pending(reservation)
        self.router.changed()

    def _detach(self, model: ModelInfo) -> tuple[bool, Reservation]:
        """Clear the active attempt and concurrency slot; return any reservation still to drop."""
        if self._active is None or self._active.key != model.key:
            return False, _NO_RESERVATION
        reservation, self._active, self._reservation = self._reservation, None, _NO_RESERVATION
        self.router._release_slot(model)
        return True, reservation

    def failed(
        self,
        model: ModelInfo,
        failure: ProviderFailure,
        started: float,
        snapshots: Iterable[RateSnapshot] = (),
        *,
        usage: Mapping[str, Any] | None = None,
    ) -> bool:
        """Record a failed attempt. Returns True when the caller should re-sync the provider's quota.

        ``usage`` is the response's token usage when the server answered (empty, rejected or
        truncated output); those tokens were spent, so they are settled instead of refunded.
        """
        _, reservation = self._detach(model)
        router = self.router
        provider = router.provider(model)
        sid = provider.state_id
        covered = _covered(failure.snapshots, snapshots, self._admitted)
        committed = False
        try:
            with router._lock, router.store.transaction() as tx:
                if reservation.keys:
                    pending_drop(tx, reservation.id, reservation.keys)
                now = router.clock()
                if failure.scope is Scope.PROVIDER:
                    cooldown = cooldown_fail(
                        tx, _provider_key(sid), failure.kind, failure.retry_after, now, router.policy
                    )
                elif failure.scope is Scope.MODEL:
                    cooldown = cooldown_fail(
                        tx, _model_key(sid, model.id), failure.kind, failure.retry_after, now, router.policy
                    )
                else:
                    cooldown = 0.0
                # Refused requests consume no tokens: give the estimate back (requests stay charged).
                used = _total_tokens(usage)
                delta = (used if used is not None else 0.0) - router.cost(provider, self.request).tokens
                limits = [lim for lim in router._limits(tx, provider, model.id) if (lim.per, lim.unit) not in covered]
                quota_adjust(tx, limits, sid, model.id, "tokens", delta, now, provider.reset_tz)
                router._observe_tx(tx, provider.name, model.id, [*failure.snapshots, *snapshots])
                if failure.scope is not Scope.REQUEST:
                    router.health.record(model.key, False, now - started)
                if failure.kind is FailureKind.UNAVAILABLE and router.retry_budget is not None:
                    router.retry_budget.on_failure()
                    self._budget_spent = not router.retry_budget.allows_retry()
            committed = True
        except BaseException:
            if reservation.keys and not committed:
                router._drop_pending(reservation)
            raise
        router.changed()
        if failure.kind is FailureKind.MODEL_GONE:
            provider.invalidate()
        if failure.kind in SKIP_FOR_REQUEST:
            self._skipped.add(model.key)
        if failure.kind is FailureKind.BAD_REQUEST:
            self._rejected_by[model.provider] = self._rejected_by.get(model.provider, 0) + 1
        if failure.kind is FailureKind.CONTEXT_EXCEEDED:
            self._min_context = max(self._min_context, self._needed_context(model))
        logger.info("%s failed (%s): %s; cooldown %.0fs", model.key, failure.kind, failure.message, cooldown)
        self._emit(
            Attempt(
                provider=model.provider,
                model=model.id,
                ok=False,
                latency=now - started,
                kind=failure.kind,
                status=failure.status,
                message=failure.message,
                cooldown=cooldown,
                scope=failure.scope,
            )
        )
        return failure.kind is FailureKind.QUOTA_EXHAUSTED and failure.scope is Scope.PROVIDER

    def succeeded(
        self,
        model: ModelInfo,
        started: float,
        *,
        snapshots: Iterable[RateSnapshot] = (),
        usage: Mapping[str, Any] | None = None,
        served_model: str | None = None,
    ) -> None:
        _, reservation = self._detach(model)
        router = self.router
        provider = router.provider(model)
        sid = provider.state_id
        snapshots = list(snapshots)
        covered = _covered(snapshots, self._admitted)
        committed = False
        try:
            with router._lock, router.store.transaction() as tx:
                if reservation.keys:
                    pending_drop(tx, reservation.id, reservation.keys)
                now = router.clock()
                for key in (_model_key(sid, model.id), _provider_key(sid)):
                    if key in tx and float(tx[key]["until"]) <= now:
                        del tx[key]  # recovered: reset the backoff (never cut a cooldown set meanwhile)
                used = _total_tokens(usage)
                if used is not None:
                    limits = [
                        lim for lim in router._limits(tx, provider, model.id) if (lim.per, lim.unit) not in covered
                    ]
                    delta = used - router.cost(provider, self.request).tokens
                    quota_adjust(tx, limits, sid, model.id, "tokens", delta, now, provider.reset_tz)
                router._observe_tx(tx, provider.name, model.id, snapshots)
                latency = now - started
                router.health.record(model.key, True, latency)
                if model.is_router and served_model and served_model != model.id:
                    router.health.record(f"{model.provider}:{served_model}", True, latency)  # credit the real model
                if router.retry_budget is not None:
                    router.retry_budget.on_success()
            committed = True
        except BaseException:
            if reservation.keys and not committed:
                router._drop_pending(reservation)
            raise
        router.changed()
        self._emit(
            Attempt(
                provider=model.provider,
                model=model.id,
                ok=True,
                latency=now - started,
                served_model=served_model,
                usage=dict(usage) if isinstance(usage, Mapping) else None,
            )
        )

    def _needed_context(self, failed: ModelInfo) -> int:
        """After a context-length rejection, require a window larger than the one that failed."""
        if failed.context_length:
            return failed.context_length + 1
        return int(self.request.cost.tokens)

    def _emit(self, attempt: Attempt) -> None:
        self.attempts.append(attempt)
        if self.router.on_attempt is not None:
            self.router.on_attempt(attempt)


# -- helpers -----------------------------------------------------------------------------------


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


def _total_tokens(usage: Mapping[str, Any] | None) -> float | None:
    if not isinstance(usage, Mapping):
        return None
    total = usage.get("total_tokens")
    if isinstance(total, (int, float)) and not isinstance(total, bool):
        return float(total)
    parts = [usage.get("prompt_tokens"), usage.get("completion_tokens")]
    if all(isinstance(p, (int, float)) and not isinstance(p, bool) for p in parts):
        return float(sum(parts))  # type: ignore[arg-type]
    return None


def _rank(model: ModelInfo, patterns: Sequence[str]) -> int:
    """Index of the first pattern matching the model's id or provider:id key, else len(patterns)."""
    for index, matcher in enumerate(_compiled_globs(patterns)):
        if matcher.match(model.key) is not None or matcher.match(model.id) is not None:
            return index
    return len(patterns)
