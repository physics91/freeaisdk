"""Routing core without I/O: candidate order, reservations, and failure bookkeeping.

Message/response parsing lives in :mod:`freeai._messages`. Attempt bookkeeping lives in
:mod:`freeai._plan`. The sync :class:`~freeai.FreeAI` and async :class:`~freeai.AsyncFreeAI`
clients are thin shells that perform the network calls this module asks for.
"""

from __future__ import annotations

import math
import random
import secrets
import threading
import time
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Literal, TypedDict

from ._messages import (
    Messages,
    _estimate_tokens,
    _input_modalities,
    _normalize_messages,
    empty_stream,
    to_chunk,
    to_response,
)
from ._order import OrderPolicy
from ._order import _rank as _rank  # re-export for ``from freeai._core import _rank``
from ._plan import Plan
from ._routing_types import (
    _NO_RESERVATION,
    Request,
    Reservation,
    Wait,
    _cooldown_bundle,
    _learned_identity,
    _learned_key,
    _model_key,
    _provider_key,
)
from .cooldown import CooldownPolicy, cooldown_failures, cooldown_ready_at
from .errors import (
    BillingWarning,
    ExhaustedError,
    FailureKind,
    ProviderFailure,
)
from .health import HealthStats
from .limits import (
    Cost,
    Limit,
    RateSnapshot,
    State,
    pending_add,
    pending_drop,
    quota_charge,
    quota_observe,
    quota_ready_at,
)
from .providers import Provider
from .retry import RetryBudget
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


class Router:
    """Shared state and policy for one client: which model to try, and what each outcome means.

    Cooldowns and quota counters live in the :class:`StateStore` (epoch seconds), so several
    processes can share them; health statistics, the retry budget, probes and concurrency slots
    are process-local.
    """

    def __init__(
        self,
        providers: Provider | Sequence[Provider],
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
        if isinstance(providers, Provider):
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
        self._order = OrderPolicy(
            providers=self.providers,
            prefer=self.prefer,
            exclude=self.exclude,
            strategy=self.strategy,
            allow_training=self.allow_training,
            by_name=self._by_name,
            health=self.health,
            rng=self._rng,
            lock=self._lock,
        )
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
        return self._order.order(request, pools, turn=turn)

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
                limits = self._limits(tx, provider, model.id, learned=learned_by_sid[sid])
                cost = self.cost(provider, request)
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

    def _limits(
        self,
        tx: State,
        provider: Provider,
        model_id: str,
        *,
        learned: Sequence[Mapping[str, Any]] | None = None,
    ) -> list[Limit]:
        """Declared limits plus those learned from response headers or quota endpoints.

        ``learned`` may be preloaded for the provider ``state_id`` within one transaction so
        :meth:`reserve` does not re-read the registry for every candidate.
        """
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



