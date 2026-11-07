"""Routing core without I/O: candidate order, reservations, and failure bookkeeping.

Message/response parsing lives in :mod:`freeai._messages`. Attempt bookkeeping lives in
:mod:`freeai._plan`. The sync :class:`~freeai.FreeAI` and async :class:`~freeai.AsyncFreeAI`
clients are thin shells that perform the network calls this module asks for.
"""

from __future__ import annotations

import random
import threading
import time
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Literal, TypedDict

from ._admission import Admission
from ._messages import Messages, empty_stream, to_chunk, to_response
from ._observe import Observe
from ._order import OrderPolicy
from ._order import _rank as _rank  # re-export for ``from freeai._core import _rank``
from ._plan import Plan
from ._prepare import ALIASES, PreparePolicy
from ._routing_types import _NO_RESERVATION, Request, Reservation, Wait
from ._routing_types import _learned_key as _learned_key  # tests import from _core
from .cooldown import CooldownPolicy
from .errors import (
    BillingWarning,
    ExhaustedError,
    FailureKind,
    ProviderFailure,
)
from .health import HealthStats
from .limits import Cost, Limit, RateSnapshot, State
from .providers import Provider
from .retry import RetryBudget
from .state import MemoryStore, StateStore
from .types import Attempt, ChatResponse, ModelInfo

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

ResponseCheck = Callable[[ChatResponse], bool | str | None]

# Model families that are listed by providers but cannot hold a conversation.
DEFAULT_EXCLUDE: tuple[str, ...] = ("*content-safety*", "*guard*", "*whisper*", "*tts*", "*embed*")
STRATEGIES: tuple[str, ...] = ("priority", "round_robin", "random", "health")


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
        self._prepare = PreparePolicy(
            aliases=self.aliases,
            allow_training=self.allow_training,
            by_name=self._by_name,
        )
        self._turn = 0
        self._warmed: set[str] = set()
        self._probing: set[str] = set()
        self._inflight: dict[str, int] = {}
        self._admission = Admission(
            lock=self._lock,
            store=self.store,
            by_name=self._by_name,
            inflight=self._inflight,
            probing=self._probing,
            clock=self.clock,
            cost=self.cost,
        )
        self._observe = Observe(
            lock=self._lock,
            store=self.store,
            by_name=self._by_name,
            providers=self.providers,
            clock=self.clock,
        )
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
        return self._prepare.prepare(messages, system, model, require, min_context, params)

    def payload_for(self, model: ModelInfo, request: Request) -> dict[str, Any]:
        return self._prepare.payload_for(model, request)

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
        self._observe.observe(provider_name, model_id, snapshots)

    def observe_many(self, items: Iterable[tuple[str, str, Iterable[RateSnapshot]]]) -> None:
        """Apply several snapshot batches in one store transaction (empty batches are skipped)."""
        self._observe.observe_many(items)

    def _observe_tx(
        self, tx: State, provider_name: str, model_id: str, snapshots: Iterable[RateSnapshot]
    ) -> None:
        """Apply ``snapshots`` into an open transaction. Caller must hold ``_lock``; no nested lock."""
        self._observe.apply_tx(tx, provider_name, model_id, snapshots)

    def status(self) -> dict[str, dict[str, float]]:
        """Remaining cooldown seconds per model key and per provider."""
        return self._observe.status()

    def health_snapshot(self) -> dict[str, dict[str, float]]:
        with self._lock:
            return self.health.snapshot()

    def cost(self, provider: Provider, request: Request) -> Cost:
        """What a request is charged on this provider's windows until the platform reports usage."""
        cost = request.cost
        overhead = provider.prompt_overhead_tokens
        if not overhead:
            return cost
        # Apply even when the estimate is 0 (e.g. ``max_tokens=0``); Groq still burns template tokens.
        return Cost(cost.requests, cost.tokens + overhead)

    def reserve(self, candidates: list[ModelInfo], request: Request) -> tuple[ModelInfo | None, float, Reservation]:
        """Pick the first candidate that is ready now and charge its quota windows.

        Returns the model (or None), when it was or will be ready, and the reservation to hand to
        :meth:`Plan.admitted` or :meth:`release`.
        """
        return self._admission.reserve(candidates, request)

    def _limits(
        self,
        tx: State,
        provider: Provider,
        model_id: str,
        *,
        learned: Sequence[Mapping[str, Any]] | None = None,
    ) -> list[Limit]:
        """Declared limits plus those learned from response headers or quota endpoints."""
        return self._admission.limits(tx, provider, model_id, learned=learned)

    def changed(self) -> None:
        if self.on_change is not None:
            self.on_change()

    def _release_slot(self, model: ModelInfo) -> None:
        """Free concurrency/probe bookkeeping only. Caller owns any ``pending_drop``."""
        self._admission.release_slot(model)

    def _drop_pending(self, reservation: Reservation) -> None:
        self._admission.drop_pending(reservation)

    def release(self, model: ModelInfo, reservation: Reservation = _NO_RESERVATION) -> None:
        """Free the concurrency slot, probe flags and any reservation :meth:`Plan.admitted` did not take."""
        self._admission.release(model, reservation)

    # -- response parsing ------------------------------------------------------------------

    def to_response(self, model: ModelInfo, data: dict[str, Any]) -> ChatResponse:
        response = to_response(model, data, retry_on_empty=self.retry_on_empty)
        if self.reject_response is not None:
            verdict = self.reject_response(response)
            if verdict:
                reason = verdict if isinstance(verdict, str) else "rejected by reject_response"
                raise ProviderFailure(FailureKind.REJECTED, f"{model.provider}: {model.id}: {reason}")
        return response



