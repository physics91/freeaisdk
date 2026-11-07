"""One-request attempt bookkeeping for :class:`~freeai._core.Router`."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

from ._http import logger
from ._routing_types import (
    _MAX_BAD_REQUESTS_PER_PROVIDER,
    _NO_RESERVATION,
    Request,
    Reservation,
    Wait,
    _covered,
    _model_key,
    _provider_key,
    _total_tokens,
)
from .cooldown import cooldown_fail
from .errors import (
    SKIP_FOR_REQUEST,
    ExhaustedError,
    FailureKind,
    ProviderFailure,
    Scope,
)
from .limits import RateSnapshot, pending_drop, quota_adjust
from .types import Attempt, ModelInfo

if TYPE_CHECKING:
    from ._core import Router


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
        if failed.context_length is not None:
            return failed.context_length + 1
        return int(self.request.cost.tokens)

    def _emit(self, attempt: Attempt) -> None:
        self.attempts.append(attempt)
        if self.router.on_attempt is not None:
            self.router.on_attempt(attempt)
