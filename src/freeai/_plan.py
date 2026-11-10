"""One-request attempt bookkeeping for :class:`~freeai._core.Router`."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

from . import _adapt
from ._http import logger
from ._routing_types import (
    _MAX_BAD_REQUESTS_PER_PROVIDER,
    _NO_RESERVATION,
    Request,
    Reservation,
    Wait,
    _covered,
    _model_key,
    _output_tokens,
    _provider_key,
    _total_tokens,
)
from .cooldown import cooldown_fail, cooldown_ready_at
from .errors import (
    SKIP_FOR_REQUEST,
    ExhaustedError,
    FailureKind,
    ProviderFailure,
    Scope,
)
from .limits import Cost, RateSnapshot, State, pending_drop, quota_adjust, quota_started, quota_used
from .providers import Provider
from .types import Attempt, ModelInfo

if TYPE_CHECKING:
    from ._core import Router


def _settlement(spent: float | None, estimate: float, accepted: bool) -> float:
    """What to add to a window's estimate when an attempt ends.

    Settle to what was spent when known. After the platform accepted the attempt (stream headers),
    keep the estimate if usage is missing so the window is not undercounted. Otherwise refund it
    (refused / never accepted / transport).
    """
    if spent is not None:
        return spent - estimate
    return 0.0 if accepted else -estimate


class Plan:
    """Attempt bookkeeping for one request: which candidate next, when to wait, when to give up."""

    def __init__(
        self,
        router: Router,
        request: Request,
        ordered: list[ModelInfo],
        limit: int,
        deadline_at: float | None,
        *,
        tiers: Mapping[str, int] | None = None,
        max_wait: float | None = None,
    ) -> None:
        self.router = router
        self.request = request
        self.ordered = ordered
        self.tiers = tiers  # static groups, when the strategy chooses inside a group by quota
        # Seconds this request may spend waiting for quota or cooldowns (inf: as long as it takes).
        self.max_wait = router.max_wait if max_wait is None else max(0.0, max_wait)
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
        self._accepted = False  # True after :meth:`admitted` (platform accepted this attempt)
        self._probed: tuple[tuple[str, float], ...] = ()  # estimated windows the active attempt goes beyond
        self._sent_at: float | None = None  # when the active attempt was reserved

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
        model, ready_at, reservation = router.reserve(remaining, self.request, self.tiers)
        if model is not None:
            self._active, self._reservation, self._admitted, self._accepted = model, reservation, [], False
            self._probed, self._sent_at = reservation.probed, ready_at
            return model
        delay = max(0.0, ready_at - now)
        if math.isinf(delay):
            raise ExhaustedError("every candidate provider is disabled; check the API keys", self.attempts)
        allowance = self.max_wait - self._waited
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
        return max(0.0, self.deadline_at - self.router.clock())

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
        self._accepted = True
        router = self.router
        with router._lock, router.store.transaction() as tx:
            # Drop pending before observe so header readings are not double-counted with the reserve.
            if reservation.keys:
                pending_drop(tx, reservation.id, reservation.keys)
            router._observe_tx(tx, model.provider, model.id, snapshots, self._sent_at)
        router.changed()

    def release(self, model: ModelInfo) -> None:
        """The attempt ended without an outcome (e.g. the caller stopped reading or was cancelled).

        If the platform never accepted the attempt (pending reservation still held), undo the local
        quota charge. After :meth:`admitted`, the platform counted the request so charges stay.
        """
        detached, reservation, probed = self._detach(model)
        router = self.router
        router._release_probes(probed)
        if not detached:
            return
        if reservation.keys:
            provider = router.provider(model)
            sid = provider.state_id
            cost = router.cost(provider, self.request)
            with router._lock, router.store.transaction() as tx:
                pending_drop(tx, reservation.id, reservation.keys)
                now = router.clock()
                limits = router._limits(tx, provider, model.id)
                if cost.tokens:
                    quota_adjust(tx, limits, sid, model.id, "tokens", -cost.tokens, now, provider.reset_tz)
                if cost.output_tokens:
                    refund = -cost.output_tokens
                    quota_adjust(tx, limits, sid, model.id, "output_tokens", refund, now, provider.reset_tz)
                if cost.requests:
                    quota_adjust(tx, limits, sid, model.id, "requests", -cost.requests, now, provider.reset_tz)
        router.changed()

    def _detach(self, model: ModelInfo) -> tuple[bool, Reservation, tuple[tuple[str, float], ...]]:
        """Clear the active attempt and concurrency slot.

        Returns any reservation still to drop and the estimated windows the attempt probed. Those
        stay marked as being probed until the caller has recorded the outcome and calls
        ``router._release_probes``, so no second probe starts in between.
        """
        if self._active is None or self._active.key != model.key:
            return False, _NO_RESERVATION, ()
        reservation, self._active, self._reservation = self._reservation, None, _NO_RESERVATION
        probed, self._probed, self._sent_at = self._probed, (), None
        self.router._release_slot(model)
        return True, reservation, probed

    def _refused(
        self,
        tx: State,
        provider: Provider,
        model: ModelInfo,
        failure: ProviderFailure,
        probed: tuple[tuple[str, float], ...],
        covered: set[tuple[str, str]],
        cost: Cost,
        now: float,
    ) -> None:
        """What a failed request says about daily and monthly limits declared as estimates.

        Any failure leaves a request charged that the platform may not have counted, so the window's
        count is not trusted for a probe until its next period has begun. Beyond that: if the
        request was sent beyond an estimate and the platform refused it for lack of allowance, however
        it words that, the probe was refused and the next one waits (a probe that failed another way,
        such as a 500, was not). Otherwise, if the platform reports its allowance exhausted, the
        fullest of these windows is smaller than assumed and is cut to what it held, provided that
        count is one of accepted requests only (nothing that failed earlier is still in it).

        Call before the failed request's own charges are settled.
        """
        sid, tz = provider.state_id, provider.reset_tz
        beyond = {key for key, _ in probed}
        soft = [
            (limit, limit.key(sid, model.id))
            for limit in self.router._limits(tx, provider, model.id)
            if _adapt.adapts(limit) and (limit.per, limit.unit) not in covered
        ]
        exact = {
            key
            for limit, key in soft
            if _adapt.unsure_since(tx, key) < quota_started(tx, limit, sid, model.id, now, tz)
        }
        refused = failure.kind in _adapt.REFUSALS
        for limit, key in soft:
            if key in beyond and refused:
                _adapt.refuse_probe(tx, limit, key, now)
            else:
                _adapt.distrust(tx, limit, key, now)
        if beyond or failure.kind is not FailureKind.QUOTA_EXHAUSTED:
            return  # explained by the probe, or not about a day's or a month's allowance
        fullest: tuple[float, Any, str, float] | None = None
        for limit, key in soft:
            accepted = max(0.0, quota_used(tx, limit, sid, model.id, now, tz) - cost.requests)
            if key in exact and accepted < limit.amount and (fullest is None or accepted / limit.amount > fullest[0]):
                fullest = (accepted / limit.amount, limit, key, accepted)
        if fullest is not None:
            _adapt.cut(tx, fullest[1], fullest[2], now, fullest[3])

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
        sent_at = self._sent_at
        _, reservation, probed = self._detach(model)
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
                # Requests: stay charged once the platform may have counted them; undo only when the
                # attempt never reached the platform (transport failure before any HTTP response).
                cost = router.cost(provider, self.request)
                if router.adapt_limits and failure.reached and cost.requests:
                    self._refused(tx, provider, model, failure, probed, covered, cost, now)
                delta = _settlement(_total_tokens(usage), cost.tokens, self._accepted)
                output_delta = _settlement(_output_tokens(usage), cost.output_tokens, self._accepted)
                limits = [lim for lim in router._limits(tx, provider, model.id) if (lim.per, lim.unit) not in covered]
                quota_adjust(tx, limits, sid, model.id, "tokens", delta, now, provider.reset_tz)
                quota_adjust(tx, limits, sid, model.id, "output_tokens", output_delta, now, provider.reset_tz)
                if not failure.reached and reservation.keys and cost.requests:
                    quota_adjust(tx, limits, sid, model.id, "requests", -cost.requests, now, provider.reset_tz)
                router._observe_tx(tx, provider.name, model.id, [*failure.snapshots, *snapshots], sent_at)
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
        finally:
            router._release_probes(probed)
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
        sent_at = self._sent_at
        _, reservation, probed = self._detach(model)
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
                    if key in tx and cooldown_ready_at(tx, key) <= now:
                        del tx[key]  # recovered: reset the backoff (never cut a cooldown set meanwhile)
                cost = router.cost(provider, self.request)
                delta = _settlement(_total_tokens(usage), cost.tokens, True)
                output_delta = _settlement(_output_tokens(usage), cost.output_tokens, True)
                limits = [lim for lim in router._limits(tx, provider, model.id) if (lim.per, lim.unit) not in covered]
                sizes = dict(probed)
                for limit in limits:  # a request sent beyond an estimated window was accepted
                    key = limit.key(sid, model.id)
                    if limit.estimate and key in sizes:
                        _adapt.grow(tx, limit, key, sizes[key])
                quota_adjust(tx, limits, sid, model.id, "tokens", delta, now, provider.reset_tz)
                quota_adjust(tx, limits, sid, model.id, "output_tokens", output_delta, now, provider.reset_tz)
                router._observe_tx(tx, provider.name, model.id, snapshots, sent_at)
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
        finally:
            router._release_probes(probed)
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
