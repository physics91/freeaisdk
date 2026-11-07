"""FreeAI: one ``chat()`` call that fails over across free models and providers."""

from __future__ import annotations

import math
import os
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from typing import Any, Self, Unpack

from ._core import (
    DEFAULT_EXCLUDE,
    Messages,
    Plan,
    Request,
    Router,
    RouterOptions,
    Wait,
    empty_stream,
    output_limit_failure,
    stream_chunk_answered,
    to_chunk,
)
from ._http import logger
from .errors import FailureKind, FreeAIError, OutputLimitError, ProviderFailure, StreamInterrupted
from .limits import Cost
from .providers import OpenRouter, Provider, Tier, build_from_env
from .types import ChatResponse, ModelInfo, Requirements, StreamChunk

__all__ = ["DEFAULT_EXCLUDE", "FreeAI", "chat", "stream"]

_NO_COST = Cost(0.0, 0.0)  # listing models reserves nothing


# How often requests waiting for quota look for changes other processes made to a shared store.
_POLL_SECONDS = 0.25


def _read_version(version: Callable[[], int] | None) -> int | None:
    if version is None:
        return None
    try:
        return version()
    except Exception:  # polling is best effort: the wait still ends on time
        logger.debug("could not poll the state store for changes", exc_info=True)
        return None


class _Wakeups:
    """Requests of this client waiting for quota, grouped by the models they can still use.

    Each change the router reports wakes the oldest waiter of every group: requests that can use
    the same models are interchangeable, so trying one of them is enough. With a store other
    processes write to (``version``), waiting threads also look for their changes every
    ``_POLL_SECONDS``, and the first to see one wakes every group.
    """

    def __init__(self, version: Callable[[], int] | None = None) -> None:
        self._lock = threading.Lock()
        self._changes = 0
        self._waiting: dict[frozenset[str], deque[threading.Event]] = {}
        self._version = version
        self._reported = -math.inf  # when a waiting thread last reported another process's change

    def mark(self) -> tuple[int, int | None]:
        """Taken before planning, so :meth:`wait` also catches changes made while the plan ran."""
        return self._changes, _read_version(self._version)

    def wake(self) -> None:
        with self._lock:
            self._changes += 1
            for queue in self._waiting.values():
                queue.popleft().set()
            self._waiting = {group: queue for group, queue in self._waiting.items() if queue}

    def wait(self, delay: float, mark: tuple[int, int | None], group: frozenset[str]) -> None:
        """Sleep up to ``delay``; return early when woken, or at once if a change came since ``mark``."""
        changes, version = mark
        event = threading.Event()
        with self._lock:
            if self._changes != changes:
                return
            self._waiting.setdefault(group, deque()).append(event)
        try:
            self._sleep(event, delay, version)
        finally:
            with self._lock:
                queue = self._waiting.get(group)
                if queue is not None and event in queue:
                    queue.remove(event)
                    if not queue:
                        del self._waiting[group]

    def _sleep(self, event: threading.Event, delay: float, version: int | None) -> None:
        deadline = time.monotonic() + delay
        if version is None:
            event.wait(delay)
            return
        looked = time.monotonic()
        while True:
            left = deadline - time.monotonic()
            if left <= 0 or event.wait(min(left, _POLL_SECONDS)):
                return
            current = _read_version(self._version)
            if current is None:  # the store failed: sleep out the rest
                event.wait(max(0.0, deadline - time.monotonic()))
                return
            now = time.monotonic()
            if current != version:
                version = current
                self._other_process_changed(since=looked, now=now)
            looked = now

    def _other_process_changed(self, since: float, now: float) -> None:
        with self._lock:
            if self._reported >= since:
                return  # another waiting thread reported a change after we last looked
            self._reported = now
        self.wake()


class _ClientBase:
    """Configuration shared by the sync and async clients (see :class:`~freeai._core.Router`)."""

    def __init__(
        self, providers: Provider | Sequence[Provider] | None = None, **options: Unpack[RouterOptions]
    ) -> None:
        if providers is None:
            providers = [OpenRouter()]
        self._router = Router(providers, **options)

    @classmethod
    def from_env(cls, *, tiers: Iterable[Tier] = ("free",), include_keyless: bool = False, **kwargs: Any) -> Self:
        """Use every preset whose environment variables are set and whose tier is in ``tiers``.

        Only ``"free"`` presets are enabled by default; pass ``tiers=("free", "trial")`` to opt into
        trial credit (Cerebras, Cohere) or ``"paid"`` for pay-as-you-go keys (Mistral).
        ``include_keyless=True`` also enables anonymous endpoints (Kilo without a key, OVHcloud).
        """
        return cls(build_from_env(tiers, include_keyless), **kwargs)

    @property
    def providers(self) -> list[Provider]:
        return self._router.providers

    def status(self) -> dict[str, dict[str, float]]:
        """Remaining cooldown seconds per model key and per provider."""
        return self._router.status()

    def health(self) -> dict[str, dict[str, float]]:
        """Smoothed success rate and latency per model key (used by ``strategy="health"``)."""
        return self._router.health_snapshot()

    def _request(
        self,
        messages: Messages,
        system: str | None,
        model: str | Sequence[str] | None,
        require: Iterable[str],
        min_context: int | None,
        params: dict[str, Any],
    ) -> Request:
        return self._router.prepare(messages, system, model, require, min_context, params)

    @staticmethod
    def _pools_from(
        results: Iterable[tuple[Provider, Sequence[ModelInfo] | FreeAIError]],
    ) -> dict[str, Sequence[ModelInfo]]:
        pools: dict[str, Sequence[ModelInfo]] = {}
        for provider, result in results:
            if isinstance(result, FreeAIError):
                logger.warning("skipping provider %s: %s", provider.name, result)
            else:
                pools[provider.name] = result
        return pools

    def _plan_after_pools(
        self, request: Request, pools: dict[str, Sequence[ModelInfo]], max_attempts: int | None, deadline: float | None
    ) -> tuple[Plan, Sequence[Provider]]:
        """Order candidates and build a plan; return warmups still needed for ``plan.providers``."""
        router = self._router
        ordered = router.order(request, pools, turn=router.next_turn())
        plan = router.plan(request, ordered, max_attempts=max_attempts, deadline=deadline)
        return plan, router.claim_warmups(plan.providers)


class FreeAI(_ClientBase):
    """Chat with whichever free model is currently available.

    Every call picks the first ready candidate, sends the request, and on failure benches that
    model (or the whole provider for account-wide limits) and moves to the next one. Options:

    - ``prefer``/``exclude``: glob patterns on model ids or ``provider:id`` keys.
    - ``strategy``: ``"priority"`` (default), ``"round_robin"``, ``"random"`` or ``"health"``
      (power-of-two choices on smoothed success rate and latency).
    - ``max_attempts``, ``max_wait`` (seconds spent waiting for cooldowns) and ``deadline`` (total
      seconds per call; also caps each HTTP read).
    - ``first_token_timeout`` (streams): give up on a model that sends no event within this many
      seconds and fail over. It is also the read timeout for the rest of that stream, so a pause
      longer than this after output started raises :class:`StreamInterrupted`.
    - ``retry_budget``: after a burst of server failures (5xx, timeouts) stop failing over until
      successes refill it; ``False`` disables it.
    - ``allow_training=False`` keeps only models that do not train on prompts (OpenRouter is asked
      per request not to collect data).
    - ``reject_response``: ``callable(ChatResponse) -> bool | str``; a truthy result fails over.
      Exceptions from the hook are treated as a rejection (not a hard abort).
    - ``store``: :class:`~freeai.SQLiteStore` to share cooldowns and quotas between processes. The
      caller owns it: :meth:`close` does not close the store.
    - ``clock``: wall-clock seconds (default :func:`time.time`). State is stored as epoch time so
      processes can share it; do not pass a monotonic clock.
    """

    def __init__(
        self,
        providers: Provider | Sequence[Provider] | None = None,
        *,
        sleep: Callable[[float], None] = time.sleep,
        **options: Unpack[RouterOptions],
    ) -> None:
        super().__init__(providers, **options)
        self._sleep = sleep
        # A custom ``sleep`` (a fake clock in tests) is used as is; time.sleep is replaced by a wait
        # that ends early when another thread's request, or another process sharing the store,
        # frees quota.
        version = getattr(self._router.store, "version", None)
        self._wakeups = _Wakeups(version) if sleep is time.sleep else None
        if self._wakeups is not None:
            self._router.on_change = self._wakeups.wake

    def _mark(self) -> tuple[int, int | None]:
        return self._wakeups.mark() if self._wakeups is not None else (0, None)

    def _pause(self, step: Wait, mark: tuple[int, int | None]) -> None:
        if self._wakeups is None:
            self._sleep(step.delay)
        else:
            self._wakeups.wait(step.delay, mark, step.group)

    # -- public API ------------------------------------------------------------------------

    def chat(
        self,
        messages: Messages,
        *,
        system: str | None = None,
        model: str | Sequence[str] | None = None,
        require: Iterable[str] = (),
        min_context: int | None = None,
        max_attempts: int | None = None,
        deadline: float | None = None,
        **params: Any,
    ) -> ChatResponse:
        """Send a chat completion, failing over across free models until one answers.

        ``model`` restricts the candidates (exact ids, ``provider:id`` keys or glob patterns) and
        orders them. Extra keyword arguments (``temperature``, ``tools``, ...) go to the API.
        """
        router = self._router
        request = self._request(messages, system, model, require, min_context, params)
        plan = self._plan(request, max_attempts, deadline)
        while True:
            mark = self._mark()
            step = plan.next()
            if isinstance(step, Wait):
                self._pause(step, mark)
                continue
            target = step
            provider = router.provider(target)
            started = router.clock()
            completion = None
            try:
                completion = provider.complete(target.id, router.payload_for(target, request), timeout=plan.time_left())
                response = router.to_response(target, completion.body)
            except ProviderFailure as failure:
                extra = completion.snapshots if completion is not None else []
                usage = completion.body.get("usage") if completion is not None else None
                if plan.failed(target, failure, started, extra, usage=usage):
                    router.observe(provider.name, "*", provider.refresh_quota())
                if failure.kind is FailureKind.OUTPUT_LIMIT:
                    raw = completion.body if completion is not None else None
                    raise OutputLimitError(failure.message, plan.attempts, raw) from failure
                continue
            except BaseException:
                if completion is not None:
                    # Platform already answered; keep request quota (do not plan.release-refund).
                    plan.failed(
                        target,
                        ProviderFailure(
                            FailureKind.REJECTED,
                            f"{target.provider}: {target.id}: aborted after response",
                        ),
                        started,
                        completion.snapshots,
                        usage=completion.body.get("usage") if isinstance(completion.body, dict) else None,
                    )
                else:
                    plan.release(target)
                raise
            plan.succeeded(
                target, started, snapshots=completion.snapshots, usage=response.usage, served_model=response.model
            )
            response.attempts = plan.attempts
            return response

    def stream(
        self,
        messages: Messages,
        *,
        system: str | None = None,
        model: str | Sequence[str] | None = None,
        require: Iterable[str] = (),
        min_context: int | None = None,
        max_attempts: int | None = None,
        deadline: float | None = None,
        first_token_timeout: float | None = None,
        **params: Any,
    ) -> Iterator[StreamChunk]:
        """Stream a chat completion. Failover happens only before the first output token;
        a failure after that raises :class:`StreamInterrupted` with the partial text."""
        request = self._request(messages, system, model, require, min_context, params)
        return self._stream(request, max_attempts, deadline, first_token_timeout)

    def models(
        self,
        *,
        require: Iterable[str] = (),
        input_modalities: Iterable[str] = (),
        min_context: int | None = None,
        refresh: bool = False,
    ) -> list[ModelInfo]:
        """Candidate models in the order a request would try them (ignoring cooldowns)."""
        if refresh:
            for provider in self.providers:
                provider.invalidate()
        requirements = Requirements(frozenset(require), frozenset(input_modalities), min_context)
        request = Request({}, requirements, (), _NO_COST)
        return self._router.order(request, self._pools(), turn=0)

    def close(self) -> None:
        for provider in self.providers:
            provider.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- internals -------------------------------------------------------------------------

    def _pools(self) -> dict[str, Sequence[ModelInfo]]:
        providers = self.providers
        # Warm catalogs stay sequential (no thread hop). Cold multi-provider fetches match async gather.
        if len(providers) <= 1 or not any(provider.catalog_may_fetch() for provider in providers):
            return self._pools_from((provider, self._snapshot(provider)) for provider in providers)
        workers = min(len(providers), 32, (os.cpu_count() or 1) + 4)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(self._snapshot, providers))
        return self._pools_from(zip(providers, results, strict=True))

    @staticmethod
    def _snapshot(provider: Provider) -> Sequence[ModelInfo] | FreeAIError:
        try:
            return provider.models_snapshot()
        except FreeAIError as exc:
            return exc

    def _plan(self, request: Request, max_attempts: int | None, deadline: float | None) -> Plan:
        router = self._router
        # Overlap account-quota HTTP with catalog discovery on the first plan.
        early = router.claim_warmups(p.name for p in self.providers if p.quota_may_fetch())
        if early:
            workers = min(1 + len(early), 32, (os.cpu_count() or 1) + 4)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                pools_future = pool.submit(self._pools)
                snapshots = list(pool.map(lambda provider: provider.refresh_quota(), early))
                pools = pools_future.result()
            router.observe_many((provider.name, "*", snaps) for provider, snaps in zip(early, snapshots, strict=True))
        else:
            pools = self._pools()
        plan, warmups = self._plan_after_pools(request, pools, max_attempts, deadline)
        self._observe_warmups(warmups)
        return plan

    def _observe_warmups(self, warmups: Sequence[Provider]) -> None:
        """Fetch each provider's quota once; parallelize only when several may hit the network."""
        if not warmups:
            return
        if len(warmups) <= 1 or not any(provider.quota_may_fetch() for provider in warmups):
            snapshots = [provider.refresh_quota() for provider in warmups]
        else:
            workers = min(len(warmups), 32, (os.cpu_count() or 1) + 4)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                snapshots = list(pool.map(lambda provider: provider.refresh_quota(), warmups))
        self._router.observe_many(
            (provider.name, "*", snaps) for provider, snaps in zip(warmups, snapshots, strict=True)
        )

    def _stream(
        self, request: Request, max_attempts: int | None, deadline: float | None, first_token_timeout: float | None
    ) -> Iterator[StreamChunk]:
        router = self._router
        plan = self._plan(request, max_attempts, deadline)
        token_timeout = first_token_timeout if first_token_timeout is not None else router.first_token_timeout
        while True:
            mark = self._mark()
            step = plan.next()
            if isinstance(step, Wait):
                self._pause(step, mark)
                continue
            target = step
            provider = router.provider(target)
            started = router.clock()
            emitted: list[str] = []
            usage: dict[str, Any] | None = None
            served: str | None = None
            delivered = False
            answered = False
            finish_reason: str | None = None
            try:
                events = provider.stream(
                    target.id,
                    router.payload_for(target, request),
                    timeout=plan.time_left(),
                    first_token_timeout=token_timeout,
                    on_snapshots=plan.admitted,  # headers mean the platform accepted it
                )
                with closing(events):
                    for event in events:
                        chunk = to_chunk(target, event)
                        if chunk.usage is not None:
                            usage = chunk.usage
                        if chunk.finish_reason is not None:
                            finish_reason = chunk.finish_reason
                        if stream_chunk_answered(chunk):
                            answered = True
                        if not delivered and not chunk.has_output:
                            continue  # role-only preambles: nothing committed to this model yet
                        delivered = True
                        served = chunk.model
                        emitted.append(chunk.text)
                        yield chunk
            except ProviderFailure as failure:
                if plan.failed(target, failure, started, usage=usage):
                    router.observe(provider.name, "*", provider.refresh_quota())
                if delivered:
                    raise StreamInterrupted(failure, "".join(emitted), plan.attempts) from failure
                continue
            except BaseException:
                plan.release(target)  # includes GeneratorExit when the caller stops iterating
                raise
            if not answered and finish_reason == "length":
                failure = output_limit_failure(target)
                plan.failed(target, failure, started, usage=usage)
                raise OutputLimitError(failure.message, plan.attempts) from None
            if not delivered and router.retry_on_empty:
                plan.failed(target, empty_stream(target), started, usage=usage)
                continue
            plan.succeeded(target, started, usage=usage, served_model=served)
            return


_default: FreeAI | None = None
_default_lock = threading.Lock()


def _default_client() -> FreeAI:
    global _default
    with _default_lock:
        if _default is None:
            _default = FreeAI.from_env()
        return _default


def chat(messages: Messages, **kwargs: Any) -> ChatResponse:
    """Module-level shortcut using a shared ``FreeAI.from_env()`` client."""
    return _default_client().chat(messages, **kwargs)


def stream(messages: Messages, **kwargs: Any) -> Iterator[StreamChunk]:
    """Module-level shortcut using a shared ``FreeAI.from_env()`` client."""
    return _default_client().stream(messages, **kwargs)
