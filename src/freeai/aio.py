"""AsyncFreeAI: the asyncio version of :class:`~freeai.FreeAI` with the same routing core."""

from __future__ import annotations

import asyncio
import math
from collections import deque
from collections.abc import AsyncGenerator, Awaitable, Callable, Iterable, Sequence
from contextlib import aclosing
from dataclasses import replace
from typing import Any, Self, Unpack

from ._core import (
    Messages,
    Plan,
    Request,
    RouterOptions,
    Wait,
    empty_stream,
    output_limit_failure,
    stream_chunk_answered,
    to_chunk,
)
from .client import _NO_COST, _POLL_SECONDS, _ClientBase, _read_version
from .errors import FailureKind, FreeAIError, OutputLimitError, ProviderFailure, StreamInterrupted
from .providers import Provider
from .types import CatalogDrift, ChatResponse, ModelInfo, Requirements, StreamChunk

__all__ = ["AsyncFreeAI"]


class _Waiter:
    """One async wait slot; ``alive`` is cleared when the wait ends or a wake claims it."""

    __slots__ = ("alive", "future")

    def __init__(self) -> None:
        self.future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self.alive = True


class _Wakeups:
    """Tasks of this client waiting for quota, grouped by the models they can still use.

    Each change the router reports wakes the oldest waiter of every group, as in the sync client.
    With a store other processes write to (``version``), one task polls it every
    ``_POLL_SECONDS`` while anything waits and wakes every group on a change.
    """

    def __init__(self, version: Callable[[], int] | None = None) -> None:
        self._waiting: dict[frozenset[str], deque[_Waiter]] = {}
        self._version = version
        self._poller: asyncio.Task[None] | None = None
        self._changes = 0

    def mark(self) -> tuple[int, int | None]:
        """Taken before planning, so :meth:`wait` also catches changes made while the plan ran."""
        return self._changes, _read_version(self._version)

    def wake(self) -> None:
        self._changes += 1
        for group in list(self._waiting):
            queue = self._waiting[group]
            while queue:
                waiter = queue.popleft()
                if waiter.alive and not waiter.future.done():
                    waiter.alive = False
                    waiter.future.set_result(None)
                    break
            if not queue:
                del self._waiting[group]

    async def wait(self, delay: float, mark: tuple[int, int | None], group: frozenset[str]) -> None:
        changes, version = mark
        if self._changes != changes:
            return
        waiter = _Waiter()
        self._waiting.setdefault(group, deque()).append(waiter)
        if version is not None and (self._poller is None or self._poller.done()):
            self._poller = asyncio.create_task(self._poll(version))
        left_early = False
        try:
            await asyncio.wait_for(asyncio.shield(waiter.future), delay)
        except TimeoutError:
            left_early = True
        except asyncio.CancelledError:
            left_early = True
            raise
        finally:
            claimed = not waiter.alive
            waiter.alive = False
            queue = self._waiting.get(group)
            if queue is not None and waiter in queue:
                queue.remove(waiter)
                if not queue:
                    del self._waiting[group]
            # Wake claimed us after timeout/cancel: pass the signal to the next peer.
            if claimed and left_early and self._waiting.get(group):
                self.wake()

    async def _poll(self, version: int) -> None:
        while self._waiting:
            await asyncio.sleep(_POLL_SECONDS)
            current = _read_version(self._version)
            if current is None:
                continue  # transient store failure: keep polling while waiters remain
            if current != version:
                version = current
                self.wake()


class AsyncFreeAI(_ClientBase):
    """``await ai.chat(...)`` / ``async for chunk in ai.stream(...)`` with failover.

    Accepts the same options as :class:`~freeai.FreeAI`. Shared state is updated synchronously in
    tiny critical sections; with :class:`~freeai.SQLiteStore` each update briefly blocks the loop.
    """

    def __init__(
        self,
        providers: Provider | Sequence[Provider] | None = None,
        *,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        **options: Unpack[RouterOptions],
    ) -> None:
        super().__init__(providers, **options)
        self._sleep = sleep
        # A custom ``sleep`` (a fake clock in tests) is used as is; asyncio.sleep is replaced by a
        # wait that ends early when another task's request, or another process sharing the store,
        # frees quota.
        version = getattr(self._router.store, "version", None)
        self._wakeups = _Wakeups(version) if sleep is asyncio.sleep else None
        if self._wakeups is not None:
            self._router.on_change = self._wakeups.wake

    def _mark(self) -> tuple[int, int | None]:
        return self._wakeups.mark() if self._wakeups is not None else (0, None)

    async def _pause(self, step: Wait, mark: tuple[int, int | None]) -> None:
        if self._wakeups is None:
            await self._sleep(step.delay)
        else:
            await self._wakeups.wait(step.delay, mark, step.group)

    async def chat(
        self,
        messages: Messages,
        *,
        system: str | None = None,
        model: str | Sequence[str] | None = None,
        require: Iterable[str] = (),
        min_context: int | None = None,
        max_attempts: int | None = None,
        deadline: float | None = None,
        max_wait: float | None = None,
        **params: Any,
    ) -> ChatResponse:
        router = self._router
        request = self._request(messages, system, model, require, min_context, params)
        plan = await self._plan(request, max_attempts, deadline, max_wait)
        while True:
            mark = self._mark()
            step = plan.next()
            if isinstance(step, Wait):
                await self._pause(step, mark)
                continue
            target = step
            provider = router.provider(target)
            started = router.clock()
            completion = None
            try:
                completion = await provider.acomplete(
                    target.id, router.payload_for(target, request), timeout=plan.time_left()
                )
                response = router.to_response(target, completion.body)
            except ProviderFailure as failure:
                extra = completion.snapshots if completion is not None else []
                usage = completion.body.get("usage") if completion is not None else None
                if plan.failed(target, failure, started, extra, usage=usage):
                    router.observe(provider.name, "*", await provider.arefresh_quota())
                if failure.kind is FailureKind.OUTPUT_LIMIT:
                    raw = completion.body if completion is not None else None
                    raise OutputLimitError(failure.message, plan.attempts, raw) from failure
                continue
            except BaseException:
                if completion is not None:
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
                    plan.release(target)  # includes task cancellation before a response
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
        max_wait: float | None = None,
        first_token_timeout: float | None = None,
        **params: Any,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Async iterator of chunks; arguments are validated before the first ``await``."""
        request = self._request(messages, system, model, require, min_context, params)
        return self._stream(request, max_attempts, deadline, first_token_timeout, max_wait)

    async def batch(
        self,
        prompts: Iterable[Messages],
        *,
        concurrency: int = 8,
        max_wait: float | None = math.inf,
        on_result: Callable[[int, ChatResponse | FreeAIError], None] | None = None,
        system: str | None = None,
        model: str | Sequence[str] | None = None,
        require: Iterable[str] = (),
        min_context: int | None = None,
        max_attempts: int | None = None,
        deadline: float | None = None,
        **params: Any,
    ) -> list[ChatResponse | FreeAIError]:
        """Answer many prompts, waiting for quota instead of giving up when it runs out.

        The asyncio version of :meth:`FreeAI.batch`: up to ``concurrency`` requests run or wait at
        a time, and the result has the response or the :class:`FreeAIError` of each prompt in the
        order given. Cancelling the call, or any other exception, cancels the requests that are
        still running or waiting.
        """
        if concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        items = list(prompts)  # before any request starts: an iterable that fails leaves nothing running
        needs = tuple(require)
        slots = asyncio.Semaphore(concurrency)

        async def answer(index: int) -> ChatResponse | FreeAIError:
            result: ChatResponse | FreeAIError
            async with slots:
                try:
                    result = await self.chat(
                        items[index],
                        system=system,
                        model=model,
                        require=needs,
                        min_context=min_context,
                        max_attempts=max_attempts,
                        deadline=deadline,
                        max_wait=max_wait,
                        **params,
                    )
                except FreeAIError as exc:
                    result = exc
            if on_result is not None:
                on_result(index, result)
            return result

        tasks = [asyncio.create_task(answer(index)) for index in range(len(items))]
        try:
            return list(await asyncio.gather(*tasks))
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    async def models(
        self,
        *,
        require: Iterable[str] = (),
        input_modalities: Iterable[str] = (),
        min_context: int | None = None,
        refresh: bool = False,
    ) -> list[ModelInfo]:
        if refresh:
            for provider in self.providers:
                provider.invalidate()
        requirements = Requirements(frozenset(require), frozenset(input_modalities), min_context)
        return self._router.order(Request({}, requirements, (), _NO_COST), await self._pools(), turn=0)

    async def audit(self) -> list[CatalogDrift]:
        """Where each pinned model list differs from the platform's catalog (read-only)."""
        drifts = await asyncio.gather(*(provider.acatalog_drift() for provider in self.providers))
        return [self._chat_models_only(drift) for drift in drifts if drift is not None]

    async def aclose(self) -> None:
        for provider in self.providers:
            await provider.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    # -- internals -------------------------------------------------------------------------

    async def _pools(self) -> dict[str, Sequence[ModelInfo]]:
        async def fetch(provider: Provider) -> Sequence[ModelInfo] | FreeAIError:
            try:
                return await provider.amodels_snapshot()
            except FreeAIError as exc:
                return exc

        results = await asyncio.gather(*(fetch(p) for p in self.providers))
        return self._pools_from(zip(self.providers, results, strict=True))

    async def _plan(
        self, request: Request, max_attempts: int | None, deadline: float | None, max_wait: float | None = None
    ) -> Plan:
        router = self._router
        # Overlap account-quota HTTP with catalog discovery on the first plan.
        early = router.claim_warmups(p.name for p in self.providers if p.quota_may_fetch())
        if early:
            pools, snapshots = await asyncio.gather(
                self._pools(),
                asyncio.gather(*(provider.arefresh_quota() for provider in early)),
            )
            router.observe_many((provider.name, "*", snaps) for provider, snaps in zip(early, snapshots, strict=True))
        else:
            pools = await self._pools()
        plan, warmups = self._plan_after_pools(request, pools, max_attempts, deadline, max_wait)
        await self._observe_warmups(warmups)
        return plan

    async def _observe_warmups(self, warmups: Sequence[Provider]) -> None:
        """Fetch each provider's quota once; parallelize only when several may hit the network."""
        if not warmups:
            return
        if len(warmups) <= 1 or not any(provider.quota_may_fetch() for provider in warmups):
            snapshots = [await provider.arefresh_quota() for provider in warmups]
        else:
            snapshots = await asyncio.gather(*(provider.arefresh_quota() for provider in warmups))
        self._router.observe_many(
            (provider.name, "*", snaps) for provider, snaps in zip(warmups, snapshots, strict=True)
        )

    async def _stream(
        self,
        request: Request,
        max_attempts: int | None,
        deadline: float | None,
        first_token_timeout: float | None,
        max_wait: float | None = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        router = self._router
        plan = await self._plan(request, max_attempts, deadline, max_wait)
        token_timeout = first_token_timeout if first_token_timeout is not None else router.first_token_timeout
        while True:
            mark = self._mark()
            step = plan.next()
            if isinstance(step, Wait):
                await self._pause(step, mark)
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
                events = provider.astream(
                    target.id,
                    router.payload_for(target, request),
                    timeout=plan.time_left(),
                    first_token_timeout=token_timeout,
                    on_snapshots=plan.admitted,  # headers mean the platform accepted it
                )
                async with aclosing(events):
                    async for event in events:
                        chunk = to_chunk(target, event)
                        raw_model = event.get("model")
                        wire_model = raw_model if isinstance(raw_model, str) and raw_model else None
                        if chunk.usage is not None:
                            usage = chunk.usage
                        if chunk.finish_reason is not None:
                            finish_reason = chunk.finish_reason
                        if stream_chunk_answered(chunk):
                            answered = True
                        if not delivered and not chunk.has_output:
                            continue
                        delivered = True
                        # Sticky wire model: later model-less events must not fall back to the request id.
                        if wire_model is not None:
                            served = wire_model
                        elif served is None:
                            served = chunk.model
                        elif chunk.model != served:
                            chunk = replace(chunk, model=served)
                        emitted.append(chunk.text)
                        yield chunk
            except ProviderFailure as failure:
                if plan.failed(target, failure, started, usage=usage):
                    router.observe(provider.name, "*", await provider.arefresh_quota())
                if delivered:
                    raise StreamInterrupted(failure, "".join(emitted), plan.attempts) from failure
                continue
            except BaseException:
                plan.release(target)
                raise
            if not answered and finish_reason == "length":
                limit_failure = output_limit_failure(target)
                plan.failed(target, limit_failure, started, usage=usage)
                raise OutputLimitError(limit_failure.message, plan.attempts) from None
            # Match chat EMPTY: reasoning-only (or other non-answer) streams fail over too.
            if not answered and router.retry_on_empty:
                plan.failed(target, empty_stream(target), started, usage=usage)
                continue
            plan.succeeded(target, started, usage=usage, served_model=served)
            return
