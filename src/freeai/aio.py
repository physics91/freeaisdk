"""AsyncFreeAI: the asyncio version of :class:`~freeai.FreeAI` with the same routing core."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Sequence
from contextlib import aclosing
from typing import Any, Self, Unpack

from ._core import Messages, Plan, Request, RouterOptions, Wait, empty_stream, to_chunk
from .client import _NO_COST, _POLL_SECONDS, _ClientBase, _read_version
from .errors import FailureKind, FreeAIError, OutputLimitError, ProviderFailure, StreamInterrupted
from .providers import Provider
from .types import ChatResponse, ModelInfo, Requirements, StreamChunk

__all__ = ["AsyncFreeAI"]


class _Wakeups:
    """Tasks of this client waiting for quota, grouped by the models they can still use.

    Each change the router reports wakes the oldest waiter of every group, as in the sync client.
    With a store other processes write to (``version``), one task polls it every
    ``_POLL_SECONDS`` while anything waits and wakes every group on a change.
    """

    def __init__(self, version: Callable[[], int] | None = None) -> None:
        self._waiting: dict[frozenset[str], deque[asyncio.Future[None]]] = {}
        self._version = version
        self._poller: asyncio.Task[None] | None = None

    def wake(self) -> None:
        for group in list(self._waiting):
            queue = self._waiting[group]
            while queue:
                future = queue.popleft()
                if not future.done():
                    future.set_result(None)
                    break
            if not queue:
                del self._waiting[group]

    async def wait(self, delay: float, group: frozenset[str]) -> None:
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._waiting.setdefault(group, deque()).append(future)
        if self._poller is None or self._poller.done():
            version = _read_version(self._version)
            if version is not None:
                self._poller = asyncio.create_task(self._poll(version))
        try:
            await asyncio.wait_for(future, delay)
        except TimeoutError:
            pass
        finally:
            queue = self._waiting.get(group)
            if queue is not None and future in queue:
                queue.remove(future)
                if not queue:
                    del self._waiting[group]

    async def _poll(self, version: int) -> None:
        while self._waiting:
            await asyncio.sleep(_POLL_SECONDS)
            current = _read_version(self._version)
            if current is None:
                return
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

    async def _pause(self, step: Wait) -> None:
        if self._wakeups is None:
            await self._sleep(step.delay)
        else:
            await self._wakeups.wait(step.delay, step.group)

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
        **params: Any,
    ) -> ChatResponse:
        router = self._router
        request = self._request(messages, system, model, require, min_context, params)
        plan = await self._plan(request, max_attempts, deadline)
        while True:
            step = plan.next()
            if isinstance(step, Wait):
                await self._pause(step)
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
                plan.release(target)  # includes task cancellation
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
    ) -> AsyncIterator[StreamChunk]:
        """Async iterator of chunks; arguments are validated before the first ``await``."""
        request = self._request(messages, system, model, require, min_context, params)
        return self._stream(request, max_attempts, deadline, first_token_timeout)

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

    async def _plan(self, request: Request, max_attempts: int | None, deadline: float | None) -> Plan:
        router = self._router
        ordered = router.order(request, await self._pools(), turn=router.next_turn())
        plan = router.plan(request, ordered, max_attempts=max_attempts, deadline=deadline)
        for provider in router.claim_warmups(plan.providers):
            router.observe(provider.name, "*", await provider.arefresh_quota())
        return plan

    async def _stream(
        self, request: Request, max_attempts: int | None, deadline: float | None, first_token_timeout: float | None
    ) -> AsyncIterator[StreamChunk]:
        router = self._router
        plan = await self._plan(request, max_attempts, deadline)
        token_timeout = first_token_timeout if first_token_timeout is not None else router.first_token_timeout
        while True:
            step = plan.next()
            if isinstance(step, Wait):
                await self._pause(step)
                continue
            target = step
            provider = router.provider(target)
            started = router.clock()
            emitted: list[str] = []
            usage: dict[str, Any] | None = None
            served: str | None = None
            delivered = False
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
                        usage = chunk.usage or usage
                        if not delivered and not chunk.has_output:
                            continue
                        delivered = True
                        served = chunk.model
                        emitted.append(chunk.text)
                        yield chunk
            except ProviderFailure as failure:
                if plan.failed(target, failure, started):
                    router.observe(provider.name, "*", await provider.arefresh_quota())
                if delivered:
                    raise StreamInterrupted(failure, "".join(emitted), plan.attempts) from failure
                continue
            except BaseException:
                plan.release(target)
                raise
            if not delivered and router.retry_on_empty:
                plan.failed(target, empty_stream(target), started, usage=usage)
                continue
            plan.succeeded(target, started, usage=usage, served_model=served)
            return
