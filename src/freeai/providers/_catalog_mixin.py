"""Provider mixin: cached /models discovery with sync and async single-flight."""

from __future__ import annotations

import asyncio
import dataclasses
import math
import threading
import time
from typing import TYPE_CHECKING, Any

from .._http import HTTPError, logger
from ..errors import FreeAIError
from ..types import CatalogDrift, ModelInfo
from ._catalog import _catalog_items

_STALE_CATALOG_RETRY = 60.0


class CatalogMixin:
    """Cached catalog discovery. Expects Provider fields/locks and ``_get_json`` / ``_aget_json``."""

    name: str
    catalog_ttl: float
    trains_on_prompts: bool | None
    _static: tuple[ModelInfo, ...] | None
    _catalog: tuple[ModelInfo, ...] | None
    _catalog_at: float
    _catalog_lock: threading.Lock
    _sync_catalog_lock: threading.Lock
    _acatalog_lock: asyncio.Lock | None
    _acatalog_loop: asyncio.AbstractEventLoop | None

    if TYPE_CHECKING:

        def _get_json(self, path: str) -> Any: ...
        async def _aget_json(self, path: str) -> Any: ...

    def list_models(self, *, refresh: bool = False) -> list[ModelInfo]:
        """Public catalog copy; callers may mutate the returned list safely."""
        return list(self.models_snapshot(refresh=refresh))

    async def alist_models(self, *, refresh: bool = False) -> list[ModelInfo]:
        """Public catalog copy; callers may mutate the returned list safely."""
        return list(await self.amodels_snapshot(refresh=refresh))

    def models_snapshot(self, *, refresh: bool = False) -> tuple[ModelInfo, ...]:
        """Cached catalog for routing. Treat as immutable; use :meth:`list_models` to copy.

        Concurrent cold calls share one discovery (single-flight) without holding
        ``_catalog_lock`` across network I/O.
        """
        if self._static is not None:
            return self._static
        with self._sync_catalog_lock:
            now = time.monotonic()
            if self._catalog_stale(now, refresh):
                try:
                    catalog = self._discover()
                except (HTTPError, ValueError, KeyError, TypeError) as exc:
                    with self._catalog_lock:
                        self._catalog_failed(exc, time.monotonic())
                        return self._catalog or ()
                with self._catalog_lock:
                    self._set_catalog(catalog, time.monotonic())
                    return self._catalog or ()
            with self._catalog_lock:
                return self._catalog or ()

    async def amodels_snapshot(self, *, refresh: bool = False) -> tuple[ModelInfo, ...]:
        """Async variant of :meth:`models_snapshot`.

        Concurrent cold calls on the same event loop share one discovery (single-flight).
        """
        if self._static is not None:
            return self._static
        async with self._async_catalog_lock():
            now = time.monotonic()
            if self._catalog_stale(now, refresh):
                try:
                    catalog = await self._adiscover()
                except (HTTPError, ValueError, KeyError, TypeError) as exc:
                    with self._catalog_lock:
                        self._catalog_failed(exc, time.monotonic())
                        return self._catalog or ()
                with self._catalog_lock:
                    self._set_catalog(catalog, time.monotonic())
                    return self._catalog or ()
            with self._catalog_lock:
                return self._catalog or ()

    def _async_catalog_lock(self) -> asyncio.Lock:
        """Event-loop-bound lock for async catalog single-flight; recreate when the loop changes."""
        loop = asyncio.get_running_loop()
        with self._catalog_lock:
            if self._acatalog_lock is None or self._acatalog_loop is not loop:
                self._acatalog_lock = asyncio.Lock()
                self._acatalog_loop = loop
            return self._acatalog_lock

    def invalidate(self) -> None:
        """Force rediscovery on the next ``list_models`` call."""
        self._catalog_at = -math.inf

    def catalog_drift(self) -> CatalogDrift | None:
        """Compare the pinned models with the platform's catalog; None when nothing is pinned.

        One read-only ``GET /models``. It changes nothing about which models are used: it shows
        where a pinned list has gone stale (models removed, models added since).
        """
        if self._static is None:
            return None
        try:
            return self._drift(self._discover())
        except (HTTPError, ValueError, KeyError, TypeError) as exc:
            return self._drift_unknown(exc)

    async def acatalog_drift(self) -> CatalogDrift | None:
        if self._static is None:
            return None
        try:
            return self._drift(await self._adiscover())
        except (HTTPError, ValueError, KeyError, TypeError) as exc:
            return self._drift_unknown(exc)

    def _drift(self, listed: list[ModelInfo]) -> CatalogDrift:
        pinned = tuple(model.id for model in self._static or ())
        live = [model.id for model in listed]
        return CatalogDrift(
            self.name,
            pinned,
            missing=tuple(model_id for model_id in pinned if model_id not in set(live)),
            unpinned=tuple(sorted(set(live) - set(pinned))),
        )

    def _drift_unknown(self, exc: Exception) -> CatalogDrift:
        pinned = tuple(model.id for model in self._static or ())
        return CatalogDrift(self.name, pinned, error=f"{type(exc).__name__}: {exc}")

    def catalog_may_fetch(self) -> bool:
        """True when the next :meth:`models_snapshot` may perform network I/O."""
        if self._static is not None:
            return False
        return self._catalog_stale(time.monotonic(), refresh=False)

    def _catalog_stale(self, now: float, refresh: bool) -> bool:
        return refresh or self._catalog is None or now - self._catalog_at > self.catalog_ttl

    def _set_catalog(self, catalog: list[ModelInfo], now: float) -> None:
        self._catalog = tuple(catalog)
        self._catalog_at = now

    def _catalog_failed(self, exc: Exception, now: float) -> None:
        if self._catalog is None:
            raise FreeAIError(f"{self.name}: model discovery failed: {exc}") from exc
        logger.warning("%s: model discovery failed, keeping stale catalog: %s", self.name, exc)
        self._catalog_at = now - self.catalog_ttl + _STALE_CATALOG_RETRY

    def _catalog_paths(self) -> tuple[str, ...]:
        """Catalog endpoints to try in order; later ones are fallbacks after an HTTP error."""
        return ("/models",)

    def _discover(self) -> list[ModelInfo]:
        paths = self._catalog_paths()
        for index, path in enumerate(paths):
            try:
                return self._parse_catalog(self._get_json(path))
            except HTTPError as exc:
                self._catalog_path_failed(path, index, paths, exc)
        return []

    async def _adiscover(self) -> list[ModelInfo]:
        paths = self._catalog_paths()
        for index, path in enumerate(paths):
            try:
                return self._parse_catalog(await self._aget_json(path))
            except HTTPError as exc:
                self._catalog_path_failed(path, index, paths, exc)
        return []

    def _catalog_path_failed(self, path: str, index: int, paths: tuple[str, ...], exc: HTTPError) -> None:
        if index == len(paths) - 1:
            raise
        logger.info("%s: %s unavailable (%s); trying the next catalog", self.name, path, exc)

    def _parse_catalog(self, body: Any) -> list[ModelInfo]:
        return [self._as_model(str(item["id"])) for item in _catalog_items(body) if item.get("id")]

    def _as_model(self, model: str | ModelInfo) -> ModelInfo:
        if isinstance(model, ModelInfo):
            trains = self.trains_on_prompts if model.trains_on_prompts is None else model.trains_on_prompts
            return dataclasses.replace(model, provider=self.name, trains_on_prompts=trains)
        return ModelInfo(self.name, model, trains_on_prompts=self.trains_on_prompts)
