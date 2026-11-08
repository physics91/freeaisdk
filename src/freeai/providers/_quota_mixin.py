"""Server-side quota endpoint I/O mixed into :class:`~freeai.providers.Provider`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .._http import HTTPError, logger
from ..limits import RateSnapshot


class QuotaMixin:
    """Optional ``GET`` quota endpoints: path, parse, and sync/async refresh."""

    name: str

    if TYPE_CHECKING:

        def _get_json(self, path: str) -> Any: ...
        async def _aget_json(self, path: str) -> Any: ...

    def quota_may_fetch(self) -> bool:
        """True when :meth:`refresh_quota` may perform network I/O."""
        return self._quota_path() is not None

    def refresh_quota(self) -> list[RateSnapshot]:
        """Read server-side quota counters, if the platform exposes them. Never raises."""
        path = self._quota_path()
        if path is None:
            return []
        try:
            return self._parse_quota(self._get_json(path))
        except (HTTPError, ValueError, KeyError, TypeError) as exc:
            return self._quota_unavailable(exc)

    async def arefresh_quota(self) -> list[RateSnapshot]:
        path = self._quota_path()
        if path is None:
            return []
        try:
            return self._parse_quota(await self._aget_json(path))
        except (HTTPError, ValueError, KeyError, TypeError) as exc:
            return self._quota_unavailable(exc)

    def _quota_unavailable(self, exc: Exception) -> list[RateSnapshot]:
        logger.info("%s: could not read the quota: %s", self.name, exc)
        return []

    def _quota_path(self) -> str | None:
        return None

    def _parse_quota(self, body: Any) -> list[RateSnapshot]:
        return []
