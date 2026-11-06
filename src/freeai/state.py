"""Where cooldowns and quota counters live.

Account-level facts (a provider is rate limited, 37 of 50 daily requests are used) must be shared by
every process using the same key, so they go through a :class:`StateStore`. Values are small JSON
dicts and timestamps are epoch seconds, because monotonic clocks differ between processes.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator, MutableMapping
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import Any, Protocol


class StateStore(Protocol):
    """Atomic read-modify-write access to a small key/value map.

    A store shared by processes may also offer ``version() -> int``, a number that changes whenever
    another process commits (see :meth:`SQLiteStore.version`). Requests waiting for quota poll it to
    wake up when another process frees some; without it they sleep until their computed delay.
    """

    def transaction(self) -> AbstractContextManager[MutableMapping[str, Any]]:
        """Exclusive view of the state; changes are committed when the block exits normally.

        Always assign modified values back (``tx[key] = value``) instead of mutating them in place.
        """
        ...

    def close(self) -> None: ...


class MemoryStore:
    """Process-local state (the default)."""

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}
        self._lock = threading.RLock()

    @contextmanager
    def transaction(self) -> Iterator[MutableMapping[str, Any]]:
        with self._lock:
            # Yield a view so callers can duck-type ``keys_with_prefix`` / ``items_with_prefix``
            # the same way they do with :class:`_SQLiteView` (a bare dict has neither).
            yield _MemoryView(self._data)

    def close(self) -> None:
        pass


class _MemoryView(MutableMapping[str, Any]):
    """MutableMapping over an in-memory dict, with the same prefix helpers as SQLite."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value

    def __delitem__(self, key: str) -> None:
        del self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def keys_with_prefix(self, prefix: str) -> Iterator[str]:
        return iter(sorted(key for key in self._data if key.startswith(prefix)))

    def items_with_prefix(self, prefix: str) -> Iterator[tuple[str, Any]]:
        return iter((key, self._data[key]) for key in self.keys_with_prefix(prefix))


class SQLiteStore:
    """State shared by processes on one host through a SQLite database in WAL mode.

    WAL needs a local filesystem (not NFS/SMB). Each thread gets its own connection; writers are
    serialized with ``BEGIN IMMEDIATE``. Operations are tiny, so blocking briefly inside asyncio
    code is acceptable.
    """

    def __init__(self, path: str | Path, *, timeout: float = 5.0) -> None:
        self.path = str(path)
        self._timeout = timeout
        self._local = threading.local()
        self._connections: list[sqlite3.Connection] = []
        self._guard = threading.Lock()
        conn = self._conn()
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE IF NOT EXISTS freeai_state (key TEXT PRIMARY KEY, value TEXT NOT NULL)")

    def _conn(self) -> sqlite3.Connection:
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is None:
            # Only this thread uses the connection; close() may run on another one.
            conn = sqlite3.connect(self.path, timeout=self._timeout, isolation_level=None, check_same_thread=False)
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
            with self._guard:
                self._connections.append(conn)
        return conn

    @contextmanager
    def transaction(self) -> Iterator[MutableMapping[str, Any]]:
        conn = self._conn()
        conn.execute("BEGIN IMMEDIATE")
        view = _SQLiteView(conn)
        try:
            yield view
            view.flush()
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise

    def version(self) -> int:
        """A number that changes whenever another connection commits: another process, or another
        thread of this one (each thread has its own connection). Clients waiting for quota poll it
        to notice what other processes freed (SQLite's ``PRAGMA data_version``)."""
        row = self._conn().execute("PRAGMA data_version").fetchone()
        return int(row[0])

    def close(self) -> None:
        with self._guard:
            for conn in self._connections:
                conn.close()
            self._connections.clear()
        self._local = threading.local()


class _SQLiteView(MutableMapping[str, Any]):
    """Lazy, write-back view of the table inside one transaction."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._cache: dict[str, Any] = {}
        self._missing: set[str] = set()
        self._dirty: set[str] = set()
        self._deleted: set[str] = set()

    def __getitem__(self, key: str) -> Any:
        if key in self._cache:
            return self._cache[key]
        if key in self._missing or key in self._deleted:
            raise KeyError(key)
        row = self._conn.execute("SELECT value FROM freeai_state WHERE key = ?", (key,)).fetchone()
        if row is None:
            self._missing.add(key)
            raise KeyError(key)
        value = json.loads(row[0])
        self._cache[key] = value
        return value

    def __setitem__(self, key: str, value: Any) -> None:
        self._cache[key] = value
        self._dirty.add(key)
        self._missing.discard(key)
        self._deleted.discard(key)

    def __delitem__(self, key: str) -> None:
        self[key]  # raise KeyError when absent
        self._cache.pop(key, None)
        self._dirty.discard(key)
        self._deleted.add(key)

    def __iter__(self) -> Iterator[str]:
        stored = {row[0] for row in self._conn.execute("SELECT key FROM freeai_state")}
        return iter(sorted((stored | set(self._cache)) - self._deleted))

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def keys_with_prefix(self, prefix: str) -> Iterator[str]:
        """Keys that start with ``prefix``, including unflushed cache overlays."""
        pattern = _like_prefix(prefix)
        stored = {
            row[0]
            for row in self._conn.execute("SELECT key FROM freeai_state WHERE key LIKE ? ESCAPE '\\'", (pattern,))
            if row[0].startswith(prefix)
        }
        return iter(self._prefix_overlay(prefix, stored))

    def items_with_prefix(self, prefix: str) -> Iterator[tuple[str, Any]]:
        """``(key, value)`` pairs for ``prefix``, one SQL scan plus cache overlays.

        Order matches :meth:`keys_with_prefix`. Values already in the write-back cache win over
        the row just read from SQLite.
        """
        pattern = _like_prefix(prefix)
        stored: set[str] = set()
        for key, raw in self._conn.execute(
            "SELECT key, value FROM freeai_state WHERE key LIKE ? ESCAPE '\\'", (pattern,)
        ):
            if not key.startswith(prefix) or key in self._deleted:
                continue
            stored.add(key)
            if key not in self._cache:
                self._cache[key] = json.loads(raw)
        keys = self._prefix_overlay(prefix, stored)
        return iter((key, self._cache[key]) for key in keys)

    def _prefix_overlay(self, prefix: str, stored: set[str]) -> list[str]:
        """Sorted keys under ``prefix`` with the same overlay contract as ``__iter__``."""
        cached = {key for key in self._cache if key.startswith(prefix)}
        return sorted((stored | cached) - self._deleted)

    def flush(self) -> None:
        if self._deleted:
            self._conn.executemany("DELETE FROM freeai_state WHERE key = ?", [(key,) for key in self._deleted])
        if self._dirty:
            self._conn.executemany(
                "INSERT INTO freeai_state (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                [(key, json.dumps(self._cache[key], separators=(",", ":"))) for key in self._dirty],
            )


def _like_prefix(prefix: str) -> str:
    """Turn a literal key prefix into a ``LIKE`` pattern with ``ESCAPE '\\'``."""
    escaped = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return escaped + "%"
