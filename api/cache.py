import threading
import time
from dataclasses import dataclass

import pandas as pd


@dataclass
class _Entry:
    df: pd.DataFrame
    created_at: float
    expires_at: float


class ScoreCache:
    """
    In-memory cache for universe score DataFrames.

    Keys are deterministic strings built from the scoring parameters
    (sorted tickers + date + period). An expired entry is dropped lazily on
    get() and, so keys that are never requested again cannot pile up, swept
    on every set(). The store is also size-bounded: when a set() would take
    it past `max_entries`, expired entries go first and then the oldest
    (closest to expiry, since the TTL is uniform) until it fits.
    `max_entries <= 0` disables the bound.
    Thread-safe for multi-worker use via a single lock.
    """

    DEFAULT_MAX_ENTRIES = 256

    def __init__(self, ttl_seconds: int = 3600, max_entries: int = DEFAULT_MAX_ENTRIES):
        self._ttl = ttl_seconds
        self._max_entries = max_entries
        self._evictions = 0
        self._store: dict[str, _Entry] = {}
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    # ── Public interface ──────────────────────────────────────────────────────

    def get(self, key: str) -> pd.DataFrame | None:
        with self._lock:
            entry = self._store.get(key)
            if entry is None or time.monotonic() > entry.expires_at:
                if entry is not None:
                    del self._store[key]
                self._misses += 1
                return None
            self._hits += 1
            return entry.df.copy()

    def set(self, key: str, df: pd.DataFrame) -> None:
        now = time.monotonic()
        with self._lock:
            # Sweep expired entries first: an expired key that is never asked
            # for again would otherwise live (and hold its DataFrame) forever.
            expired = [k for k, e in self._store.items() if now > e.expires_at]
            for k in expired:
                del self._store[k]
            self._evictions += len(expired)

            # Re-inserting moves a refreshed key to the back of the order.
            self._store.pop(key, None)
            self._store[key] = _Entry(
                df=df.copy(),
                created_at=now,
                expires_at=now + self._ttl,
            )

            if self._max_entries > 0:
                while len(self._store) > self._max_entries:
                    del self._store[next(iter(self._store))]  # oldest insertion
                    self._evictions += 1

    def invalidate(self, key: str | None = None) -> int:
        """
        Clears cache entries. Pass a key to remove one entry,
        or None to flush everything. Returns the count cleared.
        """
        with self._lock:
            if key is not None:
                removed = 1 if self._store.pop(key, None) is not None else 0
            else:
                removed = len(self._store)
                self._store.clear()
        return removed

    def stats(self) -> dict:
        now = time.monotonic()
        with self._lock:
            entries = [
                {
                    "key": k,
                    "age_seconds": round(now - e.created_at, 1),
                    "expires_in_seconds": round(max(0.0, e.expires_at - now), 1),
                    "rows": len(e.df),
                }
                for k, e in self._store.items()
            ]
        return {
            "ttl_seconds": self._ttl,
            "max_entries": self._max_entries,
            "evictions": self._evictions,
            "entries": len(entries),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / max(1, self._hits + self._misses), 3),
            "cache": entries,
        }


# ── Key helpers ───────────────────────────────────────────────────────────────


def make_score_key(tickers: list[str], date: str | None, period: str) -> str:
    """Deterministic cache key for a scored universe."""
    tickers_part = ",".join(sorted(tickers))
    date_part = date or "current"
    return f"{tickers_part}|{date_part}|{period}"


def make_universe_key() -> str:
    """Separate cache key for the universe ticker list itself."""
    return "__universe__"
