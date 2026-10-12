"""
tests/test_score_cache_bounds.py -- QoL 5: ScoreCache is size-bounded and
sweeps expired entries on insert.

Before this, an expired entry was only removed when that same key was asked
for again, so keys that never recur (every distinct date/ticker-set/period
combination) accumulated DataFrames for the life of the process.
"""

import pandas as pd
import pytest

import api.cache as cache_mod
from api.cache import ScoreCache


def _df(n: int = 1) -> pd.DataFrame:
    return pd.DataFrame({"a": range(n)})


@pytest.fixture
def clock(monkeypatch):
    now = {"t": 1000.0}
    monkeypatch.setattr(cache_mod.time, "monotonic", lambda: now["t"])
    return now


def test_size_bound_evicts_oldest(clock):
    c = ScoreCache(ttl_seconds=3600, max_entries=3)
    for i in range(5):
        clock["t"] += 1
        c.set(f"k{i}", _df())
    assert c.stats()["entries"] == 3
    assert c.get("k0") is None and c.get("k1") is None
    assert c.get("k2") is not None and c.get("k4") is not None
    assert c.stats()["evictions"] == 2


def test_overwriting_a_key_does_not_evict_others(clock):
    c = ScoreCache(ttl_seconds=3600, max_entries=2)
    c.set("a", _df())
    c.set("b", _df())
    c.set("a", _df(5))  # refresh in place
    assert c.stats()["entries"] == 2
    assert len(c.get("a")) == 5 and c.get("b") is not None


def test_refreshed_key_becomes_newest(clock):
    c = ScoreCache(ttl_seconds=3600, max_entries=2)
    c.set("a", _df())
    c.set("b", _df())
    c.set("a", _df())  # a is now newest, b is oldest
    c.set("c", _df())
    assert c.get("b") is None
    assert c.get("a") is not None and c.get("c") is not None


def test_expired_entries_swept_on_insert_without_being_requested(clock):
    """Fails on the old implementation: never-requested expired keys stayed."""
    c = ScoreCache(ttl_seconds=60, max_entries=100)
    for i in range(10):
        c.set(f"old{i}", _df())
    clock["t"] += 61
    c.set("fresh", _df())
    assert list(c._store) == ["fresh"]


def test_expired_swept_before_live_entries_are_evicted(clock):
    c = ScoreCache(ttl_seconds=60, max_entries=2)
    c.set("stale", _df())
    clock["t"] += 30
    c.set("live", _df())
    clock["t"] += 31  # stale expired, live still valid
    c.set("new", _df())
    assert c.get("live") is not None and c.get("new") is not None
    assert c.get("stale") is None


def test_zero_disables_the_bound(clock):
    c = ScoreCache(ttl_seconds=3600, max_entries=0)
    for i in range(500):
        c.set(f"k{i}", _df())
    assert c.stats()["entries"] == 500


def test_default_bound_is_applied(clock):
    c = ScoreCache(ttl_seconds=3600)
    for i in range(ScoreCache.DEFAULT_MAX_ENTRIES + 10):
        c.set(f"k{i}", _df())
    assert c.stats()["entries"] == ScoreCache.DEFAULT_MAX_ENTRIES


def test_stats_reports_bound_and_keeps_existing_fields(clock):
    s = ScoreCache(ttl_seconds=60, max_entries=7).stats()
    assert s["max_entries"] == 7 and s["evictions"] == 0
    for key in ("ttl_seconds", "entries", "hits", "misses", "hit_rate", "cache"):
        assert key in s


def test_invalidate_and_get_unchanged(clock):
    c = ScoreCache(ttl_seconds=60, max_entries=5)
    c.set("a", _df())
    assert c.get("a") is not None
    assert c.invalidate("a") == 1 and c.invalidate() == 0


def test_config_default_and_env(monkeypatch):
    import importlib

    import config

    monkeypatch.delenv("API_SCORE_CACHE_MAX_ENTRIES", raising=False)
    assert importlib.reload(config).API_SCORE_CACHE_MAX_ENTRIES == 256
    monkeypatch.setenv("API_SCORE_CACHE_MAX_ENTRIES", "12")
    assert importlib.reload(config).API_SCORE_CACHE_MAX_ENTRIES == 12
    monkeypatch.delenv("API_SCORE_CACHE_MAX_ENTRIES")
    importlib.reload(config)
