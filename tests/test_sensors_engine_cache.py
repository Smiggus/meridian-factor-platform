"""
Tests for sensors._get_engine() module-level caching.

The Dagster daemon ticks request_sensor every 30 seconds. Without caching,
each tick opened a fresh PostgreSQL connection (~120/hour from this sensor
alone). _get_engine() now caches a singleton at module level - first call
constructs, subsequent calls return the same instance.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_DAGSTER_DIR = Path(__file__).resolve().parents[1] / "dagster"
if str(_DAGSTER_DIR) not in sys.path:
    sys.path.insert(0, str(_DAGSTER_DIR))

# Ensure PGPASS exists before importing the module (its only required env var).
os.environ.setdefault("PGPASS", "test-pg-pass")
os.environ.setdefault("PARTITION_START_DATE", "2010-01-02")


def _dagster_complete() -> bool:
    """CI's slim test image installs requirements.txt + requirements-test.txt
    only, neither of which pulls Dagster. Skip when Dagster is absent."""
    try:
        import dagster
    except ImportError:
        return False
    return all(hasattr(dagster, n) for n in ("RunRequest", "sensor"))


pytestmark = pytest.mark.skipif(
    not _dagster_complete(),
    reason="Dagster not fully installed (slim CI test image)",
)


@pytest.fixture
def reset_engine_cache():
    """Clear sensors._ENGINE before and after each test."""
    from meridian_pipeline import sensors

    sensors._ENGINE = None
    yield sensors
    sensors._ENGINE = None


def test_get_engine_returns_cached_instance(reset_engine_cache, monkeypatch):
    """Two calls to _get_engine() return the identical object."""
    sensors = reset_engine_cache

    sentinel = object()
    monkeypatch.setattr(sensors.db, "_engine", lambda **kw: sentinel)

    first = sensors._get_engine()
    second = sensors._get_engine()

    assert first is second
    assert first is sentinel


def test_get_engine_calls_factory_once(reset_engine_cache, monkeypatch):
    """db._engine() is invoked at most once across N _get_engine() calls."""
    sensors = reset_engine_cache

    call_count = {"n": 0}

    def fake_engine(**kw):
        call_count["n"] += 1
        return object()

    monkeypatch.setattr(sensors.db, "_engine", fake_engine)

    for _ in range(5):
        sensors._get_engine()

    assert call_count["n"] == 1
