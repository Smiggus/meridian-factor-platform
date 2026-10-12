"""
Unit tests for sensors.composite_cache_warm_sensor (Meridian #145).

DB-free: exercises the sensor against a real ephemeral DagsterInstance whose
fetch_materializations method is monkeypatched to return controlled results.
The HTTP call is intercepted via monkeypatched requests.post so no real
network traffic is made.

Three scenarios:
  1. New composite partition triggers exactly one warm call with the right date.
  2. Empty tick (no new materialisations) yields SkipReason, makes no HTTP calls.
  3. HTTP error (exception or non-2xx) is swallowed - sensor does not raise.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

_DAGSTER_DIR = Path(__file__).resolve().parents[1] / "dagster"
if str(_DAGSTER_DIR) not in sys.path:
    sys.path.insert(0, str(_DAGSTER_DIR))

os.environ.setdefault("PGPASS", "test-pg-pass")
os.environ.setdefault("PARTITION_START_DATE", "2010-01-02")


def _dagster_complete() -> bool:
    try:
        import dagster
    except ImportError:
        return False
    return all(
        hasattr(dagster, n)
        for n in ("RunRequest", "sensor", "SkipReason", "build_sensor_context", "DagsterInstance")
    )


pytestmark = pytest.mark.skipif(
    not _dagster_complete(),
    reason="Dagster not fully installed (slim CI test image)",
)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────


def _make_record(partition_key: str | None) -> SimpleNamespace:
    """Minimal stand-in for an EventLogRecord."""
    return SimpleNamespace(partition_key=partition_key, run_id="run-test-123")


def _make_fetch_result(records, cursor="next-cursor-123"):
    """Minimal stand-in for a FetchMaterializations result object."""
    return SimpleNamespace(records=records, cursor=cursor)


@pytest.fixture
def ephemeral_instance():
    from dagster import DagsterInstance

    return DagsterInstance.ephemeral()


def _run_sensor(
    instance, monkeypatch, records, cursor_out="next-cursor-123", initial_cursor=None, env=None
):
    """Drive one tick of composite_cache_warm_sensor with a patched fetch.

    Returns (emitted_items, new_cursor).
    """
    from meridian_pipeline import sensors

    from dagster import build_sensor_context

    fetch_result = _make_fetch_result(records, cursor=cursor_out)
    monkeypatch.setattr(instance, "fetch_materializations", lambda *a, **kw: fetch_result)

    env_patch = {
        "MERIDIAN_API_TOKEN": "test-token",
        "MERIDIAN_API_BASE_URL": "http://meridian-test:8100",
    }
    if env:
        env_patch.update(env)

    with patch.dict(os.environ, env_patch):
        ctx = build_sensor_context(instance=instance, cursor=initial_cursor)
        emitted = list(sensors.composite_cache_warm_sensor(ctx))
        return emitted, ctx.cursor


# ─────────────────────────────────────────────
# Scenario 1: new partition triggers exactly one warm call
# ─────────────────────────────────────────────


def test_new_partition_triggers_warm_call(ephemeral_instance, monkeypatch):
    """A single new composite materialisation fires exactly one POST /admin/cache/warm."""
    records = [_make_record("2026-06-20")]
    posted = []

    def _fake_post(url, params=None, headers=None, timeout=10):
        posted.append({"url": url, "params": params, "headers": headers})
        resp = MagicMock()
        resp.ok = True
        resp.json.return_value = {"date": "2026-06-20", "cleared": 1}
        return resp

    with patch("meridian_pipeline.sensors.requests.post", side_effect=_fake_post):
        emitted, new_cursor = _run_sensor(ephemeral_instance, monkeypatch, records)

    # Cursor advances to the value returned by fetch_materializations.
    assert new_cursor == "next-cursor-123"

    # Exactly one POST was made.
    assert len(posted) == 1
    assert "admin/cache/warm" in posted[0]["url"]
    assert posted[0]["params"] == {"date": "2026-06-20"}
    assert "Bearer test-token" in posted[0]["headers"]["Authorization"]

    # No SkipReason emitted when records exist.
    from dagster import SkipReason

    assert not any(isinstance(e, SkipReason) for e in emitted)


# ─────────────────────────────────────────────
# Scenario 2: empty tick yields SkipReason, no HTTP calls
# ─────────────────────────────────────────────


def test_empty_tick_yields_skip_reason(ephemeral_instance, monkeypatch):
    """When fetch_materializations returns no records, sensor yields SkipReason."""
    posted = []

    def _fake_post(*args, **kwargs):
        posted.append(True)
        resp = MagicMock()
        resp.ok = True
        resp.json.return_value = {}
        return resp

    with patch("meridian_pipeline.sensors.requests.post", side_effect=_fake_post):
        emitted, _cursor = _run_sensor(ephemeral_instance, monkeypatch, records=[], cursor_out=None)

    assert len(posted) == 0, "No HTTP call expected on empty tick"

    from dagster import SkipReason

    assert any(isinstance(e, SkipReason) for e in emitted), "Expected SkipReason on empty tick"


# ─────────────────────────────────────────────
# Scenario 3: HTTP error (exception) is swallowed - sensor does not raise
# ─────────────────────────────────────────────


def test_http_error_is_swallowed(ephemeral_instance, monkeypatch):
    """A ConnectionError from requests.post must not propagate out of the sensor."""
    records = [_make_record("2026-06-21")]

    def _failing_post(url, params=None, headers=None, timeout=10):
        raise ConnectionError("API unreachable")

    with patch("meridian_pipeline.sensors.requests.post", side_effect=_failing_post):
        # Must not raise; sensor tick completes normally.
        emitted, new_cursor = _run_sensor(ephemeral_instance, monkeypatch, records)

    # Cursor still advances even though the HTTP call failed.
    assert new_cursor == "next-cursor-123"

    # No SkipReason - we did process records, just the HTTP side-effect failed.
    from dagster import SkipReason

    assert not any(isinstance(e, SkipReason) for e in emitted)


def test_http_non_2xx_is_swallowed(ephemeral_instance, monkeypatch):
    """A non-OK HTTP response is logged but does not raise."""
    records = [_make_record("2026-06-21")]

    def _bad_response(url, params=None, headers=None, timeout=10):
        resp = MagicMock()
        resp.ok = False
        resp.status_code = 503
        resp.text = "Service Unavailable"
        return resp

    with patch("meridian_pipeline.sensors.requests.post", side_effect=_bad_response):
        emitted, new_cursor = _run_sensor(ephemeral_instance, monkeypatch, records)

    assert new_cursor == "next-cursor-123"


# ─────────────────────────────────────────────
# Edge case: record with no partition_key is skipped
# ─────────────────────────────────────────────


def test_record_with_no_partition_key_is_skipped(ephemeral_instance, monkeypatch):
    """A materialisation without a partition_key (e.g. un-partitioned run) is ignored."""
    records = [_make_record(None)]
    posted = []

    def _fake_post(*args, **kwargs):
        posted.append(True)
        resp = MagicMock()
        resp.ok = True
        resp.json.return_value = {}
        return resp

    with patch("meridian_pipeline.sensors.requests.post", side_effect=_fake_post):
        _run_sensor(ephemeral_instance, monkeypatch, records)

    assert len(posted) == 0, "Un-partitioned record must not trigger a warm call"
