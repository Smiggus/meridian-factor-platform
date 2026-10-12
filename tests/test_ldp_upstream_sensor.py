"""
Multi-tick behavioural tests for sensors.ldp_ohlcv_upstream_sensor
(Meridian #141).

DB-free: uses an ephemeral DagsterInstance with a toy ticker-partitioned
ohlcv_prices asset (StaticPartitionsDefinition, mirroring LDP's ticker-
partitioned shape). No PostgreSQL connection is required; the engine and
DB helpers are monkeypatched.

Covers:
  1. Empty tick (no new ohlcv_prices records) -> no RunRequest, cursor stable.
  2. Fresh batch post-close on a weekday -> exactly ONE RunRequest for the
     resolved trading day, cursor advances.
  3. Second tick with same (already-seen) records -> no RunRequest (cursor
     already past them).
  4. Weekend tick with fresh records -> cursor advances, no RunRequest.
  5. Pre-close tick with fresh records -> cursor advances, no RunRequest.
  6. Cursor threads correctly across three sequential ticks.
  7. Readiness gate (Meridian #395): a fresh batch that reaches the
     post-close/weekday gates but whose target date is not yet ready per
     db.trigger_cross_section_ready fires no RunRequest, even though the
     cursor still advances past the materialisations - this sensor's whole
     premise (a single new ohlcv_prices record is sufficient signal) is
     exactly the premature-trigger inference that can fire into
     assert_cross_section_single_dated.

db.trigger_cross_section_ready is monkeypatched to (True, "") by default in
_run_tick so tests 1-6 (written before #395) exercise the same behaviour as
before; test 7 overrides it to (False, ...).
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date, datetime
from pathlib import Path

import pytest

_DAGSTER_DIR = Path(__file__).resolve().parents[1] / "dagster"
if str(_DAGSTER_DIR) not in sys.path:
    sys.path.insert(0, str(_DAGSTER_DIR))

# sensors.py reads PGPASS at _get_engine() time, not import time, but set a
# sentinel so any incidental import path stays happy.
os.environ.setdefault("PGPASS", "test-pg-pass")
os.environ.setdefault("PARTITION_START_DATE", "2010-01-02")


def _dagster_complete() -> bool:
    try:
        import dagster
    except ImportError:
        return False
    return all(
        hasattr(dagster, n)
        for n in ("RunRequest", "sensor", "AssetMaterialization", "Output", "build_sensor_context")
    )


if not _dagster_complete():
    pytest.skip("Dagster not fully installed (slim CI test image)", allow_module_level=True)

from zoneinfo import ZoneInfo  # noqa: E402

ET = ZoneInfo("America/New_York")

# A weekday post-close datetime (Tuesday 17:30 ET).
_WEEKDAY_POST_CLOSE = datetime(2026, 6, 16, 17, 30, 0, tzinfo=ET)  # Tuesday
# A weekday pre-close datetime (Tuesday 14:00 ET).
_WEEKDAY_PRE_CLOSE = datetime(2026, 6, 16, 14, 0, 0, tzinfo=ET)  # Tuesday
# A Saturday.
_WEEKEND = datetime(2026, 6, 20, 10, 0, 0, tzinfo=ET)  # Saturday

# The trading day we expect to resolve for _WEEKDAY_POST_CLOSE.
# 2026-06-16 is a Tuesday; no holiday override in mocks -> resolves to itself.
_EXPECTED_DATE = date(2026, 6, 16)

# Ticker partitions that simulate LDP's ticker-partitioned ohlcv_prices shape.
_TICKERS = ["AAPL", "MSFT", "GOOG"]


# ─────────────────────────────────────────────
# Toy ohlcv_prices asset (ticker-partitioned, DB-free)
# ─────────────────────────────────────────────


def _toy_ohlcv_asset():
    """Minimal ticker-partitioned ohlcv_prices asset for the ephemeral instance."""
    from dagster import AssetMaterialization, Output, StaticPartitionsDefinition, asset

    parts = StaticPartitionsDefinition(_TICKERS)

    @asset(name="ohlcv_prices", partitions_def=parts)
    def ohlcv_prices(context):
        for pk in context.partition_keys:
            yield AssetMaterialization(asset_key=context.asset_key, partition=pk)
        yield Output(value=None)

    return ohlcv_prices


def _materialize_tickers(asset_def, tickers, instance):
    """Write ohlcv_prices materialisations for the given tickers."""
    from dagster import materialize

    for ticker in tickers:
        assert materialize([asset_def], partition_key=ticker, instance=instance).success


# ─────────────────────────────────────────────
# Sensor runner
# ─────────────────────────────────────────────


class _FakeDatetime:
    """Replaces sensors.datetime so sensors.datetime.now(tz) returns a fixed instant."""

    def __init__(self, fixed: datetime):
        self._fixed = fixed

    def now(self, tz=None):
        return self._fixed


def _run_tick(
    instance, cursor: str | None, now_et: datetime, monkeypatch, *, ready: bool = True
) -> tuple:
    """Run one sensor evaluation; return (list[RunRequest], new_cursor_str).

    `ready` stands in for db.trigger_cross_section_ready's decision
    (Meridian #395) - defaults to True so tests written before that gate
    existed keep exercising the same pre-#395 firing behaviour.
    """
    from meridian_pipeline import sensors

    from dagster import RunRequest, build_sensor_context

    monkeypatch.setattr(sensors, "_get_engine", lambda: object())
    monkeypatch.setattr(sensors.db, "is_trading_day", lambda eng, d: True)
    monkeypatch.setattr(
        sensors.db,
        "trigger_cross_section_ready",
        lambda eng, d: (ready, "" if ready else "not ready"),
    )
    monkeypatch.setattr(sensors, "datetime", _FakeDatetime(now_et))

    ctx = build_sensor_context(instance=instance, cursor=cursor)
    emitted = list(sensors.ldp_ohlcv_upstream_sensor(ctx))
    run_requests = [e for e in emitted if isinstance(e, RunRequest)]
    return run_requests, ctx.cursor


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────


@pytest.fixture
def instance():
    from dagster import DagsterInstance

    return DagsterInstance.ephemeral()


@pytest.fixture
def ohlcv_asset():
    return _toy_ohlcv_asset()


# ─────────────────────────────────────────────
# Test 1: empty tick - no records, no RunRequest
# ─────────────────────────────────────────────


def test_empty_tick_fires_nothing(instance, ohlcv_asset, monkeypatch):
    # No materialisations at all in the instance.
    rrs, cur = _run_tick(instance, None, _WEEKDAY_POST_CLOSE, monkeypatch)

    assert rrs == [], "empty tick must fire no RunRequests"
    # Cursor must be set (not None string) even on an empty tick.
    assert cur is not None


# ─────────────────────────────────────────────
# Test 2: fresh batch post-close -> exactly one RunRequest
# ─────────────────────────────────────────────


def test_fresh_batch_post_close_fires_one_run_request(instance, ohlcv_asset, monkeypatch):
    # First tick: empty instance -> seed cursor, no RunRequest.
    rrs_0, cur_0 = _run_tick(instance, None, _WEEKDAY_POST_CLOSE, monkeypatch)
    assert rrs_0 == []

    # Materialise some tickers (simulating LDP's daily OHLCV run).
    _materialize_tickers(ohlcv_asset, ["AAPL", "MSFT"], instance)

    # Second tick: cursor advanced past seed -> picks up the new records.
    rrs, cur = _run_tick(instance, cur_0, _WEEKDAY_POST_CLOSE, monkeypatch)

    assert len(rrs) == 1, "must fire exactly one RunRequest for the resolved trading day"
    rr = rrs[0]
    assert rr.partition_key == str(_EXPECTED_DATE)
    assert rr.run_key == f"ldp_ohlcv_{_EXPECTED_DATE}"
    assert rr.tags["dagster/priority"] == "50"
    assert rr.tags["priority_class"] == "high"

    # Cursor must have advanced past the new records.
    cursor_obj = json.loads(cur)
    assert cursor_obj.get("ohlcv_cursor") is not None


# ─────────────────────────────────────────────
# Test 3: second tick with same records -> no RunRequest (cursor already past)
# ─────────────────────────────────────────────


def test_second_tick_no_new_records_fires_nothing(instance, ohlcv_asset, monkeypatch):
    # Seed, materialise, and consume the first batch.
    _rrs_0, cur_0 = _run_tick(instance, None, _WEEKDAY_POST_CLOSE, monkeypatch)
    _materialize_tickers(ohlcv_asset, ["AAPL"], instance)
    rrs_1, cur_1 = _run_tick(instance, cur_0, _WEEKDAY_POST_CLOSE, monkeypatch)
    assert len(rrs_1) == 1  # first real tick fires

    # Third tick: no new records since cur_1 -> must fire nothing.
    rrs_2, cur_2 = _run_tick(instance, cur_1, _WEEKDAY_POST_CLOSE, monkeypatch)
    assert rrs_2 == [], "second tick with no new records must fire nothing"

    # Cursor must not regress.
    assert json.loads(cur_2)["ohlcv_cursor"] == json.loads(cur_1)["ohlcv_cursor"]


# ─────────────────────────────────────────────
# Test 4: weekend tick with fresh records -> cursor advances, no RunRequest
# ─────────────────────────────────────────────


def test_weekend_tick_advances_cursor_but_no_run_request(instance, ohlcv_asset, monkeypatch):
    _rrs_0, cur_0 = _run_tick(instance, None, _WEEKEND, monkeypatch)

    # Materialise new tickers while on a weekend.
    _materialize_tickers(ohlcv_asset, ["GOOG"], instance)

    rrs, cur = _run_tick(instance, cur_0, _WEEKEND, monkeypatch)

    assert rrs == [], "weekend tick must fire no RunRequests"
    # Cursor must have advanced (so the records are not re-seen on Monday).
    cursor_obj = json.loads(cur)
    assert cursor_obj.get("ohlcv_cursor") is not None


# ─────────────────────────────────────────────
# Test 5: pre-close tick -> cursor advances, no RunRequest
# ─────────────────────────────────────────────


def test_pre_close_tick_advances_cursor_but_no_run_request(instance, ohlcv_asset, monkeypatch):
    _rrs_0, cur_0 = _run_tick(instance, None, _WEEKDAY_PRE_CLOSE, monkeypatch)

    _materialize_tickers(ohlcv_asset, ["AAPL", "MSFT"], instance)

    rrs, cur = _run_tick(instance, cur_0, _WEEKDAY_PRE_CLOSE, monkeypatch)

    assert rrs == [], "pre-close tick must fire no RunRequests"
    cursor_obj = json.loads(cur)
    assert cursor_obj.get("ohlcv_cursor") is not None


# ─────────────────────────────────────────────
# Test 6: cursor threads across 3 ticks (integration)
# ─────────────────────────────────────────────


def test_cursor_threads_across_three_ticks(instance, ohlcv_asset, monkeypatch):
    """
    Tick A: no records (empty instance) -> nothing fired, cursor seeded.
    Tick B: 2 new ticker records -> 1 RunRequest, cursor advances.
    Tick C: no records since tick B -> nothing fired, cursor preserved.
    """
    # Tick A: empty
    rrs_a, cur_a = _run_tick(instance, None, _WEEKDAY_POST_CLOSE, monkeypatch)
    assert rrs_a == []
    assert cur_a is not None

    # Tick B: 2 fresh ticker records
    _materialize_tickers(ohlcv_asset, ["AAPL", "GOOG"], instance)
    rrs_b, cur_b = _run_tick(instance, cur_a, _WEEKDAY_POST_CLOSE, monkeypatch)
    assert len(rrs_b) == 1
    assert rrs_b[0].partition_key == str(_EXPECTED_DATE)
    ohlcv_cur_b = json.loads(cur_b)["ohlcv_cursor"]
    assert ohlcv_cur_b is not None

    # Tick C: empty (cursor already past the tick-B records)
    rrs_c, cur_c = _run_tick(instance, cur_b, _WEEKDAY_POST_CLOSE, monkeypatch)
    assert rrs_c == []
    # Cursor must not regress.
    assert json.loads(cur_c)["ohlcv_cursor"] == ohlcv_cur_b


# ─────────────────────────────────────────────
# Test 7: readiness gate (Meridian #395) - not-ready defers even though
# every other gate (weekday, post-close, fresh records) would otherwise fire
# ─────────────────────────────────────────────


def test_not_ready_defers_even_with_fresh_post_close_batch(instance, ohlcv_asset, monkeypatch):
    """This sensor's whole premise - a single new ohlcv_prices materialisation
    is sufficient signal - is exactly the premature-trigger inference that
    can fire into assert_cross_section_single_dated when the upstream ingest
    finishes past this sensor's post-close gate.
    db.trigger_cross_section_ready says not-ready here even though
    every other gate is satisfied; the sensor must still fire no RunRequest,
    though the cursor still advances so the batch isn't re-processed forever."""
    rrs_0, cur_0 = _run_tick(instance, None, _WEEKDAY_POST_CLOSE, monkeypatch, ready=False)
    assert rrs_0 == []

    _materialize_tickers(ohlcv_asset, ["AAPL", "MSFT"], instance)

    rrs, cur = _run_tick(instance, cur_0, _WEEKDAY_POST_CLOSE, monkeypatch, ready=False)

    assert rrs == [], "not-ready cross-section must defer, not fire"
    cursor_obj = json.loads(cur)
    assert cursor_obj.get("ohlcv_cursor") is not None


# ─────────────────────────────────────────────
# Test 8: cross-sensor partition dedup (Meridian #358) - a run for the
# resolved trading day already QUEUED/STARTED (launched by a different
# sensor, e.g. stale_factor_sensor or request_sensor) must suppress firing.
# ─────────────────────────────────────────────


def _add_core_run(instance, partition_date, status, job_name="daily_factor_core_job"):
    from dagster._core.remote_origin import (
        RegisteredCodeLocationOrigin,
        RemoteJobOrigin,
        RemoteRepositoryOrigin,
    )
    from dagster._core.storage.dagster_run import DagsterRun

    from dagster import DagsterRunStatus

    # QUEUED runs require a remote_job_origin (DagsterRun.__new__ enforces
    # this - it's how a real queued run would eventually be launched). A
    # minimal fabricated origin satisfies the check without needing a real
    # code location.
    remote_job_origin = None
    if status == DagsterRunStatus.QUEUED:
        loc = RegisteredCodeLocationOrigin(location_name="test_location")
        repo = RemoteRepositoryOrigin(code_location_origin=loc, repository_name="test_repo")
        remote_job_origin = RemoteJobOrigin(repository_origin=repo, job_name=job_name)

    instance.run_storage.add_run(
        DagsterRun(
            job_name=job_name,
            status=status,
            tags={"dagster/partition": str(partition_date)},
            remote_job_origin=remote_job_origin,
        )
    )


@pytest.mark.parametrize(
    "status_name",
    ["NOT_STARTED", "QUEUED", "STARTING", "STARTED"],
)
def test_does_not_fire_when_partition_already_in_flight(
    status_name, instance, ohlcv_asset, monkeypatch
):
    """Every other gate (weekday, post-close, fresh records, readiness) is
    satisfied, but a daily_factor_core_job run for the resolved date is
    already QUEUED/STARTED - launched by a different sensor that beat this
    one to it (Meridian #358: a race between the two sensors). Fails on
    pre-#358 code, which has no notion of runs
    launched by another sensor.

    Parametrized (rather than looped) so each status gets a fresh instance -
    a shared instance across iterations would leak materialisations and runs
    from earlier statuses into later ones."""
    from dagster import DagsterRunStatus

    status = getattr(DagsterRunStatus, status_name)

    rrs_0, cur_0 = _run_tick(instance, None, _WEEKDAY_POST_CLOSE, monkeypatch)
    assert rrs_0 == []

    _materialize_tickers(ohlcv_asset, ["AAPL", "MSFT"], instance)
    _add_core_run(instance, _EXPECTED_DATE, status)

    rrs, _cur = _run_tick(instance, cur_0, _WEEKDAY_POST_CLOSE, monkeypatch)
    assert rrs == [], f"must not fire while a {status} run is in flight"


@pytest.mark.parametrize("status_name", ["SUCCESS", "FAILURE"])
def test_fires_when_only_prior_run_for_date_is_terminal(
    status_name, instance, ohlcv_asset, monkeypatch
):
    """A prior daily_factor_core_job run for the resolved date exists but
    ended SUCCESS or FAILURE - not a duplicate risk, and a failed run must
    remain retriable."""
    from dagster import DagsterRunStatus

    status = getattr(DagsterRunStatus, status_name)

    rrs_0, cur_0 = _run_tick(instance, None, _WEEKDAY_POST_CLOSE, monkeypatch)
    assert rrs_0 == []

    _materialize_tickers(ohlcv_asset, ["AAPL", "MSFT"], instance)
    _add_core_run(instance, _EXPECTED_DATE, status)

    rrs, _cur = _run_tick(instance, cur_0, _WEEKDAY_POST_CLOSE, monkeypatch)
    assert len(rrs) == 1, f"a {status} prior run must not block a retry"
    assert rrs[0].partition_key == str(_EXPECTED_DATE)
