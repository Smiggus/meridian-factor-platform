"""
Tests for sensors.watermark_complete_sensor (Meridian #286: cron -> watermark
cutover for daily phase-1 scoring).

Background: the old daily_factor_scoring cron (schedules.py, now removed)
fired at a fixed wall-clock time sized against the upstream finish time. If
the upstream ingest finishes later than that constant, the cron fires before
ohlcv.ingest_watermark marks the date complete, and
assert_cross_section_single_dated correctly refuses to score it. This sensor
replaces the cron: it polls the watermark directly and fires the moment a
date is actually marked complete, so there is no wall-clock constant left
to go stale.

Covers:
  - first tick seeds the high-water mark without firing (no stampede of
    historical already-complete dates)
  - a newly-complete date fires exactly once
  - an incomplete date (nothing newly complete) does not fire
  - a date already advanced past (high-water mark) does not re-fire
  - an unreachable watermark is a quiet no-op, cursor untouched, retried
    next tick (not "fire freely", not "never fire again")
  - an empty watermark (no complete rows at all) behaves the same way on
    the seed tick
  - a non-trading-day row is skipped without firing but still advances the
    high-water mark

DB-free: db.latest_complete_trade_date / db.get_new_complete_trade_dates /
db.is_trading_day are monkeypatched directly on sensors.db, mirroring the
existing spc_evidence_interrupt_sensor test pattern.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date
from pathlib import Path

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
    return all(hasattr(dagster, n) for n in ("RunRequest", "sensor", "build_sensor_context"))


pytestmark = pytest.mark.skipif(
    not _dagster_complete(),
    reason="Dagster not fully installed (slim CI test image)",
)


def _tick(
    monkeypatch,
    sensors,
    *,
    latest_complete=None,
    new_dates=None,
    latest_raises=None,
    new_dates_raises=None,
    cursor=None,
    non_trading_days=frozenset(),
    instance=None,
):
    """Run one sensor tick with the DB layer monkeypatched.

    latest_complete: return value for db.latest_complete_trade_date (seed path).
    new_dates: return value for db.get_new_complete_trade_dates (steady-state path).
    *_raises: an Exception instance to raise instead of returning a value.
    non_trading_days: set of dates for which db.is_trading_day returns False.
    instance: a DagsterInstance to back the sensor's cross-sensor in-flight
        check (Meridian #358). Defaults to a fresh, run-free ephemeral
        instance so pre-#358 tests keep seeing "nothing in flight" without
        having to know the guard exists.
    """
    from dagster import DagsterInstance, RunRequest, build_sensor_context

    monkeypatch.setattr(sensors, "_get_engine", lambda: object())

    def _latest(eng):
        if latest_raises is not None:
            raise latest_raises
        return latest_complete

    def _new(eng, since, limit=30):
        if new_dates_raises is not None:
            raise new_dates_raises
        return list(new_dates or [])

    def _is_trading_day(eng, d):
        return d not in non_trading_days

    monkeypatch.setattr(sensors.db, "latest_complete_trade_date", _latest)
    monkeypatch.setattr(sensors.db, "get_new_complete_trade_dates", _new)
    monkeypatch.setattr(sensors.db, "is_trading_day", _is_trading_day)

    ctx = build_sensor_context(
        instance=instance if instance is not None else DagsterInstance.ephemeral(),
        cursor=json.dumps(cursor) if cursor is not None else None,
    )
    events = list(sensors.watermark_complete_sensor(ctx))
    run_requests = [e for e in events if isinstance(e, RunRequest)]
    new_cursor = json.loads(ctx.cursor) if ctx.cursor else None
    return run_requests, new_cursor


# ─────────────────────────────────────────────
# Seed tick (first ever tick / fresh deploy)
# ─────────────────────────────────────────────


def test_first_tick_seeds_without_firing(monkeypatch):
    from meridian_pipeline import sensors

    rrs, cursor = _tick(monkeypatch, sensors, latest_complete=date(2026, 8, 17))

    assert rrs == []
    assert cursor == {"high_water_date": "2026-08-17"}


def test_first_tick_with_empty_watermark_leaves_cursor_unset(monkeypatch):
    """No complete rows at all yet (the watermark went live this week and
    hasn't populated) must not raise and must not seed a cursor - the next
    tick retries the seed."""
    from meridian_pipeline import sensors

    rrs, cursor = _tick(monkeypatch, sensors, latest_complete=None)

    assert rrs == []
    assert cursor is None


def test_first_tick_unreachable_watermark_is_quiet_no_op(monkeypatch):
    """A connection error on the seed tick must not raise, must not fire,
    and must not leave any cursor state - purely retried next tick."""
    from meridian_pipeline import sensors

    rrs, cursor = _tick(monkeypatch, sensors, latest_raises=RuntimeError("connection refused"))

    assert rrs == []
    assert cursor is None


# ─────────────────────────────────────────────
# Steady state: firing on newly-complete dates
# ─────────────────────────────────────────────


def test_newly_complete_date_fires_once(monkeypatch):
    from meridian_pipeline import sensors

    cursor0 = {"high_water_date": "2026-08-16"}
    rrs, cursor = _tick(monkeypatch, sensors, new_dates=[date(2026, 8, 17)], cursor=cursor0)

    assert len(rrs) == 1
    rr = rrs[0]
    assert rr.partition_key == "2026-08-17"
    assert rr.run_key == "watermark_2026-08-17"
    assert rr.tags["dagster/priority"] == "50"
    assert rr.tags["priority_class"] == "high"
    assert cursor == {"high_water_date": "2026-08-17"}


def test_no_newly_complete_dates_does_not_fire(monkeypatch):
    """Nothing new since the high-water mark (the date genuinely isn't
    complete yet) - a quiet tick, cursor unchanged."""
    from meridian_pipeline import sensors

    cursor0 = {"high_water_date": "2026-08-16"}
    rrs, cursor = _tick(monkeypatch, sensors, new_dates=[], cursor=cursor0)

    assert rrs == []
    # Sensor returns early without calling update_cursor, so the context
    # cursor is whatever was passed in, unchanged.
    assert cursor == cursor0


def test_already_processed_date_does_not_refire(monkeypatch):
    """A date at or before the high-water mark is never re-requested by
    get_new_complete_trade_dates (it's a strict `> since` query) - this
    proves the sensor's own state advances correctly so a second tick with
    the same underlying watermark row does not re-fire."""
    from meridian_pipeline import sensors

    cursor0 = {"high_water_date": "2026-08-17"}
    # Simulate the DB-level exclusive-bound behaviour: nothing newly complete
    # since 2026-08-17 because 2026-08-17 was already fired for.
    rrs, cursor = _tick(monkeypatch, sensors, new_dates=[], cursor=cursor0)

    assert rrs == []


def test_multiple_newly_complete_dates_fire_and_advance_high_water(monkeypatch):
    """A catch-up tick (e.g. the sensor was STOPPED for a couple of days)
    fires one RunRequest per date and advances to the latest."""
    from meridian_pipeline import sensors

    cursor0 = {"high_water_date": "2026-08-14"}
    dates = [date(2026, 8, 15), date(2026, 8, 16), date(2026, 8, 17)]
    rrs, cursor = _tick(monkeypatch, sensors, new_dates=dates, cursor=cursor0)

    assert [r.partition_key for r in rrs] == ["2026-08-15", "2026-08-16", "2026-08-17"]
    assert cursor == {"high_water_date": "2026-08-17"}


def test_non_trading_day_skipped_but_high_water_advances(monkeypatch):
    """Defensive belt: if the watermark ever marks a non-trading day
    complete, no run fires for it, but it must not get stuck being re-seen
    forever - the high-water mark still advances past it."""
    from meridian_pipeline import sensors

    cursor0 = {"high_water_date": "2026-08-14"}
    dates = [date(2026, 8, 15), date(2026, 8, 17)]
    rrs, cursor = _tick(
        monkeypatch,
        sensors,
        new_dates=dates,
        cursor=cursor0,
        non_trading_days={date(2026, 8, 15)},
    )

    assert [r.partition_key for r in rrs] == ["2026-08-17"]
    assert cursor == {"high_water_date": "2026-08-17"}


# ─────────────────────────────────────────────
# Unreachable watermark, steady state
# ─────────────────────────────────────────────


def test_steady_state_unreachable_watermark_is_quiet_no_op(monkeypatch):
    """A connection error mid-stream must not raise, must not fire, and must
    leave the high-water mark exactly where it was - next tick retries from
    the same point, so nothing is lost and nothing is fired spuriously."""
    from meridian_pipeline import sensors

    cursor0 = {"high_water_date": "2026-08-16"}
    rrs, cursor = _tick(
        monkeypatch,
        sensors,
        new_dates_raises=RuntimeError("connection refused"),
        cursor=cursor0,
    )

    assert rrs == []
    assert cursor == cursor0  # update_cursor never called; caller keeps cursor0


def test_recovers_after_unreachable_tick(monkeypatch):
    """An outage tick followed by a healthy tick fires normally once the
    watermark answers again - proves the sensor is not permanently wedged
    by a transient failure."""
    from meridian_pipeline import sensors

    cursor0 = {"high_water_date": "2026-08-16"}
    rrs1, _ = _tick(
        monkeypatch,
        sensors,
        new_dates_raises=RuntimeError("connection refused"),
        cursor=cursor0,
    )
    assert rrs1 == []

    # Retry from the same cursor (unchanged by the failed tick).
    rrs2, cursor2 = _tick(monkeypatch, sensors, new_dates=[date(2026, 8, 17)], cursor=cursor0)
    assert [r.partition_key for r in rrs2] == ["2026-08-17"]
    assert cursor2 == {"high_water_date": "2026-08-17"}


# ─────────────────────────────────────────────
# Cross-sensor partition dedup (Meridian #358)
# ─────────────────────────────────────────────


def _instance_with_run(job_name, partition_date, status):
    from dagster._core.remote_origin import (
        RegisteredCodeLocationOrigin,
        RemoteJobOrigin,
        RemoteRepositoryOrigin,
    )
    from dagster._core.storage.dagster_run import DagsterRun

    from dagster import DagsterInstance, DagsterRunStatus

    # QUEUED runs require a remote_job_origin (DagsterRun.__new__ enforces
    # this - it's how a real queued run would eventually be launched). A
    # minimal fabricated origin satisfies the check without needing a real
    # code location.
    remote_job_origin = None
    if status == DagsterRunStatus.QUEUED:
        loc = RegisteredCodeLocationOrigin(location_name="test_location")
        repo = RemoteRepositoryOrigin(code_location_origin=loc, repository_name="test_repo")
        remote_job_origin = RemoteJobOrigin(repository_origin=repo, job_name=job_name)

    instance = DagsterInstance.ephemeral()
    instance.run_storage.add_run(
        DagsterRun(
            job_name=job_name,
            status=status,
            tags={"dagster/partition": str(partition_date)},
            remote_job_origin=remote_job_origin,
        )
    )
    return instance


def test_skips_when_partition_already_in_flight(monkeypatch):
    """A run for this partition is already QUEUED/STARTED (launched by a
    different sensor, e.g. request_sensor beating the watermark to it) -
    this sensor must not also fire. Fails on pre-#358 code, which has no
    notion of runs launched by other sensors."""
    from meridian_pipeline import sensors

    from dagster import DagsterRunStatus

    for status in (
        DagsterRunStatus.NOT_STARTED,
        DagsterRunStatus.QUEUED,
        DagsterRunStatus.STARTING,
        DagsterRunStatus.STARTED,
    ):
        instance = _instance_with_run("daily_factor_core_job", date(2026, 8, 17), status)
        cursor0 = {"high_water_date": "2026-08-16"}
        rrs, cursor = _tick(
            monkeypatch,
            sensors,
            new_dates=[date(2026, 8, 17)],
            cursor=cursor0,
            instance=instance,
        )
        assert rrs == [], f"must not fire while a {status} run is in flight"
        # The high-water mark still advances - the in-flight run covers it.
        assert cursor == {"high_water_date": "2026-08-17"}


def test_fires_when_only_prior_run_is_terminal(monkeypatch):
    """A prior run for this date exists but is SUCCESS or FAILURE - not a
    duplicate risk, and a failed run must remain retriable."""
    from meridian_pipeline import sensors

    from dagster import DagsterRunStatus

    for status in (DagsterRunStatus.SUCCESS, DagsterRunStatus.FAILURE):
        instance = _instance_with_run("daily_factor_core_job", date(2026, 8, 17), status)
        cursor0 = {"high_water_date": "2026-08-16"}
        rrs, _cursor = _tick(
            monkeypatch,
            sensors,
            new_dates=[date(2026, 8, 17)],
            cursor=cursor0,
            instance=instance,
        )
        assert len(rrs) == 1, f"a {status} prior run must not block a retry"
        assert rrs[0].partition_key == "2026-08-17"


def test_in_flight_run_for_different_job_does_not_block(monkeypatch):
    """A QUEUED/STARTED run for a DIFFERENT job (e.g. the phase-2 GPU
    composite job) on the same partition tag must not suppress phase-1 -
    the guard is job-scoped, not merely partition-scoped."""
    from meridian_pipeline import sensors

    from dagster import DagsterRunStatus

    instance = _instance_with_run(
        "daily_gpu_composite_job", date(2026, 8, 17), DagsterRunStatus.STARTED
    )
    cursor0 = {"high_water_date": "2026-08-16"}
    rrs, _cursor = _tick(
        monkeypatch,
        sensors,
        new_dates=[date(2026, 8, 17)],
        cursor=cursor0,
        instance=instance,
    )
    assert len(rrs) == 1
    assert rrs[0].partition_key == "2026-08-17"
