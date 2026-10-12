"""
Tests for sensors.stale_factor_sensor's readiness gate (Meridian #286, #395).

stale_factor_sensor is now the SECONDARY safety net behind
watermark_complete_sensor (the primary trigger, tested in
test_watermark_complete_sensor.py). Before #286 it fired purely on a 30-min
timer + "no scores yet" check, with no notion of whether LDP considered the
date's ingest complete - which meant it could race the same premature-scoring
failure the old daily_factor_scoring cron hit, and because its run_key is
date-scoped (not tick-scoped), a failed run would dedup-lock every later tick
out of ever retrying that date. #286 added a watermark consult that deferred
on an explicit NOT-complete answer but proceeded on an UNKNOWN one (no row
yet for this date) - which #395 recognised as indistinguishable
from "still mid-ingest" (LDP only writes the watermark row once a date's
ingest actually finishes) and able to fire straight into
assert_cross_section_single_dated's guard. The sensor now
consults db.trigger_cross_section_ready (watermark-first, falling back to the
SAME coverage-ratio predicate the guard applies) instead of the raw
watermark status.

Covers:
  - fires when scores are missing and the predicate says ready
  - does NOT fire when the predicate says not ready (covers both the
    explicit-watermark-incomplete case and the watermark-unknown-but-
    coverage-looks-mid-ingest case - the sensor treats them identically,
    by construction, since both flow through the same predicate)
  - stays quiet when scores already exist, regardless of predicate state
  - weekday/post-close gates and holiday rollback are unaffected

DB-free: db presence/readiness checks are monkeypatched; wall-clock is
pinned by patching sensors.datetime (mirrors test_stale_composite_sensor.py).
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
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


def _fake_clock(monkeypatch, sensors, *, year=2026, month=5, day=28, hour=17, minute=30):
    """2026-05-28 is a Thursday; 17:30 ET is past the 17:00 post-close gate."""
    real = datetime

    class _FakeDateTime:
        @classmethod
        def now(cls, tz=None):
            return real(year, month, day, hour, minute, tzinfo=tz)

    monkeypatch.setattr(sensors, "datetime", _FakeDateTime)


def _run(
    monkeypatch,
    sensors,
    *,
    has_scores: bool,
    ready,  # True | False - db.trigger_cross_section_ready's first return value
    clock=None,
    is_trading: bool = True,
    instance=None,
):
    from dagster import DagsterInstance, RunRequest, build_sensor_context

    monkeypatch.setattr(sensors, "_get_engine", lambda: object())
    monkeypatch.setattr(sensors.db, "has_scores_for_date", lambda eng, d: has_scores)
    monkeypatch.setattr(sensors.db, "is_trading_day", lambda eng, d: is_trading)
    monkeypatch.setattr(
        sensors.db,
        "trigger_cross_section_ready",
        lambda eng, d: (ready, "" if ready else "not ready"),
    )
    if clock is None:
        _fake_clock(monkeypatch, sensors)
    else:
        _fake_clock(monkeypatch, sensors, **clock)

    # Meridian #358: the sensor now consults context.instance for the
    # cross-sensor in-flight guard. Default to a fresh, run-free ephemeral
    # instance so tests that don't care about this guard see "nothing in
    # flight" (unchanged pre-#358 behaviour).
    ctx = build_sensor_context(
        instance=instance if instance is not None else DagsterInstance.ephemeral()
    )
    return [e for e in sensors.stale_factor_sensor(ctx) if isinstance(e, RunRequest)]


def test_fires_when_scores_missing_and_predicate_ready(monkeypatch):
    from meridian_pipeline import sensors

    rrs = _run(monkeypatch, sensors, has_scores=False, ready=True)
    assert len(rrs) == 1
    assert rrs[0].partition_key == "2026-05-28"
    assert rrs[0].run_key == "stale_2026-05-28"
    assert rrs[0].tags["dagster/priority"] == "50"
    assert rrs[0].tags["priority_class"] == "high"


def test_does_not_fire_when_predicate_says_not_ready(monkeypatch):
    """The gate (Meridian #286, tightened #395): a not-ready answer - whether
    from an explicit watermark-incomplete or a mid-ingest coverage-ratio
    reading - must defer rather than fire and hit the same
    assert_cross_section_single_dated failure the old cron did, and because
    run_key is date-scoped, firing here on a bad date would permanently
    dedup-lock every future tick out of retrying it."""
    from meridian_pipeline import sensors

    rrs = _run(monkeypatch, sensors, has_scores=False, ready=False)
    assert rrs == []


def test_quiet_when_scores_already_present_regardless_of_readiness(monkeypatch):
    from meridian_pipeline import sensors

    for ready in (True, False):
        assert _run(monkeypatch, sensors, has_scores=True, ready=ready) == []


def test_readiness_not_consulted_when_scores_already_present(monkeypatch):
    """The readiness predicate must be short-circuited by the has_scores
    check - no reason to touch it at all once scores already exist."""
    from meridian_pipeline import sensors

    from dagster import build_sensor_context

    def _boom(eng, d):
        raise AssertionError("readiness predicate should not have been consulted")

    monkeypatch.setattr(sensors, "_get_engine", lambda: object())
    monkeypatch.setattr(sensors.db, "has_scores_for_date", lambda eng, d: True)
    monkeypatch.setattr(sensors.db, "is_trading_day", lambda eng, d: True)
    monkeypatch.setattr(sensors.db, "trigger_cross_section_ready", _boom)
    _fake_clock(monkeypatch, sensors)

    ctx = build_sensor_context(instance=None)
    rrs = list(sensors.stale_factor_sensor(ctx))
    assert rrs == []


def test_quiet_before_close(monkeypatch):
    from meridian_pipeline import sensors

    rrs = _run(
        monkeypatch,
        sensors,
        has_scores=False,
        ready=True,
        clock={"hour": 16, "minute": 0},
    )
    assert rrs == []


def test_quiet_on_weekend(monkeypatch):
    from meridian_pipeline import sensors

    # 2026-05-30 is a Saturday.
    rrs = _run(
        monkeypatch,
        sensors,
        has_scores=False,
        ready=True,
        clock={"day": 30, "hour": 17, "minute": 30},
    )
    assert rrs == []


def test_holiday_rolls_back_and_readiness_gate_applies_to_rolled_back_date(monkeypatch):
    """2026-05-25 is Memorial Day (Monday). The sensor fires at 17:30 ET that
    day; _latest_trading_day_et returns 2026-05-25, then _ensure_trading_day
    rolls back to 2026-05-22 (the prior Friday). The readiness gate must key
    off the ROLLED-BACK date, not the holiday."""
    from datetime import date

    from meridian_pipeline import sensors

    from dagster import DagsterInstance, RunRequest, build_sensor_context

    holiday = date(2026, 5, 25)
    rolled_back = date(2026, 5, 22)

    def _is_trading(eng, d):
        return d != holiday

    ready_by_date = {rolled_back: False}

    monkeypatch.setattr(sensors, "_get_engine", lambda: object())
    monkeypatch.setattr(sensors.db, "has_scores_for_date", lambda eng, d: False)
    monkeypatch.setattr(sensors.db, "is_trading_day", _is_trading)
    monkeypatch.setattr(
        sensors.db,
        "trigger_cross_section_ready",
        lambda eng, d: (ready_by_date.get(d, True), ""),
    )
    _fake_clock(monkeypatch, sensors, day=25, month=5, hour=17, minute=30)

    ctx = build_sensor_context(instance=DagsterInstance.ephemeral())
    rrs = [e for e in sensors.stale_factor_sensor(ctx) if isinstance(e, RunRequest)]
    # Predicate says the rolled-back date (May 22) is not ready -> quiet.
    assert rrs == []

    # Once the date is ready, the same rolled-back date fires.
    ready_by_date[rolled_back] = True
    ctx2 = build_sensor_context(instance=DagsterInstance.ephemeral())
    rrs2 = [e for e in sensors.stale_factor_sensor(ctx2) if isinstance(e, RunRequest)]
    assert len(rrs2) == 1
    assert rrs2[0].partition_key == "2026-05-22"


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


def test_does_not_fire_when_partition_already_in_flight(monkeypatch):
    """A different sensor (e.g. ldp_ohlcv_upstream_sensor) already has a
    daily_factor_core_job run QUEUED/STARTED for this date - stale_factor_sensor
    must not also fire. Fails on pre-#358 code: it only dedups via its own
    date-scoped run_key and has no awareness of other sensors' runs."""
    from meridian_pipeline import sensors

    from dagster import DagsterRunStatus

    for status in (
        DagsterRunStatus.NOT_STARTED,
        DagsterRunStatus.QUEUED,
        DagsterRunStatus.STARTING,
        DagsterRunStatus.STARTED,
    ):
        instance = _instance_with_run("daily_factor_core_job", "2026-05-28", status)
        rrs = _run(monkeypatch, sensors, has_scores=False, ready=True, instance=instance)
        assert rrs == [], f"must not fire while a {status} run is in flight"


def test_fires_when_only_prior_run_for_date_is_terminal(monkeypatch):
    """A prior run for this date exists but ended SUCCESS or FAILURE - not a
    duplicate risk, and a failed run must remain retriable (run_key alone
    would already allow this; the in-flight guard must not regress it)."""
    from meridian_pipeline import sensors

    from dagster import DagsterRunStatus

    for status in (DagsterRunStatus.SUCCESS, DagsterRunStatus.FAILURE):
        instance = _instance_with_run("daily_factor_core_job", "2026-05-28", status)
        rrs = _run(monkeypatch, sensors, has_scores=False, ready=True, instance=instance)
        assert len(rrs) == 1, f"a {status} prior run must not block a retry"
        assert rrs[0].partition_key == "2026-05-28"
