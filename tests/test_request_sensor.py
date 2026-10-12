"""
Tests for sensors.request_sensor's cross-sensor in-flight guard (Meridian
#358).

request_sensor polls factors.compute_requests for pending dates and fires
daily_factor_core_job for each. Before #358 it deduped only via its own
per-date run_key, so a date that another sensor (watermark_complete_sensor,
stale_factor_sensor, ldp_ohlcv_upstream_sensor) had already launched a run
for would get a SECOND concurrent run from this sensor. The sensor now
consults the shared _partition_in_flight helper before firing.

DB-free: db.get_pending_requests / db.mark_request are monkeypatched; the
in-flight check runs against a real ephemeral DagsterInstance seeded with
fake DagsterRun rows, mirroring test_watermark_complete_sensor.py and
test_stale_factor_sensor.py.
"""

from __future__ import annotations

import os
import sys
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


def _run(monkeypatch, sensors, *, pending, instance=None, marked=None):
    from dagster import DagsterInstance, RunRequest, build_sensor_context

    monkeypatch.setattr(sensors, "_get_engine", lambda: object())
    monkeypatch.setattr(sensors.db, "get_pending_requests", lambda eng: list(pending))

    marked_calls = marked if marked is not None else []

    def _mark(eng, score_date, status, error_msg=None):
        marked_calls.append((score_date, status))

    monkeypatch.setattr(sensors.db, "mark_request", _mark)

    ctx = build_sensor_context(
        instance=instance if instance is not None else DagsterInstance.ephemeral()
    )
    events = list(sensors.request_sensor(ctx))
    return [e for e in events if isinstance(e, RunRequest)], marked_calls


def test_fires_for_pending_date_with_nothing_in_flight(monkeypatch):
    from meridian_pipeline import sensors

    rrs, marked = _run(monkeypatch, sensors, pending=["2026-06-01"])

    assert len(rrs) == 1
    assert rrs[0].partition_key == "2026-06-01"
    assert rrs[0].run_key == "request_2026-06-01"
    assert rrs[0].tags["dagster/priority"] == "50"
    assert rrs[0].tags["priority_class"] == "high"
    assert marked == [("2026-06-01", "computing")]


@pytest.mark.parametrize("status_name", ["NOT_STARTED", "QUEUED", "STARTING", "STARTED"])
def test_skips_when_partition_already_in_flight(status_name, monkeypatch):
    """A different sensor already has a daily_factor_core_job run in flight
    for this date - request_sensor must not also fire, and must NOT mark
    the request 'computing' (it stays 'pending' so composite_scores' later
    unconditional mark-done still resolves it). Fails on pre-#358 code."""
    from meridian_pipeline import sensors

    from dagster import DagsterRunStatus

    status = getattr(DagsterRunStatus, status_name)
    instance = _instance_with_run("daily_factor_core_job", "2026-06-01", status)

    rrs, marked = _run(monkeypatch, sensors, pending=["2026-06-01"], instance=instance)

    assert rrs == [], f"must not fire while a {status} run is in flight"
    assert marked == [], "must not mark 'computing' when skipping - leave the request pending"


@pytest.mark.parametrize("status_name", ["SUCCESS", "FAILURE"])
def test_fires_when_only_prior_run_for_date_is_terminal(status_name, monkeypatch):
    """A prior run for this date exists but ended SUCCESS or FAILURE - not a
    duplicate risk, and a failed run must remain retriable."""
    from meridian_pipeline import sensors

    from dagster import DagsterRunStatus

    status = getattr(DagsterRunStatus, status_name)
    instance = _instance_with_run("daily_factor_core_job", "2026-06-01", status)

    rrs, marked = _run(monkeypatch, sensors, pending=["2026-06-01"], instance=instance)

    assert len(rrs) == 1, f"a {status} prior run must not block a retry"
    assert rrs[0].partition_key == "2026-06-01"
    assert marked == [("2026-06-01", "computing")]


def test_in_flight_check_is_per_date(monkeypatch):
    """Two pending dates, only one has a run in flight - the other must
    still fire. Proves the guard is keyed on partition, not a blanket halt."""
    from meridian_pipeline import sensors

    from dagster import DagsterRunStatus

    instance = _instance_with_run("daily_factor_core_job", "2026-06-01", DagsterRunStatus.STARTED)

    rrs, marked = _run(
        monkeypatch, sensors, pending=["2026-06-01", "2026-06-02"], instance=instance
    )

    assert [r.partition_key for r in rrs] == ["2026-06-02"]
    assert marked == [("2026-06-02", "computing")]
