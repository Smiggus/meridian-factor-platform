"""
Multi-tick idempotency tests for sensors.iceberg_gold_sync_sensor (Meridian#265).

Drives the real sensor over an ephemeral sensor context, threading the JSON
cursor between ticks the way the daemon does, with the gold promotion table's
current snapshot id stubbed via lakehouse.gold_promotion_current_snapshot_id.

Contract:
  * no snapshot change -> SKIP (no RunRequest, cursor unchanged);
  * a new snapshot -> exactly ONE RunRequest with run_key = snapshot_id and the
    snapshot_id threaded into run config, QUEUED in the cursor (not
    acknowledged: FND-004);
  * last_snapshot_id advances only once the run for that snapshot SUCCEEDED;
  * a failed run is re-requested under a new run_key, rate-limited and capped,
    then logged at ERROR; a run in flight is waited for;
  * the same snapshot again -> no duplicate RunRequest;
  * table unreachable/empty (current id None) -> no-op tick.

Dagster is only partially installed on the slim CI image, so the module skips
there via a hasattr check (NOT importorskip - see
a repo convention for slim CI). Lint is the real gate in CI.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

_DAGSTER_DIR = Path(__file__).resolve().parents[1] / "dagster"
if str(_DAGSTER_DIR) not in sys.path:
    sys.path.insert(0, str(_DAGSTER_DIR))

# sensors.py reads PGPASS lazily (at _get_engine()), but set a sentinel so any
# incidental import path stays happy.
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


T0 = datetime(2026, 10, 9, 12, 0, tzinfo=timezone.utc)
S1 = 5941684647689869972
S2 = 3696722753718291214


def _run_tick(cursor, snapshot_id, monkeypatch, *, state="none", now=T0):
    """Run one sensor evaluation with a stubbed current snapshot id and a stubbed
    run state for the queued snapshot; return (list[RunRequest], new_cursor_str)."""
    from meridian_pipeline import lakehouse, sensors

    from dagster import DagsterInstance, RunRequest, build_sensor_context

    monkeypatch.setattr(lakehouse, "gold_promotion_current_snapshot_id", lambda: snapshot_id)
    monkeypatch.setattr(sensors, "_gold_snapshot_run_state", lambda inst, sid: state)
    monkeypatch.setattr(sensors, "_queue_now", lambda: now)
    ctx = build_sensor_context(instance=DagsterInstance.ephemeral(), cursor=cursor)
    emitted = list(sensors.iceberg_gold_sync_sensor(ctx))
    run_requests = [e for e in emitted if isinstance(e, RunRequest)]
    return run_requests, ctx.cursor


def test_snapshot_is_queued_not_acknowledged_on_enqueue(monkeypatch):
    """The old sensor stored last_snapshot_id=current the moment it yielded, so
    a run that then failed was never seen again (FND-004)."""
    reqs, cursor = _run_tick(None, S1, monkeypatch)
    assert len(reqs) == 1
    assert reqs[0].run_key == str(S1)
    cfg = reqs[0].run_config["ops"]["gold_scores_pg_sync"]["config"]
    assert cfg["snapshot_id"] == S1
    assert reqs[0].tags["meridian/gold_snapshot_id"] == str(S1)
    c = json.loads(cursor)
    assert c["last_snapshot_id"] is None
    assert c["queued"]["snapshot_id"] == S1 and c["queued"]["attempts"] == 1


def test_success_acknowledges_and_stops_firing(monkeypatch):
    _, cursor = _run_tick(None, S1, monkeypatch)
    reqs, cursor2 = _run_tick(cursor, S1, monkeypatch, state="success")
    assert reqs == []
    c = json.loads(cursor2)
    assert c["last_snapshot_id"] == S1 and c["queued"] is None
    # and it stays quiet afterwards
    reqs, cursor3 = _run_tick(cursor2, S1, monkeypatch, state="none")
    assert reqs == [] and cursor3 == cursor2


def test_success_then_new_snapshot_fires_in_same_tick(monkeypatch):
    _, cursor = _run_tick(None, S1, monkeypatch)
    reqs, cursor2 = _run_tick(cursor, S2, monkeypatch, state="success")
    assert [r.run_key for r in reqs] == [str(S2)]
    c = json.loads(cursor2)
    assert c["last_snapshot_id"] == S1 and c["queued"]["snapshot_id"] == S2


def test_in_flight_run_is_waited_for(monkeypatch):
    _, cursor = _run_tick(None, S1, monkeypatch)
    later = T0 + timedelta(hours=5)
    reqs, cursor2 = _run_tick(cursor, S1, monkeypatch, state="in_flight", now=later)
    assert reqs == [] and cursor2 == cursor


def test_failed_run_is_rerequested_with_new_run_key_after_spacing(monkeypatch):
    """Regression for FND-004: on the old code the failed snapshot was acked at
    enqueue and the next tick returned zero requests forever."""
    from meridian_pipeline import sensors

    _, cursor = _run_tick(None, S1, monkeypatch)
    later = T0 + timedelta(minutes=sensors._QUEUE_RETRY_MINUTES + 1)
    reqs, cursor2 = _run_tick(cursor, S1, monkeypatch, state="failed", now=later)
    assert len(reqs) == 1
    assert reqs[0].run_key == f"{S1}_a{int(later.timestamp() // 60)}"
    assert reqs[0].run_config["ops"]["gold_scores_pg_sync"]["config"]["snapshot_id"] == S1
    c = json.loads(cursor2)
    assert c["last_snapshot_id"] is None and c["queued"]["attempts"] == 2


def test_failed_retry_is_rate_limited(monkeypatch):
    _, cursor = _run_tick(None, S1, monkeypatch)
    reqs, cursor2 = _run_tick(
        cursor, S1, monkeypatch, state="failed", now=T0 + timedelta(minutes=10)
    )
    assert reqs == [] and cursor2 == cursor


def test_retry_cap_stops_requests_and_keeps_snapshot_queued(monkeypatch):
    from meridian_pipeline import sensors

    _, cursor = _run_tick(None, S1, monkeypatch)
    now = T0
    for _ in range(sensors._SENSOR_MAX_ATTEMPTS - 1):
        now += timedelta(minutes=sensors._QUEUE_RETRY_MINUTES + 1)
        reqs, cursor = _run_tick(cursor, S1, monkeypatch, state="failed", now=now)
        assert len(reqs) == 1
    assert json.loads(cursor)["queued"]["attempts"] == sensors._SENSOR_MAX_ATTEMPTS

    now += timedelta(minutes=sensors._QUEUE_RETRY_MINUTES + 1)
    reqs, cursor = _run_tick(cursor, S1, monkeypatch, state="failed", now=now)
    assert reqs == []
    assert json.loads(cursor)["queued"]["gave_up"] is True
    # still queued (never dropped), and not re-requested on later ticks
    now += timedelta(hours=3)
    reqs, cursor_after = _run_tick(cursor, S1, monkeypatch, state="failed", now=now)
    assert reqs == []
    after = json.loads(cursor_after)
    assert after["last_snapshot_id"] is None and after["queued"]["snapshot_id"] == S1
    assert after["queued"]["attempts"] == sensors._SENSOR_MAX_ATTEMPTS


def test_newer_snapshot_never_supersedes_a_failed_one(monkeypatch):
    """Snapshots carry different owned_factors scopes (read_gold_snapshot reads
    each by its own scope), so abandoning a failed one for a newer one would
    silently lose its factors. The failed snapshot is retried first and the newer
    one waits behind it."""
    from meridian_pipeline import sensors

    _, cursor = _run_tick(None, S1, monkeypatch)
    # Newer snapshot lands before the retry spacing: nothing fires, S1 stays queued.
    reqs, cursor2 = _run_tick(cursor, S2, monkeypatch, state="failed")
    assert reqs == [] and cursor2 == cursor
    # Once the spacing has elapsed it is S1 that is retried, not S2.
    later = T0 + timedelta(minutes=sensors._QUEUE_RETRY_MINUTES + 1)
    reqs, cursor3 = _run_tick(cursor2, S2, monkeypatch, state="failed", now=later)
    assert [r.run_key for r in reqs] == [f"{S1}_a{int(later.timestamp() // 60)}"]
    assert reqs[0].run_config["ops"]["gold_scores_pg_sync"]["config"]["snapshot_id"] == S1
    c = json.loads(cursor3)
    assert c["queued"]["snapshot_id"] == S1 and c["last_snapshot_id"] is None


def test_capped_snapshot_blocks_newer_one_and_alerts_hourly(monkeypatch, caplog):
    from meridian_pipeline import sensors

    _, cursor = _run_tick(None, S1, monkeypatch)
    now = T0
    for _ in range(sensors._SENSOR_MAX_ATTEMPTS - 1):
        now += timedelta(minutes=sensors._QUEUE_RETRY_MINUTES + 1)
        _, cursor = _run_tick(cursor, S2, monkeypatch, state="failed", now=now)
    now += timedelta(minutes=sensors._QUEUE_RETRY_MINUTES + 1)
    reqs, cursor = _run_tick(cursor, S2, monkeypatch, state="failed", now=now)
    assert reqs == []
    q = json.loads(cursor)["queued"]
    assert q["snapshot_id"] == S1 and q["gave_up"] is True and q["last_alerted"]
    # inside the hour: silent and unchanged
    reqs, same = _run_tick(cursor, S2, monkeypatch, state="failed", now=now + timedelta(minutes=5))
    assert reqs == [] and same == cursor
    # past the hour: still no S2 request, alert timestamp refreshed
    later = now + timedelta(minutes=sensors._QUEUE_RETRY_MINUTES + 1)
    reqs, cursor2 = _run_tick(cursor, S2, monkeypatch, state="failed", now=later)
    assert reqs == []
    q2 = json.loads(cursor2)["queued"]
    assert q2["snapshot_id"] == S1 and q2["last_alerted"] != q["last_alerted"]
    assert json.loads(cursor2)["last_snapshot_id"] is None


def test_hand_launched_success_releases_the_held_snapshot(monkeypatch):
    """A tagged manual run that SUCCEEDED acknowledges the capped snapshot and the
    held newer one fires in the same tick."""
    from meridian_pipeline import sensors

    _, cursor = _run_tick(None, S1, monkeypatch)
    now = T0
    for _ in range(sensors._SENSOR_MAX_ATTEMPTS):
        now += timedelta(minutes=sensors._QUEUE_RETRY_MINUTES + 1)
        _, cursor = _run_tick(cursor, S2, monkeypatch, state="failed", now=now)
    reqs, cursor2 = _run_tick(cursor, S2, monkeypatch, state="success", now=now)
    assert [r.run_key for r in reqs] == [str(S2)]
    c = json.loads(cursor2)
    assert c["last_snapshot_id"] == S1 and c["queued"]["snapshot_id"] == S2


def test_legacy_cursor_without_queued_is_honoured(monkeypatch):
    legacy = json.dumps({"last_snapshot_id": S1})
    reqs, cursor = _run_tick(legacy, S1, monkeypatch)
    assert reqs == [] and cursor == legacy
    reqs, cursor = _run_tick(legacy, S2, monkeypatch)
    assert [r.run_key for r in reqs] == [str(S2)]


def test_run_state_helper_reads_runs_by_tag():
    from meridian_pipeline import sensors

    from dagster import DagsterRunStatus as St

    seen = {}

    def _inst(*statuses):
        def get_runs(filters=None, limit=None):
            seen["filters"] = filters
            return [SimpleNamespace(status=s) for s in statuses]

        return SimpleNamespace(get_runs=get_runs)

    assert sensors._gold_snapshot_run_state(_inst(St.FAILURE, St.SUCCESS), S1) == "success"
    assert seen["filters"].tags == {"meridian/gold_snapshot_id": str(S1)}
    assert sensors._gold_snapshot_run_state(_inst(St.FAILURE, St.STARTED), S1) == "in_flight"
    assert sensors._gold_snapshot_run_state(_inst(St.FAILURE, St.CANCELED), S1) == "failed"
    assert sensors._gold_snapshot_run_state(_inst(), S1) == "failed"
    # and the real instance accepts the filter shape
    from dagster import DagsterInstance

    assert sensors._gold_snapshot_run_state(DagsterInstance.ephemeral(), S1) == "failed"


def test_three_tick_cursor_idempotency(monkeypatch):
    # Tick 1: first ever, snapshot s1 present -> one RunRequest keyed on s1.
    reqs, cursor = _run_tick(None, S1, monkeypatch)
    assert len(reqs) == 1 and reqs[0].run_key == str(S1)

    # Tick 2: s1's run succeeded -> acknowledged, no RunRequest.
    reqs, cursor2 = _run_tick(cursor, S1, monkeypatch, state="success")
    assert reqs == []
    assert json.loads(cursor2)["last_snapshot_id"] == S1

    # Tick 3: new snapshot s2 -> exactly one RunRequest keyed on s2.
    reqs, cursor3 = _run_tick(cursor2, S2, monkeypatch)
    assert len(reqs) == 1 and reqs[0].run_key == str(S2)
    assert reqs[0].run_config["ops"]["gold_scores_pg_sync"]["config"]["snapshot_id"] == S2
    assert json.loads(cursor3)["queued"]["snapshot_id"] == S2


def test_no_snapshot_is_noop(monkeypatch):
    """Table unreachable/empty (current id None) -> no RunRequest, and the
    cursor is left untouched (a prior last_snapshot_id survives a blip)."""
    prior = json.dumps({"last_snapshot_id": 42})
    reqs, cursor = _run_tick(prior, None, monkeypatch)
    assert reqs == []
    # Cursor unchanged (sensor returns early without update_cursor).
    assert cursor == prior


# ── ntfy alerts and the skipped-run guard (MR !725 second review) ─────────────


def _capture_ntfy(monkeypatch):
    from meridian_pipeline import sensors

    sent = []
    monkeypatch.setattr(sensors, "send_ntfy", lambda message, title, **kw: sent.append(title))
    return sent


def test_retry_cap_sends_ntfy_once_per_hour(monkeypatch):
    from meridian_pipeline import sensors

    sent = _capture_ntfy(monkeypatch)
    _, cursor = _run_tick(None, S1, monkeypatch)
    now = T0
    for _ in range(sensors._SENSOR_MAX_ATTEMPTS - 1):
        now += timedelta(minutes=sensors._QUEUE_RETRY_MINUTES + 1)
        _, cursor = _run_tick(cursor, S1, monkeypatch, state="failed", now=now)
    assert sent == []  # still retrying, nothing to page about yet

    now += timedelta(minutes=sensors._QUEUE_RETRY_MINUTES + 1)
    _, cursor = _run_tick(cursor, S1, monkeypatch, state="failed", now=now)
    assert len(sent) == 1
    # inside the hour: silent
    _, cursor = _run_tick(cursor, S1, monkeypatch, state="failed", now=now + timedelta(minutes=5))
    assert len(sent) == 1
    # past the hour: alerts again
    _run_tick(
        cursor,
        S1,
        monkeypatch,
        state="failed",
        now=now + timedelta(minutes=sensors._QUEUE_ALERT_REPEAT_MINUTES + 1),
    )
    assert len(sent) == 2


def test_run_stuck_in_flight_over_6h_alerts_hourly_and_is_not_cancelled(monkeypatch):
    sent = _capture_ntfy(monkeypatch)
    _, cursor = _run_tick(None, S1, monkeypatch)
    # under the threshold: quiet, cursor untouched
    reqs, same = _run_tick(
        cursor, S1, monkeypatch, state="in_flight", now=T0 + timedelta(hours=5, minutes=59)
    )
    assert reqs == [] and same == cursor and sent == []

    t = T0 + timedelta(hours=6, minutes=1)
    reqs, cursor2 = _run_tick(cursor, S1, monkeypatch, state="in_flight", now=t)
    assert reqs == [] and len(sent) == 1
    q = json.loads(cursor2)["queued"]
    assert q["snapshot_id"] == S1 and q["last_alerted"] == t.isoformat()
    # rate limited
    _, cursor3 = _run_tick(
        cursor2, S1, monkeypatch, state="in_flight", now=t + timedelta(minutes=10)
    )
    assert len(sent) == 1 and cursor3 == cursor2
    _run_tick(cursor3, S1, monkeypatch, state="in_flight", now=t + timedelta(minutes=61))
    assert len(sent) == 2


def test_skipped_success_run_does_not_count_as_promoted():
    """A SUCCESS run whose materialization says skipped=True promoted nothing, so
    it must not acknowledge the snapshot. Uses a real run in an ephemeral
    instance so the event-log shape is the real one."""
    from meridian_pipeline import sensors

    from dagster import (
        DagsterInstance,
        MetadataValue,
        Output,
        asset,
        materialize,
    )

    def _run(skipped: bool):
        @asset(name="gold_scores_pg_sync")
        def _a():
            return Output(None, metadata={"skipped": MetadataValue.bool(skipped)})

        inst = DagsterInstance.ephemeral()
        res = materialize([_a], instance=inst)
        assert res.success
        return inst, inst.get_run_by_id(res.run_id)

    inst, run = _run(skipped=True)
    assert sensors._gold_run_promoted(inst, run) is False
    inst, run = _run(skipped=False)
    assert sensors._gold_run_promoted(inst, run) is True

    # state helper: only skipped SUCCESS runs -> not success
    class _Inst:
        def __init__(self, inst, runs):
            self._inst, self._runs = inst, runs

        def get_runs(self, filters=None, limit=None):
            return self._runs

        def all_logs(self, *a, **k):
            return self._inst.all_logs(*a, **k)

    inst, run = _run(skipped=True)
    assert sensors._gold_snapshot_run_state(_Inst(inst, [run]), S1) == "failed"
    inst, run = _run(skipped=False)
    assert sensors._gold_snapshot_run_state(_Inst(inst, [run]), S1) == "success"
