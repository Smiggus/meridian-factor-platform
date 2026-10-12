"""
Tests for sensors.backfill_completion_sensor (Meridian #285 Tier 3).

Drives the real sensor over an ephemeral sensor context with a stubbed
instance, threading the JSON cursor between ticks the way the daemon does.

Contract:
  * first tick SEEDS the watermark and fires nothing (no historical stampede);
  * terminal backfills fire, non-terminal ones do not;
  * COMPLETED_FAILED is treated as terminal and DOES fire (a partially failed
    backfill is exactly when the completeness sweep matters most);
  * already-processed backfill ids never re-fire;
  * non-Meridian asset selections are ignored;
  * the cursor's processed_ids list stays BOUNDED across many ticks;
  * run_key is stable and derived from backfill_id;
  * the range is min/max of the backfill's partition names;
  * a backfill with zero or unparseable partition names is skipped, not raised.

The Dagster instance API contracts this depends on (get_backfills' filters/
limit kwargs, BulkActionsFilter's shape, PartitionBackfill's attributes) are
asserted directly in test_instance_api_contract rather than assumed.

Dagster is only partially installed on the slim CI image, so the module skips
there via a hasattr check (NOT importorskip - see
a repo convention for slim CI).
"""

from __future__ import annotations

import json
import os
import sys
import types
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
    if not all(hasattr(dagster, n) for n in ("RunRequest", "sensor", "build_sensor_context")):
        return False
    try:
        from dagster._core.execution.backfill import (  # noqa: F401
            BulkActionsFilter,
            BulkActionStatus,
            PartitionBackfill,
        )
    except ImportError:
        return False
    return True


if not _dagster_complete():
    pytest.skip(
        "Dagster not fully installed (slim CI test image)",
        allow_module_level=True,
    )

from dagster._core.execution.backfill import (  # noqa: E402
    BulkActionsFilter,
    BulkActionStatus,
    PartitionBackfill,
)
from meridian_pipeline import sensors  # noqa: E402
from meridian_pipeline.sensors import (  # noqa: E402
    _MAX_PROCESSED_BACKFILL_IDS,
    _backfill_partition_names,
    _meridian_backfill,
    _partition_range,
    backfill_completion_sensor,
)

import dagster  # noqa: E402

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _FakeBackfill:
    """Minimal PartitionBackfill stand-in.

    Deliberately duck-typed rather than a real PartitionBackfill: constructing
    a genuine asset backfill requires a serialized AssetBackfillData bound to a
    live asset graph, which is exactly the coupling the sensor's
    _backfill_partition_names helper exists to tolerate the absence of. The
    real type's attribute surface is pinned separately in
    test_instance_api_contract.
    """

    def __init__(self, backfill_id, status, timestamp, asset_names, partition_names):
        self.backfill_id = backfill_id
        self.status = status
        self.backfill_timestamp = timestamp
        self.asset_selection = [dagster.AssetKey(n) for n in asset_names]
        self.partition_names = partition_names


def _stub_instance(backfills):
    """A REAL DagsterInstance with only get_backfills stubbed.

    build_sensor_context type-checks its `instance` argument, so a duck-typed
    fake is rejected outright - which is a useful contract check in itself.
    Patching one method on an ephemeral instance keeps every other API the
    sensor might touch genuine.
    """
    inst = dagster.DagsterInstance.ephemeral()
    inst.calls = []

    def _get_backfills(self, filters=None, cursor=None, limit=None, status=None):
        inst.calls.append({"filters": filters, "limit": limit, "status": status})
        selected = list(backfills)
        if filters is not None and filters.statuses:
            allowed = set(filters.statuses)
            selected = [b for b in selected if b.status in allowed]
        if limit is not None:
            selected = selected[:limit]
        return selected

    inst.get_backfills = types.MethodType(_get_backfills, inst)
    return inst


def _tick(instance, cursor=None):
    """Run one sensor tick, returning (run_requests, new_cursor)."""
    ctx = dagster.build_sensor_context(instance=instance, cursor=cursor)
    results = list(backfill_completion_sensor(ctx) or [])
    requests = [r for r in results if isinstance(r, dagster.RunRequest)]
    return requests, ctx.cursor


def _mk(bid, status=None, ts=1000.0, assets=("value_scores",), parts=("2020-01-02", "2020-01-03")):
    return _FakeBackfill(
        bid,
        status or BulkActionStatus.COMPLETED_SUCCESS,
        ts,
        list(assets),
        list(parts) if parts is not None else None,
    )


# ---------------------------------------------------------------------------
# Dagster API contract (verified, not assumed)
# ---------------------------------------------------------------------------


def test_instance_api_contract():
    """Pin the private-module Dagster surface the sensor depends on."""
    # Both terminal statuses the owner asked for must exist.
    assert BulkActionStatus.COMPLETED_SUCCESS
    assert BulkActionStatus.COMPLETED_FAILED

    # BulkActionsFilter must accept a statuses list of real enum members.
    f = BulkActionsFilter(
        statuses=[BulkActionStatus.COMPLETED_SUCCESS, BulkActionStatus.COMPLETED_FAILED]
    )
    assert list(f.statuses) == [
        BulkActionStatus.COMPLETED_SUCCESS,
        BulkActionStatus.COMPLETED_FAILED,
    ]

    # get_backfills must accept the filters + limit kwargs the sensor passes.
    from dagster._core.instance import DagsterInstance

    params = __import__("inspect").signature(DagsterInstance.get_backfills).parameters
    assert "filters" in params
    assert "limit" in params

    # PartitionBackfill must carry the fields the sensor reads.
    for fieldname in (
        "backfill_id",
        "status",
        "backfill_timestamp",
        "backfill_end_timestamp",
        "asset_selection",
        "partition_names",
    ):
        assert fieldname in PartitionBackfill._fields, fieldname
    assert hasattr(PartitionBackfill, "get_asset_backfill_data")


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_partition_range_min_max():
    assert _partition_range(["2020-03-05", "2020-01-02", "2020-02-11"]) == (
        "2020-01-02",
        "2020-03-05",
    )


def test_partition_range_empty_and_unparseable():
    assert _partition_range([]) is None
    assert _partition_range(None) is None
    assert _partition_range(["not-a-date", None, 17]) is None
    # Mixed input keeps the parseable subset rather than failing outright.
    assert _partition_range(["nope", "2021-06-01"]) == ("2021-06-01", "2021-06-01")


def test_meridian_backfill_filter():
    assert _meridian_backfill(_mk("a", assets=["value_scores"]))
    assert _meridian_backfill(_mk("b", assets=["some_other_asset", "composite_scores"]))
    assert not _meridian_backfill(_mk("c", assets=["ldp_ohlcv_prices"]))
    # No asset selection at all (an op/job backfill) is not ours.
    bf = _mk("d")
    bf.asset_selection = None
    assert not _meridian_backfill(bf)


def test_backfill_partition_names_never_raises():
    """A backfill whose names cannot be resolved returns None, not an error."""

    class _Exploding:
        partition_names = None

        def get_asset_backfill_data(self, asset_graph):
            raise RuntimeError("definition changed since the backfill ran")

    ctx = dagster.build_sensor_context()
    assert _backfill_partition_names(ctx, _Exploding()) is None


# ---------------------------------------------------------------------------
# Sensor behaviour
# ---------------------------------------------------------------------------


def test_first_tick_seeds_without_firing():
    inst = _stub_instance([_mk("bf-1", ts=100.0), _mk("bf-2", ts=200.0)])
    requests, cursor = _tick(inst)
    assert requests == []
    state = json.loads(cursor)
    # Ids seen at seed time are recorded so the newest of them cannot sail
    # through the STRICT watermark comparison and fire on the next tick.
    assert set(state["processed_ids"]) == {"bf-1", "bf-2"}
    assert state["high_water_ts"] == 200.0


def test_terminal_backfill_fires_after_seed():
    inst = _stub_instance([_mk("bf-1", ts=100.0)])
    _, cursor = _tick(inst)

    inst = _stub_instance(
        [_mk("bf-1", ts=100.0), _mk("bf-2", ts=300.0, parts=["2021-01-04", "2021-02-09"])]
    )
    requests, cursor = _tick(inst, cursor)
    assert len(requests) == 1
    req = requests[0]
    assert req.run_key == "verify-bf-2"
    cfg = req.run_config["ops"]["verify_range_op"]["config"]
    assert cfg["start_date"] == "2021-01-04"
    assert cfg["end_date"] == "2021-02-09"


def test_completed_failed_is_terminal_and_fires():
    """Owner-approved: a partially failed backfill is exactly when the
    completeness sweep matters most, so it must not be filtered out."""
    inst = _stub_instance([_mk("bf-0", ts=50.0)])
    _, cursor = _tick(inst)

    inst = _stub_instance(
        [
            _mk("bf-0", ts=50.0),
            _mk("bf-fail", status=BulkActionStatus.COMPLETED_FAILED, ts=400.0),
        ]
    )
    requests, _ = _tick(inst, cursor)
    assert [r.run_key for r in requests] == ["verify-bf-fail"]

    # And the filter the sensor hands the instance asks for both statuses.
    statuses = set(inst.calls[-1]["filters"].statuses)
    assert statuses == {BulkActionStatus.COMPLETED_SUCCESS, BulkActionStatus.COMPLETED_FAILED}


def test_non_terminal_backfills_are_not_fired():
    """The instance-side filter is the primary gate; assert the sensor asks
    for terminal statuses only and that in-flight backfills never fire."""
    inst = _stub_instance(
        [
            _mk("bf-seed", ts=10.0),
            _mk("bf-running", status=BulkActionStatus.REQUESTED, ts=500.0),
            _mk("bf-canceling", status=BulkActionStatus.CANCELING, ts=600.0),
        ]
    )
    _, cursor = _tick(inst)
    requests, _ = _tick(inst, cursor)
    assert requests == []


def test_already_processed_ids_skipped():
    inst = _stub_instance([_mk("bf-1", ts=100.0)])
    _, cursor = _tick(inst)

    inst = _stub_instance([_mk("bf-1", ts=100.0), _mk("bf-2", ts=300.0)])
    requests, cursor = _tick(inst, cursor)
    assert [r.run_key for r in requests] == ["verify-bf-2"]

    # Same instance state on the next tick: no duplicate RunRequest.
    requests, cursor = _tick(inst, cursor)
    assert requests == []
    assert "bf-2" in json.loads(cursor)["processed_ids"]


def test_run_key_is_stable_across_ticks():
    inst = _stub_instance([_mk("bf-seed", ts=1.0)])
    _, cursor = _tick(inst)
    inst = _stub_instance([_mk("bf-seed", ts=1.0), _mk("bf-stable", ts=900.0)])
    first, cursor_a = _tick(inst, cursor)
    # Re-run the same tick from the SAME pre-tick cursor: identical run_key.
    second, _ = _tick(inst, cursor)
    assert [r.run_key for r in first] == [r.run_key for r in second] == ["verify-bf-stable"]


def test_zero_partition_names_is_skipped_not_raised():
    inst = _stub_instance([_mk("bf-seed", ts=1.0)])
    _, cursor = _tick(inst)

    inst = _stub_instance(
        [
            _mk("bf-seed", ts=1.0),
            _mk("bf-empty", ts=800.0, parts=[]),
            _mk("bf-junk", ts=850.0, parts=["not-a-date", "also-bad"]),
        ]
    )
    requests, cursor = _tick(inst, cursor)
    assert requests == []
    # Both are recorded so they are not re-inspected every tick.
    assert {"bf-empty", "bf-junk"} <= set(json.loads(cursor)["processed_ids"])


def test_missing_partition_names_attribute_is_skipped():
    inst = _stub_instance([_mk("bf-seed", ts=1.0)])
    _, cursor = _tick(inst)

    bad = _mk("bf-none", ts=700.0)
    bad.partition_names = None  # asset backfill with no resolvable graph here
    inst = _stub_instance([_mk("bf-seed", ts=1.0), bad])
    requests, cursor = _tick(inst, cursor)
    assert requests == []
    assert "bf-none" in json.loads(cursor)["processed_ids"]


def test_non_meridian_backfill_ignored():
    inst = _stub_instance([_mk("bf-seed", ts=1.0)])
    _, cursor = _tick(inst)

    inst = _stub_instance(
        [_mk("bf-seed", ts=1.0), _mk("bf-other", ts=999.0, assets=["ldp_ohlcv_prices"])]
    )
    requests, cursor = _tick(inst, cursor)
    assert requests == []
    # Not ours, so not recorded beyond the seed - it costs nothing to re-skip.
    assert "bf-other" not in json.loads(cursor)["processed_ids"]


def test_cursor_stays_bounded_across_many_ticks():
    """Simulate 500 backfills over many ticks and assert the cursor's
    processed_ids never exceeds the documented hard bound. Unbounded cursor
    state is what wedged core_complete_sensor in Meridian #99."""
    inst = _stub_instance([_mk("seed", ts=0.0)])
    _, cursor = _tick(inst)

    max_seen = 0
    for i in range(500):
        ts = 100.0 + i
        # One fresh backfill per tick; the instance only ever returns the
        # recent window, exactly like a real get_backfills(limit=...) page.
        inst = _stub_instance([_mk(f"bf-{i}", ts=ts)])
        requests, cursor = _tick(inst, cursor)
        assert [r.run_key for r in requests] == [f"verify-bf-{i}"]
        state = json.loads(cursor)
        max_seen = max(max_seen, len(state["processed_ids"]))
        assert len(state["processed_ids"]) <= _MAX_PROCESSED_BACKFILL_IDS

    # The bound was actually exercised, not vacuously satisfied.
    assert max_seen == _MAX_PROCESSED_BACKFILL_IDS
    # And the watermark advanced monotonically to the newest backfill.
    assert json.loads(cursor)["high_water_ts"] == 599.0


def test_evicted_id_below_watermark_does_not_refire():
    """An id aged out of the bounded list must still never re-fire, because
    its timestamp is below the high-water mark. This is the half of the
    design that makes bounding the list safe."""
    inst = _stub_instance([_mk("seed", ts=0.0)])
    _, cursor = _tick(inst)

    for i in range(_MAX_PROCESSED_BACKFILL_IDS + 20):
        inst = _stub_instance([_mk(f"bf-{i}", ts=100.0 + i)])
        _, cursor = _tick(inst, cursor)

    state = json.loads(cursor)
    assert "bf-0" not in state["processed_ids"]  # evicted

    # bf-0 reappears in the instance's page; it must not fire again.
    inst = _stub_instance([_mk("bf-0", ts=100.0)])
    requests, _ = _tick(inst, cursor)
    assert requests == []


def test_sensor_ships_stopped():
    """Registered STOPPED for a watched first enable (the #99 precedent)."""
    assert backfill_completion_sensor.default_status == dagster.DefaultSensorStatus.STOPPED


def test_sensor_is_not_a_multi_asset_sensor():
    """CRITICAL_RULES rule 18: multi_asset_sensor caps unconsumed events at 25
    and dead-locks on partial-completion patterns."""
    assert isinstance(backfill_completion_sensor, dagster.SensorDefinition)
    assert not isinstance(
        backfill_completion_sensor,
        getattr(dagster, "MultiAssetSensorDefinition", ()) or (),
    )


def test_sensor_targets_the_verification_job():
    assert sensors.backfill_verification_job.name == "backfill_verification_job"
