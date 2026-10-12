"""
Multi-tick behavioural tests for sensors.core_complete_sensor (Meridian #99).

Drives the real sensor function over an ephemeral DagsterInstance with toy
assets named exactly like the CORE factor assets, threading the JSON cursor
between ticks the way the daemon does. Covers the three failure modes that
wedged the live sensor on 2026-05-28:

  1. A newly-ADDED CORE asset must be seeded to its latest record, NOT crawled
     from the start of its backfilled history (the crawl that stopped phase-2
     from ever firing).
  2. Never-completing partials (e.g. a single-asset / cancelled backfill) must
     be GC'd by wall-clock TTL instead of accumulating forever.
  3. The happy path (complete partition fires; re-score re-fires) still works.

DB-free: toy assets only emit AssetMaterialization, no engine needed. The
sensor reaches PG only via context.instance.fetch_materializations, which the
ephemeral instance serves from its in-memory event log.
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

# sensors.py reads PGPASS at _get_engine() time, not import time, but set a
# sentinel so any incidental import path stays happy.
os.environ.setdefault("PGPASS", "test-pg-pass")
os.environ.setdefault("PARTITION_START_DATE", "2010-01-02")


# A start date early enough to host synthetic 2009 "history" partitions.
_PARTS_START = "2004-01-01"


def _toy_assets():
    """One toy asset per CORE key, mirroring the production emit pattern:
    per-partition explicit AssetMaterialization plus a trailing bare
    Output(None). Returns (dict[name -> AssetsDefinition], ordered key tuple)."""
    from meridian_pipeline.asset_selections import CORE_FACTOR_ASSET_KEYS

    from dagster import AssetMaterialization, DailyPartitionsDefinition, Output, asset

    parts = DailyPartitionsDefinition(start_date=_PARTS_START)
    out = {}
    for name in CORE_FACTOR_ASSET_KEYS:

        @asset(name=name, partitions_def=parts)
        def _a(context):
            for pk in context.partition_keys:
                yield AssetMaterialization(asset_key=context.asset_key, partition=pk)
            yield Output(value=None)

        out[name] = _a
    return out, CORE_FACTOR_ASSET_KEYS


def _materialize(assets, names, partition_key, instance):
    from dagster import materialize

    selected = [assets[n] for n in names]
    assert materialize(selected, partition_key=partition_key, instance=instance).success


def _run_tick(instance, cursor):
    """Run one sensor evaluation; return (list[RunRequest], new_cursor_str)."""
    from meridian_pipeline import sensors

    from dagster import RunRequest, build_sensor_context

    ctx = build_sensor_context(instance=instance, cursor=cursor)
    emitted = list(sensors.core_complete_sensor(ctx))
    run_requests = [e for e in emitted if isinstance(e, RunRequest)]
    return run_requests, ctx.cursor


@pytest.fixture
def instance():
    from dagster import DagsterInstance

    return DagsterInstance.ephemeral()


@pytest.fixture(autouse=True)
def _freeze_age_gate_today(monkeypatch):
    """Freeze the age gate's notion of "today" near the fixed 2026-05-2x
    partition dates used throughout this file's pre-existing fixtures, so the
    partition age gate (added for the #99 follow-up backfill-storm fix)
    doesn't age those fixtures out from under them. Age-gate-specific tests
    below override this within their own test body via the same monkeypatch
    fixture instance."""
    from meridian_pipeline import sensors

    monkeypatch.setattr(sensors, "_today_utc", lambda: date(2026, 5, 28))


# ─────────────────────────────────────────────
# Happy path
# ─────────────────────────────────────────────


def test_complete_partition_fires_phase2(instance):
    assets, keys = _toy_assets()
    # Bootstrap tick on empty cursor: seeds every CORE asset to its current
    # end-of-log position without processing, and fires nothing.
    rrs, cur = _run_tick(instance, None)
    assert rrs == []
    state = json.loads(cur)
    assert set(state["asset_cursors"]) == set(keys)  # all seeded
    assert state["partitions_seen"] == {}

    # All CORE assets materialise for one partition.
    _materialize(assets, keys, "2026-05-26", instance)
    rrs, cur = _run_tick(instance, cur)

    assert [r.partition_key for r in rrs] == ["2026-05-26"]
    assert rrs[0].run_key.startswith("core_complete_2026-05-26_")
    assert rrs[0].tags["dagster/priority"] == "50"
    # Fired partition is dropped from tracker state.
    assert "2026-05-26" not in json.loads(cur)["partitions_seen"]


def test_rescore_refires_from_seeded_cursor(instance):
    assets, keys = _toy_assets()
    _materialize(assets, keys, "2026-05-26", instance)
    _, cur = _run_tick(instance, None)  # bootstrap seeds to current latest

    _materialize(assets, keys, "2026-05-26", instance)  # re-score
    rrs, _ = _run_tick(instance, cur)
    assert [r.partition_key for r in rrs] == ["2026-05-26"]


# ─────────────────────────────────────────────
# #99 fix 2: TTL GC of never-completing partials
# ─────────────────────────────────────────────


def test_stale_partials_are_gc_but_fresh_partials_survive(instance, monkeypatch):
    from meridian_pipeline import sensors

    assets, keys = _toy_assets()
    # Seed cursors cleanly on the empty instance (no partitions tracked yet).
    _, cur = _run_tick(instance, None)
    assert json.loads(cur)["partitions_seen"] == {}

    # Inject a mix of stale and fresh partials directly into the cursor.
    base = json.loads(cur)
    NOW = 1_000_000.0
    ttl = sensors._PARTIAL_PARTITION_TTL_SECONDS
    base["partitions_seen"] = {
        "2009-01-02": ["growth_scores"],  # stale single-asset partial
        "2009-01-03": ["growth_scores"],  # stale single-asset partial
        "2026-05-27": ["value_scores", "quality_scores"],  # fresh in-flight partial
    }
    base["partition_first_seen"] = {
        "2009-01-02": NOW - ttl - 10,  # older than TTL -> GC
        "2009-01-03": NOW - ttl - 10,  # older than TTL -> GC
        "2026-05-27": NOW - 60,  # well within TTL -> keep
    }
    cur = json.dumps(base)

    monkeypatch.setattr(sensors, "_now_epoch", lambda: NOW)
    rrs, cur = _run_tick(instance, cur)
    seen = json.loads(cur)["partitions_seen"]

    assert rrs == []  # none complete
    assert "2009-01-02" not in seen and "2009-01-03" not in seen  # reaped
    assert "2026-05-27" in seen  # fresh partial preserved
    # first_seen stays in lockstep with partitions_seen.
    fs = json.loads(cur)["partition_first_seen"]
    assert set(fs) == set(seen)


# ─────────────────────────────────────────────
# #99 follow-up: guard partial single-asset backfills
# ─────────────────────────────────────────────


class _FakeRun:
    def __init__(self, asset_selection):
        self.asset_selection = asset_selection


class _FakeRecord:
    def __init__(self, asset_selection):
        self.dagster_run = _FakeRun(asset_selection)


class _FakeInstance:
    """Returns a fixed run record; or raises if `boom`."""

    def __init__(self, asset_selection, boom=False):
        self._sel = asset_selection
        self._boom = boom

    def get_run_record_by_id(self, run_id):
        if self._boom:
            raise RuntimeError("lookup failed")
        return _FakeRecord(self._sel)


def _core_keys_as_assetkeys():
    from meridian_pipeline.asset_selections import CORE_FACTOR_ASSET_KEYS

    from dagster import AssetKey

    return CORE_FACTOR_ASSET_KEYS, {AssetKey(k) for k in CORE_FACTOR_ASSET_KEYS}


def test_run_covers_core_whole_job_selections_count():
    from meridian_pipeline import sensors

    keys, _ = _core_keys_as_assetkeys()
    core_set = set(keys)
    # None (whole-job run, the prod daily/backfill case) and empty frozenset
    # (some in-process paths) must both count - never block a legit run.
    assert sensors._run_covers_core(_FakeInstance(None), "r1", core_set, {}) is True
    assert sensors._run_covers_core(_FakeInstance(frozenset()), "r2", core_set, {}) is True


def test_run_covers_core_full_selection_counts():
    from meridian_pipeline import sensors

    keys, ak_set = _core_keys_as_assetkeys()
    assert sensors._run_covers_core(_FakeInstance(ak_set), "r3", set(keys), {}) is True


def test_run_covers_core_single_asset_backfill_skipped():
    from meridian_pipeline import sensors

    from dagster import AssetKey

    keys, _ = _core_keys_as_assetkeys()
    sel = {AssetKey("growth_scores")}  # the canceled-backfill shape
    assert sensors._run_covers_core(_FakeInstance(sel), "r4", set(keys), {}) is False


def test_run_covers_core_permissive_on_lookup_failure():
    from meridian_pipeline import sensors

    keys, _ = _core_keys_as_assetkeys()
    assert sensors._run_covers_core(_FakeInstance(None, boom=True), "r5", set(keys), {}) is True


def test_run_covers_core_caches_per_run():
    from meridian_pipeline import sensors

    keys, _ = _core_keys_as_assetkeys()
    cache = {"r6": False}  # pre-seeded; must be returned without a lookup
    boom = _FakeInstance(None, boom=True)  # would raise if consulted
    assert sensors._run_covers_core(boom, "r6", set(keys), cache) is False


# ─────────────────────────────────────────────
# Age gate: historical phase-2 must not fire from this sensor
# (Meridian #99 follow-up, 2026-07-04/05 backfill-storm incidents)
# ─────────────────────────────────────────────


def test_old_partition_dropped_no_run_but_fresh_partition_still_fires(instance, monkeypatch):
    from meridian_pipeline import sensors

    assets, keys = _toy_assets()
    # Bootstrap tick on empty cursor: seeds every CORE asset, fires nothing.
    _, cur = _run_tick(instance, None)

    monkeypatch.setattr(sensors, "_today_utc", lambda: date(2026, 7, 5))

    # A historical CORE backfill partition completes ALL CORE assets - the
    # exact shape that fired many high-priority GPU runs during a
    # historical rerun.
    _materialize(assets, keys, "2014-06-01", instance)
    # A live, recent partition also completes in the same tick.
    _materialize(assets, keys, "2026-07-04", instance)

    rrs, cur = _run_tick(instance, cur)

    fired = {r.partition_key for r in rrs}
    assert fired == {"2026-07-04"}, "old historical partition must not fire phase 2"


def test_dropped_old_partition_absent_from_cursor_state(instance, monkeypatch):
    from meridian_pipeline import sensors

    assets, keys = _toy_assets()
    _, cur = _run_tick(instance, None)

    monkeypatch.setattr(sensors, "_today_utc", lambda: date(2026, 7, 5))

    _materialize(assets, keys, "2014-06-01", instance)
    rrs, cur = _run_tick(instance, cur)

    state = json.loads(cur)
    assert rrs == []
    assert "2014-06-01" not in state["partitions_seen"]
    assert "2014-06-01" not in state["partition_first_seen"]


def test_age_gate_env_override_respected(instance, monkeypatch):
    """With a widened window, a partition that would otherwise be age-gated
    is treated as live and fires normally."""
    from meridian_pipeline import sensors

    assets, keys = _toy_assets()
    _, cur = _run_tick(instance, None)

    fixed_today = date(2026, 7, 5)
    monkeypatch.setattr(sensors, "_today_utc", lambda: fixed_today)

    # 12 calendar days old: gated out under the default 10-day window...
    old_pk = "2026-06-23"
    _materialize(assets, keys, old_pk, instance)
    rrs, cur = _run_tick(instance, cur)
    assert rrs == []

    # ...but fires once the operator widens the window past 12 days via env.
    monkeypatch.setenv("CORE_COMPLETE_MAX_PARTITION_AGE_DAYS", "30")
    _, cur = _run_tick(instance, None)  # fresh bootstrap for a clean re-test
    _materialize(assets, keys, old_pk, instance)
    rrs, cur = _run_tick(instance, cur)
    assert [r.partition_key for r in rrs] == [old_pk]


def test_age_gate_purges_preexisting_tracked_partition(instance, monkeypatch):
    """A partition already sitting in partitions_seen from before the gate
    was deployed (or from a prior tick) is purged even if it isn't touched
    by new materialisations this tick."""
    from meridian_pipeline import sensors

    _, cur = _run_tick(instance, None)
    base = json.loads(cur)
    base["partitions_seen"] = {"2014-06-01": list(_toy_assets()[1])}
    base["partition_first_seen"] = {"2014-06-01": 0.0}
    cur = json.dumps(base)

    monkeypatch.setattr(sensors, "_today_utc", lambda: date(2026, 7, 5))

    rrs, cur = _run_tick(instance, cur)
    state = json.loads(cur)
    assert rrs == []
    assert "2014-06-01" not in state["partitions_seen"]


def test_rescore_changes_run_key_but_tick_replay_does_not(instance):
    from dagster._core.storage.tags import RUN_KEY_TAG, SENSOR_NAME_TAG

    from dagster import RunsFilter, job

    @job
    def phase2():
        pass

    assets, keys = _toy_assets()
    _, seeded = _run_tick(instance, None)
    _materialize(assets, keys, "2026-05-26", instance)
    first, cursor = _run_tick(instance, seeded)
    replay, _ = _run_tick(instance, seeded)
    assert first[0].run_key == replay[0].run_key
    # Persist the key as the daemon does. Replaying this request finds a run.
    tags = {RUN_KEY_TAG: first[0].run_key, SENSOR_NAME_TAG: "core_complete_sensor"}
    instance.create_run_for_job(phase2, tags=tags)
    assert instance.get_runs(filters=RunsFilter(tags=tags))

    _materialize(assets, keys, "2026-05-26", instance)
    second, cursor = _run_tick(instance, cursor)
    assert second[0].run_key != first[0].run_key
    tags[RUN_KEY_TAG] = second[0].run_key
    assert instance.get_runs(filters=RunsFilter(tags=tags)) == []
    assert json.loads(cursor)["partition_run_ids"] == {}


def _tick_materialization_events(events, cursor=None):
    from types import SimpleNamespace
    from unittest.mock import MagicMock, patch

    from meridian_pipeline import sensors

    from dagster import RunRequest

    keys = sensors.CORE_FACTOR_ASSET_KEYS
    if cursor is None:
        cursor = json.dumps({"asset_cursors": {k: "seed" for k in keys}})
    instance = MagicMock()

    def fetch(asset_key, **kwargs):
        entries = events.get(asset_key.to_user_string(), [])
        records = [
            SimpleNamespace(partition_key="2026-05-26", run_id=run, storage_id=ident)
            for run, ident in entries
        ]
        return SimpleNamespace(records=records, cursor=str(entries[-1][1]) if entries else None)

    instance.fetch_materializations.side_effect = fetch
    context = SimpleNamespace(
        cursor=cursor,
        instance=instance,
        log=MagicMock(),
        resources=SimpleNamespace(),
        update_cursor=MagicMock(),
    )
    with patch.object(sensors, "_run_covers_core", return_value=True):
        requests = [
            r
            for r in sensors.core_complete_sensor._evaluation_fn(context)
            if isinstance(r, RunRequest)
        ]
    return requests, context.update_cursor.call_args.args[0]


def test_multiple_events_for_one_upstream_run_keep_one_deduplication_key():
    from meridian_pipeline import sensors

    keys = sensors.CORE_FACTOR_ASSET_KEYS
    first, cursor = _tick_materialization_events(
        {k: [("run-a", i + 1)] for i, k in enumerate(keys)}
    )
    # Explicit materializations and Dagster's implicit Output events can arrive
    # in separate ticks, but still describe one upstream generation.
    duplicate, _ = _tick_materialization_events(
        {k: [("run-a", i + 101)] for i, k in enumerate(keys)}, cursor
    )
    assert duplicate[0].run_key == first[0].run_key


def test_partial_events_from_different_runs_do_not_complete_a_generation():
    from meridian_pipeline import sensors

    keys = sensors.CORE_FACTOR_ASSET_KEYS
    first, cursor = _tick_materialization_events({keys[0]: [("old-run", 1)]})
    assert first == []
    mixed, cursor = _tick_materialization_events(
        {k: [("new-run", i + 2)] for i, k in enumerate(keys[1:])}, cursor
    )
    assert mixed == []
    completed, _ = _tick_materialization_events({keys[0]: [("new-run", 100)]}, cursor)
    assert len(completed) == 1


def test_late_old_run_event_does_not_erase_new_run_progress():
    from meridian_pipeline import sensors

    keys = sensors.CORE_FACTOR_ASSET_KEYS
    _, cursor = _tick_materialization_events({keys[0]: [("new-run", 1)]})
    events = {k: [("new-run", i + 3)] for i, k in enumerate(keys[1:])}
    events[keys[0]] = [("old-run", 2)]
    completed, cursor = _tick_materialization_events(events, cursor)
    assert len(completed) == 1
    assert completed[0].tags["meridian/upstream_run_id"] == "new-run"
    assert json.loads(cursor)["partition_run_ids"] == {"2026-05-26": {keys[0]: ["old-run"]}}
