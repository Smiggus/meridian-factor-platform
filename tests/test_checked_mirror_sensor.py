"""Post-commit sensor consumes each materialization once and carries its scope."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

_DAGSTER_DIR = Path(__file__).resolve().parents[1] / "dagster"
if str(_DAGSTER_DIR) not in sys.path:
    sys.path.insert(0, str(_DAGSTER_DIR))
_dagster = pytest.importorskip("dagster")
# `dagster/` at the repo root is importable as a bare namespace package, so
# importorskip alone passes on the slim CI image; check a real symbol.
if not hasattr(_dagster, "DagsterInstance"):
    pytest.skip("Dagster not fully installed (slim CI test image)", allow_module_level=True)
from dagster import DagsterInstance, build_sensor_context  # noqa: E402

sensors = pytest.importorskip("meridian_pipeline.sensors")


def test_checked_sensor_seeds_without_replaying_old_events():
    instance = DagsterInstance.ephemeral()
    instance.fetch_materializations = MagicMock(
        return_value=SimpleNamespace(cursor="latest", records=[])
    )
    context = build_sensor_context(instance=instance)
    assert list(sensors.checked_score_mirror_sensor(context)) == []
    assert set(json.loads(context.cursor)) == {"factor_group_scores"}


def test_checked_sensor_dispatches_only_group_commit(monkeypatch):
    from datetime import date

    monkeypatch.setattr(sensors, "_today_utc", lambda: date(2026, 9, 30))
    instance = DagsterInstance.ephemeral()
    group = SimpleNamespace(partition_key="2026-09-24", storage_id=10)
    instance.fetch_materializations = MagicMock(
        side_effect=[
            SimpleNamespace(cursor="next-group", records=[group]),
        ]
    )
    context = build_sensor_context(
        instance=instance,
        cursor=json.dumps({"factor_group_scores": "old-group"}),
    )
    requests = list(sensors.checked_score_mirror_sensor(context))
    assert len(requests) == 1
    configs = [r.run_config["ops"]["daily_scores_mirror_reconcile"]["config"] for r in requests]
    assert configs == [{"score_dates": ["2026-09-24"]}]
    assert json.loads(context.cursor) == {
        "factor_group_scores": "next-group",
    }


def test_checked_sensor_batches_month_and_skips_old_partitions(monkeypatch):
    from datetime import date

    monkeypatch.setattr(sensors, "_today_utc", lambda: date(2026, 10, 2))
    instance = DagsterInstance.ephemeral()
    records = [
        SimpleNamespace(partition_key=day, storage_id=i)
        for i, day in enumerate(["2026-09-01", "2026-09-24", "2026-09-25", "2026-10-01"], 1)
    ]
    instance.fetch_materializations = MagicMock(
        return_value=SimpleNamespace(cursor="next", records=records)
    )
    context = build_sensor_context(instance=instance, cursor='{"factor_group_scores":"old"}')
    requests = list(sensors.checked_score_mirror_sensor(context))
    assert len(requests) == 2
    assert [r.run_config["ops"]["daily_scores_mirror_reconcile"]["config"] for r in requests] == [
        {"score_dates": ["2026-09-24", "2026-09-25"]},
        {"score_dates": ["2026-10-01"]},
    ]
    assert json.loads(context.cursor)["factor_group_scores"] == "next"
