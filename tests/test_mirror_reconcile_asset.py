"""The Dagster boundary must fail closed and only check completed dates."""

import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

_DAGSTER_DIR = Path(__file__).resolve().parents[1] / "dagster"
if str(_DAGSTER_DIR) not in sys.path:
    sys.path.insert(0, str(_DAGSTER_DIR))


def _dagster_complete():
    try:
        import dagster
    except ImportError:
        return False
    return all(hasattr(dagster, name) for name in ("Config", "asset", "build_asset_context"))


pytestmark = pytest.mark.skipif(not _dagster_complete(), reason="Dagster not fully installed")

if not _dagster_complete():
    # pytestmark skips tests, but the import below runs at collection time.
    pytest.skip("Dagster not fully installed", allow_module_level=True)

from meridian_pipeline import mirror_reconcile_asset as mra  # noqa: E402


@pytest.fixture(autouse=True)
def enable_checked_mirror(monkeypatch):
    monkeypatch.setenv("MERIDIAN_CHECKED_MIRROR_ENABLED", "true")


def test_stored_running_schedule_cannot_bypass_rollout_gate(monkeypatch):
    from dagster import build_asset_context

    monkeypatch.delenv("MERIDIAN_CHECKED_MIRROR_ENABLED")
    monkeypatch.setattr(mra.db, "engine_from_env", lambda: pytest.fail("must not connect"))
    with pytest.raises(RuntimeError, match="disabled pending approved"):
        mra.daily_scores_mirror_reconcile(build_asset_context())


def test_stale_watermark_does_not_hide_missing_recent_dates(monkeypatch):
    from dagster import build_asset_context

    queried = []

    class InventoryConnection(FakeConnection):
        def execute(self, query, params):
            queried.append(params)
            return (
                SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: [date(2026, 9, 23)]))
                if "EXISTS" not in str(query)
                else SimpleNamespace(scalar_one=lambda: True)
            )

    engine = SimpleNamespace(connect=lambda: InventoryConnection(True))
    monkeypatch.setattr(mra.db, "engine_from_env", lambda: engine)
    monkeypatch.setattr(mra.db, "get_latest_score_date", lambda engine: date(2026, 8, 3))
    monkeypatch.setattr(mra, "_sweep_end_date", lambda: date(2026, 9, 24))
    monkeypatch.setattr(mra.db, "is_trading_day", lambda engine, day: day.weekday() < 5)
    monkeypatch.setattr(mra.checked_mirror, "reconcile_both", lambda *args: {})
    with pytest.raises(RuntimeError, match="missing trading days"):
        mra.daily_scores_mirror_reconcile(build_asset_context())
    assert queried[0] == {"lo": date(2026, 9, 17), "hi": date(2026, 9, 24)}


class FakeConnection:
    def __init__(self, complete):
        self.complete = complete

    def execute(self, query, params):
        return SimpleNamespace(scalar_one=lambda: self.complete)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


@pytest.mark.parametrize("complete", [True, False])
def test_explicit_date_requires_completion_before_repair(monkeypatch, complete):
    from dagster import build_asset_context

    day = date(2026, 9, 24)
    engine = SimpleNamespace(connect=lambda: FakeConnection(complete))
    monkeypatch.setattr(mra.db, "engine_from_env", lambda: engine)
    calls = []
    monkeypatch.setattr(
        mra.checked_mirror,
        "reconcile_both",
        lambda engine, day: (
            calls.append(day)
            or {"daily_scores": {"verdict": "matched"}, "group_scores": {"verdict": "repaired"}}
        ),
    )
    context = build_asset_context(asset_config={"score_date": str(day)})
    if not complete:
        with pytest.raises(RuntimeError, match="incomplete"):
            mra.daily_scores_mirror_reconcile(context)
        assert calls == []
    else:
        result = mra.daily_scores_mirror_reconcile(context)
        assert calls == [day]
        assert result.metadata["dates_checked"].value == 1


def test_sweep_refuses_missing_watermark(monkeypatch):
    from dagster import build_asset_context

    monkeypatch.setattr(mra.db, "engine_from_env", lambda: object())
    monkeypatch.setattr(mra.db, "get_latest_score_date", lambda engine: None)
    with pytest.raises(RuntimeError, match="no completed"):
        mra.daily_scores_mirror_reconcile(build_asset_context())


def test_range_missing_completed_trading_date_fails_after_valid_repairs(monkeypatch):
    from dagster import build_asset_context

    class InventoryConnection(FakeConnection):
        def execute(self, query, params):
            if "EXISTS" in str(query):
                return SimpleNamespace(scalar_one=lambda: True)
            return SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: [date(2026, 9, 23)]))

    engine = SimpleNamespace(connect=lambda: InventoryConnection(True))
    monkeypatch.setattr(mra.db, "engine_from_env", lambda: engine)
    monkeypatch.setattr(mra.db, "is_trading_day", lambda engine, day: True)
    calls = []
    monkeypatch.setattr(
        mra.checked_mirror, "reconcile_both", lambda *args: calls.append(args) or {}
    )
    context = build_asset_context(
        asset_config={"start_date": "2026-09-23", "end_date": "2026-09-24"}
    )
    with pytest.raises(RuntimeError, match="missing trading days"):
        mra.daily_scores_mirror_reconcile(context)
    assert [args[1] for args in calls] == [date(2026, 9, 23)]
