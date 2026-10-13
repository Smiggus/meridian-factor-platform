"""Tests for the trading-day invariant asset check (Meridian #174 follow-up).

The check is a detective control for the trading-day-only invariant: no
scored table may hold a score_date that is not a trading day. See
`dagster/meridian_pipeline/checks/trading_day_checks.py` for the incident
history that motivated it.
"""

import inspect
from datetime import date
from unittest.mock import MagicMock

import pytest

pytest.importorskip("sqlalchemy")

dagster = pytest.importorskip("dagster")
if not hasattr(dagster, "asset_check"):  # slim CI image
    pytest.skip("dagster asset_check unavailable", allow_module_level=True)

from meridian_pipeline import db  # noqa: E402
from meridian_pipeline.checks import trading_day_checks as tdc  # noqa: E402

# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_non_trading_dates_flags_weekend_and_holiday():
    cal = {date(2026, 5, 1), date(2026, 5, 4), date(2026, 5, 5)}
    table_dates = [
        date(2026, 5, 1),  # Fri, trading
        date(2026, 5, 2),  # Sat
        date(2026, 5, 3),  # Sun
        date(2026, 5, 4),  # Mon, trading
    ]
    assert tdc.non_trading_dates(table_dates, cal) == [date(2026, 5, 2), date(2026, 5, 3)]


def test_non_trading_dates_clean_window_is_empty():
    cal = {date(2026, 5, 1), date(2026, 5, 4)}
    assert tdc.non_trading_dates([date(2026, 5, 1), date(2026, 5, 4)], cal) == []


def test_non_trading_dates_deduplicates_and_sorts():
    cal = {date(2026, 5, 1)}
    dirty = [date(2026, 5, 3), date(2026, 5, 2), date(2026, 5, 3)]
    assert tdc.non_trading_dates(dirty, cal) == [date(2026, 5, 2), date(2026, 5, 3)]


def test_invariant_ok_passes_when_all_tables_clean():
    passed, reason = tdc.invariant_ok({t: [] for t in tdc.SCORED_TABLES})
    assert passed is True
    assert "clean" in reason


def test_invariant_ok_has_zero_tolerance():
    """A single non-trading date in a single table must fail. No threshold."""
    offenders = {t: [] for t in tdc.SCORED_TABLES}
    offenders["factors.risk_metrics"] = [date(2026, 5, 25)]
    passed, reason = tdc.invariant_ok(offenders)
    assert passed is False
    assert "factors.risk_metrics" in reason
    assert "2026-05-25" in reason


def test_invariant_ok_reports_every_dirty_table():
    offenders = {
        "factors.daily_scores": [date(2010, 1, 9)],
        "factors.composite_scores": [date(2010, 1, 9), date(2010, 1, 10)],
        "factors.group_scores": [],
    }
    passed, reason = tdc.invariant_ok(offenders)
    assert passed is False
    assert "factors.daily_scores" in reason
    assert "factors.composite_scores" in reason
    assert "factors.group_scores" not in reason


# ---------------------------------------------------------------------------
# Calendar parity with db.is_trading_day
# ---------------------------------------------------------------------------


def test_calendar_sql_mirrors_is_trading_day_definition():
    """The check builds its own calendar SET in one round trip rather than
    calling db.is_trading_day per date. Pin the two definitions together so a
    change to one cannot silently diverge from the other."""
    src = inspect.getsource(db.is_trading_day)
    assert "security_type = 'equity'" in src
    assert "> 100" in src or "100" in src

    sql = tdc._CALENDAR_SQL
    assert "security_type = 'equity'" in sql
    assert "FROM ohlcv.prices" in sql
    assert tdc._TRADING_DAY_MIN_EQUITY_ROWS == 100
    # Same sargable half-open range form as db.is_trading_day (db.py perf fix).
    assert "ts_event >= :s" in sql
    assert "ts_event < :e_next" in sql
    assert "ts_event::date = " not in sql


def test_calendar_sql_and_is_trading_day_filter_resolution_daily():
    """Meridian #275 Phase 0: ohlcv.prices co-mingles intraday (hourly/minute)
    bars now, which would otherwise count toward the >100-row threshold and
    could manufacture a phantom trading day. Both the check's calendar SQL and
    db.is_trading_day must filter to resolution = 'daily'; guard against one
    site drifting from the other."""
    src = inspect.getsource(db.is_trading_day)
    assert "resolution = 'daily'" in src

    sql = tdc._CALENDAR_SQL
    assert "resolution = 'daily'" in sql


def test_load_trading_calendar_binds_half_open_range_and_threshold():
    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = [(date(2026, 5, 1),)]
    engine = MagicMock()
    engine.connect.return_value.__enter__.return_value = conn
    engine.connect.return_value.__exit__.return_value = False

    cal = tdc.load_trading_calendar(engine, date(2026, 5, 1), date(2026, 5, 4))

    assert cal == {date(2026, 5, 1)}
    _sql, params = conn.execute.call_args[0]
    assert params["s"] == date(2026, 5, 1)
    # Half-open upper bound: end + 1 day.
    assert params["e_next"] == date(2026, 5, 5)
    assert params["min_rows"] == 100


# ---------------------------------------------------------------------------
# Table sweep
# ---------------------------------------------------------------------------


def test_scored_tables_covers_every_score_date_keyed_output_table():
    """Regression guard: a purge of daily_scores alone would leave the other
    score-date-keyed output tables unchecked. The sweep must not silently
    shrink back to one table."""
    for expected in (
        "factors.daily_scores",
        "factors.group_scores",
        "factors.composite_scores",
        "factors.risk_metrics",
        "factors.etf_features",
        "factors.specific_returns",
        "factors.specific_risk",
        "factors.factor_returns",
        "factors.factor_covariance",
        # Meridian#379: regime_snapshot's own per-date output table.
        "factors.regime_snapshots",
    ):
        assert expected in tdc.SCORED_TABLES


def test_load_table_dates_rejects_unknown_table():
    """Table name is string-interpolated into SQL (no bind param for
    identifiers), so the allow-list check matters."""
    with pytest.raises(ValueError):
        tdc.load_table_dates(
            MagicMock(), "factors.daily_scores; DROP TABLE x", date(2026, 1, 1), date(2026, 1, 2)
        )


def test_load_table_dates_uses_loose_index_scan_not_select_distinct():
    """factors.daily_scores is a very large table; SELECT DISTINCT over the
    window would scan a very large number of index entries. The recursive
    skip-scan costs O(distinct dates) probes instead."""
    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = [(date(2026, 5, 2),)]
    engine = MagicMock()
    engine.connect.return_value.__enter__.return_value = conn
    engine.connect.return_value.__exit__.return_value = False

    dates = tdc.load_table_dates(engine, "factors.daily_scores", date(2026, 5, 1), date(2026, 5, 4))

    assert dates == [date(2026, 5, 2)]
    sql = str(conn.execute.call_args[0][0])
    assert "WITH RECURSIVE" in sql
    assert "SELECT DISTINCT" not in sql
    assert "factors.daily_scores" in sql


def test_scan_window_flags_only_non_trading_dates(monkeypatch):
    cal = {date(2026, 5, 1), date(2026, 5, 4)}
    per_table = {
        "factors.daily_scores": [date(2026, 5, 1), date(2026, 5, 4)],
        "factors.risk_metrics": [date(2026, 5, 1), date(2026, 5, 2), date(2026, 5, 4)],
    }
    monkeypatch.setattr(tdc, "load_trading_calendar", lambda e, s, en: cal)
    monkeypatch.setattr(
        tdc, "load_table_dates", lambda e, t, s, en: per_table.get(t, [date(2026, 5, 1)])
    )

    out = tdc.scan_window(MagicMock(), date(2026, 5, 1), date(2026, 5, 4))

    assert out["factors.daily_scores"] == []
    assert out["factors.risk_metrics"] == [date(2026, 5, 2)]
    assert set(out) == set(tdc.SCORED_TABLES)


# ---------------------------------------------------------------------------
# Check wiring
# ---------------------------------------------------------------------------


def _ctx(end_key: str):
    ctx = MagicMock()
    ctx.partition_key_range = MagicMock(start=end_key, end=end_key)
    return ctx


def _invoke(ctx):
    """Call the undecorated check body (mirrors test_data_quality_checks)."""
    return tdc.check_trading_day_invariant.op.compute_fn.decorated_fn(ctx)


def test_check_passes_on_clean_window(monkeypatch):
    monkeypatch.setattr(tdc, "_get_engine", lambda: MagicMock())
    monkeypatch.setattr(tdc, "scan_window", lambda e, s, en: {t: [] for t in tdc.SCORED_TABLES})

    res = _invoke(_ctx("2026-07-24"))

    assert res.passed is True
    assert res.metadata["dirty_table_count"].value == 0


def test_check_fails_and_reports_offenders(monkeypatch):
    offenders = {t: [] for t in tdc.SCORED_TABLES}
    offenders["factors.composite_scores"] = [date(2026, 5, 2), date(2026, 5, 3)]
    monkeypatch.setattr(tdc, "_get_engine", lambda: MagicMock())
    monkeypatch.setattr(tdc, "scan_window", lambda e, s, en: offenders)

    ctx = _ctx("2026-07-24")
    res = _invoke(ctx)

    assert res.passed is False
    assert res.severity == dagster.AssetCheckSeverity.ERROR
    assert res.metadata["dirty_table_count"].value == 1
    assert res.metadata["offending_dates"].data == {
        "factors.composite_scores": ["2026-05-02", "2026-05-03"]
    }
    ctx.log.error.assert_called_once()


def test_check_window_is_bounded_lookback(monkeypatch):
    """The daily path must not scan full history; the standalone script owns
    the unbounded audit."""
    seen = {}

    def _scan(engine, start, end):
        seen["start"], seen["end"] = start, end
        return {t: [] for t in tdc.SCORED_TABLES}

    monkeypatch.setattr(tdc, "_get_engine", lambda: MagicMock())
    monkeypatch.setattr(tdc, "scan_window", _scan)

    _invoke(_ctx("2026-07-24"))

    assert seen["end"] == date(2026, 7, 24)
    assert (seen["end"] - seen["start"]).days == tdc._LOOKBACK_DAYS


def test_check_is_non_blocking_and_registered():
    """Contamination is historical rows written by another path; halting
    today's composite would punish the wrong partition."""
    from meridian_pipeline.checks import ALL_FACTOR_CHECKS

    assert tdc.check_trading_day_invariant in ALL_FACTOR_CHECKS
    spec = next(iter(tdc.check_trading_day_invariant.check_specs))
    assert spec.blocking is False
