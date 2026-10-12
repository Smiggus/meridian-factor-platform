"""
test_lakehouse_parity.py
-------------------------
Unit tests for lakehouse_parity.compare_stats and staleness_report
(Meridian M14.6 / #415).

Pure-dict / pure-function tests only - no DB, no Iceberg catalog, no Dagster
import. Neither compare_stats nor staleness_report has a dependency on
dagster, DB connections or the Iceberg catalog (staleness_report takes an
injected `is_trading_day` callable instead of an engine), so none of the
slim-CI dagster-skip guards used elsewhere in this test suite are needed
here; this file runs unconditionally.
"""

import sys
from datetime import date
from pathlib import Path

_DAGSTER_DIR = Path(__file__).resolve().parents[1] / "dagster"
if str(_DAGSTER_DIR) not in sys.path:
    sys.path.insert(0, str(_DAGSTER_DIR))

from meridian_pipeline import db as _db_module  # noqa: E402
from meridian_pipeline.lakehouse_parity import (  # noqa: E402
    compare_stats,
    rolling_spotcheck_dates,
    staleness_report,
)


def _stats(rows=100, factors=10, tickers=20, z_checksum=1.234567, raw_checksum=9.876543):
    return {
        "rows": rows,
        "factors": factors,
        "tickers": tickers,
        "z_checksum": z_checksum,
        "raw_checksum": raw_checksum,
    }


def test_compare_stats_exact_match_is_ok():
    pg = _stats()
    ice = _stats()
    result = compare_stats(pg, ice)
    assert result["ok"] is True
    assert result["diffs"] == {}


def test_compare_stats_checksum_within_tolerance_is_ok():
    pg = _stats(z_checksum=1.0000, raw_checksum=5.0000)
    ice = _stats(z_checksum=1.0005, raw_checksum=5.0003)
    result = compare_stats(pg, ice, checksum_tol=1e-3)
    assert result["ok"] is True
    assert result["diffs"] == {}


def test_compare_stats_large_magnitude_reordering_passes():
    """A large-magnitude case: raw_value sums ~1e12 differ by ~0.7 in absolute
    terms (float reordering) but ~7e-13 relative. The scale-aware band
    (1e-9 * magnitude ~= 1000) must absorb this and PASS, where a fixed 1e-3
    absolute floor would spuriously FAIL."""
    pg = _stats(raw_checksum=1_000_000_000_000.1234)
    ice = _stats(raw_checksum=1_000_000_000_000.8000)
    result = compare_stats(pg, ice)
    assert result["ok"] is True
    assert result["diffs"] == {}


def test_compare_stats_large_magnitude_genuine_divergence_fails():
    """A genuinely divergent value set (e.g. a corrupt/missing ~1e6-scale row)
    still fails even at large magnitude: 1e6 diff on a 1e12 sum is 1e-6 relative,
    well beyond the 1e-9 relative band."""
    pg = _stats(raw_checksum=1_000_000_000_000.0)
    ice = _stats(raw_checksum=1_000_000_000_000.0 + 1_000_000.0)
    result = compare_stats(pg, ice)
    assert result["ok"] is False
    assert "raw_checksum" in result["diffs"]


def test_compare_stats_checksum_beyond_tolerance_fails():
    pg = _stats(z_checksum=1.0000)
    ice = _stats(z_checksum=1.01)
    result = compare_stats(pg, ice, checksum_tol=1e-3)
    assert result["ok"] is False
    assert "z_checksum" in result["diffs"]
    assert result["diffs"]["z_checksum"] == (1.0000, 1.01)
    # raw_checksum matched exactly - only z_checksum should be flagged.
    assert "raw_checksum" not in result["diffs"]


def test_compare_stats_row_count_mismatch_fails_regardless_of_tolerance():
    pg = _stats(rows=100)
    ice = _stats(rows=99)
    result = compare_stats(pg, ice, checksum_tol=1e6)  # absurdly loose checksum tol
    assert result["ok"] is False
    assert result["diffs"]["rows"] == (100, 99)


def test_compare_stats_multiple_mismatches_all_reported():
    pg = _stats(rows=100, factors=10, tickers=20, z_checksum=1.0, raw_checksum=5.0)
    ice = _stats(rows=100, factors=9, tickers=21, z_checksum=1.0, raw_checksum=50.0)
    result = compare_stats(pg, ice, checksum_tol=1e-3)
    assert result["ok"] is False
    assert set(result["diffs"].keys()) == {"factors", "tickers", "raw_checksum"}
    assert result["diffs"]["factors"] == (10, 9)
    assert result["diffs"]["tickers"] == (20, 21)


def test_compare_stats_handles_missing_keys_as_none_or_zero():
    pg = _stats()
    ice = {"rows": pg["rows"], "factors": pg["factors"], "tickers": pg["tickers"]}
    # missing checksums in ice default to 0.0 via .get(..., None) -> float(None or 0.0)
    result = compare_stats(pg, ice, checksum_tol=1e-3)
    assert result["ok"] is False
    assert "z_checksum" in result["diffs"]
    assert "raw_checksum" in result["diffs"]


# ─────────────────────────────────────────────
# staleness_report (#415)
# ─────────────────────────────────────────────
#
# A fixed real calendar week so trading-day fakes are easy to reason about:
# 2026-09-14 (Mon) .. 2026-09-20 (Sun). weekdays_only() treats Sat/Sun as
# non-trading, matching db.is_trading_day's real-world shape closely enough
# for these pure-function tests (they never touch db.is_trading_day itself).
def _weekdays_only(d: date) -> bool:
    return d.weekday() < 5


def test_staleness_report_clean_parity_is_ok():
    """(c) Clean parity: both sides share the same latest date."""
    d = date(2026, 9, 17)
    result = staleness_report(pg_max=d, ice_max=d, is_trading_day=_weekdays_only)
    assert result["ok"] is True
    assert result["lag_trading_days"] == 0


def test_staleness_report_mirror_ahead_of_pg_is_ok():
    """Mirror can be equal-or-ahead (e.g. it also holds a promoted batch PG
    hasn't seen yet via a different write path) without that being a
    staleness finding - this check is about the mirror falling BEHIND."""
    result = staleness_report(
        pg_max=date(2026, 9, 16),
        ice_max=date(2026, 9, 17),
        is_trading_day=_weekdays_only,
    )
    assert result["ok"] is True
    assert result["lag_trading_days"] == 0


def test_staleness_report_stale_mirror_max_date_fails():
    """(b) The #415 shape exactly: every date the mirror holds compares
    clean (per-date counts would all match), but the mirror's newest date is
    behind PG's. This is what a pure per-date trailing-window count
    comparison over dates BOTH sides already have cannot see - it must be
    caught by comparing MAX(score_date) directly."""
    pg_max = date(2026, 9, 17)  # Thursday
    ice_max = date(2026, 9, 16)  # Wednesday - one trading day behind
    result = staleness_report(pg_max=pg_max, ice_max=ice_max, is_trading_day=_weekdays_only)
    assert result["ok"] is False
    assert result["lag_trading_days"] == 1
    assert result["pg_max"] == pg_max
    assert result["ice_max"] == ice_max


def test_staleness_report_lag_counts_trading_days_not_calendar_days():
    """Mirror stopped at Friday close, PG's latest is the following Monday:
    3 CALENDAR days elapsed but only 1 TRADING day (Sat/Sun don't count). A
    naive calendar-day delta would over-count this as 3; the real lag is 1
    (the single missing Monday session)."""
    friday = date(2026, 9, 18)
    monday = date(2026, 9, 21)
    result = staleness_report(
        pg_max=monday, ice_max=friday, is_trading_day=_weekdays_only, max_lag_trading_days=0
    )
    assert result["lag_trading_days"] == 1
    assert (monday - friday).days == 3  # the calendar delta this must NOT be mistaken for
    assert result["ok"] is False


def test_staleness_report_lag_beyond_tolerance_fails():
    pg_max = date(2026, 9, 21)  # Monday
    ice_max = date(2026, 9, 16)  # Wednesday - 3 trading days behind (Th, Fr, Mo)
    result = staleness_report(
        pg_max=pg_max, ice_max=ice_max, is_trading_day=_weekdays_only, max_lag_trading_days=1
    )
    assert result["ok"] is False
    assert result["lag_trading_days"] == 3


def test_staleness_report_lag_within_explicit_tolerance_passes():
    pg_max = date(2026, 9, 17)  # Thursday
    ice_max = date(2026, 9, 16)  # Wednesday - 1 trading day behind
    result = staleness_report(
        pg_max=pg_max, ice_max=ice_max, is_trading_day=_weekdays_only, max_lag_trading_days=1
    )
    assert result["ok"] is True


def test_staleness_report_missing_mirror_data_fails():
    """Iceberg side returned nothing at all in the scanned window (e.g. the
    catalog answered but the window floor missed all its data, or the table
    is otherwise empty) while PG has a latest date - can't be ok."""
    result = staleness_report(pg_max=date(2026, 9, 17), ice_max=None, is_trading_day=_weekdays_only)
    assert result["ok"] is False
    assert result["lag_trading_days"] is None


def test_staleness_report_no_pg_data_is_ok_trivially():
    """No score_date in PG at all (fresh/empty environment) - nothing to be
    stale against, so this must not be a finding (matches the asset's own
    pre-check: it skips clean before staleness_check is ever called when
    get_latest_score_date() returns None)."""
    result = staleness_report(pg_max=None, ice_max=None, is_trading_day=_weekdays_only)
    assert result["ok"] is True


def test_staleness_report_reproduces_415_incident_shape():
    """End-to-end shape of a dual-write-skip incident (#415): a date was
    written to PG but the dual-write to the mirror was skipped, so the
    mirror's latest date stalls while PG advances over the following days.
    Confirms the lag accumulates day over day until the
    repair lands, rather than being masked by a one-off tolerance."""
    ice_max = date(2026, 9, 16)
    day_after = staleness_report(
        pg_max=date(2026, 9, 17), ice_max=ice_max, is_trading_day=_weekdays_only
    )
    two_days_after = staleness_report(
        pg_max=date(2026, 9, 18), ice_max=ice_max, is_trading_day=_weekdays_only
    )
    assert day_after["ok"] is False
    assert day_after["lag_trading_days"] == 1
    assert two_days_after["ok"] is False
    assert two_days_after["lag_trading_days"] == 2


def test_staleness_and_count_dimensions_are_independent():
    """A fabricated pair where per-date counts match for every date the
    mirror holds (compare_stats would report clean on all of them) but the
    mirror is simply missing the newest date - the failure a count-only
    trailing-window comparison over existing dates cannot see, only the
    staleness dimension catches it."""
    shared_stats = {
        "rows": 250000,
        "factors": 50,
        "tickers": 3000,
        "z_checksum": 12.5,
        "raw_checksum": 998877.1,
    }
    # Every date both sides have compares clean.
    for _ in range(5):
        assert compare_stats(shared_stats, dict(shared_stats))["ok"] is True

    # But the mirror never received the newest date at all.
    result = staleness_report(
        pg_max=date(2026, 9, 17), ice_max=date(2026, 9, 16), is_trading_day=_weekdays_only
    )
    assert result["ok"] is False


# ─────────────────────────────────────────────
# rolling_spotcheck_dates (#420 detection gap)
# ─────────────────────────────────────────────
#
# The trailing MERIDIAN_PARITY_WINDOW_DAYS window (default 5) can never see a
# defect outside itself - exactly how months of #420 orphan rows sat
# undetected. These tests exercise the bounded, rotating full-history spot
# check that closes that gap, with `meridian_pipeline.db.is_trading_day`
# faked so no DB connection is needed.


def _all_days_are_trading_days(engine, d):
    return True


def test_rolling_spotcheck_dates_empty_range_returns_nothing(monkeypatch):
    monkeypatch.setattr(_db_module, "is_trading_day", _all_days_are_trading_days)
    result = rolling_spotcheck_dates(
        engine=None,
        floor_date=date(2020, 1, 2),
        ceiling_date=date(2019, 1, 1),  # ceiling before floor
        n=5,
        anchor=date(2020, 1, 1),
    )
    assert result == []


def test_rolling_spotcheck_dates_zero_n_returns_nothing(monkeypatch):
    monkeypatch.setattr(_db_module, "is_trading_day", _all_days_are_trading_days)
    result = rolling_spotcheck_dates(
        engine=None,
        floor_date=date(2020, 1, 1),
        ceiling_date=date(2020, 12, 31),
        n=0,
        anchor=date(2020, 6, 1),
    )
    assert result == []


def test_rolling_spotcheck_dates_bounded_call_count(monkeypatch):
    """Cost bound (#420): at most n dates returned, and at most
    n * (bucket_search_days + 1) is_trading_day calls regardless of how long
    the requested history range is - a 16-year range costs the same as a
    1-year range. This bound is what makes the spot check safe to run every
    scheduled tick."""
    calls = []

    def _counting_is_trading_day(engine, d):
        calls.append(d)
        return True

    monkeypatch.setattr(_db_module, "is_trading_day", _counting_is_trading_day)
    floor_date = date(2010, 1, 1)
    ceiling_date = date(2026, 1, 1)
    n = 10
    result = rolling_spotcheck_dates(
        engine=None, floor_date=floor_date, ceiling_date=ceiling_date, n=n, anchor=date(2026, 9, 22)
    )
    assert len(result) <= n
    assert len(calls) <= n * (5 + 1)  # default bucket_search_days=5
    assert all(floor_date <= d <= ceiling_date for d in result)


def test_rolling_spotcheck_dates_deduplicates_and_sorts(monkeypatch):
    monkeypatch.setattr(_db_module, "is_trading_day", _all_days_are_trading_days)
    result = rolling_spotcheck_dates(
        engine=None,
        floor_date=date(2020, 1, 1),
        ceiling_date=date(2020, 1, 10),
        n=20,
        anchor=date(2020, 1, 1),
    )
    assert result == sorted(set(result))


def test_rolling_spotcheck_dates_rotates_with_anchor(monkeypatch):
    """The rotation property that gives eventual full-history coverage
    without a persisted cursor (#420): two different anchors over the same
    range probe different points, so consecutive scheduled ticks sample
    different history."""
    monkeypatch.setattr(_db_module, "is_trading_day", _all_days_are_trading_days)
    floor_date = date(2010, 1, 1)
    ceiling_date = date(2026, 1, 1)
    n = 5
    d1 = rolling_spotcheck_dates(
        engine=None, floor_date=floor_date, ceiling_date=ceiling_date, n=n, anchor=date(2026, 1, 1)
    )
    d2 = rolling_spotcheck_dates(
        engine=None, floor_date=floor_date, ceiling_date=ceiling_date, n=n, anchor=date(2026, 1, 2)
    )
    assert d1 != d2


def test_rolling_spotcheck_dates_skips_bucket_with_no_trading_day(monkeypatch):
    """A bucket that never lands on a trading day within bucket_search_days
    is skipped rather than raising or returning a non-trading day."""
    monkeypatch.setattr(_db_module, "is_trading_day", lambda engine, d: False)
    result = rolling_spotcheck_dates(
        engine=None,
        floor_date=date(2020, 1, 1),
        ceiling_date=date(2020, 12, 31),
        n=5,
        anchor=date(2020, 6, 1),
    )
    assert result == []
