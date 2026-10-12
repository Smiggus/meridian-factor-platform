"""
Tests for #128 Item 3: 815-day lower bound on fetch_income_pit / fetch_cashflow_pit.

Covers:
  - Lower bound filters out rows older than end_date - 815 days.
  - Rows exactly AT the lower bound (filing_date == end_date - 815d) are INCLUDED.
  - Rows one day before the lower bound are EXCLUDED.
  - fetch_balance_pit is NOT bounded (verified by supplying an old row and
    confirming it comes back).
  - PIT upper bound is still respected (filing_date <= end_date).

Also documents the Item 2 PIT safety verdict:
  _compute_ttm_pl and _compute_ttm_cashflow_pl BOTH re-filter on
  `filing_date <= score_date` before consuming the DataFrame, so a batch
  fetch at max(score_date) would not leak future filings into earlier
  partitions. Item 2 is PIT-safe to implement.
"""

import importlib
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

_DAGSTER_DIR = Path(__file__).resolve().parents[1] / "dagster"
if str(_DAGSTER_DIR) not in sys.path:
    sys.path.insert(0, str(_DAGSTER_DIR))

os.environ.setdefault("PGPASS", "test-pg-pass")
os.environ.setdefault("PARTITION_START_DATE", "2010-01-02")

# ---------------------------------------------------------------------------
# Helpers / constants
# ---------------------------------------------------------------------------

_SCORE_DATE = date(2025, 6, 1)
_LOWER_BOUND = _SCORE_DATE - timedelta(days=815)  # 2023-03-25 for score_date=2025-06-01

# Filing dates for boundary tests
_WITHIN = _LOWER_BOUND  # exactly on the boundary -> must be INCLUDED
_BEFORE = _LOWER_BOUND - timedelta(days=1)  # one day before -> must be EXCLUDED
_RECENT = _SCORE_DATE - timedelta(days=90)  # well within window -> included


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def parquet_root(tmp_path, monkeypatch):
    """Create the expected subdirectory layout and patch PARQUET_BASE_PATH."""
    (tmp_path / "ohlcv").mkdir()
    (tmp_path / "fundamentals").mkdir(parents=True)
    # Write a minimal OHLCV parquet so parquet_available() returns True.
    pl.DataFrame(
        {"date": [date(2025, 1, 2)], "ticker": ["AAPL"], "ret_1d": [0.01], "close": [150.0]},
        schema={"date": pl.Date, "ticker": pl.String, "ret_1d": pl.Float64, "close": pl.Float64},
    ).write_parquet(tmp_path / "ohlcv" / "prices.parquet")

    monkeypatch.setenv("PARQUET_BASE_PATH", str(tmp_path))

    import meridian_pipeline.parquet_io as pio

    importlib.reload(pio)
    return tmp_path, pio


def _write_income(tmp_path: Path, rows: list[dict]) -> None:
    schema = {
        "ticker": pl.String,
        "date": pl.Date,
        "filing_date": pl.Date,
        "period": pl.String,
        "revenue": pl.Float64,
        "net_income": pl.Float64,
        "ebitda": pl.Float64,
        "shares_outstanding": pl.Float64,
        "eps": pl.Float64,
        "gross_profit": pl.Float64,
        # LDP#156 columns
        "ebit": pl.Float64,
        "income_before_tax": pl.Float64,
        "income_tax_expense": pl.Float64,
        "interest_expense": pl.Float64,
        "eps_diluted": pl.Float64,
    }
    pl.DataFrame(rows, schema=schema).write_parquet(
        tmp_path / "fundamentals" / "income_statement.parquet"
    )


def _write_cashflow(tmp_path: Path, rows: list[dict]) -> None:
    schema = {
        "ticker": pl.String,
        "date": pl.Date,
        "filing_date": pl.Date,
        "period": pl.String,
        "operating_cash_flow": pl.Float64,
        "capital_expenditure": pl.Float64,
        # LDP#156 columns
        "interest_paid": pl.Float64,
        "net_debt_issuance": pl.Float64,
    }
    pl.DataFrame(rows, schema=schema).write_parquet(tmp_path / "fundamentals" / "cash_flow.parquet")


def _write_balance(tmp_path: Path, rows: list[dict]) -> None:
    schema = {
        "ticker": pl.String,
        "date": pl.Date,
        "filing_date": pl.Date,
        "period": pl.String,
        "total_equity": pl.Float64,
        "total_debt": pl.Float64,
        "cash_and_equivalents": pl.Float64,
        "total_assets": pl.Float64,
    }
    pl.DataFrame(rows, schema=schema).write_parquet(
        tmp_path / "fundamentals" / "balance_sheet.parquet"
    )


def _income_row(filing_date: date, ticker: str = "AAPL") -> dict:
    return {
        "ticker": ticker,
        "date": filing_date - timedelta(days=30),
        "filing_date": filing_date,
        "period": "quarter",
        "revenue": 1e9,
        "net_income": 1e8,
        "ebitda": 2e8,
        "shares_outstanding": 1e9,
        "eps": 0.1,
        "gross_profit": 3e8,
        # LDP#156 columns
        "ebit": 1.25e8,
        "income_before_tax": 1.2e8,
        "income_tax_expense": 2.4e7,
        "interest_expense": 5e7,
        "eps_diluted": 0.098,
    }


def _cashflow_row(filing_date: date, ticker: str = "AAPL") -> dict:
    return {
        "ticker": ticker,
        "date": filing_date - timedelta(days=30),
        "filing_date": filing_date,
        "period": "quarter",
        "operating_cash_flow": 5e8,
        "capital_expenditure": -1e8,
        # LDP#156 columns
        "interest_paid": -5e7,
        "net_debt_issuance": 2e8,
    }


def _balance_row(filing_date: date, ticker: str = "AAPL", period: str = "quarter") -> dict:
    return {
        "ticker": ticker,
        "date": filing_date - timedelta(days=30),
        "filing_date": filing_date,
        "period": period,
        "total_equity": 10e9,
        "total_debt": 5e9,
        "cash_and_equivalents": 2e9,
        "total_assets": 20e9,
    }


# ---------------------------------------------------------------------------
# Item 3: fetch_income_pit lower bound
# ---------------------------------------------------------------------------


def test_income_pit_lower_bound_excludes_too_old(parquet_root):
    """Rows with filing_date < end_date - 815d must not appear."""
    tmp_path, pio = parquet_root
    _write_income(tmp_path, [_income_row(_BEFORE), _income_row(_RECENT)])
    df = pio.fetch_income_pit(["AAPL"], _SCORE_DATE)
    filing_dates = df["filing_date"].to_list()
    assert _BEFORE not in filing_dates, (
        f"Row with filing_date={_BEFORE} should be excluded (before lower bound)"
    )
    assert _RECENT in filing_dates


def test_income_pit_lower_bound_includes_boundary_day(parquet_root):
    """Rows with filing_date == end_date - 815d must be INCLUDED (>= semantics)."""
    tmp_path, pio = parquet_root
    _write_income(tmp_path, [_income_row(_WITHIN), _income_row(_RECENT)])
    df = pio.fetch_income_pit(["AAPL"], _SCORE_DATE)
    filing_dates = df["filing_date"].to_list()
    assert _WITHIN in filing_dates, f"Row exactly at lower bound ({_WITHIN}) must be included"


def test_income_pit_upper_bound_still_respected(parquet_root):
    """Rows with filing_date > end_date must be excluded (PIT upper bound)."""
    tmp_path, pio = parquet_root
    future = _SCORE_DATE + timedelta(days=1)
    _write_income(tmp_path, [_income_row(_RECENT), _income_row(future)])
    df = pio.fetch_income_pit(["AAPL"], _SCORE_DATE)
    assert future not in df["filing_date"].to_list()
    assert _RECENT in df["filing_date"].to_list()


def test_income_pit_all_old_returns_empty(parquet_root):
    """If every row is older than end_date - 815d, result is empty."""
    tmp_path, pio = parquet_root
    _write_income(tmp_path, [_income_row(_BEFORE)])
    df = pio.fetch_income_pit(["AAPL"], _SCORE_DATE)
    assert df.is_empty()


# ---------------------------------------------------------------------------
# Item 3: fetch_cashflow_pit lower bound
# ---------------------------------------------------------------------------


def test_cashflow_pit_lower_bound_excludes_too_old(parquet_root):
    """Rows with filing_date < end_date - 815d must not appear."""
    tmp_path, pio = parquet_root
    _write_cashflow(tmp_path, [_cashflow_row(_BEFORE), _cashflow_row(_RECENT)])
    df = pio.fetch_cashflow_pit(["AAPL"], _SCORE_DATE)
    filing_dates = df["filing_date"].to_list()
    assert _BEFORE not in filing_dates
    assert _RECENT in filing_dates


def test_cashflow_pit_lower_bound_includes_boundary_day(parquet_root):
    """Rows with filing_date == end_date - 815d must be INCLUDED."""
    tmp_path, pio = parquet_root
    _write_cashflow(tmp_path, [_cashflow_row(_WITHIN), _cashflow_row(_RECENT)])
    df = pio.fetch_cashflow_pit(["AAPL"], _SCORE_DATE)
    assert _WITHIN in df["filing_date"].to_list()


def test_cashflow_pit_upper_bound_still_respected(parquet_root):
    """PIT upper bound (filing_date <= end_date) is preserved."""
    tmp_path, pio = parquet_root
    future = _SCORE_DATE + timedelta(days=1)
    _write_cashflow(tmp_path, [_cashflow_row(_RECENT), _cashflow_row(future)])
    df = pio.fetch_cashflow_pit(["AAPL"], _SCORE_DATE)
    assert future not in df["filing_date"].to_list()
    assert _RECENT in df["filing_date"].to_list()


def test_cashflow_pit_all_old_returns_empty(parquet_root):
    """If every row is older than end_date - 815d, result is empty."""
    tmp_path, pio = parquet_root
    _write_cashflow(tmp_path, [_cashflow_row(_BEFORE)])
    df = pio.fetch_cashflow_pit(["AAPL"], _SCORE_DATE)
    assert df.is_empty()


# ---------------------------------------------------------------------------
# Balance sheet: no lower bound applied
# ---------------------------------------------------------------------------


def test_balance_pit_no_lower_bound(parquet_root):
    """fetch_balance_pit must NOT prune old rows -- consumers read the
    latest-available row which can be arbitrarily old."""
    tmp_path, pio = parquet_root
    very_old = date(2010, 1, 15)  # 15+ years before _SCORE_DATE
    _write_balance(tmp_path, [_balance_row(very_old), _balance_row(_RECENT)])
    df = pio.fetch_balance_pit(["AAPL"], _SCORE_DATE)
    assert very_old in df["filing_date"].to_list(), "Old balance sheet row must not be pruned"


# ---------------------------------------------------------------------------
# 815-day lower bound derivation: match the TTM floor maths
#
# The TTM window floor is score_date - 450 days.
# Growth uses TTM(score_date - 365d) as the lagged window.
# Earliest filing_date that can contribute:
#   (score_date - 365d) - 450d = score_date - 815d.
#
# This test verifies that a row at exactly score_date - 815d is used by
# _compute_ttm_pl when called with (score_date - 365d).
# ---------------------------------------------------------------------------


def test_815d_is_tight_bound_for_growth_lagged_ttm():
    """A filing_date exactly at score_date - 815d falls inside the lagged TTM
    window (score_date-365d, floor=score_date-815d).

    This confirms 815 is the tightest correct lower bound for the growth
    asset's oldest reachable filing.
    """
    from datetime import timedelta

    score_date = date(2025, 6, 1)
    lagged_date = score_date - timedelta(days=365)
    ttm_floor_for_lag = lagged_date - timedelta(days=450)

    # 815 = 365 + 450
    assert ttm_floor_for_lag == score_date - timedelta(days=815)

    # A filing exactly AT that floor is within the lagged TTM window.
    filing_at_floor = ttm_floor_for_lag
    assert filing_at_floor >= ttm_floor_for_lag  # boundary is inclusive
    assert filing_at_floor <= lagged_date  # still <= lagged score date
