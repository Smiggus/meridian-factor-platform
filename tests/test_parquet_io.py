"""
Integration tests for parquet_io.py using real Parquet files in a temp directory.

No DB, no network. Creates minimal Parquet fixtures with Polars and exercises
every public function:
  - parquet_available()
  - fetch_ohlcv_ret1d()   (strict_start True and False)
  - fetch_latest_prices()
  - fetch_income_pit()
  - fetch_balance_pit()
  - fetch_cashflow_pit()
  - fetch_ohlcv_prices()

PARQUET_BASE_PATH is overridden per-test via monkeypatch + module-level reload.
"""

import importlib
import os
import sys
from datetime import date
from pathlib import Path

import polars as pl
import pytest

_DAGSTER_DIR = Path(__file__).resolve().parents[1] / "dagster"
if str(_DAGSTER_DIR) not in sys.path:
    sys.path.insert(0, str(_DAGSTER_DIR))

os.environ.setdefault("PGPASS", "test-pg-pass")
os.environ.setdefault("PARTITION_START_DATE", "2010-01-02")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def parquet_root(tmp_path, monkeypatch):
    """Create the expected subdirectory layout and patch PARQUET_BASE_PATH."""
    (tmp_path / "ohlcv").mkdir()
    (tmp_path / "fundamentals").mkdir(parents=True)

    monkeypatch.setenv("PARQUET_BASE_PATH", str(tmp_path))

    # Re-import so the module-level PARQUET_BASE_PATH constant picks up the env var.
    import meridian_pipeline.parquet_io as pio

    importlib.reload(pio)
    return tmp_path, pio


@pytest.fixture()
def ohlcv_parquet(parquet_root):
    """Write a small OHLCV Parquet file and return (tmp_path, pio)."""
    tmp_path, pio = parquet_root
    rows = [
        {
            "date": date(2025, 1, 2),
            "ticker": "AAPL",
            "ret_1d": 0.01,
            "close": 150.0,
            "adj_close": 150.0,
            "raw_close": 150.0,
        },
        {
            "date": date(2025, 1, 3),
            "ticker": "AAPL",
            "ret_1d": -0.005,
            "close": 149.25,
            "adj_close": 149.25,
            "raw_close": 149.25,
        },
        {
            "date": date(2025, 1, 6),
            "ticker": "AAPL",
            "ret_1d": 0.02,
            "close": 152.24,
            "adj_close": 152.24,
            "raw_close": 152.24,
        },
        {
            "date": date(2025, 1, 2),
            "ticker": "MSFT",
            "ret_1d": 0.005,
            "close": 300.0,
            "adj_close": 300.0,
            "raw_close": 300.0,
        },
        {
            "date": date(2025, 1, 3),
            "ticker": "MSFT",
            "ret_1d": 0.008,
            "close": 302.4,
            "adj_close": 302.4,
            "raw_close": 302.4,
        },
    ]
    df = pl.DataFrame(
        rows,
        schema={
            "date": pl.Date,
            "ticker": pl.String,
            "ret_1d": pl.Float64,
            "close": pl.Float64,
            "adj_close": pl.Float64,
            "raw_close": pl.Float64,
        },
    )
    df.write_parquet(tmp_path / "ohlcv" / "prices.parquet")
    return tmp_path, pio


@pytest.fixture()
def income_parquet(parquet_root):
    tmp_path, pio = parquet_root
    rows = [
        {
            "ticker": "AAPL",
            "date": date(2024, 9, 30),
            "filing_date": date(2024, 11, 1),
            "period": "quarter",
            "revenue": 90e9,
            "net_income": 20e9,
            "ebitda": 25e9,
            "shares_outstanding": 15e9,
            "eps": 1.33,
            "gross_profit": 35e9,
            "ebit": 22e9,
            "income_before_tax": 21e9,
            "income_tax_expense": 4e9,
            "interest_expense": 1e9,
            "eps_diluted": 1.30,
        },
        {
            "ticker": "AAPL",
            "date": date(2024, 6, 30),
            "filing_date": date(2024, 8, 1),
            "period": "quarter",
            "revenue": 85e9,
            "net_income": 19e9,
            "ebitda": 23e9,
            "shares_outstanding": 15e9,
            "eps": 1.27,
            "gross_profit": 33e9,
            "ebit": 21e9,
            "income_before_tax": 20e9,
            "income_tax_expense": 3.8e9,
            "interest_expense": 1e9,
            "eps_diluted": 1.24,
        },
        {
            "ticker": "MSFT",
            "date": date(2024, 9, 30),
            "filing_date": date(2024, 10, 31),
            "period": "quarter",
            "revenue": 65e9,
            "net_income": 22e9,
            "ebitda": 28e9,
            "shares_outstanding": 7.4e9,
            "eps": 2.97,
            "gross_profit": 44e9,
            "ebit": 25e9,
            "income_before_tax": 24e9,
            "income_tax_expense": 4.5e9,
            "interest_expense": 1e9,
            "eps_diluted": 2.90,
        },
    ]
    df = pl.DataFrame(
        rows,
        schema={
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
            "ebit": pl.Float64,
            "income_before_tax": pl.Float64,
            "income_tax_expense": pl.Float64,
            "interest_expense": pl.Float64,
            "eps_diluted": pl.Float64,
        },
    )
    df.write_parquet(tmp_path / "fundamentals" / "income_statement.parquet")
    return tmp_path, pio


@pytest.fixture()
def balance_parquet(parquet_root):
    tmp_path, pio = parquet_root
    rows = [
        {
            "ticker": "AAPL",
            "date": date(2024, 9, 30),
            "filing_date": date(2024, 11, 1),
            "period": "quarter",
            "total_equity": 50e9,
            "total_debt": 100e9,
            "cash_and_equivalents": 30e9,
            "total_assets": 350e9,
        },
        {
            "ticker": "MSFT",
            "date": date(2024, 9, 30),
            "filing_date": date(2024, 10, 31),
            "period": "annual",
            "total_equity": 200e9,
            "total_debt": 50e9,
            "cash_and_equivalents": 80e9,
            "total_assets": 420e9,
        },
    ]
    df = pl.DataFrame(
        rows,
        schema={
            "ticker": pl.String,
            "date": pl.Date,
            "filing_date": pl.Date,
            "period": pl.String,
            "total_equity": pl.Float64,
            "total_debt": pl.Float64,
            "cash_and_equivalents": pl.Float64,
            "total_assets": pl.Float64,
        },
    )
    df.write_parquet(tmp_path / "fundamentals" / "balance_sheet.parquet")
    return tmp_path, pio


@pytest.fixture()
def cashflow_parquet(parquet_root):
    tmp_path, pio = parquet_root
    rows = [
        {
            "ticker": "AAPL",
            "date": date(2024, 9, 30),
            "filing_date": date(2024, 11, 1),
            "period": "quarter",
            "operating_cash_flow": 25e9,
            "capital_expenditure": -3e9,
            "interest_paid": -1e9,
            "net_debt_issuance": 2e9,
        },
        {
            "ticker": "MSFT",
            "date": date(2024, 9, 30),
            "filing_date": date(2024, 10, 31),
            "period": "quarter",
            "operating_cash_flow": 30e9,
            "capital_expenditure": -5e9,
            "interest_paid": -0.8e9,
            "net_debt_issuance": 1e9,
        },
    ]
    df = pl.DataFrame(
        rows,
        schema={
            "ticker": pl.String,
            "date": pl.Date,
            "filing_date": pl.Date,
            "period": pl.String,
            "operating_cash_flow": pl.Float64,
            "capital_expenditure": pl.Float64,
            "interest_paid": pl.Float64,
            "net_debt_issuance": pl.Float64,
        },
    )
    df.write_parquet(tmp_path / "fundamentals" / "cash_flow.parquet")
    return tmp_path, pio


# ---------------------------------------------------------------------------
# parquet_available
# ---------------------------------------------------------------------------


def test_parquet_available_true(parquet_root):
    _, pio = parquet_root
    assert pio.parquet_available() is True


def test_parquet_available_false_missing_ohlcv(tmp_path, monkeypatch):
    # Base path exists but ohlcv subdir absent.
    monkeypatch.setenv("PARQUET_BASE_PATH", str(tmp_path))
    import meridian_pipeline.parquet_io as pio

    importlib.reload(pio)
    assert pio.parquet_available() is False


def test_parquet_available_false_nonexistent_root(tmp_path, monkeypatch):
    monkeypatch.setenv("PARQUET_BASE_PATH", str(tmp_path / "does_not_exist"))
    import meridian_pipeline.parquet_io as pio

    importlib.reload(pio)
    assert pio.parquet_available() is False


# ---------------------------------------------------------------------------
# LDP_MINIO_ENDPOINT resolution + startup logging (Meridian#239)
# ---------------------------------------------------------------------------


def test_ldp_minio_endpoint_env_wins_over_default(monkeypatch, caplog):
    """When LDP_MINIO_ENDPOINT is set, it wins and the source is logged as
    'env:LDP_MINIO_ENDPOINT' -- not the (possibly stale) code default."""
    monkeypatch.setenv("LDP_MINIO_ENDPOINT", "minio-new.example.internal:9000")
    import meridian_pipeline.parquet_io as pio

    with caplog.at_level("INFO"):
        importlib.reload(pio)

    assert pio.LDP_MINIO_ENDPOINT == "minio-new.example.internal:9000"
    assert pio._LDP_MINIO_ENDPOINT_SOURCE == "env:LDP_MINIO_ENDPOINT"
    assert any(
        "minio-new.example.internal:9000" in rec.message
        and "source=env:LDP_MINIO_ENDPOINT" in rec.message
        for rec in caplog.records
    )
    # Env-sourced value must not carry the code-default warning suffix.
    assert not any("using code default" in rec.message for rec in caplog.records)


def test_ldp_minio_endpoint_falls_back_to_default_when_unset(monkeypatch, caplog):
    """When LDP_MINIO_ENDPOINT is unset, the code default is used as a LAST
    RESORT and the startup log flags it as a code default (not env-
    configured), per Meridian#239's diagnosability ask. The default itself
    (object-store.example.internal:9500) is the current in-cluster
    topology as of the #239 repo-side fix, not the pre-2026-07-11 dead
    hostname."""
    monkeypatch.delenv("LDP_MINIO_ENDPOINT", raising=False)
    import meridian_pipeline.parquet_io as pio

    with caplog.at_level("INFO"):
        importlib.reload(pio)

    assert pio.LDP_MINIO_ENDPOINT == pio._LDP_MINIO_ENDPOINT_DEFAULT
    assert pio._LDP_MINIO_ENDPOINT_DEFAULT == "object-store.example.internal:9500"
    assert pio._LDP_MINIO_ENDPOINT_SOURCE == "code-default"
    assert any(
        "source=code-default" in rec.message and "using code default" in rec.message
        for rec in caplog.records
    )


# ---------------------------------------------------------------------------
# fetch_ohlcv_ret1d
# ---------------------------------------------------------------------------


def test_fetch_ohlcv_ret1d_strict_start_excludes_boundary(ohlcv_parquet):
    _, pio = ohlcv_parquet
    # start_date = 2025-01-02 with strict_start=True -> date > 2025-01-02
    df = pio.fetch_ohlcv_ret1d(["AAPL"], date(2025, 1, 2), date(2025, 1, 6), strict_start=True)
    assert date(2025, 1, 2) not in df["date"].to_list()
    assert date(2025, 1, 3) in df["date"].to_list()
    assert date(2025, 1, 6) in df["date"].to_list()


def test_fetch_ohlcv_ret1d_inclusive_start(ohlcv_parquet):
    _, pio = ohlcv_parquet
    df = pio.fetch_ohlcv_ret1d(["AAPL"], date(2025, 1, 2), date(2025, 1, 6), strict_start=False)
    assert date(2025, 1, 2) in df["date"].to_list()


def test_fetch_ohlcv_ret1d_columns(ohlcv_parquet):
    _, pio = ohlcv_parquet
    df = pio.fetch_ohlcv_ret1d(["AAPL"], date(2025, 1, 1), date(2025, 1, 6))
    assert set(df.columns) == {"date", "ticker", "ret_1d"}


def test_fetch_ohlcv_ret1d_ticker_filter(ohlcv_parquet):
    _, pio = ohlcv_parquet
    df = pio.fetch_ohlcv_ret1d(["AAPL"], date(2025, 1, 1), date(2025, 1, 6))
    assert set(df["ticker"].unique().to_list()) == {"AAPL"}


def test_fetch_ohlcv_ret1d_multiple_tickers(ohlcv_parquet):
    _, pio = ohlcv_parquet
    df = pio.fetch_ohlcv_ret1d(["AAPL", "MSFT"], date(2025, 1, 1), date(2025, 1, 6))
    assert {"AAPL", "MSFT"} == set(df["ticker"].unique().to_list())


def test_fetch_ohlcv_ret1d_values(ohlcv_parquet):
    _, pio = ohlcv_parquet
    df = pio.fetch_ohlcv_ret1d(["AAPL"], date(2025, 1, 2), date(2025, 1, 3), strict_start=False)
    row = df.filter(pl.col("date") == date(2025, 1, 2))
    assert float(row["ret_1d"][0]) == pytest.approx(0.01)


def test_fetch_ohlcv_ret1d_empty_ticker(ohlcv_parquet):
    _, pio = ohlcv_parquet
    df = pio.fetch_ohlcv_ret1d(["NOTEXIST"], date(2025, 1, 1), date(2025, 1, 6))
    assert df.is_empty()


# ---------------------------------------------------------------------------
# fetch_latest_prices
# ---------------------------------------------------------------------------


def test_fetch_latest_prices_returns_latest(ohlcv_parquet):
    _, pio = ohlcv_parquet
    df = pio.fetch_latest_prices(["AAPL"], date(2025, 1, 6))
    assert len(df) == 1
    assert float(df["close"][0]) == pytest.approx(152.24)


def test_fetch_latest_prices_as_of_filter(ohlcv_parquet):
    _, pio = ohlcv_parquet
    # As of 2025-01-02 -> should return only the 2025-01-02 row.
    df = pio.fetch_latest_prices(["AAPL"], date(2025, 1, 2))
    assert float(df["close"][0]) == pytest.approx(150.0)


def test_fetch_latest_prices_columns(ohlcv_parquet):
    _, pio = ohlcv_parquet
    df = pio.fetch_latest_prices(["AAPL"], date(2025, 1, 6))
    assert set(df.columns) == {"ticker", "close"}


def test_fetch_latest_prices_missing_ticker(ohlcv_parquet):
    _, pio = ohlcv_parquet
    df = pio.fetch_latest_prices(["NOTEXIST"], date(2025, 1, 6))
    assert df.is_empty()


def test_fetch_latest_prices_multiple_tickers(ohlcv_parquet):
    _, pio = ohlcv_parquet
    df = pio.fetch_latest_prices(["AAPL", "MSFT"], date(2025, 1, 6))
    assert set(df["ticker"].to_list()) == {"AAPL", "MSFT"}


# ---------------------------------------------------------------------------
# fetch_income_pit
# ---------------------------------------------------------------------------


def test_fetch_income_pit_pit_filter(income_parquet):
    _, pio = income_parquet
    # filing_date <= 2024-09-01 -> only the June quarter (Aug 1 filing).
    df = pio.fetch_income_pit(["AAPL"], date(2024, 9, 1))
    assert len(df) == 1
    assert date(2024, 8, 1) in df["filing_date"].to_list()


def test_fetch_income_pit_includes_boundary(income_parquet):
    _, pio = income_parquet
    df = pio.fetch_income_pit(["AAPL"], date(2024, 11, 1))
    assert len(df) == 2


def test_fetch_income_pit_columns(income_parquet):
    _, pio = income_parquet
    df = pio.fetch_income_pit(["AAPL"], date(2024, 12, 31))
    expected = {
        "ticker",
        "date",
        "filing_date",
        "revenue",
        "net_income",
        "ebitda",
        "shares_outstanding",
        "eps",
        "gross_profit",
        # LDP#156 columns
        "ebit",
        "income_before_tax",
        "income_tax_expense",
        "interest_expense",
        "eps_diluted",
    }
    assert expected.issubset(set(df.columns))


def test_fetch_income_pit_ticker_filter(income_parquet):
    _, pio = income_parquet
    df = pio.fetch_income_pit(["MSFT"], date(2024, 12, 31))
    assert set(df["ticker"].unique().to_list()) == {"MSFT"}


def test_fetch_income_pit_empty_result(income_parquet):
    _, pio = income_parquet
    # end_date before any filing.
    df = pio.fetch_income_pit(["AAPL"], date(2024, 1, 1))
    assert df.is_empty()


# ---------------------------------------------------------------------------
# fetch_balance_pit
# ---------------------------------------------------------------------------


def test_fetch_balance_pit_pit_filter(balance_parquet):
    _, pio = balance_parquet
    df = pio.fetch_balance_pit(["AAPL", "MSFT"], date(2024, 10, 31))
    # MSFT filing 2024-10-31 should be included; AAPL filing 2024-11-01 excluded.
    assert date(2024, 10, 31) in df["filing_date"].to_list()
    assert date(2024, 11, 1) not in df["filing_date"].to_list()


def test_fetch_balance_pit_columns(balance_parquet):
    _, pio = balance_parquet
    df = pio.fetch_balance_pit(["AAPL"], date(2024, 12, 31))
    expected = {
        "ticker",
        "date",
        "filing_date",
        "period",
        "total_equity",
        "total_debt",
        "cash_and_equivalents",
        "total_assets",
    }
    assert expected.issubset(set(df.columns))


def test_fetch_balance_pit_includes_annual_and_quarterly(balance_parquet):
    _, pio = balance_parquet
    # balance_pit returns both annual and quarterly (no period filter).
    df = pio.fetch_balance_pit(["AAPL", "MSFT"], date(2024, 12, 31))
    periods = set(df["period"].to_list())
    assert "quarter" in periods
    assert "annual" in periods


# ---------------------------------------------------------------------------
# fetch_cashflow_pit
# ---------------------------------------------------------------------------


def test_fetch_cashflow_pit_pit_filter(cashflow_parquet):
    _, pio = cashflow_parquet
    # Only MSFT has filing_date <= 2024-10-31.
    df = pio.fetch_cashflow_pit(["AAPL", "MSFT"], date(2024, 10, 31))
    assert set(df["ticker"].to_list()) == {"MSFT"}


def test_fetch_cashflow_pit_columns(cashflow_parquet):
    _, pio = cashflow_parquet
    df = pio.fetch_cashflow_pit(["AAPL"], date(2024, 12, 31))
    expected = {
        "ticker",
        "date",
        "filing_date",
        "operating_cash_flow",
        "capital_expenditure",
        # LDP#156 columns
        "interest_paid",
        "net_debt_issuance",
    }
    assert expected.issubset(set(df.columns))


def test_fetch_cashflow_pit_values(cashflow_parquet):
    _, pio = cashflow_parquet
    df = pio.fetch_cashflow_pit(["AAPL"], date(2024, 12, 31))
    row = df.filter(pl.col("ticker") == "AAPL")
    assert float(row["operating_cash_flow"][0]) == pytest.approx(25e9)


# ---------------------------------------------------------------------------
# fetch_ohlcv_prices
# ---------------------------------------------------------------------------


def test_fetch_ohlcv_prices_default_inclusive_start(ohlcv_parquet):
    _, pio = ohlcv_parquet
    df = pio.fetch_ohlcv_prices(["AAPL"], date(2025, 1, 2), date(2025, 1, 6))
    assert date(2025, 1, 2) in df["date"].to_list()


def test_fetch_ohlcv_prices_strict_start(ohlcv_parquet):
    _, pio = ohlcv_parquet
    df = pio.fetch_ohlcv_prices(["AAPL"], date(2025, 1, 2), date(2025, 1, 6), strict_start=True)
    assert date(2025, 1, 2) not in df["date"].to_list()


def test_fetch_ohlcv_prices_uses_adj_close(ohlcv_parquet):
    _, pio = ohlcv_parquet
    df = pio.fetch_ohlcv_prices(["AAPL"], date(2025, 1, 2), date(2025, 1, 2))
    # adj_close == close in the fixture -> close column value should match.
    assert float(df["close"][0]) == pytest.approx(150.0)


def test_fetch_ohlcv_prices_columns(ohlcv_parquet):
    _, pio = ohlcv_parquet
    df = pio.fetch_ohlcv_prices(["AAPL"], date(2025, 1, 1), date(2025, 1, 6))
    assert set(df.columns) == {"date", "ticker", "close"}


def test_fetch_ohlcv_prices_ticker_filter(ohlcv_parquet):
    _, pio = ohlcv_parquet
    df = pio.fetch_ohlcv_prices(["MSFT"], date(2025, 1, 1), date(2025, 1, 6))
    assert set(df["ticker"].unique().to_list()) == {"MSFT"}


# ---------------------------------------------------------------------------
# #125: raw_close / adj_close column selection (split-correct market cap)
# ---------------------------------------------------------------------------


@pytest.fixture()
def ohlcv_split_parquet(parquet_root):
    """OHLCV fixture where close / raw_close / adj_close DIVERGE (a pre-split day),
    so we can assert the right column is selected. Mirrors an AAPL-style 4:1 split:
    raw_close = observed pre-split price, close = split-adjusted, adj_close = +div."""
    tmp_path, pio = parquet_root
    rows = [
        {
            "date": date(2025, 1, 2),
            "ticker": "AAPL",
            "ret_1d": 0.0,
            "close": 150.0,
            "raw_close": 600.0,
            "adj_close": 148.0,
        },
        {
            "date": date(2025, 1, 3),
            "ticker": "AAPL",
            "ret_1d": 0.01,
            "close": 152.24,
            "raw_close": 608.96,
            "adj_close": 150.2,
        },
    ]
    pl.DataFrame(
        rows,
        schema={
            "date": pl.Date,
            "ticker": pl.String,
            "ret_1d": pl.Float64,
            "close": pl.Float64,
            "raw_close": pl.Float64,
            "adj_close": pl.Float64,
        },
    ).write_parquet(tmp_path / "ohlcv" / "prices.parquet")
    return tmp_path, pio


@pytest.fixture()
def ohlcv_legacy_parquet(parquet_root):
    """OHLCV fixture WITHOUT raw_close/adj_close (pre-LDP#137 export). The fetches
    must fall back to plain `close` rather than error on a missing column."""
    tmp_path, pio = parquet_root
    rows = [
        {"date": date(2025, 1, 2), "ticker": "AAPL", "ret_1d": 0.0, "close": 150.0},
        {"date": date(2025, 1, 3), "ticker": "AAPL", "ret_1d": 0.01, "close": 152.24},
    ]
    pl.DataFrame(
        rows,
        schema={
            "date": pl.Date,
            "ticker": pl.String,
            "ret_1d": pl.Float64,
            "close": pl.Float64,
        },
    ).write_parquet(tmp_path / "ohlcv" / "prices.parquet")
    return tmp_path, pio


def test_latest_prices_uses_raw_close(ohlcv_split_parquet):
    _, pio = ohlcv_split_parquet
    assert pio.parquet_available()
    df = pio.fetch_latest_prices(["AAPL"], date(2025, 1, 3))
    # latest row 2025-01-03 -> raw_close 608.96, NOT split-adjusted close 152.24
    assert float(df["close"][0]) == pytest.approx(608.96)


def test_ohlcv_prices_uses_adj_close(ohlcv_split_parquet):
    _, pio = ohlcv_split_parquet
    assert pio.parquet_available()
    df = pio.fetch_ohlcv_prices(["AAPL"], date(2025, 1, 2), date(2025, 1, 2))
    # adj_close 148.0, NOT close 150.0 / raw_close 600.0
    assert float(df["close"][0]) == pytest.approx(148.0)


def test_latest_prices_falls_back_to_close_when_no_raw(ohlcv_legacy_parquet):
    _, pio = ohlcv_legacy_parquet
    assert pio.parquet_available()
    df = pio.fetch_latest_prices(["AAPL"], date(2025, 1, 3))
    assert float(df["close"][0]) == pytest.approx(152.24)


def test_ohlcv_prices_falls_back_to_close_when_no_adj(ohlcv_legacy_parquet):
    _, pio = ohlcv_legacy_parquet
    assert pio.parquet_available()
    df = pio.fetch_ohlcv_prices(["AAPL"], date(2025, 1, 2), date(2025, 1, 2))
    assert float(df["close"][0]) == pytest.approx(150.0)


# ---------------------------------------------------------------------------
# #128: DuckDB connection caching (build once per process)
# ---------------------------------------------------------------------------


def test_flat_connection_is_cached(ohlcv_parquet):
    _, pio = ohlcv_parquet
    pio.parquet_available()
    c1 = pio._con()
    c2 = pio._con()
    assert c1 is c2  # same cached connection object, not rebuilt


def test_flat_connection_built_once_across_fetches(ohlcv_parquet, monkeypatch):
    _, pio = ohlcv_parquet
    calls = {"n": 0}
    real_build = pio._build_con

    def _counting_build():
        calls["n"] += 1
        return real_build()

    monkeypatch.setattr(pio, "_build_con", _counting_build)
    monkeypatch.setattr(pio, "_flat_con_cache", None)
    pio.fetch_latest_prices(["AAPL"], date(2025, 1, 6))
    pio.fetch_ohlcv_prices(["AAPL"], date(2025, 1, 1), date(2025, 1, 6))
    pio.fetch_ohlcv_ret1d(["AAPL"], date(2025, 1, 1), date(2025, 1, 6))
    assert calls["n"] == 1  # three fetches, one connection build


# ---------------------------------------------------------------------------
# #170 Phase 3b: fetch_ohlcv_window / ohlcv_window_fresh_for
# ---------------------------------------------------------------------------


def test_fetch_ohlcv_window_ret1d_full_universe(ohlcv_parquet):
    _, pio = ohlcv_parquet
    df = pio.fetch_ohlcv_window("ret_1d", date(2025, 1, 1), date(2025, 1, 6))
    # No ticker filter -- both AAPL and MSFT rows are returned.
    assert {"AAPL", "MSFT"} == set(df["ticker"].unique().to_list())
    assert set(df.columns) == {"ticker", "date", "value"}


def test_fetch_ohlcv_window_ret1d_values(ohlcv_parquet):
    _, pio = ohlcv_parquet
    df = pio.fetch_ohlcv_window("ret_1d", date(2025, 1, 1), date(2025, 1, 6))
    row = df.filter((pl.col("ticker") == "AAPL") & (pl.col("date") == date(2025, 1, 2)))
    assert float(row["value"][0]) == pytest.approx(0.01)


def test_fetch_ohlcv_window_raw_close_uses_mcap_column(ohlcv_split_parquet):
    _, pio = ohlcv_split_parquet
    df = pio.fetch_ohlcv_window("raw_close", date(2025, 1, 2), date(2025, 1, 2))
    row = df.filter(pl.col("ticker") == "AAPL")
    # raw_close (600.0), not split-adjusted close (150.0) or adj_close (148.0).
    assert float(row["value"][0]) == pytest.approx(600.0)


def test_fetch_ohlcv_window_date_bounds_inclusive(ohlcv_parquet):
    _, pio = ohlcv_parquet
    df = pio.fetch_ohlcv_window("ret_1d", date(2025, 1, 2), date(2025, 1, 3))
    dates = set(df["date"].to_list())
    assert dates == {date(2025, 1, 2), date(2025, 1, 3)}


def test_fetch_ohlcv_window_rejects_unknown_column(ohlcv_parquet):
    _, pio = ohlcv_parquet
    with pytest.raises(ValueError):
        pio.fetch_ohlcv_window("close", date(2025, 1, 1), date(2025, 1, 6))


def test_ohlcv_window_fresh_for_true_when_mirror_covers_buffer(ohlcv_parquet):
    _, pio = ohlcv_parquet
    # Mirror max date is 2025-01-06 (fixture's last row).
    assert pio.ohlcv_window_fresh_for(date(2025, 1, 1), buffer_days=5) is True


def test_ohlcv_window_fresh_for_false_when_within_buffer(ohlcv_parquet):
    _, pio = ohlcv_parquet
    # Mirror max date (2025-01-06) is within 5 days of this window end.
    assert pio.ohlcv_window_fresh_for(date(2025, 1, 3), buffer_days=5) is False


def test_ohlcv_window_fresh_for_false_when_unavailable(tmp_path, monkeypatch):
    monkeypatch.setenv("PARQUET_BASE_PATH", str(tmp_path / "does_not_exist"))
    import meridian_pipeline.parquet_io as pio

    importlib.reload(pio)
    assert pio.ohlcv_window_fresh_for(date(2025, 1, 1)) is False


# ---------------------------------------------------------------------------
# Audit batch C (#171): fetch_ohlcv_window resolution filter
# ---------------------------------------------------------------------------


@pytest.fixture()
def ohlcv_mixed_resolution_parquet(parquet_root):
    """OHLCV fixture that carries a `resolution` column with a non-daily row,
    mirroring the flat-Parquet mirror's documented mixed-resolution content."""
    tmp_path, pio = parquet_root
    rows = [
        {
            "date": date(2025, 1, 2),
            "ticker": "AAPL",
            "ret_1d": 0.01,
            "close": 150.0,
            "adj_close": 150.0,
            "raw_close": 150.0,
            "resolution": "daily",
        },
        {
            "date": date(2025, 1, 2),
            "ticker": "AAPL",
            "ret_1d": 999.0,
            "close": 150.0,
            "adj_close": 150.0,
            "raw_close": 150.0,
            "resolution": "1hour",
        },
    ]
    df = pl.DataFrame(
        rows,
        schema={
            "date": pl.Date,
            "ticker": pl.String,
            "ret_1d": pl.Float64,
            "close": pl.Float64,
            "adj_close": pl.Float64,
            "raw_close": pl.Float64,
            "resolution": pl.String,
        },
    )
    df.write_parquet(tmp_path / "ohlcv" / "prices.parquet")
    return tmp_path, pio


def test_fetch_ohlcv_window_excludes_non_daily_resolution(ohlcv_mixed_resolution_parquet):
    _, pio = ohlcv_mixed_resolution_parquet
    df = pio.fetch_ohlcv_window("ret_1d", date(2025, 1, 2), date(2025, 1, 2))
    # Only the 'daily' row should survive; the '1hour' row (ret_1d=999.0) must
    # not leak into a full-universe window read.
    assert df["value"].to_list() == pytest.approx([0.01])


def test_fetch_ohlcv_window_no_resolution_column_is_unfiltered(ohlcv_parquet):
    """Fixture has no `resolution` column - _resolution_clause() must emit ''
    (not filter everything to NULL), preserving prior behaviour."""
    _, pio = ohlcv_parquet
    df = pio.fetch_ohlcv_window("ret_1d", date(2025, 1, 1), date(2025, 1, 6))
    assert len(df) > 0


# ---------------------------------------------------------------------------
# Audit batch C (#171): _ohlcv_columns does not cache a transient probe failure
# ---------------------------------------------------------------------------


def test_ohlcv_columns_does_not_cache_transient_failure(ohlcv_parquet, monkeypatch):
    _, pio = ohlcv_parquet
    assert pio._ohlcv_columns_cache is None

    real_con = pio._con()
    calls = {"n": 0}

    class _FlakyThenOk:
        def execute(self, sql):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("transient S3/catalog blip")
            return real_con.execute(sql)

    monkeypatch.setattr(pio, "_con", lambda: _FlakyThenOk())

    # First call hits the simulated transient failure - must NOT cache.
    cols_first = pio._ohlcv_columns()
    assert cols_first == set()
    assert pio._ohlcv_columns_cache is None  # not cached - old bug cached set()

    # Second call succeeds and IS cached.
    cols_second = pio._ohlcv_columns()
    assert "raw_close" in cols_second
    assert pio._ohlcv_columns_cache == cols_second

    # A close-column expression must reflect the real (recovered) columns,
    # not be permanently degraded to plain 'close' by the first failure.
    assert pio._mcap_close_expr() == "COALESCE(raw_close, close)"


def test_ohlcv_columns_caches_on_success(ohlcv_parquet):
    _, pio = ohlcv_parquet
    first = pio._ohlcv_columns()
    assert pio._ohlcv_columns_cache == first
    # Second call must be served from cache (same object), not re-probed.
    second = pio._ohlcv_columns()
    assert second is pio._ohlcv_columns_cache
    assert first == second


# ---------------------------------------------------------------------------
# Audit batch C (#171): dualwrite dedup precedence in fetch_factor_scores_chunks
# ---------------------------------------------------------------------------


@pytest.fixture()
def factor_parquet_root(tmp_path, monkeypatch, parquet_root):
    """Local (non-S3) meridian-parquet factor feature table for dedup tests.

    Bypasses the hardcoded s3:// glob / S3-only connection builder by
    monkeypatching _use_meridian_s3 / _factor_glob / _meridian_con so the
    dedup SQL can be exercised against local files with no network/creds.
    """
    _, pio = parquet_root
    factor_dir = tmp_path / "factors"
    factor_dir.mkdir()

    monkeypatch.setattr(pio, "_use_meridian_s3", lambda: True)
    monkeypatch.setattr(pio, "_factor_glob", lambda: str(factor_dir / "**" / "*.parquet"))
    monkeypatch.setattr(pio, "_meridian_con", lambda: pio._con())

    # The computed_at probe is process-cached per glob; each test gets a fresh
    # directory, so clear it to avoid leaking a previous test's answer.
    pio._factor_computed_at_cache.clear()

    def _write(year: int, fname: str, rows: list[dict], computed_at=None):
        d = factor_dir / f"year={year}"
        d.mkdir(parents=True, exist_ok=True)
        schema = {
            "score_date": pl.Date,
            "ticker": pl.String,
            "factor_name": pl.String,
            "z_score": pl.Float64,
        }
        if computed_at is not None:
            rows = [{**r, "computed_at": computed_at} for r in rows]
            schema["computed_at"] = pl.Datetime("us")
        pl.DataFrame(rows, schema=schema).write_parquet(d / fname)

    return pio, _write


def test_dedup_prefers_materialized_file_over_dualwrite(factor_parquet_root):
    """A stale pre-fix dualwrite row must lose to the materialized (data_*)
    rebuild for the same (score_date, ticker, factor_name) key."""
    pio, write = factor_parquet_root
    sd = date(2025, 6, 13)

    # Stale pre-fix dualwrite fragment.
    write(
        2025,
        "dualwrite__2025-06-13_2025-06-13__aaa111111111.parquet",
        [{"score_date": sd, "ticker": "AAPL", "factor_name": "pe_ratio", "z_score": -9.99}],
    )
    # Materialized full-rebuild file with the corrected value.
    write(
        2025,
        "data_0.parquet",
        [{"score_date": sd, "ticker": "AAPL", "factor_name": "pe_ratio", "z_score": 0.42}],
    )

    chunks = list(pio.fetch_factor_scores_chunks(["pe_ratio"], sd, sd))
    df = chunks[0]
    assert len(df) == 1
    assert df["z_score"].iloc[0] == pytest.approx(0.42)


def test_dedup_dualwrite_tie_break_is_deterministic(factor_parquet_root):
    """Two dualwrite fragments carrying the SAME key (a re-score whose
    non-null factor set differs across runs) must resolve to ONE
    deterministic winner - not a random pick across repeated calls."""
    pio, write = factor_parquet_root
    sd = date(2025, 6, 13)

    write(
        2025,
        "dualwrite__2025-06-13_2025-06-13__aaa111111111.parquet",
        [{"score_date": sd, "ticker": "AAPL", "factor_name": "accruals", "z_score": 1.11}],
    )
    write(
        2025,
        "dualwrite__2025-06-13_2025-06-13__zzz999999999.parquet",
        [{"score_date": sd, "ticker": "AAPL", "factor_name": "accruals", "z_score": 2.22}],
    )

    winners = set()
    for _ in range(5):
        chunks = list(pio.fetch_factor_scores_chunks(["accruals"], sd, sd))
        df = chunks[0]
        assert len(df) == 1  # exactly one survivor, not two duplicate rows
        winners.add(float(df["z_score"].iloc[0]))

    # Deterministic: repeated calls always pick the same survivor. The
    # winner is whichever file sorts last by filename text (documented,
    # stable, not a true recency guarantee) - here that is the "zzz..." file.
    assert winners == {2.22}


def test_dedup_prefers_newest_computed_at_over_filename(factor_parquet_root):
    """With generation stamps present (#275), recency decides - even when the
    stale row sits in the file that sorts last by name and in the `data_` tier."""
    from datetime import datetime as _dt

    pio, write = factor_parquet_root
    sd = date(2025, 6, 13)

    # Corrected re-score: earlier-sorting name, newest stamp.
    write(
        2025,
        "dualwrite__2025-06-13_2025-06-13__aaa111111111.parquet",
        [{"score_date": sd, "ticker": "AAPL", "factor_name": "accruals", "z_score": 2.22}],
        computed_at=_dt(2026, 7, 25, 9, 0, 0),
    )
    # Pre-correction generation: later-sorting name AND the `data_` tier.
    write(
        2025,
        "data_zzz.parquet",
        [{"score_date": sd, "ticker": "AAPL", "factor_name": "accruals", "z_score": 1.11}],
        computed_at=_dt(2026, 7, 3, 9, 0, 0),
    )

    df = list(pio.fetch_factor_scores_chunks(["accruals"], sd, sd))[0]
    assert len(df) == 1
    assert df["z_score"].iloc[0] == pytest.approx(2.22)


def test_dedup_stamped_row_beats_unstamped_row(factor_parquet_root):
    """A stamped (post-#275) row wins over a legacy row with no stamp: the
    unstamped row is by construction from an older generation of the writer."""
    from datetime import datetime as _dt

    pio, write = factor_parquet_root
    sd = date(2025, 6, 13)

    write(
        2025,
        "dualwrite__2025-06-13_2025-06-13__zzz999999999.parquet",
        [{"score_date": sd, "ticker": "AAPL", "factor_name": "accruals", "z_score": 1.11}],
    )
    write(
        2025,
        "dualwrite__2025-06-13_2025-06-13__aaa111111111.parquet",
        [{"score_date": sd, "ticker": "AAPL", "factor_name": "accruals", "z_score": 3.33}],
        computed_at=_dt(2026, 7, 25, 9, 0, 0),
    )

    df = list(pio.fetch_factor_scores_chunks(["accruals"], sd, sd))[0]
    assert len(df) == 1
    assert df["z_score"].iloc[0] == pytest.approx(3.33)


def test_year_range_clause_bounds_to_queried_years():
    import meridian_pipeline.parquet_io as pio

    clause = pio._year_range_clause(date(2023, 6, 1), date(2024, 3, 1))
    assert clause == "AND year >= 2023 AND year <= 2024"
