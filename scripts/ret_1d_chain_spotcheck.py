"""Spot-check for chained ret_1d.

Chain ret_1d over a dividend-paying ticker for N trading days and the result should be the
close ratio plus whatever dividends were paid in the window, to within floating-point
tolerance. A chain that comes out lower than the close ratio has the window-start
off-by-one described in the LDP DATABASE.md: it includes ret_1d[start], the return into
close[start] from the previous close, where it should start from close[start] onward.

The correct window is `ts_event > start AND ts_event <= end`.

A consumer that chains ret_1d, as opposed to taking log1p of single rows for vol and
covariance, should call these helpers from its tests or a debug harness before trusting
what it produces.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import date

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool


def _engine():
    return create_engine(
        f"postgresql://{os.environ.get('PGUSER', 'dbuser')}:"
        f"{os.environ['PGPASS']}@"
        f"{os.environ.get('PGHOST', 'db.internal')}:5432/"
        f"{os.environ.get('PGDB', 'marketdata')}",
        poolclass=NullPool,
        pool_pre_ping=True,
    )


@dataclass
class ChainSpotCheckResult:
    ticker: str
    start_date: date
    end_date: date
    n_days: int
    close_ratio: float  # (close[end]/close[start]) - 1
    chained_ret: float  # prod(1 + ret_1d[start<t<=end]) - 1
    implied_div: float  # chained_ret - close_ratio
    passed: bool

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] {self.ticker} {self.start_date}->{self.end_date} "
            f"({self.n_days}d): close_ratio={self.close_ratio:+.4f} "
            f"chained={self.chained_ret:+.4f} implied_div={self.implied_div:+.4f}"
        )


def chain_spotcheck(
    engine,
    ticker: str,
    start_date: date,
    end_date: date,
    tol_abs: float = 0.0005,
) -> ChainSpotCheckResult:
    """Check that the ret_1d chain on a ticker equals close_ratio plus dividends.

    Passes when chained_ret >= close_ratio - tol_abs, since the chain can only be lower
    than the close ratio if it has the off-by-one. The implied dividend, chained_ret minus
    close_ratio, must also stay under 0.10. Over a quarter that is far above any yield
    among the tickers in SPOTCHECK_CASES, so a larger gap points at bad data or a bug.
    """
    df = pd.read_sql(
        text("""
            SELECT ts_event::date AS d, close, ret_1d
            FROM ohlcv.prices
            WHERE ticker = :t
              AND resolution = 'daily'
              AND ts_event::date BETWEEN :s AND :e
            ORDER BY ts_event
        """),
        engine,
        params={"t": ticker, "s": start_date, "e": end_date},
    )
    if len(df) < 2:
        raise RuntimeError(f"not enough rows for {ticker} [{start_date}, {end_date}]")

    df["d"] = pd.to_datetime(df["d"])
    df = df.drop_duplicates("d").sort_values("d")

    close_start = df.iloc[0]["close"]
    close_end = df.iloc[-1]["close"]
    close_ratio = float(close_end / close_start - 1.0)

    # The first row's ret_1d is the return into close_start, so the chain
    # starts at the second row (ts_event > start_date, strict).
    chain_df = df.iloc[1:]
    chained = float((1.0 + chain_df["ret_1d"]).prod() - 1.0)

    implied_div = chained - close_ratio

    # The chain may fall short of the close ratio by at most tol_abs, and may
    # exceed it by at most a plausible dividend yield.
    passed = (implied_div >= -tol_abs) and (implied_div <= 0.10)

    return ChainSpotCheckResult(
        ticker=ticker,
        start_date=df.iloc[0]["d"].date(),
        end_date=df.iloc[-1]["d"].date(),
        n_days=len(df),
        close_ratio=close_ratio,
        chained_ret=chained,
        implied_div=implied_div,
        passed=passed,
    )


# Dividend payers: two broad-market ETF windows, a sector ETF and three single stocks.
# implied_div over a quarter should be about one quarterly dividend yield, a few tenths of a
# percent for the ETFs and somewhat more for the stocks.
SPOTCHECK_CASES = [
    ("SPY", date(2024, 1, 2), date(2024, 4, 1)),  # Q1 2024, ~60-62 trading days
    ("SPY", date(2024, 4, 1), date(2024, 7, 1)),  # Q2 2024
    ("XLK", date(2024, 1, 2), date(2024, 4, 1)),  # sector ETF
    ("XOM", date(2024, 1, 2), date(2024, 4, 1)),  # single stock, large dividend
    ("KO", date(2024, 1, 2), date(2024, 4, 1)),  # single stock, steady payer
    ("T", date(2024, 1, 2), date(2024, 4, 1)),  # single stock, highest yield of the set
]


def run_default_spotchecks() -> list[ChainSpotCheckResult]:
    engine = _engine()
    results = []
    for ticker, s, e in SPOTCHECK_CASES:
        r = chain_spotcheck(engine, ticker, s, e)
        results.append(r)
        print(r)
    return results


def assert_all_pass() -> None:
    """Raise AssertionError if any spotcheck fails - consumers should
    call this from their test suite / debug harness."""
    fails = [r for r in run_default_spotchecks() if not r.passed]
    if fails:
        lines = "\n  ".join(repr(f) for f in fails)
        raise AssertionError(
            f"ret_1d chain spot-check failed for {len(fails)} tickers:\n  {lines}\n"
            "Likely the window-start off-by-one: consumer is including "
            "ret_1d[start] in its chain (should start at ret_1d[start+1])."
        )


if __name__ == "__main__":
    try:
        assert_all_pass()
        print("\nall spot-checks passed")
        sys.exit(0)
    except AssertionError as e:
        print(f"\n{e}")
        sys.exit(1)
