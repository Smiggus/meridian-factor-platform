"""
ret_1d_chain_spotcheck.py

Reusable spot-check: chaining ret_1d over a dividend-paying ETF for N
trading days should equal (close ratio + dividend yield) within
floating-point tolerance. If LESS, the consumer has the window-start
off-by-one (per LDP DATABASE.md): it's including ret_1d[start] which
measures the return INTO close[start] from close[start-1], instead of
FROM close[start] onward.

Correct chain: `ts_event > start AND ts_event <= end`.

Each consumer that chains ret_1d (as opposed to just taking log1p of
individual rows for vol/cov math) should import + call these helpers
during tests / in a debug harness before trusting its output.
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
    """Verify ret_1d chain on a ticker matches close_ratio + dividend_yield.

    Passing = chained_ret >= close_ratio - tol_abs. (Must be at least as
    large as close ratio; if less, the chain has the off-by-one bug.)

    Further asserts chained_ret - close_ratio is within [0, +0.10] - any
    larger positive implied dividend over 3 months is implausible for a
    broad US equity ETF (absurd dividend yield or another bug).
    """
    # Pull close + ret_1d for [start_date, end_date]
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

    # Close ratio: use first and last rows in the window
    close_start = df.iloc[0]["close"]
    close_end = df.iloc[-1]["close"]
    close_ratio = float(close_end / close_start - 1.0)

    # Chained ret: exclude the first row's ret_1d (which measured
    # return INTO close_start, not from it), include all subsequent.
    # Filter: ts_event > start_date (strict) AND ts_event <= end_date.
    chain_df = df.iloc[1:]  # equivalent to ts_event > first row's date
    chained = float((1.0 + chain_df["ret_1d"]).prod() - 1.0)

    implied_div = chained - close_ratio

    # Passes if chain at least matches close ratio (never less)
    # and the excess is within a plausible dividend yield.
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


# Known dividend-paying ETFs for spot-checks. Expected implied_div
# (90d) should be ~quarterly dividend yield; broad US equity ETFs pay
# ~0.3-0.6% per quarter.
SPOTCHECK_CASES = [
    ("SPY", date(2024, 1, 2), date(2024, 4, 1)),  # Q1 2024, ~60-62 trading days
    ("SPY", date(2024, 4, 1), date(2024, 7, 1)),  # Q2 2024
    ("XLK", date(2024, 1, 2), date(2024, 4, 1)),  # Tech sector ETF (user's example)
    ("XOM", date(2024, 1, 2), date(2024, 4, 1)),  # Heavy dividend single-name
    ("KO", date(2024, 1, 2), date(2024, 4, 1)),  # Classic dividend payer
    ("T", date(2024, 1, 2), date(2024, 4, 1)),  # Very high yield, big expected gap
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
