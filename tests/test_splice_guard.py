"""
Unit tests for splice_guard.py.

DB-free, dagster-free: exercises the pandas/numpy/polars gap-detection
helpers directly against synthetic date/mask arrays. No dagster import is
needed here (splice_guard.py has no asset defs), so this file does not use
the dagster-availability skip guard the other tests use.
"""

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

_DAGSTER_DIR = Path(__file__).resolve().parents[1] / "dagster"
if str(_DAGSTER_DIR) not in sys.path:
    sys.path.insert(0, str(_DAGSTER_DIR))

from meridian_pipeline.splice_guard import (  # noqa: E402
    MAX_INTERNAL_GAP_DAYS,
    numpy_window_has_gap,
    pandas_window_has_gap,
    polars_window_gap_mask,
)


def _bdate_range(start: date, n: int) -> list[date]:
    out = []
    d = start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


# ---------------------------------------------------------------------------
# numpy_window_has_gap
# ---------------------------------------------------------------------------


def test_numpy_no_gap_continuous_series():
    dates = np.array(_bdate_range(date(2024, 1, 1), 60), dtype="datetime64[D]")
    valid = np.ones((60, 1), dtype=bool)
    flags = numpy_window_has_gap(dates, valid)
    assert flags.tolist() == [False]


def test_numpy_flags_large_internal_gap():
    dates = np.array(_bdate_range(date(2020, 1, 1), 300), dtype="datetime64[D]")
    valid = np.zeros((300, 1), dtype=bool)
    # First 40 rows (old company), gap, last 40 rows (new company).
    valid[:40, 0] = True
    valid[260:, 0] = True
    flags = numpy_window_has_gap(dates, valid)
    assert flags.tolist() == [True]


def test_numpy_boundary_gap_exactly_at_threshold_not_flagged():
    # Two observations exactly MAX_INTERNAL_GAP_DAYS apart: not flagged (uses
    # strict > in the implementation).
    d0 = date(2024, 1, 1)
    d1 = d0 + timedelta(days=MAX_INTERNAL_GAP_DAYS)
    dates = np.array([d0, d1], dtype="datetime64[D]")
    valid = np.ones((2, 1), dtype=bool)
    flags = numpy_window_has_gap(dates, valid)
    assert flags.tolist() == [False]


def test_numpy_boundary_gap_one_day_over_threshold_flagged():
    d0 = date(2024, 1, 1)
    d1 = d0 + timedelta(days=MAX_INTERNAL_GAP_DAYS + 1)
    dates = np.array([d0, d1], dtype="datetime64[D]")
    valid = np.ones((2, 1), dtype=bool)
    flags = numpy_window_has_gap(dates, valid)
    assert flags.tolist() == [True]


def test_numpy_fewer_than_two_valid_obs_not_flagged():
    dates = np.array(_bdate_range(date(2024, 1, 1), 10), dtype="datetime64[D]")
    valid = np.zeros((10, 1), dtype=bool)
    valid[0, 0] = True
    flags = numpy_window_has_gap(dates, valid)
    assert flags.tolist() == [False]


def test_numpy_multi_ticker_independent():
    dates = np.array(_bdate_range(date(2020, 1, 1), 300), dtype="datetime64[D]")
    valid = np.zeros((300, 2), dtype=bool)
    # Ticker 0: continuous.
    valid[:, 0] = True
    # Ticker 1: spliced.
    valid[:40, 1] = True
    valid[260:, 1] = True
    flags = numpy_window_has_gap(dates, valid)
    assert flags.tolist() == [False, True]


# ---------------------------------------------------------------------------
# pandas_window_has_gap
# ---------------------------------------------------------------------------


def test_pandas_continuous_column_not_flagged():
    idx = pd.DatetimeIndex(_bdate_range(date(2024, 1, 1), 60))
    df = pd.DataFrame({"AAPL": np.arange(60, dtype=float)}, index=idx)
    flags = pandas_window_has_gap(df)
    assert not flags["AAPL"]


def test_pandas_spliced_column_flagged():
    idx = pd.DatetimeIndex(_bdate_range(date(2020, 1, 1), 300))
    vals = np.full(300, np.nan)
    vals[:40] = 1.0
    vals[260:] = 1.0
    df = pd.DataFrame({"REUSED1": vals}, index=idx)
    flags = pandas_window_has_gap(df)
    assert flags["REUSED1"]


def test_pandas_empty_window_returns_all_false():
    df = pd.DataFrame(columns=["AAPL"])
    flags = pandas_window_has_gap(df)
    assert not flags["AAPL"]


# ---------------------------------------------------------------------------
# polars_window_gap_mask
# ---------------------------------------------------------------------------


def _polars_long_frame(rows: list[tuple[date, str, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={"date": pl.Date, "ticker": pl.String, "ret_1d": pl.Float64},
        orient="row",
    ).sort(["ticker", "date"])


def test_polars_continuous_ticker_not_flagged():
    dates = _bdate_range(date(2020, 1, 1), 260)
    rows = [(d, "AAPL", 0.001) for d in dates]
    df = _polars_long_frame(rows)
    df = df.with_columns(polars_window_gap_mask(window_size=252, min_samples=200).alias("gap"))
    assert not df["gap"].any()


def test_polars_spliced_ticker_flagged_after_gap():
    # 40 rows of "old company" data, then a multi-year gap, then 220 rows of
    # "new company" data - together >= 200 rows, so min_samples alone would
    # not null the window; the gap guard must flag it.
    old_dates = _bdate_range(date(2000, 1, 1), 40)
    new_dates = _bdate_range(date(2025, 1, 1), 220)
    rows = [(d, "REUSED1", 0.001) for d in old_dates] + [(d, "REUSED1", 0.001) for d in new_dates]
    df = _polars_long_frame(rows)
    df = df.with_columns(polars_window_gap_mask(window_size=252, min_samples=200).alias("gap"))
    # The rows in the "new" segment fall inside a 252-row window that also
    # includes the tail of the "old" segment, so at least one row in the new
    # segment must be flagged.
    tail = df.filter(pl.col("date") >= new_dates[0])
    assert tail["gap"].any()


def test_polars_short_gap_not_flagged():
    # A 30-day gap (e.g. a long trading halt) must stay well under the
    # splice threshold and not flag the window.
    part1 = _bdate_range(date(2020, 1, 1), 130)
    part2 = _bdate_range(part1[-1] + timedelta(days=30), 130)
    rows = [(d, "HALT", 0.001) for d in part1] + [(d, "HALT", 0.001) for d in part2]
    df = _polars_long_frame(rows)
    df = df.with_columns(polars_window_gap_mask(window_size=252, min_samples=200).alias("gap"))
    assert not df["gap"].any()


def _three_variant_flags(dates: list[date], window: int) -> tuple[list, list, list]:
    """Row-by-row gap flag for one ticker from the polars, pandas and numpy
    variants, each given the same trailing `window`-row slice."""
    df = _polars_long_frame([(d, "X", 0.001) for d in dates])
    polars_flags = (
        df.with_columns(polars_window_gap_mask(window_size=window).alias("gap"))["gap"]
    ).to_list()
    pandas_flags, numpy_flags = [], []
    for i in range(len(dates)):
        win = dates[max(0, i - window + 1) : i + 1]
        frame = pd.DataFrame({"X": 0.001}, index=pd.DatetimeIndex(win))
        pandas_flags.append(bool(pandas_window_has_gap(frame)["X"]))
        numpy_flags.append(
            bool(
                numpy_window_has_gap(
                    np.array(win, dtype="datetime64[D]"), np.ones((len(win), 1), dtype=bool)
                )[0]
            )
        )
    return polars_flags, pandas_flags, numpy_flags


def test_polars_matches_pandas_and_numpy_around_a_splice():
    # 10 old rows, a multi-year gap, 12 new rows; window of 5 rows.
    old = _bdate_range(date(2000, 1, 3), 10)
    new = _bdate_range(date(2025, 1, 6), 12)
    polars_flags, pandas_flags, numpy_flags = _three_variant_flags(old + new, window=5)
    assert polars_flags == pandas_flags == numpy_flags
    first_new = len(old)
    # Windows that hold both old and new rows are flagged; the window that
    # starts on the first new row holds only new rows and is not.
    assert polars_flags[first_new]
    assert polars_flags[first_new + 3]
    assert not polars_flags[first_new + 4]
    assert not polars_flags[-1]


def test_polars_window_starting_just_after_a_splice_is_not_flagged():
    # Exactly `window` rows since the splice: the window's first row is the
    # first post-splice row. The gap before it is outside the window.
    window = 252
    old = _bdate_range(date(2000, 1, 3), 40)
    new = _bdate_range(date(2025, 1, 6), window)
    df = _polars_long_frame([(d, "REUSED1", 0.001) for d in old + new])
    df = df.with_columns(polars_window_gap_mask(window_size=window, min_samples=200).alias("gap"))
    assert not df["gap"][-1]
    # One row earlier the window still reaches back across the splice.
    assert df["gap"][-2]


def test_polars_leading_gap_flags_window_starting_on_first_post_splice_row():
    # A return-sum caller measures the first return in the window
    # from the row before it, so that window still spans the splice.
    window = 252
    old = _bdate_range(date(2000, 1, 3), 40)
    new = _bdate_range(date(2025, 1, 6), window)
    df = _polars_long_frame([(d, "REUSED1", 0.001) for d in old + new])
    df = df.with_columns(
        polars_window_gap_mask(window_size=window, min_samples=200, include_leading_gap=True).alias(
            "gap"
        )
    )
    assert df["gap"][-2]
    assert df["gap"][-1]


def test_polars_leading_gap_clears_one_row_later():
    window = 252
    old = _bdate_range(date(2000, 1, 3), 40)
    new = _bdate_range(date(2025, 1, 6), window + 1)
    df = _polars_long_frame([(d, "REUSED1", 0.001) for d in old + new])
    df = df.with_columns(
        polars_window_gap_mask(window_size=window, min_samples=200, include_leading_gap=True).alias(
            "gap"
        )
    )
    assert df["gap"][-2]
    assert not df["gap"][-1]


def test_polars_gap_mask_is_per_ticker():
    old = _bdate_range(date(2000, 1, 3), 10)
    new = _bdate_range(date(2025, 1, 6), 10)
    rows = [(d, "SPLICED", 0.001) for d in old + new]
    rows += [(d, "CLEAN", 0.001) for d in _bdate_range(date(2024, 1, 1), 20)]
    df = _polars_long_frame(rows).with_columns(polars_window_gap_mask(window_size=5).alias("gap"))
    assert df.filter(pl.col("ticker") == "SPLICED")["gap"].any()
    assert not df.filter(pl.col("ticker") == "CLEAN")["gap"].any()


def test_polars_one_row_window_has_no_internal_gap():
    old = _bdate_range(date(2000, 1, 3), 3)
    new = _bdate_range(date(2025, 1, 6), 3)
    df = _polars_long_frame([(d, "X", 0.001) for d in old + new])
    df = df.with_columns(polars_window_gap_mask(window_size=1).alias("gap"))
    assert not df["gap"].any()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
