"""
splice_guard.py
----------------
Interim ticker-reuse ("splice") guard for rolling/lookback window
computations that are partitioned by ticker SYMBOL rather than by company.

BACKGROUND: ticker symbols get reused. When a delisted company's symbol is
later reused by an unrelated company (a symbol last traded decades earlier,
with a gap of thousands of calendar days), any window
computation partitioned only by `ticker` can span straight across the gap and
silently blend two different companies' return/price series into one number.
After LDP's CRSP delisted-tape ingest, a ticker can map to more than one
entity_id and carry a coverage gap > 365 calendar days - meaning rolling
windows for such names can span straight across the gap and blend two
companies' returns into one factor value.

UPDATE: LDP #189 shipped entity_id into the OHLCV gold/parquet
export. entity_id is now the PRIMARY guard - the fetch queries in assets.py
(_compute_momentum_pl's caller), risk_factor_scores_asset.py, and
risk_metrics_asset.py resolve each ticker's CURRENT entity_id (the entity_id
of its own most recent row as of score_date) and restrict the fetch to rows
matching that entity_id, so an old, unrelated company's rows for a reused
ticker symbol never reach the rolling window at all. This module's date-gap
guard is now the SECONDARY / fallback layer: a small share of ohlcv.prices rows
have a NULL entity_id and cannot be filtered
this way, so those tickers fall through to the date-gap heuristic below. The
guard also remains cheap, independent defense-in-depth for entity_id data
quality issues generally, not just the NULL case.

THRESHOLD: MAX_INTERNAL_GAP_DAYS = 120 calendar days.

Any two chronologically consecutive observations for the same ticker that
are more than 120 calendar days apart invalidate any window spanning across
them - the affected aggregate is set to NULL/NaN rather than a blended
number. Justification:
  - A continuously-traded name has a ret_1d row on (almost) every trading
    day; ordinary weekends/holidays never produce a gap anywhere near this
    size, and even an unusually long single trading halt (a few weeks) stays
    well under 120 days.
  - A genuine ticker-reuse splice is unambiguous at > 365 calendar days
    (the reused symbol resumes years later - years of dormancy, not weeks).
  - 120 days sits with real margin below the 365-day "unambiguous splice"
    bar, so that shorter, more ambiguous multi-month suspensions also fail
    toward NULL rather than risk silently blending two return series. Per
    the design directive: a missing factor is handled downstream (the
    composite is a weighted nanmean); a wrong factor is not - so this
    threshold is deliberately conservative in the "when in doubt, NULL"
    direction rather than tuned to maximize retained coverage.

This module intentionally does not try to be clever about *which* side of
the gap is "real" - it just refuses to compute the aggregate that would span
it. Downstream consumers already tolerate NaN/NULL factor values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

# See module docstring for calibration rationale.
MAX_INTERNAL_GAP_DAYS = 120


def polars_window_gap_mask(
    *,
    window_size: int,
    min_samples: int = 1,
    date_col: str = "date",
    ticker_col: str = "ticker",
    max_gap_days: int = MAX_INTERNAL_GAP_DAYS,
) -> pl.Expr:
    """Boolean Polars expression, True where the trailing row-count window
    (window_size, matching the caller's own rolling_sum/rolling_mean call)
    contains an internal date gap larger than max_gap_days - i.e. the window
    straddles a ticker-reuse splice and the value computed over it should be
    discarded.

    The input frame must be sorted by (ticker_col, date_col) before this
    expression is evaluated (same precondition as any rolling_*().over(...)
    call). min_samples is accepted for signature symmetry with the caller's
    rolling_sum but is not itself used for gap detection - a window with
    fewer than the caller's own min_samples is already nulled by the
    caller's own rolling aggregate; this expression only adds the
    orthogonal "does the window straddle a big gap" check.
    """
    day_gap = pl.col(date_col).diff().over(ticker_col).dt.total_days()
    max_gap_in_window = day_gap.rolling_max(window_size=window_size, min_samples=1).over(ticker_col)
    return (max_gap_in_window > max_gap_days).fill_null(False)


def pandas_window_has_gap(
    window: pd.DataFrame,
    *,
    max_gap_days: int = MAX_INTERNAL_GAP_DAYS,
) -> pd.Series:
    """window: a (date-index, ticker-columns) slice of a wide panel - one
    rolling window as already sliced by the caller (e.g. `rets.iloc[-60:]`).

    Returns a boolean Series indexed by column (ticker), True where that
    column's non-null observations within `window` contain an internal gap
    larger than max_gap_days calendar days - the window straddles a
    ticker-reuse splice and the aggregate computed over it should be
    discarded (set to NaN) rather than trusted.
    """
    if window.empty:
        return pd.Series(False, index=window.columns)
    idx = pd.DatetimeIndex(window.index)
    day_vals = idx.values.astype("datetime64[D]")
    mask = window.notna().to_numpy()
    flags = numpy_window_has_gap(day_vals, mask, max_gap_days=max_gap_days)
    return pd.Series(flags, index=window.columns)


def numpy_window_has_gap(
    dates: np.ndarray,
    valid_mask: np.ndarray,
    *,
    max_gap_days: int = MAX_INTERNAL_GAP_DAYS,
) -> np.ndarray:
    """Vectorised-per-column variant (used directly by the GPU risk_metrics
    asset, which builds its own numpy/torch masks ahead of the pivot).

    dates: 1-D array of length n_rows (datetime64[D] or convertible), the
      panel's dense date index (shared across all tickers/columns).
    valid_mask: (n_rows, n_tickers) bool array, True where that ticker has a
      non-null observation on that row's date.

    Returns a 1-D bool array of length n_tickers, True where that ticker's
    valid observations contain an internal gap larger than max_gap_days.
    """
    dates = np.asarray(dates).astype("datetime64[D]")
    n_tickers = valid_mask.shape[1]
    flags = np.zeros(n_tickers, dtype=bool)
    for i in range(n_tickers):
        col_mask = valid_mask[:, i]
        if col_mask.sum() < 2:
            continue
        d = dates[col_mask]
        gaps = np.diff(d).astype("timedelta64[D]").astype(np.int64)
        if gaps.size and gaps.max() > max_gap_days:
            flags[i] = True
    return flags


# ── Segment-start SQL (ATH / 52w-high style unbounded-or-windowed MAX
#    aggregates that don't go through a pandas/polars panel at all) ────────
#
# These aggregates (example_level_a, example_level_b) are computed directly
# in SQL/DuckDB as MAX(price) GROUP BY ticker with no per-row visibility for
# a pandas/polars-side gap check. The fix there is a CTE that restricts the
# MAX to rows on/after the ticker's own most recent internal gap larger than
# max_gap_days (its "current trading segment"); see the CTE embedded in
# risk_factor_scores_asset._fetch_ath / _fetch_52w_high and
# parquet_io.fetch_ath / fetch_52w_high. Kept alongside this constant so the
# threshold is defined in exactly one place.
SEGMENT_GAP_THRESHOLD_DAYS = MAX_INTERNAL_GAP_DAYS
