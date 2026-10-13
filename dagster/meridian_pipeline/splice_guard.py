"""Guard against ticker reuse in rolling-window computations.

Ticker symbols get reused. When a delisted company's symbol is later taken by an unrelated one,
years after the first stopped trading, a window computed per `ticker` can run straight across
the gap and mix two companies' returns into one number. Since LDP's CRSP delisted-tape ingest, a
ticker can map to more than one entity_id and carry a coverage gap of more than 365 calendar
days.

The main defence is upstream of this module. LDP now exports entity_id with the OHLCV data (LDP
#189). The price fetch queries resolve each ticker's current entity_id, the one on its most
recent row as of score_date, and drop rows that belong to any other entity, so the earlier
company never reaches the window. This module covers what that filter cannot: the small share of
`ohlcv.prices` rows with a NULL entity_id, and any other entity_id quality problem.

It does so with a date-gap test. If two consecutive observations of a ticker are more than
MAX_INTERNAL_GAP_DAYS (120) calendar days apart, any window that spans them is invalid and its
aggregate is set to NULL or NaN. Holidays and even a trading halt of a few weeks stay well under
120 days, while a real splice is over 365. The threshold sits at 120 so that a multi-month
suspension is dropped as well. A missing factor is handled downstream, because the composite is
a weighted nanmean, and a wrong one is not, so the guard errs toward NULL.

Which side of the gap is the real company is not decided here. Any aggregate that would span the
gap is simply not computed.
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


# Segment-start SQL. The aggregates drawdown_all and drawdown_52w are plain
# MAX(price) GROUP BY ticker in SQL or DuckDB, so there is no pandas or polars frame to run a
# gap check on. They restrict the MAX to rows on or after the ticker's most recent internal gap
# larger than max_gap_days, its current trading segment. The CTE is in
# risk_factor_scores_asset._fetch_ath / _fetch_52w_high and parquet_io.fetch_ath /
# fetch_52w_high. The threshold is defined here so it exists in one place.
SEGMENT_GAP_THRESHOLD_DAYS = MAX_INTERNAL_GAP_DAYS
