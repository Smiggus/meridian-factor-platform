"""
adj_close_guard.py
------------------
Defence-in-depth guard against corrupted `ohlcv.prices.adj_close` values
feeding downstream price-level computations.

THE FAILURE MODE
    A single historical row with an adj_close many orders of magnitude
    too large relative to its close. Any price-level measure built on a historical extreme would be
    distorted by a single poisoned row for the rest of that
    ticker's life. The values are FINITE, so np.isfinite() gates pass them
    straight through; only a same-row ratio test can see it.

THE DETECTOR
    Same-row ratio `adj_close / close`, NOT a time-series (rolling-median)
    test and NOT a ratio against `raw_close`:

    * Same-row, split-adjusted-vs-split-adjusted. `close` is the
      split-adjusted series and `adj_close` is nominally the same series with
      dividend adjustment applied, so the legitimate ratio is a pure
      corporate-action factor of order 1 (and essentially 1.000 for tickers
      that never split, where raw_close == close for their entire history).

    * Price-move invariant. A stock that genuinely 10x's in a year moves
      `close` and `adj_close` together, so the ratio does not budge. A
      rolling-252d-median test, by contrast, needs K > 10 just to survive
      a genuine 10x - and would still MISS this corruption, because the
      corruption is not a one-row spike: an affected ticker can have
      a long run of consecutive corrupted rows, so
      its own trailing median is equally corrupted and the ratio-to-median
      is ~1.

    * Split invariant. A ratio against `raw_close` cannot separate
      corruption from a legitimate reverse split: a ticker's pre-reverse-split
      rows legitimately carry adj_close/raw_close of order 1e5. An adj/raw
      guard therefore nulls legitimately reverse-split tickers outright (false
      positives) and still leaves some tickers with an ATH far above their max
      split-adjusted close.

THE THRESHOLD (K = 1.1)
    A loose bound such as 3.0 only catches gross multi-fold corruption. The
    rows this guard targets are those where adj_close sits above close, which
    no legitimate adjustment produces, so the bound is set just above 1:
    1.1 rather than 1.001 leaves headroom for rounding accumulation over long
    adjustment chains.

    The direction is the invariant that makes a one-sided test correct:
    dividend back-adjustment and every corporate action (rights issues,
    spin-offs, return of capital) adjusts historical prices DOWNWARD.
    Nothing legitimate pushes `adj_close` ABOVE the same row's `close`.

SEMANTICS (substitute, do not null)
    A rejected row's adjusted close is replaced by the SAME row's
    `close` - the vendor split-adjusted price - not by NULL.

    Nulling (the original !471 behaviour) removes the observation
    entirely, so a MAX over the survivors is biased LOW for any ticker
    whose true high sits on a rejected row (the nulled ATH is lower
    than the true one). Substituting keeps the observation and removes that
    [redacted: internal research note]
    arguably the more correct one anyway, and it is exactly what
    COALESCE(adj_close, close) already falls back to when adj_close is
    missing. Substitution never invents a value: `close` is always
    present and positive on a rejected row (the predicate requires
    `close > 0` before it can reject).

    This does change factor VALUES, not just their safety - see the WARN
    below. It is still a mask, not a repair: the real fix stays upstream.
    Every substitution is logged at WARN so the upstream problem remains
    visible.

NULL VS NaN (same substituted value, DIFFERENT branch)
    A NULL adj_close and a NaN adj_close both end up serving the row's
    `close`, but they take different branches to get there, and the two
    are not interchangeable - PostgreSQL (and IEEE 754 generally) treats
    NaN as NOT NULL.

    A NULL adj_close PASSES the guard via the NULL branch: `guard_sql`'s
    `adj_col IS NULL` clause is TRUE, so the row is accepted as
    "plausible" and `guarded_px_sql` falls back to `close` through
    `COALESCE(adj_close, close)` on the accepted side of the CASE.

    A NaN adj_close is REJECTED by the ratio branch, NOT accepted by the
    NULL branch: NaN IS NULL is FALSE (it is a real float value, not a
    SQL NULL), so the NULL clause does not apply, and `NaN <= 1.1 *
    close` evaluates to FALSE (NaN is not `<=` anything, including
    itself) - so `guard_sql` as a whole is FALSE and the row is routed
    through the CASE's ELSE (rejected) branch, which also substitutes
    `close`. The numpy mirror (`is_rejected` / `guarded_px`) reaches the
    same outcome via `np.isfinite(adj)`, which is FALSE for NaN.

    Both branches happen to substitute the same value (`close`), so this
    asymmetry is not itself a correctness bug here - but assuming a NaN
    would be silently waved through the NULL path exactly like a real
    NULL is the mistake that caused Meridian#318: a NaN was written to
    `factors.group_scores.group_score` on the assumption a downstream
    `group_score IS NOT NULL` completeness check would treat it the same
    as NULL. It does not - PostgreSQL's NaN is NOT NULL - so the row read
    as "covered" by the SQL check and "uncovered" by pandas' `.notna()`
    elsewhere, and the two disagreed permanently. Never assume NaN and
    NULL take the same code path just because they can look - or, as
    here, actually be - equivalent in the final output.
"""

import logging
from datetime import date

logger = logging.getLogger(__name__)

# Maximum plausible adj_close / close ratio. See module docstring for the
# derivation. Nothing legitimate pushes adj_close ABOVE close, so the true
# bound is 1.0; 1.1 is that bound plus headroom for rounding accumulation
# over multi-thousand-row adjustment chains.
ADJ_CLOSE_MAX_RATIO = 1.1

# Cap on how many individual rejected rows a single WARN line enumerates.
# The count is always reported in full; only the per-row detail is capped.
MAX_REJECTS_LOGGED = 15


def guard_sql(
    adj_col: str = "adj_close",
    close_col: str = "close",
    ratio: float = ADJ_CLOSE_MAX_RATIO,
) -> str:
    """SQL predicate that is TRUE for rows whose adj_close is plausible.

    Never evaluates to NULL: rows where the ratio cannot be formed (NULL
    adj_close, NULL/non-positive close) pass, so `NOT guard_sql(...)` is a
    safe complement. Valid ANSI SQL - used verbatim by both the PostgreSQL
    path (risk_factor_scores_asset) and the DuckDB/Parquet path
    (parquet_io), so the two implementations cannot drift.
    """
    return (
        f"({adj_col} IS NULL OR {close_col} IS NULL OR {close_col} <= 0"
        f" OR {adj_col} <= {ratio} * {close_col})"
    )


def guarded_px_sql(
    adj_col: str = "adj_close",
    close_col: str = "close",
    ratio: float = ADJ_CLOSE_MAX_RATIO,
) -> str:
    """Adjusted-close expression that SUBSTITUTES close for rejected rows.

    Mirrors COALESCE(adj_close, close) - the canonical adjusted basis for
    the price-level factors (#122) - except that an implausible adj_close
    falls back to the same row's split-adjusted close rather than
    poisoning the series (or, as before the substitution change, being nulled and
    biasing every MAX over the series LOW). See the module docstring's
    SEMANTICS section.

    The output is NULL only where COALESCE(adj_close, close) would itself
    be NULL: a rejected row always has close > 0 by construction of the
    predicate. Used verbatim by the PostgreSQL path, the DuckDB/Parquet
    path and the burst kernel so none of them can drift.
    """
    return (
        f"CASE WHEN {guard_sql(adj_col, close_col, ratio)}"
        f" THEN COALESCE({adj_col}, {close_col}) ELSE {close_col} END"
    )


def is_rejected(adj_close, close, ratio: float = ADJ_CLOSE_MAX_RATIO):
    """Pandas/numpy mirror of `NOT guard_sql(...)`.

    Returns a boolean array: True where the row's adj_close is implausible.
    Kept in lockstep with the SQL predicate by tests/test_adj_close_guard.py,
    which evaluates both over the same fixture.
    """
    import numpy as np
    import pandas as pd

    adj = pd.to_numeric(pd.Series(adj_close), errors="coerce").to_numpy(dtype=float)
    cl = pd.to_numeric(pd.Series(close), errors="coerce").to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        return np.asarray(
            np.isfinite(adj) & np.isfinite(cl) & (cl > 0) & (adj > ratio * cl),
            dtype=bool,
        )


def guarded_px(adj_close, close, ratio: float = ADJ_CLOSE_MAX_RATIO):
    """Pandas/numpy mirror of `guarded_px_sql(...)`.

    Returns a float array: COALESCE(adj_close, close) where the row is
    plausible, the row's own `close` where it is not. Kept in lockstep
    with the SQL expression by tests/test_adj_close_guard.py, which
    evaluates both over the same fixture.
    """
    import numpy as np
    import pandas as pd

    adj = pd.to_numeric(pd.Series(adj_close), errors="coerce").to_numpy(dtype=float)
    cl = pd.to_numeric(pd.Series(close), errors="coerce").to_numpy(dtype=float)
    coalesced = np.where(np.isnan(adj), cl, adj)
    return np.where(is_rejected(adj, cl, ratio), cl, coalesced)


# ---------------------------------------------------------------------------
# WARN logging
# ---------------------------------------------------------------------------
# Requirement: upstream corruption must stay VISIBLE. Both loggers below emit
# a single WARN per (caller, score_date) carrying the row count, the affected
# ticker count, and a bounded sample of (ticker, date, value) triples.


def log_rejected_aggregate(
    df,
    caller: str,
    score_date: date,
    log: "logging.Logger | None" = None,
) -> int:
    """WARN on rejected rows summarised by a MAX/COUNT ... FILTER aggregate.

    Expects the columns emitted by the guarded ATH / 52w-high queries:
    `ticker`, `rejected_rows`, `rejected_max_px`, `rejected_max_date`.
    Accepts either a pandas or a Polars frame (the PostgreSQL path produces
    the former, the DuckDB/Parquet path the latter). Returns the total number
    of rejected rows.
    """
    import pandas as pd

    log = log or logger
    if df is None or len(df) == 0:
        return 0
    if not isinstance(df, pd.DataFrame) and hasattr(df, "to_pandas"):
        df = df.to_pandas()
    if "rejected_rows" not in df.columns:
        return 0
    hits = df[df["rejected_rows"].fillna(0) > 0]
    if hits.empty:
        return 0
    total = int(hits["rejected_rows"].sum())
    worst = hits.sort_values("rejected_max_px", ascending=False).head(MAX_REJECTS_LOGGED)
    sample = ", ".join(
        f"{r.ticker}@{r.rejected_max_date}={r.rejected_max_px:.6g}" for r in worst.itertuples()
    )
    log.warning(
        "%s(%s): adj_close guard substituted close for %d row(s) across %d ticker(s) "
        "(adj_close > %.1fx close - implausible upstream adj_close). Highs for these "
        "tickers are computed on the "
        "split-adjusted close basis for the affected rows. "
        "Worst offenders (ticker@date=adj_close): %s%s",
        caller,
        score_date,
        total,
        len(hits),
        ADJ_CLOSE_MAX_RATIO,
        sample,
        "" if len(hits) <= MAX_REJECTS_LOGGED else f" ... (+{len(hits) - MAX_REJECTS_LOGGED} more)",
    )
    return total


def log_rejected_rows(
    df,
    caller: str,
    score_date: date,
    value_col: str = "adj_px_rejected",
    ticker_col: str = "ticker",
    date_col: str = "d",
    log: "logging.Logger | None" = None,
) -> int:
    """WARN on rejected rows carried row-wise in a fetched price frame.

    `value_col` holds the offending adj_close for rejected rows and NULL
    everywhere else. Returns the total number of rejected rows.
    """
    log = log or logger
    if df is None or len(df) == 0 or value_col not in df.columns:
        return 0
    hits = df[df[value_col].notna()]
    if hits.empty:
        return 0
    worst = hits.sort_values(value_col, ascending=False).head(MAX_REJECTS_LOGGED)
    sample = ", ".join(
        f"{getattr(r, ticker_col)}@{getattr(r, date_col)}={getattr(r, value_col):.6g}"
        for r in worst.itertuples()
    )
    n_tickers = int(hits[ticker_col].nunique())
    log.warning(
        "%s(%s): adj_close guard substituted close for %d price row(s) across %d ticker(s) "
        "(adj_close > %.1fx close - implausible upstream adj_close). The same row's "
        "split-adjusted close is used instead. "
        "Worst offenders (ticker@date=adj_close): %s%s",
        caller,
        score_date,
        len(hits),
        n_tickers,
        ADJ_CLOSE_MAX_RATIO,
        sample,
        "" if n_tickers <= MAX_REJECTS_LOGGED else " ...",
    )
    return int(len(hits))
