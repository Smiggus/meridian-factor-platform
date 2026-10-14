"""Guard against corrupted `ohlcv.prices.adj_close` values.

Now and then a historical row arrives with an adj_close many orders of magnitude above its
close. The value is finite, so np.isfinite() gates let it through, and anything built on a
historical extreme, a running maximum for instance, is wrong for that ticker from that row on.
Only a comparison within the row sees it.

The test is `adj_close <= K * close` with K = 1.1. close is the split-adjusted series and
adj_close is the same series with dividends applied, so an honest ratio never exceeds 1:
dividend back-adjustment and the other corporate actions (rights issues, spin-offs, returns of
capital) all move history down. The extra 0.1 absorbs rounding that builds up over long
adjustment chains.

Two other tests were tried and dropped. A rolling median needs K above 10 to survive a stock
that really does 10x in a year, and it still misses the corruption, because an affected ticker
can carry a long run of bad rows and then its trailing median is bad too. A ratio against
raw_close flags legitimate reverse splits, whose pre-split rows have adj_close/raw_close near
1e5. The same-row ratio is unmoved by price changes and by splits.

A rejected row gets its own `close` in place of adj_close. It is not nulled. The first version
nulled, which drops the observation and biases a MAX over the survivors low whenever the true
high sat on a rejected row. Substituting keeps the observation and matches what
COALESCE(adj_close, close) already does when adj_close is missing. close is present and positive
on any rejected row, since the predicate needs close > 0 before it can reject. This changes
factor values. It masks the problem and does not repair it, so every substitution is logged at
WARN.

NaN is a value in PostgreSQL, not a NULL (IEEE 754 has no NULL), and it sorts above every other
number. A NULL
adj_close passes through the `IS NULL` clause and COALESCE returns close. A NaN fails `adj_close
<= 1.1 * close`, so it counts as rejected and is also replaced by close. The numpy mirror splits
this up differently: is_rejected says False for NaN and True for +inf, and guarded_px swaps NaN
for close in a separate step. Both paths end with the same price. The guard is one-sided, so
-inf, zero and negative values are not policed.
"""

import logging
from datetime import date

logger = logging.getLogger(__name__)

# Largest plausible adj_close / close. The true bound is 1.0; see the module docstring for the margin.
ADJ_CLOSE_MAX_RATIO = 1.1

# A WARN line reports the full count but lists at most this many rows.
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
    poisoning the series. See the module docstring for why it substitutes
    rather than nulls.

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
    A +inf adj_close is rejected, as in SQL. A NaN adj_close is not flagged
    here, but `guarded_px` still serves `close` for it, which is what the SQL
    expression does. Kept in lockstep with the SQL predicate.
    """
    import numpy as np
    import pandas as pd

    adj = pd.to_numeric(pd.Series(adj_close), errors="coerce").to_numpy(dtype=float)
    cl = pd.to_numeric(pd.Series(close), errors="coerce").to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        return np.asarray(
            ~np.isnan(adj) & np.isfinite(cl) & (cl > 0) & (adj > ratio * cl),
            dtype=bool,
        )


def guarded_px(adj_close, close, ratio: float = ADJ_CLOSE_MAX_RATIO):
    """Pandas/numpy mirror of `guarded_px_sql(...)`.

    Returns a float array: COALESCE(adj_close, close) where the row is
    plausible, the row's own `close` where it is not. Kept in lockstep
    with the SQL expression.
    """
    import numpy as np
    import pandas as pd

    adj = pd.to_numeric(pd.Series(adj_close), errors="coerce").to_numpy(dtype=float)
    cl = pd.to_numeric(pd.Series(close), errors="coerce").to_numpy(dtype=float)
    coalesced = np.where(np.isnan(adj), cl, adj)
    return np.where(is_rejected(adj, cl, ratio), cl, coalesced)


# Both loggers below emit a single WARN per (caller, score_date) carrying the
# row count, the affected ticker count, and a bounded sample of
# (ticker, date, value) triples, so upstream corruption shows up in the logs.


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
