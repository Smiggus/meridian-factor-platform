"""Shared floor for bogus prints and illiquid names, for code that reads `ohlcv.prices` in
PostgreSQL directly (Meridian #275, LDP #182).

Some bad close prints reach PG `ohlcv.prices`: sub-penny or zero-volume prints that produce
impossible `ret_1d` values. A synthetic example:

    day 1  close=0.000001  volume=1000  ret_1d=-0.999999   (~-100%)
    day 2  close=0.5       volume=0     ret_1d=499999.0    (huge positive)

The bad value on day 2 sits on a row whose own close is fine, since it was the prior day's close
that was bogus. Flooring on `close < MIN_PRICE_FLOOR OR volume <= 0` catches both rows: day 1 by
its sub-penny close and day 2 by its zero volume, which marks a stale print and not a trade.

The floor was first written for the risk factor group (#259: `_MIN_PRICE_FLOOR` and
`_fetch_delisted_tickers` in `risk_factor_scores_asset.py`). It is collected here and extended
with a dollar-volume (ADV) floor, so that `check_ret1d_input_quality` and `risk_metrics` exclude
the same rows the risk group does and no consumer invents its own thresholds. Most
extreme-|ret_1d| rows are bogus prints or illiquid shells. What survives the floor is
single-name outliers in liquid tickers, real large moves or data errors, and that is what the
check should keep judging. Applied to the check's own aggregate, the floor leaves a small
residue from one name, so the check's sensitivity to broad corruption (the `n_extreme > 10` /
std bound) is unchanged.

Delistings have two guards. `fetch_delisted_tickers` reads `entities.companies.delisting_date`,
a nullable upstream column. Where it is NULL, `NULL < score_date` is never true and the guard
lets the ticker through, so the cross-section can grow with names that have no recent price
(#291). `stale_tickers_from_last_seen` does not depend on that column: a ticker with no traded
bar within `STALE_TRADING_DAY_TOLERANCE` trading days of `score_date` is excluded whether or not
`delisting_date` is set. Callers OR the two, and either is enough to exclude a ticker for the
partition.

`fetch_delisted_tickers` also has to avoid the opposite error, a false positive on a reused
ticker. Symbols are recycled after the original issuer dies, so a query of the form `SELECT
DISTINCT p.ticker ... JOIN entities.companies c ON c.entity_id = p.entity_id WHERE
c.delisting_date < :score_date` gives a live issuer its dead predecessor's delisting, and the
name drops out of the scored universe. Instead the function resolves each ticker to the entity
behind its most recent daily bar as of score_date (`DISTINCT ON (p.ticker) ... ORDER BY
p.ticker, p.ts_event DESC, p.entity_id DESC NULLS LAST`) and tests only that entity's
delisting_date. Its docstring explains the windowing decision.

When the scored universe shrinks after a change to the delisting inputs, part of the drop is
usually correct, because newly visible dead tickers are properly excluded. The thing to look for
is the residual: tickers whose current issuer is alive and printing but were excluded because a
dead predecessor carried a delisting_date.
"""

from __future__ import annotations

import logging
import os
from datetime import date, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import text

_log = logging.getLogger(__name__)

# Minimum traded (raw, unadjusted) close below which a row is treated as a
# bogus print. Matches #259's `_MIN_PRICE_FLOOR` (risk_factor_scores_asset.py) -
# kept in sync by hand; if one changes, both should.
MIN_PRICE_FLOOR = 0.01

# Minimum trailing-day dollar volume (raw_close * volume) below which a row
# is treated as an illiquid shell rather than a real print. Calibrated well
# below any name Meridian actually scores so this only drops the genuinely
# dead/illiquid tail.
MIN_DOLLAR_VOLUME = 10_000.0

# SQL predicate fragment shared by both direct-PG consumers. `close` and
# `volume` are the raw (unadjusted) ohlcv.prices columns; dollar volume uses
# raw_close (consistent with the risk-group and risk_metrics convention).
BOGUS_PRINT_SQL_GUARD = """
    close >= :min_price_floor
    AND volume > 0
    AND (raw_close * volume) >= :min_dollar_volume
"""

BOGUS_PRINT_SQL_PARAMS = {
    "min_price_floor": MIN_PRICE_FLOOR,
    "min_dollar_volume": MIN_DOLLAR_VOLUME,
}


# Implausible ret_1d magnitude (Meridian #336)
# The floor above catches BOGUS PRINTS - sub-penny closes, zero-volume stale
# prints, dead illiquid shells. It does NOT catch the OTHER corruption class:
# a perfectly liquid, perfectly priced name whose `ret_1d` is discontinuous
# across a split day, or matches neither the raw nor the adjusted price
# series. Such rows can carry an absurd ret_1d on names that pass every
# close/volume/ADV test, so
# a magnitude bound is the only predicate that sees them.
#
# 5.0 (+500% in ONE session) is the repo's existing bad-print bound: the
# portfolio-sort research harness has used `BAD_PRINT_ABS = 5.0` since it was
# written (scripts/portfolio_sort.py), on exactly this rationale - "a single
# bad print can hugely distort a mean cumulative return", and it DROPS rather
# than clips. Reusing that number keeps the research harness and the evidence
# engine judging the same panel. It sits far above any real single-session
# equity move a scored name could plausibly print and orders of magnitude
# below a corrupted-row scale.
#
# Note this is tighter than the 50.0 gate in
# checks/ret1d_checks.py. That gate is aligned to LDP #182's CLEANING boundary
# (LDP NULLs |ret_1d| > 50 upstream, so a check that re-flagged them would fire
# on rows that are already gone); this is a CONSUMER-side exclusion bound, and
# a consumer that averages returns equal-weight across a decile is hurt long
# before 50.
MAX_ABS_RET_1D = 5.0

# SQL predicate fragment for direct-PG consumers that want the same exclusion
# pushed down. Use with IMPLAUSIBLE_RET1D_SQL_PARAMS.
PLAUSIBLE_RET1D_SQL_GUARD = """
    ret_1d IS NOT NULL
    AND ABS(ret_1d) <= :max_abs_ret_1d
"""

PLAUSIBLE_RET1D_SQL_PARAMS = {"max_abs_ret_1d": MAX_ABS_RET_1D}


def implausible_ret1d_mask(ret_1d: pd.Series) -> pd.Series:
    """Boolean mask (index-aligned), True where the ret_1d value is
    implausible and must be EXCLUDED (not clipped) before any return
    aggregation. True for non-finite values and for |ret_1d| > MAX_ABS_RET_1D.

    Exclusion rather than clipping is on purpose: clipping a 4e6 return to
    +500% still lets a corrupt row masquerade as a real extreme and still
    moves an equal-weight decile mean materially, whereas
    dropping it costs one name-day out of hundreds of thousands.
    """
    values = pd.to_numeric(ret_1d, errors="coerce")
    finite = pd.Series(np.isfinite(values.to_numpy(dtype=float)), index=values.index, dtype=bool)
    return (~finite) | (values.abs() > MAX_ABS_RET_1D)


# Calendar-day lookback shared by every hygiene query in this module (the
# delisting resolver below and the bulk last-seen / trading-calendar queries
# further down). STALE_TRADING_DAY_TOLERANCE (15 trading days) is ~21 calendar
# days including weekends; 60 calendar days leaves a wide margin (holidays,
# thin names that print sparsely) while keeping every query a narrow, sargable
# range scan rather than an unbounded table scan.
HYGIENE_LOOKBACK_CALENDAR_DAYS = 60

# Meridian#454: wrong-entity-link exemption for the delisting guard.
#
# A wrong entity link can carry a live ticker onto the
# ticker's most recently ENDED span, i.e. a dead predecessor listing, so the
# delisting guard would exclude a live ticker because of someone else's
# delisting_date. A bar printed more than DELISTING_LINK_TOLERANCE_DAYS
# calendar days after the linked entity's delisting_date contradicts the link,
# and the ticker is then NOT treated as delisted.
#
# The tolerance absorbs the few trailing bars a genuinely delisted name can
# print around its delisting date.
DELISTING_LINK_TOLERANCE_DAYS = 5

# The exemption applies only to score dates on or after this date; before
# it the earlier semantics are kept exactly. Remove this bound (and the date
# gate in `fetch_delisted_tickers` and the staged mirror) when it is no
# longer needed (Meridian#454).
DELISTING_LINK_CHECK_SINCE = date(2026, 1, 1)

# A kept (link-contradicted) ticker whose delisting_date is within this many
# calendar days of the score date is "recent": it may be a REAL delisting that is
# still printing OTC / ghost bars, so the warning separates it from the "old"
# group, where the delisting predates the score date by so long that a wrong
# entity link is almost the only explanation (Meridian#466).
DELISTING_LINK_RECENT_DAYS = 90

# Per-process memo so the contradicted-link warning is emitted at most once per
# score_date (the guard runs once per partition and again in batch scope).
# Bounded: cleared wholesale when it reaches the cap.
_LINK_WARN_MEMO_CAP = 512
_link_warned: set = set()
_LINK_WARN_LIST_MAX = 25


# Reason codes that are NOT delistings (Meridian#325)
#
# CRSP's `dlstcd` 100 means "issue still trading at end of file". It is an
# ACTIVE marker, the opposite of a delisting. A loader that copied it into
# `entities.companies.delisting_date` would stamp still-live companies with
# the file's cut-off date:
#
#     delisting_reason = 'crsp_dlstcd_100'
#     delisting_date   = <the file's cut-off date>   <- IDENTICAL across all such rows
#
# A single shared date across many companies is a cut-off, not an event. Real
# codes spread naturally over many years.
#
# Without this filter a re-score would drop all such tickers from EVERY
# staged date, and promoting would delete their factors across the whole
# history. This guard stays regardless of upstream behaviour: a delisting
# flag is an ABSENCE-shaped input, so when it is wrong names vanish silently
# and nothing errors. We should not be one bad upstream batch away from
# deleting live names from the whole factor history.
NON_DELISTING_REASONS: frozenset = frozenset({"crsp_dlstcd_100"})


def fetch_delisted_tickers(
    engine,
    score_date: date,
    lookback_days: int = HYGIENE_LOOKBACK_CALENDAR_DAYS,
) -> frozenset[str]:
    """Return the set of tickers whose CURRENT issuer is confirmed delisted
    before score_date.

    WRONG-LINK EXEMPTION (Meridian#454). For `score_date >=
    DELISTING_LINK_CHECK_SINCE`, a ticker whose latest bar is more
    than DELISTING_LINK_TOLERANCE_DAYS (5) calendar days after the linked
    entity's delisting_date is NOT returned: a bar that late contradicts the
    link (an entity-enrichment link attaches a live ticker to a dead
    predecessor listing). Such tickers are logged once per score_date.
    For earlier score dates the guard is unchanged: delisted iff the latest
    bar's entity has delisting_date < score_date, excluding
    NON_DELISTING_REASONS.

    TRADE-OFF. In the 2026+ regime a genuine ghost ticker that keeps printing
    bars more than 5 days after a real delisting is no longer caught here; it
    is caught only by the stale-by-bars rule (once it stops printing). The time bound
    exists so that older history, where such ghost bars are real and
    believable, keeps the full guard. The check is bounded in time rather than
    in tickers because the defect itself only affects recent dates.

    "Current issuer" = the entity behind the ticker's most recent daily bar
    at or before score_date. Ticker symbols get recycled after the original
    issuer dies, so a form of this query that matched any entity that ever
    used the symbol would wrongly exclude live names. Resolving to the latest bar's entity
    is what makes a recycled symbol judged on its live issuer instead of on
    a company that died two decades ago.

    Uses entities.companies.delisting_date (populated by the LDP
    delisting_refresh_job). Only covers tickers whose ohlcv.prices rows
    carry an entity_id linkage. A ticker whose latest bar has no entity_id
    cannot be checked and remains unguarded - assumed active (conservative,
    unchanged behaviour: the JOIN simply drops it).

    PIT-correct: the `ts_event < score_date + 1 day` half-open bound means a
    historical partition resolves the ticker to whichever issuer was printing
    bars AS OF that date, never a later one (`ts_event` is stored at midnight
    UTC and the pipeline runs in UTC, so the half-open date bound is exact;
    same idiom as CRITICAL_RULES rule 13's `fetched_at < :next_day`).

    WINDOWING (perf): the latest-bar subquery is bounded to
    `lookback_days` calendar days rather than resolving over the whole of
    ohlcv.prices. The windowed form is an index range scan; the
    unbounded per-ticker resolution needs a full index scan and
    takes minutes per partition, which is unusable in a per-date asset
    loop. The window costs NOTHING in effective coverage, because a ticker is
    only ABSENT from the window if it printed no bar in the last
    `lookback_days` calendar days - i.e. it is already far past
    STALE_TRADING_DAY_TOLERANCE (15 trading days, ~21 calendar days) and is
    therefore excluded by the independent stale-by-bars guard that every
    caller OR-s with this one (risk_factor_scores_asset, risk_metrics_asset,
    partition_scoring, and `hygiene_exclusions` below). The union the callers
    actually apply is unchanged; only the split between the two guards moves.
    What the window preserves is exactly the case the delisting guard exists
    for: a name that KEEPS printing ghost bars after its delisting date, which
    the stale guard cannot see because the bars are fresh.

    (`entities.ticker_history` was evaluated as an alternative, unbounded,
    resolver and rejected: its intervals overlap - e.g. a recycled ticker can map to two
    issuers over the same years - so it cannot answer "which
    issuer owned this symbol on date D" unambiguously.)

    Returns a `DelistedTickers`: a frozenset (compares and combines as one) that
    also carries `link_kept`, the tickers the wrong-link rule kept instead of
    delisting, so `hygiene_exclusions` can report how many it rescued. A bar-day
    tie between two entities resolves to the highest entity_id (NULL lowest),
    the same pick the staged mirror (scripts/burst/stage_inputs.py) makes.

    Moved here from risk_factor_scores_asset.py (#259) so
    check_ret1d_input_quality and risk_metrics can share it (#275).
    """
    window_start = score_date - timedelta(days=lookback_days)
    next_day = score_date + timedelta(days=1)
    df = pd.read_sql(
        text("""
            WITH latest_bar AS (
                SELECT DISTINCT ON (p.ticker)
                       p.ticker,
                       p.entity_id,
                       p.ts_event
                FROM ohlcv.prices p
                WHERE p.resolution = 'daily'
                  AND p.security_type IN ('equity', 'etf')
                  AND p.ts_event >= :window_start
                  AND p.ts_event < :next_day
                -- entity_id DESC NULLS LAST: a same-day tie must resolve to the
                -- same entity the staged mirror picks (highest entity_id, an
                -- absent id sorting lowest). Postgres puts NULLs FIRST on DESC,
                -- which would hand the tie to the unlinked row and drop the
                -- ticker at the JOIN (Meridian#466).
                ORDER BY p.ticker, p.ts_event DESC, p.entity_id DESC NULLS LAST
            )
            SELECT lb.ticker,
                   lb.ts_event::date AS last_bar_date,
                   c.delisting_date
            FROM latest_bar lb
            JOIN entities.companies c ON c.entity_id = lb.entity_id
            WHERE c.delisting_date IS NOT NULL
              AND c.delisting_date < :score_date
              -- A reason that is not actually a delisting must not delist.
              -- See NON_DELISTING_REASONS (Meridian#325).
              AND (c.delisting_reason IS NULL
                   OR NOT (c.delisting_reason = ANY(:non_delisting)))
        """),
        engine,
        params={
            "score_date": score_date,
            "window_start": window_start,
            "next_day": next_day,
            "non_delisting": sorted(NON_DELISTING_REASONS),
        },
    )
    if df.empty:
        return DelistedTickers()
    contradicted = _link_contradicted_mask(df, score_date)
    if contradicted.any():
        kept = df[contradicted]
        recent_cut = pd.Timestamp(score_date) - pd.Timedelta(days=DELISTING_LINK_RECENT_DAYS)
        is_recent = pd.to_datetime(kept["delisting_date"]) >= recent_cut
        _warn_contradicted_links(
            score_date,
            sorted(kept.loc[~is_recent, "ticker"]),
            sorted(kept.loc[is_recent, "ticker"]),
        )
    return DelistedTickers(
        df.loc[~contradicted, "ticker"].tolist(),
        link_kept=df.loc[contradicted, "ticker"].tolist(),
    )


class DelistedTickers(frozenset):
    """The `fetch_delisted_tickers` result: a plain frozenset of delisted
    tickers (equal to, and usable as, any other frozenset) that also carries
    `link_kept`, the tickers the #454 wrong-link rule KEPT instead of delisting.

    The extra attribute lets `hygiene_exclusions` report how many tickers the
    rule rescued without a second query and without changing the return type
    callers and test doubles already rely on. Set operations (`&`, `|`)
    return a plain frozenset, so read `link_kept` off the direct result
    (`getattr(result, "link_kept", frozenset())`)."""

    link_kept: frozenset

    def __new__(cls, delisted=(), link_kept=()):
        obj = super().__new__(cls, delisted)
        obj.link_kept = frozenset(link_kept)
        return obj


def _link_contradicted_mask(df: pd.DataFrame, score_date: date) -> pd.Series:
    """True where the latest bar is more than DELISTING_LINK_TOLERANCE_DAYS
    after the linked entity's delisting_date, for score dates on or after
    DELISTING_LINK_CHECK_SINCE only (all False before it)."""
    if score_date < DELISTING_LINK_CHECK_SINCE:
        return pd.Series(False, index=df.index)
    return bar_contradicts_delisting_link(df["last_bar_date"], df["delisting_date"])


def bar_contradicts_delisting_link(bar_date, delisting_date) -> pd.Series:
    """The #454 wrong-link test, shared by every consumer: True where a bar
    dated `bar_date` is more than DELISTING_LINK_TOLERANCE_DAYS calendar days
    after the linked entity's `delisting_date`, i.e. the bar contradicts the
    link. NaT on either side is False (nothing to contradict). It does NOT
    apply the DELISTING_LINK_CHECK_SINCE gate: each caller gates on its own
    notion of "as of" (the score date here, the row date in
    feature_panel._universe_wide)."""
    gap = pd.to_datetime(bar_date) - pd.to_datetime(delisting_date)
    return gap > pd.Timedelta(days=DELISTING_LINK_TOLERANCE_DAYS)


def _warn_contradicted_links(
    score_date: date, old: list[str], recent: list[str] | tuple = ()
) -> None:
    """Log the contradicted-link tickers at most once per score_date per
    process (bounded memo), with a "+N more" marker when truncated and the
    Dagster run id when one is set.

    Split by the age of the linked delisting_date (Meridian#466): `old`
    (delisting more than DELISTING_LINK_RECENT_DAYS before the score date) is
    almost certainly a wrong entity link, `recent` may be a REAL delisting
    that is still printing OTC bars and deserves a human look."""
    if score_date in _link_warned:
        return
    if len(_link_warned) >= _LINK_WARN_MEMO_CAP:
        _link_warned.clear()
    _link_warned.add(score_date)
    run_id = os.environ.get("DAGSTER_RUN_ID")
    for tickers, label in (
        (old, "old delisting date, almost certainly a wrong entity link"),
        (
            recent,
            f"delisting within {DELISTING_LINK_RECENT_DAYS}d, possibly a real delisting still printing",
        ),
    ):
        if not tickers:
            continue
        shown = tickers[:_LINK_WARN_LIST_MAX]
        more = len(tickers) - len(shown)
        _log.warning(
            "delisting guard %s: %d ticker(s) print bars more than %d days after their "
            "linked entity's delisting date - entity link treated as wrong, NOT delisted "
            "(Meridian#454) [%s]%s: %s%s",
            score_date,
            len(tickers),
            DELISTING_LINK_TOLERANCE_DAYS,
            label,
            f" [run {run_id}]" if run_id else "",
            ", ".join(shown),
            f" +{more} more" if more > 0 else "",
        )


def bogus_row_mask(close: pd.Series, volume: pd.Series, raw_close: pd.Series) -> pd.Series:
    """Boolean mask (index-aligned), True where the row is a bogus/illiquid
    print per the same floor as BOGUS_PRINT_SQL_GUARD. For in-pandas
    filtering when a caller already has the frame in memory (avoids a
    second round-trip query)."""
    dollar_volume = raw_close * volume
    return (close < MIN_PRICE_FLOOR) | (volume <= 0) | (dollar_volume < MIN_DOLLAR_VOLUME)


# Trading-day tolerance for the price-staleness fallback (see the module
# docstring's PRICE-STALENESS FALLBACK section). Two bounds pin this number:
#   - LOWER bound: normal ingest lag. The LDP Parquet mirror can lag by a few
#     CALENDAR days (_MIRROR_FRESHNESS_BUFFER_DAYS in
#     risk_factor_scores_asset.py), roughly 3-4 trading days, and a
#     legitimately active name can also sit through an SEC Rule 12(k) trading
#     suspension, which is capped at 10 TRADING days by statute. A tolerance
#     at or below ~10 trading days would false-positive on a name mid-
#     suspension that is going to resume trading.
#   - UPPER bound: a genuinely dead ticker. Ghost names have no price for a
#     long time - a gap of many trading days well beyond this threshold - so
#     there is no tension between
#     "tolerant enough for halts/lag" and "tight enough to catch dead names";
#     any value comfortably above 10 and well below the dead-name gap works.
# 15 trading days (~3 calendar weeks) sits with margin above the 10-day
# suspension cap while staying far below the multi-month
# gaps it actually needs to catch.
STALE_TRADING_DAY_TOLERANCE = 15


def stale_tickers_from_last_seen(
    last_seen: pd.Series,
    trading_days: "pd.Series | np.ndarray | pd.Index",
    score_date: date,
    max_gap_trading_days: int = STALE_TRADING_DAY_TOLERANCE,
) -> frozenset[str]:
    """Tickers whose most recent observed bar is more than
    `max_gap_trading_days` TRADING days before `score_date`.

    `last_seen`: Series indexed by ticker, values = each ticker's most
      recent observed date within the caller's already-fetched lookback
      window (e.g. `prices.groupby("ticker")["d"].max()`). PIT-safe by
      construction as long as the caller's own fetch query already bounds
      rows to `ts_event::date <= score_date` (true for every caller in this
      module as of #291) - a historical backfill partition only ever sees
      rows up to its own score_date, so this never looks into the future.

    `trading_days`: the set of distinct dates present in that SAME fetched
      frame (e.g. `prices["d"].unique()`), used as a trading-calendar proxy.
      No new query: on any date where at least one liquid instrument
      (SPY / a sector ETF / any actively-traded name) has a row, that date is
      a real trading day, and the callers here always include such names in
      the same fetch. This means `last_seen` values are always members of
      the resulting sorted calendar, so no interpolation/nearest-date logic
      is needed - position lookups are exact.

    `score_date` is included explicitly (appended if not already present)
    so the trading-day gap is always measured relative to score_date itself,
    even on the rare partition where nothing in the fetched frame traded
    exactly on score_date.

    Returns the frozenset of stale tickers (empty if `last_seen` is empty).
    """
    if last_seen.empty:
        return frozenset()

    days = pd.to_datetime(pd.Index(trading_days).unique()).normalize()
    score_ts = pd.Timestamp(score_date)
    if score_ts not in days:
        days = days.append(pd.DatetimeIndex([score_ts]))
    days = days.sort_values()
    days_arr = days.to_numpy()

    score_pos = int(np.searchsorted(days_arr, np.datetime64(score_ts), side="left"))
    last_seen_arr = pd.to_datetime(last_seen).dt.normalize().to_numpy()
    last_pos = np.searchsorted(days_arr, last_seen_arr, side="left")

    gap_trading_days = score_pos - last_pos
    stale_mask = gap_trading_days > max_gap_trading_days
    return frozenset(last_seen.index[stale_mask])


# CORE FACTOR ASSET EXTENSION (#259 follow-up, flagged in !453's "other
# #259-exposed sites" audit): risk_factor_scores_asset.py / risk_metrics_asset.py
# already had a `prices` frame in memory to derive `stale_tickers_from_last_seen`
# for free. The CORE factor assets mostly do NOT fetch a multi-day OHLCV
# window at all - the fundamentals-based ones read fundamentals only, so
# there is nothing to derive last-seen dates
# from without a dedicated query. `fetch_last_seen_dates` + `fetch_trading_calendar`
# below are that dedicated query - ONE bulk GROUP BY over the ticker list (never
# per-ticker) bounded to a short recent window, so the added cost per partition
# is a single cheap indexed scan, the same shape as `fetch_delisted_tickers`.
# `hygiene_exclusions` composes both signals (delisted OR stale-by-bars) into
# the one call CORE assets need right after they resolve their ticker universe.

# The calendar-day lookback for the bulk last-seen / trading-calendar queries
# below is HYGIENE_LOOKBACK_CALENDAR_DAYS, defined near the top of this module
# (it is shared with fetch_delisted_tickers' latest-bar window).

# Liquid proxy ticker used as the trading-calendar signal when the caller has
# no ticker-agnostic date frame already in memory. SPY trades every session
# the market is open, so "dates SPY printed a bar" is a faithful trading
# calendar for this purpose (same idiom the risk-group callers get for free
# from their own already-fetched `prices` frame, which always includes SPY).
_CALENDAR_PROXY_TICKER = "SPY"


def fetch_last_seen_dates(
    engine,
    tickers: list[str],
    score_date: date,
    lookback_days: int = HYGIENE_LOOKBACK_CALENDAR_DAYS,
) -> pd.Series:
    """Bulk (single query, never per-ticker) most-recent-observed-bar date per
    ticker, PIT-bounded to score_date.

    Returns a Series indexed by ticker (datetime64 values). A ticker absent
    from the result had NO bar anywhere in [score_date - lookback_days,
    score_date] - since STALE_TRADING_DAY_TOLERANCE (15 trading days, ~21
    calendar days) is well inside the default 60-day lookback, an absent
    ticker is unambiguously stale; callers should treat "not in this Series"
    the same as "gap > tolerance" (see `stale_or_never_seen_tickers`, which
    does this).

    PIT-safe: bounded to `ts_event::date <= score_date`, so a historical
    backfill partition only ever sees bars available as of that date.
    """
    if not tickers:
        return pd.Series(dtype="datetime64[ns]")
    lookback_start = score_date - timedelta(days=lookback_days)
    # Window bounds are written against the raw timestamptz column, never
    # `ts_event::date > :start` - a cast on the column is not sargable, so the
    # planner walked EVERY ticker's full history under idx_ohlcv_ticker_res_ts
    # and filtered afterwards (many times the cost; a whole-table read per call once
    # burst workers + the daily CORE pods ran it concurrently).
    # `(:start)::date + 1 <= ts_event < (:end)::date + 1` is the same half-open
    # (start, end] window, folded to constants and pushed into the index.
    # The three sibling queries below follow the same form.
    df = pd.read_sql(
        text("""
            SELECT ticker, MAX(ts_event::date) AS last_seen
            FROM ohlcv.prices
            WHERE ticker = ANY(:tickers)
              AND resolution = 'daily'
              AND ts_event >= (:start)::date + 1
              AND ts_event < (:score_date)::date + 1
            GROUP BY ticker
        """),
        engine,
        params={"tickers": list(tickers), "start": lookback_start, "score_date": score_date},
    )
    if df.empty:
        return pd.Series(dtype="datetime64[ns]")
    return pd.to_datetime(df.set_index("ticker")["last_seen"])


# RANGE FORMS (Meridian #295). A caller that scores a CONTIGUOUS BLOCK of dates
# - the burst re-score kernel's chunk loop - would otherwise pay the two queries
# above once per date, over 60-day windows that overlap in 59 of 60 days. The
# four functions below let it pay once per BLOCK instead: two fetches over the
# union of the block's windows, plus two pure restrictions back down to one
# date. They live here, beside the per-date primitives they mirror, so a change
# to one is visibly a change the other has to follow - the hoist that broke in
# 2026-08 was a COPY of a primitive that later changed underneath it.
#
# They are additive: nothing in the per-date path calls them, and the burst
# kernel still routes every list-dependent step through the same
# `stale_tickers_from_last_seen` + never-seen composition
# `stale_or_never_seen_tickers` uses.


def _empty_bars() -> pd.DataFrame:
    """Empty (ticker, d) frame with the dtypes fetch_last_seen_bars_range emits."""
    return pd.DataFrame({"ticker": pd.Series(dtype=object), "d": pd.Series(dtype="datetime64[ns]")})


def fetch_last_seen_bars_range(
    engine,
    tickers: list[str],
    start: date,
    end: date,
    lookback_days: int = HYGIENE_LOOKBACK_CALENDAR_DAYS,
) -> pd.DataFrame:
    """RANGE form of `fetch_last_seen_dates`: the DISTINCT (ticker, date) bars
    every score_date in [start, end] could possibly draw on, in ONE query.

    Returns a DataFrame with columns `ticker` (object) and `d`
    (datetime64[ns]), for bars in (start - lookback_days, end].

    THE RESTRICTION PROPERTY (this is the whole point, so it is stated
    precisely). For any D in [start, end] and any ticker list L that is a
    SUBSET of `tickers`,

        fetch_last_seen_dates(engine, L, D, lookback_days)
          == this frame, restricted to rows with `ticker in L`
             and `D - lookback_days < d <= D`, grouped by ticker, MAX(d)

    which holds on three independent counts:

      * TICKER LIST. The per-date query's only ticker predicate is
        `ticker = ANY(:tickers)` - a ROW-level filter - and its aggregate is
        `GROUP BY ticker`, so each ticker's answer depends on that ticker's
        rows alone. Restricting a superset's rows to a sublist therefore
        yields exactly the sublist's own result. (Callers must still CHECK the
        subset relation: a ticker outside `tickers` is simply absent here,
        which a caller would otherwise misread as "never seen".)
      * WINDOW BOUND. The per-date window is the half-open
        (D - lookback_days, D]. For D in [start, end], D - lookback_days >=
        start - lookback_days and D <= end, so that window is contained in
        this query's (start - lookback_days, end]. Every row the per-date
        query would see is present; no row it would exclude survives the
        per-date re-filter, which uses the identical strict-lower /
        inclusive-upper comparison.
      * DUPLICATES. The per-date form aggregates MAX over rows; this one
        SELECTs DISTINCT (ticker, date) first. MAX over a set is unchanged by
        collapsing duplicates, and the two queries share every other
        predicate verbatim (`resolution = 'daily'`, no security_type filter).

    PIT-safe for every D in the range by construction: the derived per-date
    result is bounded to `d <= D`, so a partition never sees a later bar even
    though the underlying fetch read up to `end`. Callers MUST NOT use this
    frame unrestricted for a date earlier than `end`.
    """
    if not tickers:
        return _empty_bars()
    window_start = start - timedelta(days=lookback_days)
    df = pd.read_sql(
        text("""
            SELECT DISTINCT ticker, ts_event::date AS d
            FROM ohlcv.prices
            WHERE ticker = ANY(:tickers)
              AND resolution = 'daily'
              AND ts_event >= (:start)::date + 1
              AND ts_event < (:end)::date + 1
        """),
        engine,
        params={"tickers": list(tickers), "start": window_start, "end": end},
    )
    if df.empty:
        return _empty_bars()
    df["d"] = pd.to_datetime(df["d"])
    return df


def last_seen_from_bars_range(
    bars: pd.DataFrame,
    tickers: list[str],
    score_date: date,
    lookback_days: int = HYGIENE_LOOKBACK_CALENDAR_DAYS,
) -> pd.Series:
    """Restrict a `fetch_last_seen_bars_range` frame back down to ONE date.

    The literal pandas restatement of `fetch_last_seen_dates`'s SQL over the
    already-fetched superset - same half-open window, same GROUP BY ticker,
    same MAX, same returned shape (Series named `last_seen`, indexed by
    `ticker`, datetime64[ns]). See `fetch_last_seen_bars_range` for why the
    restriction is exact rather than approximate.

    `tickers` must be a subset of the list the frame was fetched for; that is
    the caller's obligation and is NOT checked here (checking it needs the
    original list, which the caller holds).
    """
    empty = pd.Series(dtype="datetime64[ns]", name="last_seen")
    empty.index.name = "ticker"
    if bars.empty:
        return empty
    lo = pd.Timestamp(score_date - timedelta(days=lookback_days))
    hi = pd.Timestamp(score_date)
    want = set(tickers)
    sub = bars[(bars["d"] > lo) & (bars["d"] <= hi) & bars["ticker"].isin(want)]
    if sub.empty:
        return empty
    out = sub.groupby("ticker", sort=False)["d"].max()
    out.name = "last_seen"
    out.index.name = "ticker"
    return out


def fetch_trading_calendar(
    engine,
    score_date: date,
    lookback_days: int = HYGIENE_LOOKBACK_CALENDAR_DAYS,
    proxy_ticker: str = _CALENDAR_PROXY_TICKER,
) -> pd.Series:
    """Distinct trading dates in (score_date - lookback_days, score_date],
    proxied via a definitely-liquid, always-trading ticker. One small query.

    PIT-safe: bounded to `ts_event::date <= score_date`.
    """
    lookback_start = score_date - timedelta(days=lookback_days)
    df = pd.read_sql(
        text("""
            SELECT DISTINCT ts_event::date AS d
            FROM ohlcv.prices
            WHERE ticker = :proxy_ticker
              AND resolution = 'daily'
              AND ts_event >= (:start)::date + 1
              AND ts_event < (:score_date)::date + 1
        """),
        engine,
        params={"proxy_ticker": proxy_ticker, "start": lookback_start, "score_date": score_date},
    )
    return pd.to_datetime(df["d"])


def fetch_trading_calendar_range(
    engine,
    start: date,
    end: date,
    lookback_days: int = HYGIENE_LOOKBACK_CALENDAR_DAYS,
    proxy_ticker: str = _CALENDAR_PROXY_TICKER,
) -> pd.Series:
    """RANGE form of `fetch_trading_calendar`: every proxy-ticker trading date
    in (start - lookback_days, end], in one query.

    Exact for the same window-containment reason as
    `fetch_last_seen_bars_range` (see its WINDOW BOUND paragraph); there is no
    ticker dimension to restrict, since the calendar probe is a single fixed
    proxy ticker and does not depend on the caller's list at all.
    """
    window_start = start - timedelta(days=lookback_days)
    df = pd.read_sql(
        text("""
            SELECT DISTINCT ts_event::date AS d
            FROM ohlcv.prices
            WHERE ticker = :proxy_ticker
              AND resolution = 'daily'
              AND ts_event >= (:start)::date + 1
              AND ts_event < (:end)::date + 1
        """),
        engine,
        params={"proxy_ticker": proxy_ticker, "start": window_start, "end": end},
    )
    return pd.to_datetime(df["d"])


def calendar_from_range(
    calendar: pd.Series,
    score_date: date,
    lookback_days: int = HYGIENE_LOOKBACK_CALENDAR_DAYS,
) -> pd.Series:
    """Restrict a `fetch_trading_calendar_range` Series back down to ONE date -
    the pandas restatement of `fetch_trading_calendar`'s half-open window."""
    lo = pd.Timestamp(score_date - timedelta(days=lookback_days))
    hi = pd.Timestamp(score_date)
    out = calendar[(calendar > lo) & (calendar <= hi)].reset_index(drop=True)
    out.name = "d"
    return out


def stale_or_never_seen_tickers(
    engine,
    tickers: list[str],
    score_date: date,
    max_gap_trading_days: int = STALE_TRADING_DAY_TOLERANCE,
) -> frozenset[str]:
    """Stale-by-bars exclusion set for callers with no already-fetched price
    frame: bulk-fetches last-seen dates + a trading calendar (both single
    queries, see the two functions above) and applies
    `stale_tickers_from_last_seen`, PLUS explicitly includes tickers entirely
    absent from the lookback window (a ticker with zero bars in
    HYGIENE_LOOKBACK_CALENDAR_DAYS is unambiguously past the tolerance - see
    `fetch_last_seen_dates` docstring).

    Fails OPEN (returns an empty set) when the trading-calendar proxy itself
    has no data in the window - e.g. a very early historical score_date
    before the proxy ticker's inception - so a thin historical partition is
    never silently emptied out by a calendar gap that has nothing to do with
    the tickers being judged.
    """
    if not tickers:
        return frozenset()
    last_seen = fetch_last_seen_dates(engine, tickers, score_date)
    calendar = fetch_trading_calendar(engine, score_date)
    if calendar.empty:
        return frozenset()
    stale = set(stale_tickers_from_last_seen(last_seen, calendar, score_date, max_gap_trading_days))
    never_seen = set(tickers) - set(last_seen.index)
    return frozenset(stale | never_seen)


def batch_hygiene_exclusions(
    engine, tickers: list[str], batch_start: date, batch_end: date
) -> tuple[frozenset[str], dict[str, int]]:
    """BATCH-SCOPE hygiene exclusions (#292): the set of tickers that
    `hygiene_exclusions` is GUARANTEED to exclude at EVERY score_date in
    [batch_start, batch_end].

    Purpose: an asset with a `BackfillPolicy.multi_run` batch does its bulk
    OHLCV fetches ONCE at batch scope, before its per-partition loop, so the
    per-partition `hygiene_exclusions` call cannot bound those fetches at
    all - they see the full delisted tape. This function is the
    filter that CAN be applied at batch scope. It is
    conservative: the surviving list is a strict SUPERSET of every
    partition's own post-hygiene universe, so the per-partition guard stays
    the sole authority over what actually gets scored and no value changes.

    MONOTONICITY (why the two halves are computed at different anchors):

    - DELISTED is NOT reliably monotone in date once the #454 wrong-link
      exemption applies (score dates >= DELISTING_LINK_CHECK_SINCE): a ticker
      can be delisted at an early partition and become link-contradicted at a
      later one when a bar prints more than DELISTING_LINK_TOLERANCE_DAYS
      after the delisting_date. So the batch leg excludes only tickers
      delisted at BOTH batch_start and batch_end. A ticker delisted at both
      ends printed no bar beyond the tolerance by batch_end, so none in
      between either, and stays delisted throughout. This is
      conservative (it can only exclude less than the per-partition guard
      does, never more) and leaves the per-partition guard as the sole
      authority. Entirely before DELISTING_LINK_CHECK_SINCE the plain
      `delisting_date < d` monotonicity holds and batch_start alone suffices.

    - STALE-BY-BARS is NOT monotone in either direction: a ticker can be
      stale at an early partition and resume trading before a later one
      (dormant -> live), and can trade early and go quiet later (live ->
      dormant). Neither `stale(batch_start)` nor `stale(batch_end)` is a
      subset of the intersection. The batch-safe test instead pairs the
      LATEST last-seen bar in the whole batch window with the EARLIEST
      partition date: if a ticker's most recent bar anywhere on/before
      batch_end is already more than `max_gap_trading_days` trading days
      before batch_start, then for every d in [batch_start, batch_end] its
      own as-of-d last bar is no later than that bar (max over a subset of
      rows) while d is no earlier than batch_start, so its as-of-d gap is at
      least as large - it is stale at d too. Tickers with no bar at all in
      the (widened) lookback window are stale at every d for the same
      reason.

    Returns (excluded_tickers, counts), same shape as `hygiene_exclusions`;
    "link_kept" counts universe tickers the #454 wrong-link rule kept at either
    batch end (informational, not part of the exclusion).
    Fails OPEN (empty exclusion set) when the trading-calendar proxy has no
    data in the window, exactly like `stale_or_never_seen_tickers`.
    """
    if not tickers:
        return frozenset(), {"delisted": 0, "stale": 0, "excluded_total": 0, "link_kept": 0}
    if batch_end < batch_start:
        raise ValueError(f"batch_end {batch_end} precedes batch_start {batch_start}")

    universe = set(tickers)
    # Delisted: the intersection of both batch ends (see docstring). Before
    # DELISTING_LINK_CHECK_SINCE the end-of-batch call adds nothing.
    start_raw = fetch_delisted_tickers(engine, batch_start)
    delisted = start_raw & universe
    link_kept = frozenset(getattr(start_raw, "link_kept", ())) & universe
    if batch_end != batch_start and batch_end >= DELISTING_LINK_CHECK_SINCE:
        end_raw = fetch_delisted_tickers(engine, batch_end)
        delisted = delisted & end_raw
        # Informational, like hygiene_exclusions: kept at either end.
        link_kept = link_kept | (frozenset(getattr(end_raw, "link_kept", ())) & universe)

    # Stale, anchored at (last bar <= batch_end) vs batch_start. The lookback
    # is widened by the batch span so the window still reaches
    # HYGIENE_LOOKBACK_CALENDAR_DAYS before batch_start.
    span_days = (batch_end - batch_start).days
    lookback = HYGIENE_LOOKBACK_CALENDAR_DAYS + span_days
    last_seen = fetch_last_seen_dates(engine, tickers, batch_end, lookback_days=lookback)
    calendar = fetch_trading_calendar(engine, batch_end, lookback_days=lookback)
    if calendar.empty:
        stale: frozenset[str] = frozenset()
    else:
        stale_by_gap = set(stale_tickers_from_last_seen(last_seen, calendar, batch_start))
        never_seen = set(tickers) - set(last_seen.index)
        stale = frozenset((stale_by_gap | never_seen) & universe)

    excluded = frozenset(delisted | stale)
    counts = {
        "delisted": len(delisted),
        "stale": len(stale),
        "excluded_total": len(excluded),
        "link_kept": len(link_kept),
    }
    return excluded, counts


def hygiene_exclusions(
    engine, tickers: list[str], score_date: date
) -> tuple[frozenset[str], dict[str, int]]:
    """Combined #259/#291 hygiene exclusion set for CORE factor assets:
    delisted (entities.companies.delisting_date < score_date) OR-ed with
    stale-by-bars (no observed print within STALE_TRADING_DAY_TOLERANCE
    trading days of score_date). Mirrors the exclusion the risk factor group
    already applies (risk_factor_scores_asset.py, risk_metrics_asset.py) -
    same two signals, same OR, just packaged for callers that do not already
    have a price frame in memory.

    Returns (excluded_tickers, counts) where counts has keys "delisted",
    "stale", "excluded_total" (union count, not a sum - a ticker can be both)
    and "link_kept": tickers in `tickers` that the #454 wrong-link rule kept
    out of "delisted" (informational; not part of the exclusion).
    """
    if not tickers:
        return frozenset(), {"delisted": 0, "stale": 0, "excluded_total": 0, "link_kept": 0}
    universe = set(tickers)
    delisted_raw = fetch_delisted_tickers(engine, score_date)
    delisted = delisted_raw & universe
    link_kept = frozenset(getattr(delisted_raw, "link_kept", ())) & universe
    stale = stale_or_never_seen_tickers(engine, tickers, score_date) & universe
    excluded = frozenset(delisted | stale)
    counts = {
        "delisted": len(delisted),
        "stale": len(stale),
        "excluded_total": len(excluded),
        "link_kept": len(link_kept),
    }
    return excluded, counts
