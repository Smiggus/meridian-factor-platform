"""
db.py - PostgreSQL helpers for the factors schema.

All writes are idempotent. Most tables use ON CONFLICT DO UPDATE; factors.daily_scores and
factors.group_scores never name a conflict target and instead use key-exact delete-then-insert
under a per-score_date advisory lock (Meridian#432). Factor scores are point-in-time immutable
in principle, but re-computation overwrites to allow correction without manual DELETE.
"""

from __future__ import annotations

import bisect
import io
import logging
import os
import random
import re
import statistics
import time
from datetime import date, datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.pool import QueuePool

from meridian_pipeline.pg_identity import PIPELINE_APPLICATION_NAME, pg_application_name

_log = logging.getLogger(__name__)

# Bounded connection pool, cached per-URL at module level.
#
# Switched from NullPool (one fresh connection per query) to a bounded
# QueuePool to avoid connection storms during heavy ops like
# gpu_asset (5 source queries x thousands of tickers x N concurrent runs).
# Caching the engine per-URL means consumers that call _engine() many
# times in one process get the same engine -- and therefore share its
# pool of up to (pool_size + max_overflow) = 5 connections. Connections
# are recycled every 5 minutes to avoid stale-connection issues.
#
# Sized for the run pod's actual concurrency: each run materialises
# assets sequentially via the multiprocess executor, so at most a handful
# of subprocess threads need a connection at once. pool_size=3 covers
# the common case; max_overflow=2 absorbs short bursts.
_POOL_SIZE = 3
_MAX_OVERFLOW = 2
_POOL_RECYCLE_SECONDS = 300

_ENGINE_CACHE: dict = {}

# Meridian#428 item 4 / #467 item 5: every connection names itself to Postgres via
# pg_identity (one helper shared with the API and the burst kernels); the env
# override is documented there.


def _engine(
    pguser: str,
    pgpass: str,
    pghost: str,
    pgport: int,
    pgdb: str,
    application_name: str | None = None,
):
    """Return a process-cached SQLAlchemy engine with a bounded QueuePool.

    Same connection params return the same engine instance; subsequent
    callers share its pool. See module-level constants for pool sizing.

    ``application_name`` is the role default (``meridian-factors`` when omitted;
    burst kernels pass ``meridian-burst``); the pg_identity env override still wins. Engines for different roles are cached separately.
    """
    url = f"postgresql://{pguser}:{pgpass}@{pghost}:{pgport}/{pgdb}"
    app_name = pg_application_name(application_name or PIPELINE_APPLICATION_NAME)
    key = (url, app_name)
    cached = _ENGINE_CACHE.get(key)
    if cached is not None:
        return cached
    engine = create_engine(
        url,
        connect_args={"application_name": app_name},
        poolclass=QueuePool,
        pool_size=_POOL_SIZE,
        max_overflow=_MAX_OVERFLOW,
        pool_recycle=_POOL_RECYCLE_SECONDS,
        pool_pre_ping=True,
    )
    _ENGINE_CACHE[key] = engine
    return engine


def engine_from_env():
    """Build an engine from PGUSER / PGPASS / PGHOST / PGPORT / PGDB env vars.

    Single-source-of-truth helper for callers that just want a db.internal engine
    using the standard env contract. Delegates to `_engine` so any pool /
    cache changes there propagate everywhere automatically.
    """
    return _engine(
        pguser=os.environ.get("PGUSER", "dbuser"),
        pgpass=os.environ["PGPASS"],
        pghost=os.environ.get("PGHOST", "db.internal"),
        pgport=int(os.environ.get("PGPORT", 5432)),
        pgdb=os.environ.get("PGDB", "marketdata"),
    )


# ─────────────────────────────────────────────
# Partition floor resolution
# ─────────────────────────────────────────────

# Hard-barrier floor query: GREATEST across every Meridian input table,
# each padded by its scoring lookback requirement. The result is the
# earliest date where ALL factor groups can produce output simultaneously.
#
# Each input table is padded by the longest lookback requirement of any
# factor group.
_PARTITION_FLOOR_SQL = """
SELECT GREATEST(
    (SELECT MIN(ts_event)::date + INTERVAL '730 days'
     FROM ohlcv.prices
     WHERE ret_1d IS NOT NULL
       AND security_type = 'equity'
       AND resolution = 'daily'),
    (SELECT MIN(filing_date)::date + INTERVAL '730 days'
     FROM fundamentals.income_statement
     WHERE period = 'quarter' AND filing_date IS NOT NULL),
    (SELECT MIN(filing_date)::date + INTERVAL '730 days'
     FROM fundamentals.balance_sheet
     WHERE filing_date IS NOT NULL),
    (SELECT MIN(filing_date)::date + INTERVAL '730 days'
     FROM fundamentals.cash_flow
     WHERE period = 'quarter' AND filing_date IS NOT NULL),
    (SELECT MIN(published_at)::date
     FROM analyst.price_targets
     WHERE price_target > 0),
    (SELECT MIN(fetched_at)::date
     FROM analyst.estimates
     WHERE period = 'annual')
)
"""

_PARTITION_FLOOR_FALLBACK = "2010-01-02"


def resolve_partition_start() -> str:
    """Return the earliest date where all Meridian factor groups can produce output.

    Checks PARTITION_START_DATE env var first (set in CI and local dev to skip
    the DB query). Falls back to _PARTITION_FLOOR_FALLBACK if db.internal is unreachable
    at import time so the code server boots regardless of DB state.

    Result is cached on the module at import time - floor changes (from LDP
    history extensions) take effect on the next code server restart.
    """
    if override := os.environ.get("PARTITION_START_DATE"):
        return override
    try:
        engine = _engine(
            pguser=os.environ.get("PGUSER", "dbuser"),
            pgpass=os.environ["PGPASS"],
            pghost=os.environ.get("PGHOST", "db.internal"),
            pgport=int(os.environ.get("PGPORT", 5432)),
            pgdb=os.environ.get("PGDB", "marketdata"),
        )
        with engine.connect() as conn:
            row = conn.execute(text(_PARTITION_FLOOR_SQL)).one()
        engine.dispose()
        if row[0] is None:
            return _PARTITION_FLOOR_FALLBACK
        return str(row[0].date() if hasattr(row[0], "date") else row[0])
    except Exception as exc:
        _log.warning("resolve_partition_start: DB query failed, using fallback. %s", exc)
        return _PARTITION_FLOOR_FALLBACK


def assert_score_date_not_future(score_date: date) -> None:
    """Reject future score_dates at the entry of any partitioned asset.

    Why: a misrouted Dagster run can stamp rows with a future score_date
    into factors.daily_scores, which downstream API consumers would then 404
    on. This guard stops that at write-time; callers should invoke it
    immediately after `date.fromisoformat(context.partition_key)`. See #20.
    """
    today = date.today()
    if score_date > today:
        raise ValueError(
            f"Refusing to score {score_date}: future date (today={today}). "
            f"Partition key must be <= today. See Meridian#20."
        )


def _assert_df_score_dates_not_future(df: pd.DataFrame) -> None:
    """Belt-and-suspenders: reject DataFrames whose score_date column contains
    any future dates. Catches callers that somehow bypass the asset guard."""
    if df.empty or "score_date" not in df.columns:
        return
    today = date.today()
    max_sd = pd.to_datetime(df["score_date"]).dt.date.max()
    if max_sd > today:
        raise ValueError(
            f"Refusing upsert: max score_date={max_sd} > today={today}. See Meridian#20."
        )


# ─────────────────────────────────────────────
# Cross-section single-dated guard (read-during-write hardening)
# ─────────────────────────────────────────────
#
# UPDATE (2026-08-15): LDP's completion watermark (ohlcv.ingest_watermark)
# is now live. `assert_cross_section_single_dated` checks it FIRST
# (`_watermark_status_for_date`) and only falls back to the coverage-ratio
# heuristic documented below when the watermark has no usable answer for
# score_date (unreachable, empty, or simply no row yet). The heuristic below
# is retained unchanged as that fallback - see `_watermark_status_for_date`'s
# docstring and the "WATERMARK-FIRST" paragraph in
# `assert_cross_section_single_dated`'s own docstring for the cutover.
#
# UPDATE (2026-09, Meridian #395): the decision logic above was extracted out
# of `assert_cross_section_single_dated` into non-raising `cross_section_ready`
# so a TRIGGER SENSOR could consult the identical predicate before launching a
# run, instead of only an asset finding out (and failing) after the run
# already started. `trigger_cross_section_ready` is the sensor-facing entry
# point - see its docstring for why it needs its own (unscoped) data-gathering
# path and for the incident this closes.
#
# UPDATE (2026-09-18, Meridian #413): `trigger_cross_section_ready`'s
# fallback (`_cross_section_present_count_unscoped` /
# `_cross_section_trailing_coverage_unscoped`) now requires a non-NULL
# `ret_1d`, not just bar presence. Bars land the moment LDP's price ingest
# writes the row; `ret_1d` is computed by a separate, later pass
# (`ohlcv_returns`) that can still be running once every bar for the date
# has landed. Counting bars let the sensor fire while returns were still
# being computed, so a run could start before the returns existed. The
# write-side guard
# (`cross_section_ready`, used by `assert_cross_section_single_dated`)
# deliberately stays bar-only - see that function's docstring and
# `trigger_cross_section_ready`'s for the full argument on why the two
# sides now count different populations despite sharing one decision
# function and one set of thresholds.
#
# LDP's nightly OHLCV ingest window rewrites/repairs `ohlcv.prices` for a
# period each night (see the ingest-window constant in
# scripts/burst/burst_common.py). A
# cross-section read taken WHILE that window is in progress sees a
# partially-rewritten table: some tickers already carry a bar dated the
# new score_date, others still carry the prior trading day's bar. A
# mid-ingest latest-per-ticker cross-section can return only a fraction of
# tickers as-of the new date, the rest one trading day
# stale, indistinguishable from "did not trade that day".
#
# This is NOT an exposure for the nightly partitioned scoring path: every
# price/fundamental lookup there is as-of bounded (`ts_event::date <=
# :score_date`, `filing_date <= :d`) and the nightly schedule scores the
# T-1 partition, so same-day-landing rows are excluded by the bound before
# they can be read. The exposure is a read-heavy rescore/backfill campaign
# started DURING the ingest window (see the module-level guard in
# scripts/burst/burst_common.py for the preflight that refuses to launch
# one), and more generally the absence of any assertion that a scored
# cross-section is actually single-dated.
# This guard is the belt to that preflight's suspenders: it fires from
# inside the per-partition scoring loop itself, so it also catches a
# campaign that was launched cleanly but overran into the next ingest
# window, or any other path that ends up reading history concurrently with
# an LDP rewrite.
#
# Threshold rationale: the per-session coverage fraction (tickers with a bar
# dated that session / universe size) is expected to stay close to its
# trailing norm under ordinary universe churn (new listings not yet
# backfilled, thin small-caps), while a mid-ingest read drops it sharply. A
# relative-drop threshold should sit with a wide margin above ordinary
# fluctuation and a clear margin below a mid-ingest reading - the same
# clean-separation reasoning
# `_PINNED_FLOOR_MAX_PCT` uses in checks/data_quality_checks.py. It is
# expressed as a RELATIVE drop against the cross-section's own trailing
# median (not an absolute constant) so it does not need retuning as the
# universe grows or shrinks over the life of the backfill.
_SINGLE_DATED_MAX_RELATIVE_DROP_PCT = 25.0

# Denominator guard: below this universe size the coverage fraction is too
# noisy for a percentage comparison to mean anything (one missing ticker in
# a 20-name universe is already a 5-point swing on its own, before any
# ingest-window effect). Mirrors the small-denominator suppression in
# `evaluate_dq_liquid_unscored` / `pinned_at_floor_ok`
# (checks/data_quality_checks.py) - both skip rather than divide on a small
# or zero denominator instead of manufacturing a false alarm.
_SINGLE_DATED_MIN_UNIVERSE = 50

# Trailing-history guard: need at least this many prior trading sessions
# with a measured coverage fraction before a median means anything. Early
# in a backfill (near the partition floor) or shortly after a fresh
# `PARTITION_START_DATE` there may be no prior data at all in the lookback
# window - "no history to compare against" must not be a failure, the same
# posture `evaluate_dq_universe_drift` takes when `prev_tickers is None`.
_SINGLE_DATED_MIN_TRAILING_DAYS = 5

# How many trading sessions to pull for the trailing baseline, and how far
# back (in calendar days) to look for them. 30 calendar days comfortably
# covers 10 trading sessions plus weekends/a holiday or two.
_SINGLE_DATED_LOOKBACK_TRADING_DAYS = 10
_SINGLE_DATED_LOOKBACK_CALENDAR_DAYS = 30


def cross_section_completeness_ok(
    current_fraction: float, trailing_fractions: list[float]
) -> tuple[bool, str]:
    """Pure decision (DB-free, unit-testable): is `current_fraction` consistent
    with the trailing norm, or does it look like a mid-ingest partial read?

    Args:
        current_fraction: share of the scoring universe with a bar dated the
            score_date under evaluation (0..1).
        trailing_fractions: the same statistic for the prior trailing trading
            sessions, most-recent first (any order is fine, only the median
            is used).

    Returns:
        (True, "") when there is not enough trailing history to judge, when
        the median trailing fraction is degenerate (<=0, nothing to compare
        against), or when the drop from the median is within tolerance.
        (False, reason) when the drop is material - see
        `_SINGLE_DATED_MAX_RELATIVE_DROP_PCT` for the threshold and its
        justification.
    """
    if len(trailing_fractions) < _SINGLE_DATED_MIN_TRAILING_DAYS:
        return True, "insufficient trailing history to judge"
    median = statistics.median(trailing_fractions)
    if median <= 0:
        return True, "trailing median is zero, no baseline to compare against"
    if current_fraction >= median:
        return True, ""
    relative_drop_pct = (median - current_fraction) / median * 100.0
    if relative_drop_pct <= _SINGLE_DATED_MAX_RELATIVE_DROP_PCT:
        return True, ""
    return (
        False,
        f"coverage fraction {current_fraction:.3f} is {relative_drop_pct:.1f}% "
        f"below the trailing {len(trailing_fractions)}-session median {median:.3f}; "
        f"threshold={_SINGLE_DATED_MAX_RELATIVE_DROP_PCT}%",
    )


def _cross_section_present_count(engine, tickers: list[str], d: date) -> int:
    """Distinct tickers (from `tickers`) with a daily equity bar dated exactly
    `d`. Sargable half-open range on ts_event (see `is_trading_day`'s
    docstring for why the `::date` cast form is avoided in WHERE)."""
    d_next = d + timedelta(days=1)
    with engine.connect() as conn:
        row = conn.execute(
            text("""
                SELECT COUNT(DISTINCT ticker) FROM ohlcv.prices
                WHERE resolution = 'daily'
                  AND ticker = ANY(:tickers)
                  AND ts_event >= :d AND ts_event < :d_next
            """),
            {"tickers": tickers, "d": d, "d_next": d_next},
        ).fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def _cross_section_trailing_coverage(
    engine,
    tickers: list[str],
    score_date: date,
    lookback_calendar_days: int,
    max_trading_days: int,
) -> list[tuple[date, int]]:
    """Distinct-ticker counts per trading session strictly before `score_date`,
    most recent first, capped at `max_trading_days` rows. One bounded,
    sargable query (range WHERE + GROUP BY on the cast column, not the WHERE
    predicate) instead of a per-day round trip.
    """
    lookback_start = score_date - timedelta(days=lookback_calendar_days)
    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT ts_event::date AS d, COUNT(DISTINCT ticker) AS n
                FROM ohlcv.prices
                WHERE resolution = 'daily'
                  AND ticker = ANY(:tickers)
                  AND ts_event >= :lookback_start AND ts_event < :score_date
                GROUP BY 1
                ORDER BY 1 DESC
                LIMIT :max_days
            """),
            {
                "tickers": tickers,
                "lookback_start": lookback_start,
                "score_date": score_date,
                "max_days": max_trading_days,
            },
        ).fetchall()
    return [(r[0], int(r[1])) for r in rows]


def _watermark_status_for_date(engine, score_date: date) -> Optional[bool]:
    """LDP's completion watermark (ohlcv.ingest_watermark, live 2026-08-15)
    for `score_date`. Returns True/False mirroring the row's `is_complete`
    column, or None if the table is unreachable (connection/relation error),
    empty, or simply has no row yet for this date - all three are UNKNOWN,
    not "incomplete", and the caller must fall back to the statistical
    coverage-ratio check below rather than treat unknown as a failure. One
    cheap indexed point lookup (`ingest_watermark_pkey` on trade_date)."""
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT is_complete FROM ohlcv.ingest_watermark WHERE trade_date = :d"),
                {"d": score_date},
            ).fetchone()
    except Exception:
        return None
    if row is None:
        return None
    return bool(row[0])


def cross_section_ready(engine, score_date: date, tickers: list[str]) -> tuple[bool, str]:
    """Non-raising decision behind `assert_cross_section_single_dated` (Meridian
    #395): is `tickers`' cross-section for `score_date` single-dated, or does it
    look like a mid-LDP-ingest partial read? Returns `(True, "")` when ready,
    `(False, reason)` when not - `assert_cross_section_single_dated` is now a
    thin wrapper that raises `ValueError(reason)` on the latter.

    This is the ONE definition of "ready" shared by the write-side guard
    (called from inside an asset's partition loop, with that asset's own
    `tickers` already in hand) and `trigger_cross_section_ready` below (called
    from a sensor deciding whether to launch a run at all, with no
    asset-specific universe yet) - see that function's docstring for why a
    second entry point exists rather than reusing this one directly, and for
    the Meridian#395 incident that motivated it. Sharing this function (and the
    `cross_section_completeness_ok` / `_SINGLE_DATED_MAX_RELATIVE_DROP_PCT`
    threshold it applies) means the two surfaces cannot independently drift on
    the DECISION FUNCTION or its thresholds.

    Measures the share of `tickers` with a `resolution='daily'` bar dated
    exactly `score_date`, and compares it to the trailing median of the same
    statistic over the prior `_SINGLE_DATED_LOOKBACK_TRADING_DAYS` trading
    sessions. Not ready when the current share falls materially below that
    trailing norm - the signature of reading `ohlcv.prices` while LDP's
    nightly ingest is still mid-rewrite (see the module-level comment above
    `_SINGLE_DATED_MAX_RELATIVE_DROP_PCT` for the incident and the threshold
    derivation).

    UPDATE (2026-09, Meridian#413): this function's own POPULATION (bar
    presence, via `_cross_section_present_count` /
    `_cross_section_trailing_coverage`) deliberately does NOT require
    `ret_1d`, unlike `trigger_cross_section_ready`'s fallback population
    below it - see that function's docstring for the full argument. In
    short: this guard runs from inside a specific asset's partition loop,
    after that asset has already resolved its own universe, and several of
    its callers don't consume `ret_1d` at all - requiring it here would refuse
    partitions those assets could score correctly.

    Callers that DO consume `ret_1d` are NOT uniformly protected by THIS
    guard - it defers the run, which is the strongest line of defense, but
    each caller does its own per-asset completeness check as a second line of
    defense. Those checks are of two kinds: a per-ticker check that drops (or
    nulls) a ticker for the partition when its own `ret_1d` is missing on
    score_date, and a frame-wide `parquet_io.require_window_ends_on` check,
    which only compares the FETCHED FRAME'S max date to score_date - it
    catches a fully-stale read, not a partial one where most tickers already
    have the row and a minority (the #413 shape) don't. Incomplete
    tickers are counted under `incomplete_ret1d` in the asset's
    materialization metadata. `trigger_cross_section_ready`'s tightened
    fallback (below) remains the primary defense - it defers the whole run
    before any of this per-asset logic even executes - and these per-asset
    checks are the second line of defense for the straggler-ticker shape a
    run-level gate cannot see.

    Silently ready (no query beyond the cheap current-day count) when the
    universe is too small to be a meaningful ratio
    (`_SINGLE_DATED_MIN_UNIVERSE`) or when there isn't enough trailing
    history to judge (`_SINGLE_DATED_MIN_TRAILING_DAYS`) - both are
    intentionally silent passes, not failures; see those constants'
    docstrings.

    WATERMARK-FIRST (2026-08-15): before falling back to the statistical
    coverage-ratio heuristic below, checks LDP's completion watermark
    (`ohlcv.ingest_watermark`) for `score_date` via `_watermark_status_for_date`.
    That table answers "is this date's ingest actually done" directly, from
    LDP's own coverage/returns thresholds, instead of us inferring it from a
    trailing-median comparison - a strictly cleaner signal when it is
    available. If the watermark says NOT complete, not-ready immediately
    without touching the (statistical, universe-size-gated) checks below. If
    it says complete, ready immediately - LDP's own gate is a stronger
    guarantee than ours, so there is nothing more to check. Only when the
    watermark is UNKNOWN for this date (unreachable, empty table, or simply no
    row yet - see `_watermark_status_for_date`'s docstring) does this fall
    through to the coverage-ratio heuristic, which is unchanged from before
    the watermark existed.

    Cheap: one indexed point lookup against `ohlcv.ingest_watermark`, plus -
    only on the fallback path - two bounded, sargable queries against
    `ohlcv.prices` (a single-day count and a single grouped range scan), same
    shape as `is_trading_day`.
    """
    watermark_status = _watermark_status_for_date(engine, score_date)
    if watermark_status is False:
        return False, (
            f"LDP's ingest watermark (ohlcv.ingest_watermark) marks this date NOT "
            f"complete. This is a direct completion signal from LDP itself, not an "
            f"inference - see the module comment above "
            f"_SINGLE_DATED_MAX_RELATIVE_DROP_PCT for the statistical fallback this "
            f"supersedes when the watermark is available. Wait for LDP to mark "
            f"{score_date} complete before scoring it."
        )
    if watermark_status is True:
        return True, ""

    if len(tickers) < _SINGLE_DATED_MIN_UNIVERSE:
        return True, ""
    trailing = _cross_section_trailing_coverage(
        engine,
        tickers,
        score_date,
        _SINGLE_DATED_LOOKBACK_CALENDAR_DAYS,
        _SINGLE_DATED_LOOKBACK_TRADING_DAYS,
    )
    if len(trailing) < _SINGLE_DATED_MIN_TRAILING_DAYS:
        return True, ""

    n_universe = len(tickers)
    trailing_fractions = [n / n_universe for _, n in trailing]
    n_present = _cross_section_present_count(engine, tickers, score_date)
    current_fraction = n_present / n_universe

    ok, msg = cross_section_completeness_ok(current_fraction, trailing_fractions)
    if ok:
        return True, ""

    prior_date, prior_n = trailing[0]  # most recent trailing trading session
    return False, (
        f"cross-section is not single-dated. "
        f"{n_present}/{n_universe} tickers ({current_fraction:.1%}) have a bar dated "
        f"{score_date}, vs {prior_n}/{n_universe} ({prior_n / n_universe:.1%}) dated "
        f"{prior_date} (the most recent trailing trading session) and a "
        f"{len(trailing)}-session trailing median of "
        f"{statistics.median(trailing_fractions):.1%}. {msg}. This is the signature of "
        f"reading ohlcv.prices while LDP's nightly ingest is still mid-rewrite (see the "
        f"module comment above _SINGLE_DATED_MAX_RELATIVE_DROP_PCT for the incident that "
        f"motivated this guard)."
    )


def assert_cross_section_single_dated(engine, score_date: date, tickers: list[str]) -> None:
    """Refuse to score a cross-section that is silently mixing as-of dates.

    Thin raising wrapper around `cross_section_ready` (Meridian #395) - see
    that function's docstring for the full watermark-first /
    coverage-ratio-fallback decision this applies. Callers should invoke it
    once per partition, alongside `assert_score_date_not_future`.
    """
    ready, reason = cross_section_ready(engine, score_date, tickers)
    if not ready:
        raise ValueError(f"Refusing to score {score_date}: {reason}")


def _cross_section_present_count_unscoped(engine, d: date) -> int:
    """Distinct EQUITY tickers with a NON-NULL `ret_1d` dated exactly `d`,
    with no external `tickers` list to scope against - the sensor-side
    counterpart to `_cross_section_present_count` (used by the
    tickers-scoped guard above).

    Requires `ret_1d IS NOT NULL`, not just bar presence (Meridian#413). A
    bar lands the moment the price row is ingested; `ret_1d` is computed by
    a separate downstream LDP pass (`ohlcv_returns`) that can still be
    running after every bar for the date has landed. Counting bars alone let
    `trigger_cross_section_ready`'s fallback read a date as ready while
    returns were still being computed. See
    `trigger_cross_section_ready`'s docstring for why this scoped-out
    population deliberately does NOT change to match
    (`_cross_section_present_count`, used by the write-side guard, stays
    bar-only).

    Same `resolution='daily' AND security_type='equity'` filter
    `tradable_universe_size` uses, now additionally gated on `ret_1d IS NOT
    NULL`. Sargable: `EXPLAIN ANALYZE` shows this
    still resolves via `idx_ohlcv_ts_res` (the `ts_event`/`resolution`
    range scan already in use pre-#413) with `ret_1d IS NOT NULL` folded
    into the same heap filter as `security_type` - no plan change, no new
    index needed. Used by `trigger_cross_section_ready`, which has no asset
    universe in hand yet - see its docstring."""
    d_next = d + timedelta(days=1)
    with engine.connect() as conn:
        row = conn.execute(
            text("""
                SELECT count(DISTINCT ticker) FROM ohlcv.prices
                WHERE resolution = 'daily' AND security_type = 'equity'
                  AND ts_event >= :d AND ts_event < :d_next
                  AND ret_1d IS NOT NULL
            """),
            {"d": d, "d_next": d_next},
        ).fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def _cross_section_trailing_coverage_unscoped(
    engine,
    score_date: date,
    lookback_calendar_days: int,
    max_trading_days: int,
) -> list[tuple[date, int]]:
    """Trailing per-session distinct EQUITY ticker counts with a NON-NULL
    `ret_1d`, unscoped to any external `tickers` list - the sensor-side
    counterpart to `_cross_section_trailing_coverage`.

    Filters on `ret_1d IS NOT NULL` for the same reason
    `_cross_section_present_count_unscoped` does (Meridian#413) - and for
    the same reason, the numerator (current-date count) and this
    denominator (trailing baseline) MUST count the identical population, or
    the ratio the fallback compares is skewed by construction. See
    `trigger_cross_section_ready`."""
    lookback_start = score_date - timedelta(days=lookback_calendar_days)
    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT ts_event::date AS d, count(DISTINCT ticker) AS n
                FROM ohlcv.prices
                WHERE resolution = 'daily' AND security_type = 'equity'
                  AND ts_event >= :lookback_start AND ts_event < :score_date
                  AND ret_1d IS NOT NULL
                GROUP BY 1
                ORDER BY 1 DESC
                LIMIT :max_days
            """),
            {
                "lookback_start": lookback_start,
                "score_date": score_date,
                "max_days": max_trading_days,
            },
        ).fetchall()
    return [(r[0], int(r[1])) for r in rows]


def trigger_cross_section_ready(engine, score_date: date) -> tuple[bool, str]:
    """Non-raising readiness predicate for TRIGGER SENSORS deciding whether to
    launch `daily_factor_core_job` for `score_date` (Meridian #395).

    Shares its decision logic with the write-side guard (`cross_section_ready`
    / `assert_cross_section_single_dated`): watermark first
    (`_watermark_status_for_date`), falling back to the SAME
    relative-drop-from-trailing-median comparison
    (`cross_section_completeness_ok`, `_SINGLE_DATED_MAX_RELATIVE_DROP_PCT`)
    when the watermark has no usable answer yet - one set of thresholds,
    shared, so a sensor and an asset cannot independently drift on the
    DECISION FUNCTION.

    Differs from `cross_section_ready` in two ways. First, mechanically: a
    sensor is deciding whether to launch a run at all, so it has no
    asset-specific `tickers` list yet - unlike the guard, which runs from
    inside a specific asset's partition loop with that asset's own universe
    already resolved. Falls back to a self-contained distinct-EQUITY-ticker
    count (`_cross_section_present_count_unscoped` /
    `_cross_section_trailing_coverage_unscoped`, the same
    `resolution='daily' AND security_type='equity'` filter
    `tradable_universe_size` uses) for `score_date` and the trailing trading
    sessions, using the trailing MEDIAN count as the reference denominator for
    the human-readable message (not just the most recent trailing session,
    which would make that one session read as a tautological 100%) - the
    decision itself only depends on the RELATIVE drop from the trailing
    median, which is invariant to that choice of denominator.

    Second, and deliberately (Meridian#413): the POPULATION counted by that
    fallback additionally requires a non-NULL `ret_1d`, where
    `cross_section_ready`'s does not. A bar lands the instant LDP's price
    ingest writes the row; `ret_1d` is computed by a separate, LATER pass
    (`ohlcv_returns`) that can still be running after the very last bar for
    the date has landed - bars for the whole universe can be single-dated
    and present while a tail of tickers still has a NULL `ret_1d`. Bar presence is
    not evidence the date is scoreable; `ret_1d` presence is, because it is
    the input the price-return-based assets actually consume. A sensor deciding
    whether to launch a run has no per-asset visibility into which of the
    assets need `ret_1d` and which don't - unlike the write-side guard, which
    runs inside one specific asset and, for SOME of those assets, can rely on a
    per-asset backstop (see `cross_section_ready`'s docstring for how much
    that backstop catches; it is uneven, and for some assets there is none).
    Because that
    per-asset coverage is uneven, THIS predicate must gate on the STRICTEST
    requirement across the job it is about to launch, or it launches a run
    that is certain to partially fail regardless of what any one asset can
    catch on its own. This is why the two predicates now count different
    populations despite sharing one decision function: `cross_section_ready`'s
    docstring argues the other half of this split (why the write-side guard
    stays bar-only) and is explicit about which callers that leaves exposed.

    The concrete fix for Meridian#395: `stale_factor_sensor` and
    `ldp_ohlcv_upstream_sensor` previously treated "no watermark row yet" as
    "proceed" (the correct posture when the watermark subsystem is genuinely
    unreachable - see `stale_factor_sensor`'s own docstring on why it must
    stay a backstop for exactly that outage). But LDP writes
    `ohlcv.ingest_watermark`'s row for a date once ingest for that date
    finishes, not while it is in progress - so during the nightly ingest
    window there is no row yet EITHER WAY, so "unknown" can mean "still
    ingesting" just as easily as "watermark subsystem down". The two
    sensors could fire straight into `assert_cross_section_single_dated`'s guard
    whenever the ingest window ran past their post-close gate
    (a fixed-clock delay cannot absorb a variable finish time, which is
    why this is a readiness-predicate fix rather than a bigger timer offset).
    Consulting this predicate before firing distinguishes the two cases: a
    genuinely mid-ingest cross-section fails the SAME coverage-ratio check the
    guard would apply (queried directly against `ohlcv.prices`, which is
    unaffected by whatever is wrong with the watermark table), so the sensor
    defers instead of launching a run that is certain to fail; a watermark
    outage on an evening where ingest actually finished normally still reads
    as ready.
    """
    watermark_status = _watermark_status_for_date(engine, score_date)
    if watermark_status is False:
        return False, (
            f"LDP's ingest watermark (ohlcv.ingest_watermark) marks {score_date} NOT complete"
        )
    if watermark_status is True:
        return True, ""

    trailing = _cross_section_trailing_coverage_unscoped(
        engine,
        score_date,
        _SINGLE_DATED_LOOKBACK_CALENDAR_DAYS,
        _SINGLE_DATED_LOOKBACK_TRADING_DAYS,
    )
    if len(trailing) < _SINGLE_DATED_MIN_TRAILING_DAYS:
        return True, ""

    # Reference denominator for the human-readable message: the trailing
    # MEDIAN count, not just the single most recent trailing session - using
    # trailing[0] alone would make that session read as a tautological 100%
    # in the message below. The decision itself (cross_section_completeness_ok)
    # only depends on the RELATIVE drop from the trailing median regardless of
    # which denominator is used here, so this choice affects only wording.
    n_reference = round(statistics.median(n for _, n in trailing))
    if n_reference < _SINGLE_DATED_MIN_UNIVERSE:
        return True, ""

    current = _cross_section_present_count_unscoped(engine, score_date)
    trailing_fractions = [n / n_reference for _, n in trailing]
    current_fraction = current / n_reference

    ok, msg = cross_section_completeness_ok(current_fraction, trailing_fractions)
    if ok:
        return True, ""

    prior_date, prior_n = trailing[0]  # most recent trailing trading session
    return False, (
        f"cross-section for {score_date} is not ready: "
        f"{current}/{n_reference} tickers ({current_fraction:.1%}) have a NON-NULL ret_1d "
        f"dated {score_date}, vs {prior_n}/{n_reference} ({prior_n / n_reference:.1%}) dated "
        f"{prior_date} (the most recent trailing trading session) and a "
        f"{len(trailing)}-session trailing median of "
        f"{statistics.median(trailing_fractions):.1%}. {msg}. This is either LDP's nightly "
        f"ingest still mid-rewrite, or bars present with ret_1d not yet computed by the "
        f"downstream ohlcv_returns pass (Meridian#413)."
    )


# ─────────────────────────────────────────────
# Universe-anchored coverage gate (see SCOPE)
# ─────────────────────────────────────────────
#
# WHAT THIS GATE CATCHES: the scored cross-section fell below the tradable
# tape THAT WAS AVAILABLE at scoring time. Factor scores are cross-sectional
# z-scores and composite quintiles are a qcut over whatever cross-section is
# present, so a cross-section missing a large slice of the tape does not
# merely omit those tickers, it makes the values written for the tickers that
# ARE present wrong too.
#
# WHAT THIS GATE DOES NOT CATCH: the tape ITSELF being short. A short tape
# shrinks the denominator and pushes the ratio ABOVE 1.0, the regime documented
# below as normal and deliberately not bounded, so the gate stays silent. A
# passing gate is therefore not evidence that the universe is whole.
#
# THE GENERAL PRINCIPLE: you cannot detect a shrinking reference by measuring
# against that reference. `_composite_group_trailing_medians` fails this way
# against factors.group_scores' own history, and this gate fails the same way
# when `ohlcv.prices` is the thing that shrank. The missing control is an
# ABSOLUTE one: an expectation for the universe size that is NOT derived from
# any artifact the ingest pipeline writes (a floor on manifest size, or a
# cross-check against an independent constituent source). That belongs with
# whoever owns the universe contract and is deliberately not bolted on here.
#
# WHY THE TRAILING-BASELINE GATES ARE THE WRONG SHAPE ON THEIR OWN:
# `_composite_group_trailing_medians` derives its baseline from
# `factors.group_scores` itself over the trailing 20 trading days. A
# SELF-REFERENTIAL baseline is structurally blind to a SUSTAINED shift: once
# the trailing median IS the new level the ratio returns to ~1.0 and the gate
# goes quiet. It can only ever see the edge of a step, never the plateau.
#
# A threshold re-tuned against the period in which it fired is a threshold
# switched off. FUTURE MAINTAINERS: if this gate fires, find out why
# coverage collapsed. Do NOT re-calibrate against the firing window.
#
# WHY A FLOOR AND NOT AN EQUALITY CHECK: the scored set is not a strict
# subset of the names carrying a bar on that exact date, and the ratio
# legitimately exceeds 1.0. Scoring is driven by presence in the factor
# inputs, which carry forward across a missing or stale bar, while the
# reference counts only names with a bar on the date itself. Only the FLOOR
# direction carries information.
#
# THRESHOLDS: the default floor is illustrative and meant to be tuned per
# deployment against the false-positive rate on known-good sessions. It is
# chosen to stay quiet in normal operation; its sensitivity to a real
# collapse is a property of the chosen value, so validate it against your own
# history before relying on it. At RUNTIME the denominator is the tape as it
# stands that evening, which can be smaller than a later backfilled table, so
# the runtime ratio is only ever higher (the gate is biased toward silence).
#
# Posture matches the sibling gates: this RAISES rather than warns (refuse
# to write bad cross-sectional output, do not annotate it after the fact),
# and a missing or zero reference passes silently - "no reference to compare
# against" is not evidence of a defect, the same stance
# `composite_group_completeness_ok` takes when there is no trailing
# baseline.
#
# TWO FLOORS: the group_scores cross-section and the composite cross-section
# are not the same size. `composite_scores` blends a narrower set (tickers
# whose WEIGHTED coverage is zero are dropped before the write), so it gets
# its own, independently chosen floor rather than sharing this constant.
_UNIVERSE_COVERAGE_MIN_FRACTION = 0.5


def universe_coverage_ok(
    scored_tickers: int, tradable_tickers: int, min_fraction: float = 0.5
) -> tuple[bool, str]:
    """Pure decision (DB-free, unit-testable): is the scored cross-section
    for a partition large enough relative to the EXTERNAL tradable universe
    (`ohlcv.prices` daily equity bars for that date) to be a credible
    cross-section, or has scoring silently dropped a large slice of the
    actively-trading tape?

    Deliberately anchored outside `factors.*` - see the module comment above
    `_UNIVERSE_COVERAGE_MIN_FRACTION` for why a trailing-history baseline
    (`_composite_group_trailing_medians`) is structurally unable to see a
    sustained coverage break, and for the measurement behind the default
    threshold. Do not re-tune this against a period in which it fired.

    SCOPE, read the module comment before citing this anywhere: this catches
    "we scored less than the tape that was AVAILABLE". It does NOT catch
    "the tape itself was short" - a short tape shrinks the denominator and
    pushes the ratio ABOVE 1.0, where this gate is silent by design. The
    missing control for that is an absolute universe-size
    expectation not derived from any ingest-written artifact; it is
    deliberately not implemented here.

    Args:
        scored_tickers: distinct tickers in this partition's scored
            cross-section.
        tradable_tickers: distinct tickers in the external reference
            universe for the same date. 0 means "no reference available".
        min_fraction: floor on scored/tradable.

    Returns:
        (True, "") when there is no reference to compare against, or
        coverage is at or above the floor - INCLUDING ratios above 1.0,
        which are normal (factor inputs carry forward across a missing bar,
        so the scored set is not a subset of names with a bar on that exact
        date). Only the floor direction is checked.
        (False, reason) naming both counts and the ratio otherwise.
    """
    if tradable_tickers <= 0:
        return True, ""  # no external reference for this date - do not block
    if scored_tickers < 0:
        return True, ""
    frac = scored_tickers / tradable_tickers
    if frac < min_fraction:
        return (
            False,
            f"scored {scored_tickers} tickers against {tradable_tickers} "
            f"tradable (ohlcv.prices daily equity bars) = {frac:.1%} "
            f"coverage; floor={min_fraction:.0%}",
        )
    return True, ""


def tradable_universe_size(engine, score_date: date) -> int:
    """Distinct tickers with a daily equity bar in `ohlcv.prices` on
    `score_date` - the external reference the coverage gate is anchored to.

    Predicate determined by inspecting the data, not assumed:
    `ohlcv.prices` mixes resolutions and security types on the same date
    (daily/equity, hourly/equity, daily/etf and daily/index rows all coexist).
    Only `resolution='daily' AND security_type='equity'` yields the tradable
    equity universe, and it is the same segregation
    the scoring path itself uses (docs/06-CRITICAL_RULES.md, LDP `security_type` column).

    `ts_event` is a timestamptz stamped at the session close (16:00Z), so
    the bound is a half-open range on the date rather than a `::date` cast -
    sargable against `idx_ohlcv_ts_res` / `idx_ohlcv_ticker_res_ts`.

    Returns 0 when the reference has no rows for the date (the tape has not
    landed yet, or a non-trading day). Callers treat 0 as "no reference",
    not as a failure.
    """
    with engine.connect() as conn:
        n = conn.execute(
            text("""
                SELECT count(DISTINCT ticker)
                FROM ohlcv.prices
                WHERE ts_event >= :score_date
                  AND ts_event < :next_day
                  AND resolution = 'daily'
                  AND security_type = 'equity'
            """),
            {"score_date": score_date, "next_day": score_date + timedelta(days=1)},
        ).scalar()
    return int(n or 0)


def assert_universe_coverage(
    engine,
    score_date: date,
    scored_tickers: int,
    min_fraction: "float | None" = None,
    target: str = "factors.group_scores",
) -> None:
    """Refuse to persist a partition whose scored cross-section covers too
    little of the external tradable universe (Meridian#307 - see the module
    comment above `_UNIVERSE_COVERAGE_MIN_FRACTION` for why the baseline is
    external rather than trailing).

    This is a FLOOR, not an equality check: the scored set is not a strict
    subset of names with a bar on the exact date and ratios above 1.0 are
    normal. A SHORT TAPE lands in that above-1.0 regime and is therefore out
    of scope - see `universe_coverage_ok`'s SCOPE note and the module
    comment.

    Two callers, two floors, deliberately - `factor_group_scores`
    (group_scores_asset.py) at the default, and `composite_scores`
    (assets.py) at `_COMPOSITE_UNIVERSE_COVERAGE_MIN_FRACTION`, because the
    composite blends a systematically narrower cross-section than
    group_scores does. See the two-floor table in the module comment.

    Args:
        engine: SQLAlchemy engine.
        score_date: the partition being scored.
        scored_tickers: distinct tickers in this partition's cross-section,
            counted as the caller's own written quantity (see each call
            site - they are NOT the same number).
        min_fraction: floor override; None uses
            `_UNIVERSE_COVERAGE_MIN_FRACTION`.
        target: the table this refusal protects, named in the message.

    Raises:
        ValueError naming the scored count, the tradable count and the ratio.
    """
    if min_fraction is None:
        min_fraction = _UNIVERSE_COVERAGE_MIN_FRACTION
    tradable = tradable_universe_size(engine, score_date)
    ok, msg = universe_coverage_ok(scored_tickers, tradable, min_fraction)
    if ok:
        return

    raise ValueError(
        f"Refusing to write {target} for {score_date}: "
        f"universe coverage gate failed - {msg}. Factor scores are "
        f"cross-sectional z-scores and composite quintiles are a qcut over "
        f"the present cross-section, so a cross-section missing a large "
        f"slice of the actively-trading tape does not merely omit those "
        f"tickers, it makes the values written for the tickers that ARE "
        f"present wrong too. This gate is anchored to ohlcv.prices "
        f"deliberately, because the trailing-history gates re-baseline onto "
        f"a sustained break and go quiet (see the module comment above "
        f"_UNIVERSE_COVERAGE_MIN_FRACTION). Find out why the scored "
        f"universe collapsed before re-running this partition; do NOT "
        f"re-tune the threshold against the broken period. Note this gate "
        f"compares against the tape AS INGESTED - if the tape itself is "
        f"short (an upstream universe-manifest defect) the "
        f"ratio goes ABOVE 1.0 and this gate stays silent, so a passing "
        f"gate is not evidence the universe is whole."
    )


# ─────────────────────────────────────────────
# Watermark-driven scoring trigger (Meridian #286: cron -> dependency cutover)
# ─────────────────────────────────────────────
#
# daily_factor_scoring used to fire on a fixed wall-clock cron, sized against
# an expected upstream finish time. If the upstream ingest finishes later than
# that constant, the schedule fires before ohlcv.ingest_watermark marks the
# date complete - and assert_cross_section_single_dated (correctly) refuses
# to score it. This is a wall-clock-constant failure class; the fix is to
# make the downstream cascade dependency-driven instead of clock-scheduled.
# sensors.py's
# watermark_complete_sensor is the Meridian equivalent: it polls
# ohlcv.ingest_watermark itself and fires the moment a date is actually
# marked complete, so there is no constant left to go stale. These two
# functions are its DB-facing half.


def latest_complete_trade_date(engine) -> Optional[date]:
    """Most recent `trade_date` marked complete in LDP's `ohlcv.ingest_watermark`.

    Returns None when the table has no complete rows yet - either because it
    is empty (LDP hasn't started populating it, or nothing has cleared its
    completion threshold today) or genuinely has no rows at all. Both are
    legitimate "nothing to report" states, not errors: the watermark went
    from zero rows to live population within the current week, so an empty
    table is an expected transient state callers must handle quietly, not a
    failure. Connection/relation errors are NOT caught here - they propagate
    so a caller can tell "checked, nothing complete" (None, no exception)
    apart from "could not check" (exception). Callers that need that
    distinction should wrap the call themselves (see
    sensors.watermark_complete_sensor)."""
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT max(trade_date) FROM ohlcv.ingest_watermark WHERE is_complete")
        ).fetchone()
    return row[0] if row and row[0] is not None else None


def get_new_complete_trade_dates(engine, since: date, limit: int = 30) -> list[date]:
    """`trade_date`s marked complete in `ohlcv.ingest_watermark` strictly
    after `since`, ascending, capped at `limit` rows per call.

    Returns [] when there simply are none yet - not an error, and
    indistinguishable in return shape from "watermark table is empty"; both
    mean "nothing new to fire". Connection/relation errors propagate (same
    contract as `latest_complete_trade_date`) so the caller can log the
    unreachable case distinctly rather than silently treating it as "no new
    dates". `since` is an exclusive lower bound so a caller tracking its own
    high-water mark never re-sees a date it already fired for."""
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT trade_date FROM ohlcv.ingest_watermark "
                "WHERE is_complete AND trade_date > :since "
                "ORDER BY trade_date ASC LIMIT :limit"
            ),
            {"since": since, "limit": limit},
        ).fetchall()
    return [r[0] for r in rows]


# ─────────────────────────────────────────────
# Schema bootstrap
# ─────────────────────────────────────────────

# Deadlock-safe DDL bootstrap.
#
# Every daily asset calls ensure_schema(engine) at the start of its run
# (see ensure_schema below), which runs idempotent CREATE TABLE IF NOT
# EXISTS / ALTER TABLE ... ADD COLUMN IF NOT EXISTS DDL. Postgres takes an
# AccessExclusiveLock for the duration of an ALTER TABLE statement even
# when the column already exists (the "IF NOT EXISTS" only short-circuits
# the actual add, not the lock acquisition). Under N-way concurrent
# backfill runs, several run pods calling ensure_schema at once can grab
# AccessExclusiveLocks on different tables in different orders within the
# same multi-statement DDL block and deadlock
# (psycopg2.errors.DeadlockDetected, pgcode 40P01).
#
# Two layers of defense:
#   1. _ddl_bootstrap_needed() - a cheap read-only existence check (no
#      locks that conflict with concurrent DDL) that lets callers skip the
#      DDL entirely once the schema is already current. This is the
#      steady-state fast path: after the first successful migration
#      anywhere, every subsequent ensure_* call in every concurrent run
#      pod is a couple of SELECTs against information_schema, never an
#      AccessExclusiveLock.
#   2. _execute_ddl_with_deadlock_retry() - belt-and-suspenders for the
#      unavoidable case where two runs race the *first* migration at once:
#      catches DeadlockDetected and retries with jittered backoff instead
#      of failing the asset outright.
_DEADLOCK_MAX_RETRIES = 4
_DEADLOCK_BASE_DELAY_SECONDS = 0.5


def _is_deadlock_error(exc: Exception) -> bool:
    """True if exc (or its DBAPI cause) is a Postgres deadlock (40P01)."""
    orig = getattr(exc, "orig", None)
    pgcode = getattr(orig, "pgcode", None)
    if pgcode == "40P01":
        return True
    return "deadlock detected" in str(exc).lower()


def _execute_ddl_with_deadlock_retry(engine, ddl: str) -> None:
    """Run `ddl` in a transaction, retrying on Postgres deadlock (40P01).

    Concurrent DDL bootstrap calls can legitimately deadlock each other
    (see module-level comment above `_DEADLOCK_MAX_RETRIES`). A deadlock
    is Postgres's normal way of breaking a lock cycle - one of the two
    transactions is rolled back and is safe to simply retry.
    """
    attempt = 0
    while True:
        try:
            with engine.begin() as conn:
                conn.execute(text(ddl))
            return
        except OperationalError as exc:
            attempt += 1
            if not _is_deadlock_error(exc) or attempt > _DEADLOCK_MAX_RETRIES:
                raise
            delay = _DEADLOCK_BASE_DELAY_SECONDS * (2 ** (attempt - 1))
            delay += random.uniform(0, delay * 0.25)
            _log.warning(
                "ensure_schema DDL hit a deadlock (attempt %d/%d), retrying in %.2fs",
                attempt,
                _DEADLOCK_MAX_RETRIES,
                delay,
            )
            time.sleep(delay)


def _table_and_columns_exist(engine, table: str, required_columns: frozenset) -> bool:
    """Read-only check: does `table` exist and carry every column in
    `required_columns`? Used as the fast-path gate so steady-state
    ensure_* calls can skip DDL (and its AccessExclusiveLock) entirely.
    """
    schema_name, _, table_name = table.partition(".")
    with engine.connect() as conn:
        reg = conn.execute(text("SELECT to_regclass(:t)"), {"t": table}).scalar()
        if reg is None:
            return False
        if not required_columns:
            return True
        rows = conn.execute(
            text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema = :s AND table_name = :t"
            ),
            {"s": schema_name, "t": table_name},
        ).fetchall()
    existing = {r[0] for r in rows}
    return required_columns.issubset(existing)


def ensure_schema(engine) -> None:
    """Creates the factors schema and tables if they don't exist."""
    from pathlib import Path

    sql_path = Path(__file__).parent.parent / "db" / "migrations" / "002_factors_schema.sql"
    if sql_path.exists():
        _execute_ddl_with_deadlock_retry(engine, sql_path.read_text())
    # Bootstrap write-campaign lineage (Meridian#294). Forward-declared below;
    # never raises, because a lineage bootstrap failure must not stop scoring.
    ensure_write_campaigns_table(engine)
    # Additive n_groups (blend width) on composite_scores (#329). Forward-declared
    # below; never raises - an additive column bootstrap must not stop scoring.
    ensure_composite_ngroups_column(engine)
    ensure_composite_blend_provenance_column(engine)


# ─────────────────────────────────────────────
# Write-campaign lineage (Meridian #294 / #295)
# ─────────────────────────────────────────────
#
# See meridian_pipeline/lineage.py for the model and
# dagster/db/migrations/2026-08-02_write_campaigns.sql for the schema rationale.
# This half is the DB surface: the campaign table, the insert/close helpers, and
# the capability probe every stamped writer consults.
#
# THE INVARIANT THAT GOVERNS EVERY LINE BELOW: lineage is additive and must
# never fail a write. Two independent guards enforce it.
#
#   1. `_campaign_column_available(engine, table)` - a cached, read-only probe.
#      Until the migration is applied the probe is False and every writer emits
#      BYTE-IDENTICAL SQL to its pre-#294 form. This is what lets the migration
#      ship unapplied and the code deploy in either order. Only a SUCCESSFUL
#      probe is cached; a probe ERROR is logged and re-raised (Meridian#432: the
#      daily/group writers delete and re-insert, so a swallowed error would
#      re-insert every row with campaign_id NULL and wipe its lineage).
#   2. Every helper here except the probe swallows its own exceptions and
#      returns None. A NULL campaign_id is a legitimate, expected state (lineage
#      disabled, column not yet present); it is never an error condition for the
#      caller. The probe is the deliberate exception: resolve_campaign_id calls
#      it, so every stamped writer (daily/group/composite scores, factor and
#      specific returns, covariance, specific risk) now fails fast on a probe
#      error instead of writing unstamped rows.

#: Tables that carry a campaign_id stamp. Kept in lockstep with the migration.
CAMPAIGN_STAMPED_TABLES = (
    "factors.daily_scores",
    "factors.group_scores",
    "factors.composite_scores",
    "factors.factor_returns",
    "factors.specific_returns",
    "factors.specific_risk",
    "factors.factor_covariance",
)

#: Sentinel for rows written before lineage existed. Mirrors
#: lineage.PRE_LINEAGE_CAMPAIGN_ID (duplicated as a literal so db.py has no
#: import-time dependency on lineage.py - the two modules import each other
#: lazily).
PRE_LINEAGE_CAMPAIGN_ID = 0

_WRITE_CAMPAIGNS_DDL = """
CREATE TABLE IF NOT EXISTS factors.write_campaigns (
    campaign_id       bigint      GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    writer            text        NOT NULL,
    code_version      text,
    run_id            text,
    input_vintage     jsonb       NOT NULL DEFAULT '{}'::jsonb,
    input_vintage_max timestamptz,
    started_at        timestamptz NOT NULL DEFAULT now(),
    finished_at       timestamptz,
    gate_result       text,
    detail            jsonb       NOT NULL DEFAULT '{}'::jsonb
);
CREATE INDEX IF NOT EXISTS idx_write_campaigns_writer_started
    ON factors.write_campaigns (writer, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_write_campaigns_vintage_max
    ON factors.write_campaigns (input_vintage_max);
CREATE INDEX IF NOT EXISTS idx_write_campaigns_run
    ON factors.write_campaigns (run_id) WHERE run_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_write_campaigns_vintage
    ON factors.write_campaigns USING gin (input_vintage);
INSERT INTO factors.write_campaigns
    (campaign_id, writer, input_vintage, started_at, finished_at, gate_result, detail)
VALUES (0, 'pre_lineage', '{}'::jsonb, now(), now(), 'unverified',
        jsonb_build_object('campaign_origin', 'sentinel',
                           'note', 'Rows written before write-campaign lineage existed '
                                   '(Meridian #294). Provenance not recoverable.'))
ON CONFLICT (campaign_id) DO NOTHING;
"""

# `ADD COLUMN ... DEFAULT 0` then `DROP DEFAULT` (issued per table in
# ensure_write_campaigns_table below) is the whole trick: the constant default
# is stored in pg_attribute.attmissingval (metadata-only since PG 11) so the
# existing rows read the sentinel without a rewrite, and dropping the default
# afterwards leaves attmissingval intact while making a NEW unstamped row NULL
# - so a post-migration write with no campaign is visibly a bug instead of
# blending into history. Filenode and relation size are unchanged.
_campaign_column_cache: "dict[str, bool]" = {}


def ensure_write_campaigns_table(engine) -> None:
    """Idempotent DDL for factors.write_campaigns + the campaign_id columns.

    Mirrors dagster/db/migrations/2026-08-02_write_campaigns.sql. Uses the same
    read-only fast-path gate as every other ensure_* helper so the steady state
    is a couple of information_schema SELECTs rather than an AccessExclusiveLock
    on a very large table (see the _DEADLOCK_MAX_RETRIES comment above).

    NEVER RAISES. A lineage bootstrap failure must not stop factor scoring, so a
    failure here just leaves the capability probe False and every writer emits
    its pre-#294 SQL.
    """
    try:
        needed = frozenset({"campaign_id"})
        if _table_and_columns_exist(engine, "factors.write_campaigns", frozenset()) and all(
            _table_and_columns_exist(engine, t, needed) for t in CAMPAIGN_STAMPED_TABLES
        ):
            return
        _execute_ddl_with_deadlock_retry(engine, _WRITE_CAMPAIGNS_DDL)
        # Column adds are attempted per table so one missing/renamed table
        # (a fresh DB mid-bootstrap) cannot block the rest.
        for table in CAMPAIGN_STAMPED_TABLES:
            if _table_and_columns_exist(engine, table, needed):
                continue
            try:
                _execute_ddl_with_deadlock_retry(
                    engine,
                    f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS campaign_id bigint DEFAULT 0;\n"
                    f"ALTER TABLE {table} ALTER COLUMN campaign_id DROP DEFAULT;",
                )
            except Exception as exc:
                _log.warning("lineage: could not add campaign_id to %s: %s", table, exc)
    except Exception as exc:
        _log.warning("lineage: write_campaigns bootstrap skipped: %s", exc)
    finally:
        _campaign_column_cache.clear()


# #329: additive n_groups (per-ticker composite blend width) on composite_scores.
# Same additive-and-never-fail discipline as the campaign_id column above: a
# read-only capability probe gates the writer, so the code deploys and the
# migration self-applies in either order (the column is absent -> writer emits
# pre-#329 SQL; present -> it writes n_groups). smallint is ample (groups <= ~20).
_ngroups_column_cache: "dict[str, bool]" = {}


def ensure_composite_ngroups_column(engine) -> None:
    """Idempotent, probe-gated ADD COLUMN for factors.composite_scores.n_groups.
    NEVER RAISES - an additive column bootstrap must not stop factor scoring."""
    try:
        if _table_and_columns_exist(engine, "factors.composite_scores", frozenset({"n_groups"})):
            return
        _execute_ddl_with_deadlock_retry(
            engine,
            "ALTER TABLE factors.composite_scores ADD COLUMN IF NOT EXISTS n_groups smallint;",
        )
    except Exception as exc:
        _log.warning("could not add n_groups to composite_scores: %s", exc)
    finally:
        _ngroups_column_cache.clear()


def ensure_composite_blend_provenance_column(engine) -> None:
    """Add calculation provenance without blocking the existing score writer."""
    try:
        if not _table_and_columns_exist(
            engine, "factors.composite_scores", frozenset({"blend_provenance"})
        ):
            _execute_ddl_with_deadlock_retry(
                engine,
                "ALTER TABLE factors.composite_scores "
                "ADD COLUMN IF NOT EXISTS blend_provenance jsonb;",
            )
    except Exception as exc:
        _log.warning("could not add composite blend provenance: %s", exc)


def _composite_has_blend_provenance(engine) -> bool:
    # Do not cache absence: migration may be applied while this process lives.
    # A probe ERROR re-raises (Meridian#432): the composite writer deletes then
    # re-inserts, so a False from a failed probe would drop blend_provenance.
    try:
        return _table_and_columns_exist(
            engine, "factors.composite_scores", frozenset({"blend_provenance"})
        )
    except Exception as exc:
        _log.warning("composite: blend_provenance probe failed (re-raising): %s", exc)
        raise


def _composite_has_ngroups(engine) -> bool:
    """Cached read-only probe: does factors.composite_scores carry n_groups?
    A successful absent answer falls back to pre-#329 SQL. A probe ERROR is
    logged and re-raised, never cached as False (Meridian#432: the composite
    writer deletes then re-inserts, so a False would drop n_groups)."""
    if "composite_scores" not in _ngroups_column_cache:
        try:
            _ngroups_column_cache["composite_scores"] = _table_and_columns_exist(
                engine, "factors.composite_scores", frozenset({"n_groups"})
            )
        except Exception as exc:
            _log.warning("composite: n_groups probe failed (not cached, re-raising): %s", exc)
            raise
    return _ngroups_column_cache["composite_scores"]


def _campaign_column_available(engine, table: str) -> bool:
    """Cached: does `table` carry a campaign_id column on this database?

    A successful probe that finds the column absent returns False, which makes
    the calling writer emit exactly the SQL it emitted before lineage existed -
    which is how the migration can ship unapplied without breaking a single
    write. Cached per (engine URL, table) so this costs at most two
    information_schema SELECTs per process.

    A probe ERROR is NOT answered with False (Meridian#432): the daily/group
    writers delete and re-insert, so a False built from a transient failure
    would re-insert every row with campaign_id NULL and silently wipe its
    lineage. The error is logged at WARNING and re-raised, and nothing is
    cached, so the next call probes again.
    """
    key = f"{getattr(engine, 'url', None) or id(engine)}|{table}"
    cached = _campaign_column_cache.get(key)
    if cached is not None:
        return cached
    try:
        present = bool(_table_and_columns_exist(engine, table, frozenset({"campaign_id"})))
    except Exception as exc:
        _log.warning(
            "lineage: campaign_id probe failed for %s (not cached, re-raising): %s", table, exc
        )
        raise
    _campaign_column_cache[key] = present
    return present


def resolve_campaign_id(engine, campaign_id: "int | None", table: str) -> "int | None":
    """The campaign id a writer should stamp: the explicit one, else the ambient
    or implicit one from lineage.py, else None. Returns None whenever the target
    table has no campaign_id column, so the caller can branch on a single value.
    """
    if not _campaign_column_available(engine, table):
        return None
    if campaign_id is not None:
        return campaign_id
    try:
        from meridian_pipeline import lineage

        return lineage.current_campaign_id(engine)
    except Exception as exc:  # pragma: no cover - defensive
        _log.debug("lineage: campaign resolution degraded to None: %s", exc)
        return None


def _campaign_sql_fragments(cid: "int | None") -> "tuple[str, str, str]":
    """(column-list, bind-list, ON CONFLICT SET) fragments for a stamped INSERT.

    Empty strings when cid is None, so the composed statement is character-for
    -character the pre-#294 one. That exact identity is what the parity test
    asserts - a stamped write and an unstamped write must differ in lineage
    columns only, never in a factor VALUE.
    """
    if cid is None:
        return "", "", ""
    return (
        ", campaign_id",
        ", :campaign_id",
        ",\n                              campaign_id = EXCLUDED.campaign_id",
    )


def _replace_derived_surface_input_campaigns(
    conn,
    surface_name: str,
    output_campaign_id: "int | None",
    output_dates: list,
    input_campaign_rows: pd.DataFrame | None,
) -> None:
    """Atomically replace one output campaign's exact per-date factor inputs.

    A missing or malformed source stamp leaves the affected date without a
    manifest after deleting that output campaign/date's older one. That is
    intentionally a consumer denial, never a partial manifest that could omit
    an unverified input. A writer without a positive output campaign ID cannot
    safely publish a manifest at all.
    """
    if output_campaign_id is None or output_campaign_id <= PRE_LINEAGE_CAMPAIGN_ID:
        return
    for score_date in output_dates:
        conn.execute(
            text("""
                DELETE FROM factors.derived_surface_input_campaigns
                WHERE surface_name = :surface_name
                  AND output_campaign_id = :output_campaign_id
                  AND score_date = :score_date
            """),
            {
                "surface_name": surface_name,
                "output_campaign_id": output_campaign_id,
                "score_date": score_date,
            },
        )
    if input_campaign_rows is None:
        return
    required = {"score_date", "factor_name", "campaign_id"}
    if not required.issubset(input_campaign_rows.columns):
        return
    manifests: list[dict] = []
    for score_date in output_dates:
        source = input_campaign_rows.loc[input_campaign_rows["score_date"] == score_date]
        if source.empty:
            continue
        pairs: set[tuple[str, int]] = set()
        valid = True
        for factor_name, campaign_id in source[["factor_name", "campaign_id"]].itertuples(
            index=False, name=None
        ):
            if not isinstance(factor_name, str) or not factor_name or isinstance(campaign_id, bool):
                valid = False
                break
            try:
                campaign = int(campaign_id)
            except (TypeError, ValueError):
                valid = False
                break
            if campaign <= PRE_LINEAGE_CAMPAIGN_ID:
                valid = False
                break
            pairs.add((factor_name, campaign))
        if valid:
            manifests.extend(
                {
                    "surface_name": surface_name,
                    "output_campaign_id": output_campaign_id,
                    "score_date": score_date,
                    "factor_name": factor_name,
                    "campaign_id": campaign_id,
                }
                for factor_name, campaign_id in sorted(pairs)
            )
    if manifests:
        conn.execute(
            text("""
                INSERT INTO factors.derived_surface_input_campaigns
                    (surface_name, output_campaign_id, score_date, factor_name, campaign_id)
                VALUES (:surface_name, :output_campaign_id, :score_date, :factor_name, :campaign_id)
                ON CONFLICT (surface_name, output_campaign_id, score_date, factor_name, campaign_id)
                DO NOTHING
            """),
            manifests,
        )


def record_write_campaign(
    engine,
    writer: str,
    code_version: "str | None" = None,
    run_id: "str | None" = None,
    input_vintage: "dict | None" = None,
    started_at=None,
    gate_result: "str | None" = None,
    detail: "dict | None" = None,
) -> "int | None":
    """Insert one factors.write_campaigns row and return its campaign_id.

    Returns None (never raises) if the table is absent or the insert fails - the
    caller then writes rows with a NULL campaign_id, which is degraded but
    correct. `input_vintage_max` is derived here from `input_vintage` via
    lineage.vintage_scalar so the scalar and the jsonb cannot disagree.
    """
    import json as _json

    try:
        from meridian_pipeline import lineage

        vintage = dict(input_vintage or {})
        vintage_max = lineage.vintage_scalar(vintage)
        with engine.begin() as conn:
            return conn.execute(
                text("""
                INSERT INTO factors.write_campaigns
                    (writer, code_version, run_id, input_vintage, input_vintage_max,
                     started_at, gate_result, detail)
                VALUES (:writer, :code_version, :run_id, CAST(:input_vintage AS jsonb),
                        :input_vintage_max, COALESCE(:started_at, now()), :gate_result,
                        CAST(:detail AS jsonb))
                RETURNING campaign_id
                """),
                {
                    "writer": writer,
                    "code_version": code_version,
                    "run_id": run_id,
                    "input_vintage": _json.dumps(vintage, default=str),
                    "input_vintage_max": vintage_max,
                    "started_at": started_at,
                    "gate_result": gate_result,
                    "detail": _json.dumps(detail or {}, default=str),
                },
            ).scalar()
    except Exception as exc:
        _log.warning("lineage: could not record write campaign (writer=%s): %s", writer, exc)
        return None


def finish_write_campaign(
    engine,
    campaign_id: int,
    gate_result: "str | None" = None,
    input_vintage: "dict | None" = None,
    detail: "dict | None" = None,
    finished_at=None,
) -> None:
    """Close a campaign: stamp finished_at, and refresh the fields that are only
    known once the work is done (the gate verdict, late-discovered vintages).

    jsonb fields are MERGED (`||`), not replaced, so a caller that adds one
    source's vintage at the end does not erase what was recorded at open time.
    Never raises.
    """
    import json as _json

    if campaign_id is None:
        return
    try:
        from meridian_pipeline import lineage

        vintage = dict(input_vintage or {})
        params = {
            "campaign_id": campaign_id,
            "gate_result": gate_result,
            "finished_at": finished_at,
            "input_vintage": _json.dumps(vintage, default=str) if vintage else None,
            "input_vintage_max": lineage.vintage_scalar(vintage) if vintage else None,
            "detail": _json.dumps(detail, default=str) if detail else None,
        }
        with engine.begin() as conn:
            conn.execute(
                text("""
                UPDATE factors.write_campaigns
                   SET finished_at   = COALESCE(:finished_at, now()),
                       gate_result   = COALESCE(:gate_result, gate_result),
                       input_vintage = CASE WHEN :input_vintage IS NULL THEN input_vintage
                                            ELSE input_vintage || CAST(:input_vintage AS jsonb) END,
                       input_vintage_max = COALESCE(:input_vintage_max, input_vintage_max),
                       detail        = CASE WHEN :detail IS NULL THEN detail
                                            ELSE detail || CAST(:detail AS jsonb) END
                 WHERE campaign_id = :campaign_id
                """),
                params,
            )
    except Exception as exc:
        _log.warning("lineage: could not close write campaign %s: %s", campaign_id, exc)


def add_write_campaign_input_vintage(engine, campaign_id: int, input_vintage: dict) -> None:
    """Merge newly discovered input provenance without closing a campaign.

    Derived writers often resolve an implicit output campaign immediately
    before their write.  They need to attach their exact input-campaign
    manifest then, not call ``finish_write_campaign`` early and make a still
    running campaign look complete.  Never raises: provenance remains
    additive observability, while a missing manifest is fail-closed downstream.
    """
    import json as _json

    if campaign_id is None or not input_vintage:
        return
    try:
        from meridian_pipeline import lineage

        vintage = dict(input_vintage)
        with engine.begin() as conn:
            conn.execute(
                text("""
                UPDATE factors.write_campaigns
                   SET input_vintage = input_vintage || CAST(:input_vintage AS jsonb),
                       input_vintage_max = CASE
                           WHEN :input_vintage_max IS NULL THEN input_vintage_max
                           WHEN input_vintage_max IS NULL THEN :input_vintage_max
                           ELSE GREATEST(input_vintage_max, :input_vintage_max)
                       END
                 WHERE campaign_id = :campaign_id
                """),
                {
                    "campaign_id": campaign_id,
                    "input_vintage": _json.dumps(vintage, default=str),
                    "input_vintage_max": lineage.vintage_scalar(vintage),
                },
            )
    except Exception as exc:
        _log.warning("lineage: could not add input vintage to campaign %s: %s", campaign_id, exc)


def read_write_campaign(engine, campaign_id: int) -> "dict | None":
    """One campaign row as a dict, or None. Read-only convenience for operators
    and tests ("what wrote this row?" after joining a score row's campaign_id)."""
    try:
        with engine.connect() as conn:
            row = (
                conn.execute(
                    text("SELECT * FROM factors.write_campaigns WHERE campaign_id = :c"),
                    {"c": campaign_id},
                )
                .mappings()
                .first()
            )
        return dict(row) if row else None
    except Exception as exc:
        _log.warning("lineage: could not read campaign %s: %s", campaign_id, exc)
        return None


# ─────────────────────────────────────────────
# Reads
# ─────────────────────────────────────────────


def has_scores_for_date(engine, score_date: date) -> bool:
    """Check if factor scores exist for a given date."""
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT 1 FROM factors.daily_scores WHERE score_date = :d LIMIT 1"),
            {"d": score_date},
        ).fetchone()
    return row is not None


def get_pending_requests(engine) -> list[date]:
    """Returns dates from compute_requests with status='pending'."""
    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT score_date FROM factors.compute_requests
                WHERE status = 'pending'
                ORDER BY score_date
            """)
        ).fetchall()
    return [r[0] for r in rows]


def get_latest_score_date(engine) -> Optional[date]:
    """Returns the most recent date with composite scores."""
    with engine.connect() as conn:
        row = conn.execute(text("SELECT MAX(score_date) FROM factors.composite_scores")).fetchone()
    return row[0] if row and row[0] else None


def is_trading_day(engine, d: date) -> bool:
    """Return True if date d was a trading day (equity OHLCV rows landed).

    Uses a threshold of >100 rows to guard against partial-ingest races
    where a handful of rows land before the daily LDP cascade finishes.

    Sargable half-open range (perf fix, 2026-07 audit: db.py:288):
    `ts_event >= :d AND ts_event < :d_next` instead of `ts_event::date = :d`.
    The cast form is not sargable - PostgreSQL cannot use any btree index on
    ts_event (timestamptz -> date is not an immutable expression the planner
    indexes on), forcing a full sequential scan of ohlcv.prices on every
    call (a full Parallel Seq Scan versus a Bitmap Index Scan for the
    equivalent range form). Called once per
    daily-schedule tick, up to 7x per sensor evaluation, and per-partition in
    every #106 trading-day gate, so this is on the hot path for both daily
    scoring and multi-hundred-partition backfills.
    """
    d_next = d + timedelta(days=1)
    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT COUNT(*) FROM ohlcv.prices "
                "WHERE ts_event >= :d AND ts_event < :d_next AND security_type = 'equity' "
                "AND resolution = 'daily'"
            ),
            {"d": d, "d_next": d_next},
        ).fetchone()
    return (row[0] if row else 0) > 100


def read_daily_scores(engine, score_date: date) -> pd.DataFrame:
    """Read all factor scores for a date. Used by composite_scores asset."""
    return pd.read_sql(
        "SELECT * FROM factors.daily_scores WHERE score_date = %(d)s",
        engine,
        params={"d": score_date},
    )


# ─────────────────────────────────────────────
# Writes
# ─────────────────────────────────────────────


# Chunk size for the executemany-style upserts below. Same idiom as
# `risk_metrics_asset._upsert_risk_metrics`. Daily scoring writes many
# rows per partition across the daily SDA assets; row-by-row inserts
# meant one round-trip per row (sub-ms on a LAN-co-located database, multi-ms
# over a cluster network). Chunking at 1000 rows
# per execute() cuts the round-trips by three orders of magnitude. Same
# `ON CONFLICT DO UPDATE` semantics, so backfill re-runs still overwrite
# cleanly. Chunk size kept under PG's 65535 parameter ceiling for
# the 7-column `daily_scores` row.
_UPSERT_CHUNK = 1000


# ---------------------------------------------------------------------------
# Cold-tier-safe writes (Meridian#432)
#
# db-host is moving old factors.daily_scores months to an Iceberg cold tier (a
# pg_lake foreign-table partition). Once a partition is foreign the PARENT table
# cannot carry a primary key or unique index, so `ON CONFLICT (score_date,
# ticker, factor_name)` stops working for hot AND cold months. The writers below
# therefore never name a conflict target:
#
#   1. take a per-score_date advisory lock (transaction scoped, sorted order);
#   2. key-exact DELETE of the payload's (score_date, ticker, factor_name) keys
#      ... RETURNING the columns the old `DO UPDATE SET` left untouched;
#   3. plain INSERT, carrying those untouched columns forward.
#
# The lock is what replaces the unique index as the duplicate guard: with no
# index, two concurrent writers of the same keys would both delete (finding
# nothing) and both insert. Serialising writers per score_date makes the second
# one's DELETE see the first one's committed rows.
# ---------------------------------------------------------------------------

# 32-bit namespace per table, placed in the high half of the 64-bit advisory key
# with the date ordinal (< 2**20 until year 2872) in the low half. Both
# namespaces are < 2**31, so the key is a positive signed bigint. pg_advisory
# keys share one global space per database: the constants spell "MD" + table id.
_ADVISORY_NS = {
    "factors.daily_scores": 0x4D440001,
    "factors.group_scores": 0x4D440002,
}


# Table-level "range" lock (Meridian#468 FND-007). The per-date keys above only
# cover the dates a writer's payload carries, but replace_date_range deletes a
# whole interval, including interior trading days with no payload rows
# (allow_interior_gaps). A concurrent writer of such a date holds no lock the
# range replace waits on, so its fresh row is silently deleted. One extra key
# per table closes that without locking every calendar day (each advisory lock
# costs a shared lock-table slot):
#
#   * every score writer takes it SHARED (pg_advisory_xact_lock_shared) before
#     its per-date locks, so ordinary writers never block each other on it;
#   * replace_date_range takes it EXCLUSIVE before deleting, so it waits for
#     every in-flight writer and holds off new ones until it commits.
#
# Order is always range key first, then per-date keys ascending, in every
# writer and for either mode, so the lock graph stays acyclic (no deadlock).
# A transaction only ever holds ONE mode of the range key, so there is no
# shared-to-exclusive upgrade. The namespaces differ from _ADVISORY_NS, so a
# range key can never equal a per-date key.
_RANGE_LOCK_NS = {
    "factors.daily_scores": 0x4D440101,
    "factors.group_scores": 0x4D440102,
}


def range_lock_key(table: str) -> int:
    """Stable 64-bit advisory key for a table's range lock (FND-007)."""
    return _RANGE_LOCK_NS[table] << 32


def _as_date(v) -> date:
    """Normalise a score_date (str / Timestamp / datetime / date) to a date."""
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    return pd.Timestamp(v).date()


def score_date_lock_key(table: str, score_date) -> int:
    """Stable 64-bit advisory-lock key for one (table, score_date)."""
    return (_ADVISORY_NS[table] << 32) | _as_date(score_date).toordinal()


# Every advisory lock occupies a slot in the shared lock table (about
# max_locks_per_transaction x max_connections, shared with every
# other session), so a transaction may not take an unbounded number. Writers
# that span more dates than this must be split by the caller (one partition or
# one year per call, which is how every promote already runs).
_MAX_LOCK_DATES = 4000


def _lock_keys(table: str, score_dates) -> "list[int]":
    keys = sorted({score_date_lock_key(table, d) for d in score_dates})
    if len(keys) > _MAX_LOCK_DATES:
        raise ValueError(
            f"{table}: one write spans {len(keys)} distinct score_dates (limit "
            f"{_MAX_LOCK_DATES}); each date takes a pg advisory lock held to commit, so "
            "split the write into smaller date ranges (Meridian#432)."
        )
    return keys


_LOCK_TIMEOUT_RE = re.compile(r"^(\d+)(ms|s|min|h)?$")
_DEFAULT_LOCK_TIMEOUT = "10min"


def _lock_timeout_setting() -> str:
    """lock_timeout for score writes: MERIDIAN_SCORE_WRITE_LOCK_TIMEOUT or 10min.

    Validated against a strict pattern because SET LOCAL cannot take a bind
    parameter. A bare integer means milliseconds (PG semantics). A zero value
    (0, 0s, 0ms, ...) is refused: in PG it DISABLES the timeout, which would let
    a stuck writer block its date forever, so it falls back to the default.
    """
    v = (os.environ.get("MERIDIAN_SCORE_WRITE_LOCK_TIMEOUT") or _DEFAULT_LOCK_TIMEOUT).strip()
    m = _LOCK_TIMEOUT_RE.match(v)
    if not m:
        raise ValueError(f"MERIDIAN_SCORE_WRITE_LOCK_TIMEOUT={v!r} is not <int>[ms|s|min|h]")
    if int(m.group(1)) == 0:
        _log.warning(
            "MERIDIAN_SCORE_WRITE_LOCK_TIMEOUT=%r would disable lock_timeout (zero means no "
            "limit in PG); using the default %s",
            v,
            _DEFAULT_LOCK_TIMEOUT,
        )
        return _DEFAULT_LOCK_TIMEOUT
    return v


def _is_lock_timeout(exc: BaseException) -> bool:
    return getattr(getattr(exc, "orig", exc), "pgcode", None) == "55P03"


def _log_lock_timeout(table: str, score_dates, exc: BaseException, stage: str = "") -> None:
    """`stage` names the lock that timed out: the table range lock or a per-date lock."""
    _log.error(
        "%s: lock_timeout waiting for %s (score_date advisory lock(s) %s) run=%s: %s",
        table,
        stage or "a lock",
        sorted({_as_date(d).isoformat() for d in score_dates}),
        _write_run_id(),
        exc,
    )


def _log_replace_scope(
    op: str, table: str, dates, n_factors: int, payload_rows: int, *, date_range=None
) -> None:
    """INFO preview of a destructive replace-delete, before it runs (QoL 12).

    Names the table, the date span, how many dates/owned factors the delete
    touches, and the row bound the owner should expect: the incoming payload
    rows (what replaces the deleted ones) and the key-space ceiling (dates x
    factors, a per-ticker multiplier away from the worst case). A delete whose
    scope is far wider than the payload shows up in the log before the commit.
    """
    ds = sorted({_as_date(d) for d in dates})
    if date_range is not None:
        lo, hi = date_range
        span_dates = (hi - lo).days + 1
    else:
        lo, hi = (ds[0], ds[-1]) if ds else (None, None)
        span_dates = len(ds)
    _log.info(
        "%s: run=%s about to DELETE table=%s dates=%s..%s span_days=%d payload_dates=%d "
        "owned_factors=%d payload_rows=%d key_space_days_x_factors=%d",
        op,
        _write_run_id(),
        table,
        lo,
        hi,
        span_dates,
        len(ds),
        n_factors,
        payload_rows,
        span_dates * n_factors,
    )


def _lock_score_dates(
    conn, table: str, score_dates, *, range_exclusive: bool = False
) -> "list[int]":
    """Take the table range lock, then pg_advisory_xact_lock per distinct score_date.

    The range lock (see _RANGE_LOCK_NS) is taken first: SHARED for an ordinary
    writer, EXCLUSIVE when `range_exclusive` (replace_date_range). The per-date
    locks follow in ascending key order. A single global order (range key, then
    ascending dates) is the deadlock guard: any two writers acquire their
    (overlapping) locks in the same order, so neither can hold a lock the other
    needs while waiting on one the other holds. Released automatically at
    COMMIT/ROLLBACK, so the lock lives exactly as long as the write
    transaction. Returns the per-date keys taken, in order (for tests / logs).

    `SET LOCAL lock_timeout` runs first (it also bounds advisory-lock waits and
    stays in force for the rest of the transaction), so a stuck writer fails
    loudly with SQLSTATE 55P03 instead of blocking its date forever; the dates
    and run id are logged and the error re-raised.
    """
    keys = _lock_keys(table, score_dates)
    rkey = range_lock_key(table)
    range_fn = "pg_advisory_xact_lock" if range_exclusive else "pg_advisory_xact_lock_shared"
    conn.execute(text(f"SET LOCAL lock_timeout = '{_lock_timeout_setting()}'"))
    stage = "the table range lock"
    try:
        conn.execute(text(f"SELECT {range_fn}(:k)"), {"k": rkey})
        stage = "the per-date advisory locks"
        for k in keys:
            conn.execute(text("SELECT pg_advisory_xact_lock(:k)"), {"k": k})
    except Exception as exc:
        if _is_lock_timeout(exc):
            _log_lock_timeout(table, score_dates, exc, stage)
        raise
    return keys


def _lock_score_dates_cursor(
    cur, table: str, score_dates, *, range_exclusive: bool = False
) -> "list[int]":
    """_lock_score_dates for a raw psycopg2 cursor (the COPY write paths)."""
    keys = _lock_keys(table, score_dates)
    rkey = range_lock_key(table)
    range_fn = "pg_advisory_xact_lock" if range_exclusive else "pg_advisory_xact_lock_shared"
    cur.execute(f"SET LOCAL lock_timeout = '{_lock_timeout_setting()}'")
    stage = "the table range lock"
    try:
        cur.execute(f"SELECT {range_fn}(%s)", (rkey,))
        stage = "the per-date advisory locks"
        for k in keys:
            cur.execute("SELECT pg_advisory_xact_lock(%s)", (k,))
    except Exception as exc:
        if _is_lock_timeout(exc):
            _log_lock_timeout(table, score_dates, exc, stage)
        raise
    return keys


def _rowcount(res) -> int:
    rc = getattr(res, "rowcount", None)
    return rc if isinstance(rc, int) and rc >= 0 else 0


def _insert_rowcount(res, n_rows: int) -> int:
    """The driver's rowcount for an INSERT when it reports one (>= 0, including a
    real 0), else the staged row count. Never `rc or n_rows`: that would report
    n_rows for an INSERT that the driver says wrote nothing."""
    rc = getattr(res, "rowcount", None)
    if isinstance(rc, int) and rc >= 0:
        return rc
    _log.debug("insert rowcount unavailable (%r); reporting the %d staged rows", rc, n_rows)
    return n_rows


def _distinct_dates(values) -> "list[date]":
    """Sorted distinct dates of a score_date column / iterable (any date-like)."""
    return sorted({_as_date(v) for v in values})


def _write_run_id() -> "str | None":
    """Dagster run id when running inside a run pod, else None (scripts)."""
    return os.environ.get("DAGSTER_RUN_ID") or None


def _dedupe_keep_last(rows: "list[dict]", key_cols: "tuple[str, ...]") -> "tuple[list[dict], int]":
    """Collapse repeated keys, last occurrence wins.

    The old chunked INSERT .. ON CONFLICT DO UPDATE ended with the last
    duplicate's values. A unique index used to make a duplicate impossible to
    store; without one a repeated key in the payload would be stored twice, so
    it is collapsed here and the count returned for logging.
    """
    seen: dict = {}
    for r in rows:
        seen[tuple(_as_date(r[c]) if c == "score_date" else r[c] for c in key_cols)] = r
    return list(seen.values()), len(rows) - len(seen)


_KEY_SQL_TYPES = {
    "score_date": "date",
    "ticker": "text",
    "factor_name": "text",
    "group_name": "text",
    "campaign_id": "bigint",
    "computed_at": "timestamptz",
}


def _pg_copy_text(v) -> str:
    """Render one value as a COPY ... FORMAT text field.

    NULL is \\N for None, pandas NA / NaT and any non-finite float (NaN and inf
    must never reach float8; Meridian rule 8). Everything else is str(v) with the
    four characters text mode treats specially escaped (backslash first, then tab,
    newline, carriage return), so a ticker or group name cannot split a row.
    """
    if (
        v is None
        or v is pd.NA
        or v is pd.NaT
        or (isinstance(v, (float, np.floating)) and not np.isfinite(v))
    ):
        return "\\N"
    s = str(v)
    return s.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n").replace("\r", "\\r")


# Integer columns: an INSERT accepted 1.0 for a smallint/bigint (assignment cast),
# a COPY does not, so a whole-valued float is rendered as the integer.
_COPY_INT_COLS = frozenset({"direction", "campaign_id", "quintile"})


def _copy_field(col: str, v) -> str:
    if col == "score_date":
        return _pg_copy_text(_as_date(v).isoformat())
    if col in _COPY_INT_COLS and isinstance(v, (float, np.floating)) and float(v).is_integer():
        v = int(v)
    return _pg_copy_text(v)


def _pg_copy_text_rows(rows: "list[dict]", cols: "list[str]") -> str:
    """Tab-separated, newline-terminated COPY text for `rows`, in `cols` order.
    score_date is normalised to an ISO date (the payload may hold str / Timestamp)."""
    return "".join("\t".join(_copy_field(c, r[c]) for c in cols) + "\n" for r in rows)


_STAGE_COPY_ROWS = 200_000


class _Staged:
    """A payload loaded into a transaction-local temp table (see _stage_rows)."""

    __slots__ = ("table", "stage", "n_rows", "cols", "key_cols", "carry_cols", "order_col", "dates")

    def __init__(self, table, stage, n_rows, cols, key_cols, carry_cols, order_col, dates):
        self.table = table
        self.stage = stage
        self.n_rows = n_rows
        self.cols = cols
        self.key_cols = key_cols
        self.carry_cols = carry_cols
        self.order_col = order_col
        self.dates = dates


def _stage_rows(
    conn,
    table: str,
    rows: "list[dict]",
    *,
    cols: "list[str]",
    key_cols: "tuple[str, ...]",
    carry_cols: "tuple[str, ...]",
    order_col: "str | None" = None,
) -> "_Staged | None":
    """Create the transaction-local stage table and load `rows` into it.

    This is the slow, row-proportional half of a score write and it touches
    nothing but a temp table, so callers run it BEFORE taking the per-score_date
    advisory locks (_lock_score_dates): the lock is then held only for the
    deletes and the one INSERT (Meridian#432). The load is ONE COPY FROM STDIN
    on the same DBAPI connection, hence the same transaction, as `conn`.

    Returns None for an empty payload.
    """
    if not rows:
        return None
    stage = "_m432_stage"
    conn.execute(
        text(f"CREATE TEMP TABLE {stage} (LIKE {table} INCLUDING DEFAULTS) ON COMMIT DROP")
    )
    cur = conn.connection.cursor()
    try:
        # One COPY per _STAGE_COPY_ROWS rows into the same stage: bounds the text
        # buffer (a StringIO costs 4 bytes/char) on multi-million-row promotes.
        for i in range(0, len(rows), _STAGE_COPY_ROWS):
            cur.copy_expert(
                f"COPY {stage} ({', '.join(cols)}) FROM STDIN",
                io.StringIO(_pg_copy_text_rows(rows[i : i + _STAGE_COPY_ROWS], cols)),
            )
    finally:
        cur.close()
    conn.execute(text(f"ANALYZE {stage}"))
    return _Staged(
        table,
        stage,
        len(rows),
        list(cols),
        key_cols,
        carry_cols,
        order_col,
        _distinct_dates(r["score_date"] for r in rows),
    )


def _delete_insert_staged(conn, st: "_Staged | None", *, key_delete: bool) -> "tuple[int, int]":
    """Key-exact delete-then-insert of a staged payload with ONE DELETE and ONE INSERT.

    Every statement that touches a cold (pg_lake / Iceberg) partition costs
    seconds because it rewrites Iceberg metadata, so a per-chunk statement loop
    is unusable there. The target table therefore sees exactly:

      1. DELETE FROM table USING stage (join on the full key, plus a literal
         score_date = ANY(<payload dates>) so the planner prunes to the touched
         partitions and pg_lake never locks the cold file catalog for hot-only
         dates) RETURNING the key and carry_cols into a second temp table
         (skipped when key_delete is False);
      2. INSERT INTO table SELECT ... FROM stage LEFT JOIN carried, writing
         COALESCE(carried value, payload value) for each carry column.

    carry_cols are the columns the old ON CONFLICT DO UPDATE SET left alone:
    a surviving row keeps its existing value. `carried` is reduced with
    DISTINCT ON so a key that already has duplicate rows cannot multiply the
    [redacted: internal research note]
    by the DELETE as an ordering column only) and then by the carry values, so
    the choice is deterministic. Returns (rows_deleted, rows_inserted).

    The caller must already hold the advisory locks for st.dates.
    """
    if st is None:
        return 0, 0
    table, stage, key_cols, carry_cols, cols = (
        st.table,
        st.stage,
        st.key_cols,
        st.carry_cols,
        st.cols,
    )
    carried = "_m432_carried"
    join_s_c = " AND ".join(f"s.{c} = c.{c}" for c in key_cols)
    deleted = 0
    if key_delete:
        cc = [*key_cols, *carry_cols]
        if st.order_col:
            cc.append(st.order_col)
        col_ddl = ", ".join(f"{c} {_KEY_SQL_TYPES[c]}" for c in cc)
        conn.execute(text(f"CREATE TEMP TABLE {carried} ({col_ddl}) ON COMMIT DROP"))
        join_d_s = " AND ".join(f"d.{c} = s.{c}" for c in key_cols)
        res = conn.execute(
            text(
                f"WITH del AS (DELETE FROM {table} AS d USING {stage} AS s "
                f"WHERE {join_d_s} AND d.score_date = ANY(CAST(:score_dates AS date[])) "
                f"RETURNING {', '.join('d.' + c for c in cc)}) "
                f"INSERT INTO {carried} ({', '.join(cc)}) SELECT {', '.join(cc)} FROM del"
            ),
            {"score_dates": st.dates},
        )
        deleted = _rowcount(res)
        order_by = [
            *key_cols,
            *([f"{st.order_col} DESC NULLS LAST"] if st.order_col else []),
            *carry_cols,
        ]
        carried_src = (
            f"(SELECT DISTINCT ON ({', '.join(key_cols)}) * FROM {carried} "
            f"ORDER BY {', '.join(order_by)}) AS c"
        )
        select_exprs = [f"COALESCE(c.{c}, s.{c})" if c in carry_cols else f"s.{c}" for c in cols]
        from_sql = f"{stage} AS s LEFT JOIN {carried_src} ON {join_s_c}"
    else:
        select_exprs = [f"s.{c}" for c in cols]
        from_sql = f"{stage} AS s"
    res = conn.execute(
        text(
            f"INSERT INTO {table} ({', '.join(cols)}) "
            f"SELECT {', '.join(select_exprs)} FROM {from_sql}"
        )
    )
    return deleted, _insert_rowcount(res, st.n_rows)


def _assert_owned_factors_present(
    df: pd.DataFrame, owned_factors: "list[str]", caller: str = "upsert_daily_scores"
) -> None:
    """Refuse a delete-without-replace (Meridian#378).

    ``owned_factors`` makes the DELETE cover names the payload may not carry.
    That is deliberate for a canonical-superset writer and fatal for a writer
    whose owned list is meant to be what it just computed: the rows are deleted,
    nothing replaces them, and the run is green. Callers that genuinely mean the
    superset pass ``allow_unwritten_owned_factors=True``.

    ``caller`` names the refusing write path in the error, because this guard is
    shared by upsert_daily_scores and copy_replace_daily_scores and the message
    is the only thing an operator sees: naming the wrong one sends them to the
    wrong call site.

    The check is PER score_date, matching the DELETE, which runs once per
    distinct score_date in the frame and removes every owned name for that date.
    A whole-frame check passes while one date inside a multi-date frame has the
    hole - which is #378 again, just harder to see. That only the opted-out
    batch promoters currently pass multi-date frames is a fact about today's
    callers, not an invariant the check may lean on. The cost is a groupby over
    a partition-sized frame.
    """
    wanted = list(dict.fromkeys(owned_factors))
    for sd, part in df.groupby("score_date", sort=False):
        present = set(part["factor_name"].unique())
        missing = [f for f in wanted if f not in present]
        if missing:
            raise ValueError(
                f"{caller}: owned_factors names {missing} but the payload "
                f"carries no rows for them on score_date {sd}, so the owned-factors "
                "DELETE would remove those rows and insert nothing (Meridian#378 "
                "delete-without-replace). Scope owned_factors to the factors this run "
                "actually produced, or pass allow_unwritten_owned_factors=True if "
                "clearing them is the intent."
            )


_DAILY_KEY = ("score_date", "ticker", "factor_name")


def _daily_stage_plan(rows, *, cid, has_campaign_col, has_computed_at):
    """(cols, carry_cols, order_col) for a daily_scores write; sets campaign_id on rows."""
    # group_name is never in the old SET list: always carried. campaign_id is
    # carried only when no campaign is bound (the old SET wrote EXCLUDED
    # .campaign_id otherwise).
    carry_cols = ["group_name"]
    cols = [
        "score_date",
        "ticker",
        "factor_name",
        "group_name",
        "raw_value",
        "z_score",
        "direction",
    ]
    if has_campaign_col:
        cols.append("campaign_id")
        if cid is None:
            carry_cols.append("campaign_id")
        for r in rows:
            r["campaign_id"] = cid
    if has_computed_at:
        cols.append("computed_at")
    return cols, tuple(carry_cols), "computed_at"


def stage_daily_scores_rows(
    conn,
    rows: "list[dict]",
    *,
    cid: "int | None",
    has_campaign_col: bool,
    has_computed_at: bool,
) -> "_Staged | None":
    """Load `rows` into the daily_scores stage table (no lock needed, temp table only).

    First half of a daily_scores write: the caller then takes the per-score_date
    locks (_lock_score_dates) and calls apply_staged_daily_scores, all in the
    same transaction, so the lock is held only for the deletes and the INSERT.
    """
    if not rows:
        return None
    cols, carry_cols, order_col = _daily_stage_plan(
        rows, cid=cid, has_campaign_col=has_campaign_col, has_computed_at=has_computed_at
    )
    return _stage_rows(
        conn,
        "factors.daily_scores",
        rows,
        cols=cols,
        key_cols=_DAILY_KEY,
        carry_cols=carry_cols,
        order_col=order_col,
    )


def apply_staged_daily_scores(
    conn, staged: "_Staged | None", *, skip_key_delete: bool = False
) -> "tuple[int, int]":
    """Key-exact delete-then-insert of a staged daily_scores payload.

    Runs on an OPEN transaction `conn`; the caller must already hold the
    per-score_date advisory locks (_lock_score_dates) so that no other writer
    can interleave between this DELETE and INSERT. Returns (deleted, inserted).

    Replaces `INSERT .. ON CONFLICT (score_date, ticker, factor_name) DO UPDATE
    SET raw_value, z_score, direction, computed_at[, campaign_id]`. The old SET
    list did NOT touch group_name, and touched campaign_id only when a campaign
    was bound, so on a key that already existed the surviving row kept its
    group_name (always) and its campaign_id (when cid is None). The DELETE
    therefore RETURNS those two columns and the INSERT writes them back, so the
    end state matches the ON CONFLICT path exactly. computed_at is rewritten:
    EXCLUDED.computed_at when the payload carries it, else the column DEFAULT
    now() (the old `computed_at = NOW()`). The DELETE also returns the old
    computed_at, as an ordering column only, so that when a key already has
    duplicate rows the carried values come from the newest one.

    skip_key_delete: the caller proves a prior scoped DELETE already removed
    every payload key (replace=True with every payload factor in owned_factors),
    so there is nothing left to match and the second pass is skipped.
    """
    return _delete_insert_staged(conn, staged, key_delete=not skip_key_delete)


def write_daily_scores_rows(
    conn,
    rows: "list[dict]",
    *,
    cid: "int | None",
    has_campaign_col: bool,
    has_computed_at: bool,
    skip_key_delete: bool = False,
) -> "tuple[int, int]":
    """Stage + key-exact delete-then-insert of `rows` into factors.daily_scores.

    For callers that already hold the per-score_date advisory locks
    (_lock_score_dates) before calling, e.g. one-off scripts. upsert_daily_scores
    stages first and locks second via stage_daily_scores_rows /
    apply_staged_daily_scores so the lock is held for less time. Returns
    (deleted, inserted); see apply_staged_daily_scores for the semantics.
    """
    staged = stage_daily_scores_rows(
        conn, rows, cid=cid, has_campaign_col=has_campaign_col, has_computed_at=has_computed_at
    )
    return apply_staged_daily_scores(conn, staged, skip_key_delete=skip_key_delete)


def upsert_daily_scores(
    engine,
    df: pd.DataFrame,
    replace: bool = True,
    owned_factors: "list[str] | None" = None,
    campaign_id: "int | None" = None,
    allow_unwritten_owned_factors: bool = False,
    replace_date_range: "tuple[date, date] | None" = None,
    allow_interior_gaps: bool = False,
) -> int:
    """
    Upserts factor scores. Expects columns:
    score_date, ticker, factor_name, group_name, raw_value, z_score, direction

    campaign_id (Meridian#294, optional): the factors.write_campaigns row this
    write belongs to. When None the ambient/implicit campaign from
    meridian_pipeline.lineage is used, so callers need no change to get
    provenance. When the DB has no campaign_id column yet (migration
    2026-08-02 unapplied) the emitted SQL is byte-identical to its pre-#294
    form - lineage is additive and can never fail a write.

    computed_at: if `df` carries a `computed_at` column it is written through as
    COMPUTE time. Otherwise the column keeps its previous behaviour (table
    DEFAULT now() on insert, NOW() on conflict). This is the fix for the defect
    that made even the timestamp write-time-only - see
    copy_replace_daily_scores, where it mattered most.

    replace_date_range: batch-only inclusive replacement interval, even when
    some dates have no rows. Requires replace=True, explicit owned_factors and
    allow_unwritten_owned_factors=True. Rows outside that scope are refused.
    The interval is forwarded to the Iceberg mirror. Flat Parquet remains an
    append mirror and retains its existing orphan/tombstone limitations.

    allow_interior_gaps (Meridian#401): replace_date_range deletes every
    owned-factor row across the WHOLE declared interval, including any date
    strictly inside it that has no rows in `df` - that date's existing rows are
    removed and nothing replaces them. A containment check (rows fall inside
    the range, all owned factors) cannot catch this: a payload missing an
    interior date is still entirely inside the range. Nothing distinguishes
    "the recompute correctly determined this date should be empty" from "the
    batch that assembled this payload silently dropped a date."

    So this function compares the payload's distinct dates against the REAL
    trading calendar (`missing_trading_days`, sourced from ohlcv.prices - never
    derived from `df` itself, which is the thing being checked) over the
    declared interval, and by default REFUSES the write when any in-scope
    trading day has zero payload rows, logging the missing dates either way.
    Pass allow_interior_gaps=True only when an empty date is a deliberate,
    reviewed outcome of the recompute (not merely "the caller expects gaps") -
    the same explicit-opt-out shape as allow_unwritten_owned_factors above.

    replace=True (default) deletes existing rows for each (score_date, group_name)
    in df before inserting, so a rerun whose universe SHRANK leaves no orphan rows
    for tickers that dropped out (#108). Plain ON CONFLICT only updates matching
    keys. DELETE + INSERT share one transaction (engine.begin), so it is atomic.

    owned_factors: when provided, the DELETE targets factor_name = ANY(owned_factors)
    instead of the df's actual factor_names. Use this when an asset may silently
    drop a factor it normally writes (e.g. swallowed by try/except) - the
    old row for that factor would survive forever if the delete only covered df's
    factors. Each asset passes its full canonical factor list so stale orphan rows
    for any owned factor are cleared even when the df is smaller than usual. When
    None (default), only factors present in df are deleted (pre-#161 behaviour).
    Fix #5 (#161).

    owned_factors deletes are scoped to (score_date, factor_name) only - NOT
    group_name - because owned_factors asserts the caller has EXCLUSIVE
    ownership of those factor names across every group, not just the groups
    that happen to appear in this payload. Fix #5's original implementation
    grouped the delete by (score_date, group_name) pairs derived from df, so a
    factor whose entire designated group was absent from the payload (e.g.
    a writer that gives each factor its own group, and a gappy input panel can
    empty a whole group) never got a DELETE issued for it at all - the
    stale row from a prior run silently survived next to its corrected
    siblings. Scoping strictly to (score_date, factor_name) closes that gap
    (finding: db.py:394 multi-group-writer orphan, 2026-07).

    allow_unwritten_owned_factors (Meridian#378): the owned_factors DELETE runs
    before the insert, so any owned name with no rows in `df` is a
    delete-WITHOUT-replace - that date loses the factor entirely, in PG and in
    both mirrors, and the run still reports SUCCESS (#378). By default this
    function therefore REFUSES a
    write whose owned_factors names a factor the payload does not carry.

    Two legitimate caller shapes need the old permissive behaviour and opt out
    explicitly:

      * canonical-superset writers pass their whole factor list, and their scoring helper
        skips a factor whose column is missing or has fewer than 5 observations.
        The superset is exactly how a stale row from a prior run gets cleared.
      * batch promoters - promote_backfill / promote_replay_out /
        incluster_bulk_rescore / gold_pg_sync hold one owned scope for a whole
        campaign and upsert it one partition at a time, so a single partition is
        legitimately a subset of that scope.

    Every other caller (one whose owned list is meant to be exactly what it just
    computed) gets the check, and a failed partition instead of a silent hole.
    The check is per score_date, matching the DELETE: a whole-frame check passes
    while one date inside a multi-date frame has the hole, which is #378 again.
    Today only the opted-out batch promoters batch dates, but that is a fact
    about the current callers, not an invariant. Cost is a groupby on a
    partition-sized frame.
    """
    # Batch snapshot promotion owns a declared interval, including dates with
    # no payload rows. Require explicit ownership and deletion permission.
    if replace_date_range is not None:
        lo, hi = replace_date_range
        if not replace or not owned_factors or not allow_unwritten_owned_factors:
            raise ValueError(
                "replace_date_range requires replace, owned_factors and deletion opt-in"
            )
        if lo > hi:
            raise ValueError("replace_date_range is reversed")
        assert_score_date_not_future(hi)
        if not df.empty:
            dates = pd.to_datetime(df["score_date"]).dt.date
            if not dates.between(lo, hi).all() or not df["factor_name"].isin(owned_factors).all():
                raise ValueError("payload falls outside declared replacement scope")
        # Meridian#401: containment (checked above) cannot catch a HOLE - a
        # payload missing an interior date is still entirely inside [lo, hi].
        # Compare against the real trading calendar instead of df itself.
        payload_days = set(pd.to_datetime(df["score_date"]).dt.date) if not df.empty else set()
        missing_days = missing_trading_days(engine, lo, hi, payload_days)
        if missing_days:
            preview = ", ".join(d.isoformat() for d in missing_days[:10])
            more = f", +{len(missing_days) - 10} more" if len(missing_days) > 10 else ""
            _log.warning(
                "upsert_daily_scores: replace_date_range %s..%s for owned_factors=%s "
                "carries %d in-scope trading day(s) with ZERO payload rows "
                "(%s%s). Those dates' existing rows are about to be DELETED with "
                "nothing inserted to replace them.",
                lo,
                hi,
                owned_factors,
                len(missing_days),
                preview,
                more,
            )
            if not allow_interior_gaps:
                raise ValueError(
                    f"upsert_daily_scores: refusing replace_date_range {lo}..{hi} for "
                    f"owned_factors={owned_factors} - {len(missing_days)} "
                    "in-scope trading day(s) have no rows in "
                    "the payload, so the delete-replace would remove those dates' "
                    "rows and insert nothing (Meridian#401 interior-date gap). "
                    "Nothing distinguishes a batch that silently dropped a date from "
                    "a date that is genuinely meant to be empty. If the empty "
                    "date(s) are a deliberate, reviewed outcome, pass "
                    "allow_interior_gaps=True to state that explicitly; otherwise "
                    f"fix the batch that assembled this payload. Missing: {preview}"
                    f"{more}"
                )
    if df.empty and replace_date_range is None:
        return 0
    _assert_df_score_dates_not_future(df)
    if replace and owned_factors and not allow_unwritten_owned_factors:
        _assert_owned_factors_present(df, owned_factors)
    rows = df.to_dict(orient="records")
    cid = resolve_campaign_id(engine, campaign_id, "factors.daily_scores")
    has_computed_at = "computed_at" in df.columns
    # Fix #2 (#161): sanitize non-finite float fields at the DB boundary.
    # NaN/inf in raw_value or z_score must not reach Postgres float8; they
    # survive pd.isna() checks and break downstream consumers (QC Insights,
    # risk model). Mirror the pattern from upsert_composite_scores.
    import math as _math

    for r in rows:
        for _col in ("raw_value", "z_score"):
            v = r.get(_col)
            if v is None:
                continue
            try:
                r[_col] = float(v) if _math.isfinite(float(v)) else None
            except (TypeError, ValueError):
                r[_col] = None
    has_campaign_col = _campaign_column_available(engine, "factors.daily_scores")
    rows, n_dup = _dedupe_keep_last(rows, _DAILY_KEY)
    if n_dup:
        _log.warning("upsert_daily_scores: collapsed %d duplicate payload key(s), last wins", n_dup)
    # The scoped replace-DELETE covers every payload key only when each payload
    # factor is in owned_factors (it deletes by (score_date, factor_name)); the
    # group-scoped form can leave a same-key row under another group_name, and
    # replace=False leaves everything, so those take the key-exact pass.
    covered = (
        replace
        and owned_factors is not None
        and (df.empty or set(df["factor_name"].unique()) <= set(owned_factors))
    )
    payload_dates = [] if df.empty else df["score_date"].drop_duplicates().tolist()
    replace_deleted = 0
    if cid is not None:
        for r in rows:
            r["campaign_id"] = cid
    with engine.begin() as conn:
        # Meridian#432: stage the payload FIRST (temp table + one COPY, slow but
        # lock-free), THEN take the locks, so the per-date advisory locks are held
        # only for the deletes and the single INSERT below.
        staged = stage_daily_scores_rows(
            conn,
            rows,
            cid=cid,
            has_campaign_col=has_campaign_col,
            has_computed_at=has_computed_at,
        )
        # No unique index on a cold-tier parent, so serialise writers per
        # score_date BEFORE any delete (sorted, deadlock-free).
        # Per-date locks cover the payload's dates only. A replace_date_range
        # batch also deletes dates that carry no payload; locking each calendar
        # day of a multi-year range would risk "out of shared memory" (every
        # advisory lock takes a shared lock-table slot). Instead (FND-007,
        # Meridian#468) the range replace takes the table's range lock
        # EXCLUSIVE, which waits for every in-flight writer (they all hold it
        # SHARED) and blocks new ones until commit, so no concurrent write can
        # land in an unpayloaded interior date and then be deleted.
        _lock_score_dates(
            conn,
            "factors.daily_scores",
            payload_dates,
            range_exclusive=replace_date_range is not None,
        )
        if replace:
            if replace_date_range is not None:
                _log_replace_scope(
                    "upsert_daily_scores(replace_date_range)",
                    "factors.daily_scores",
                    payload_dates,
                    len(owned_factors),
                    len(rows),
                    date_range=(lo, hi),
                )
                replace_deleted += _rowcount(
                    conn.execute(
                        text(
                            "DELETE FROM factors.daily_scores "
                            "WHERE score_date BETWEEN :date_min AND :date_max "
                            "AND factor_name = ANY(:factor_names)"
                        ),
                        {"date_min": lo, "date_max": hi, "factor_names": owned_factors},
                    )
                )
            elif owned_factors is not None:
                # Fix #5 follow-up (2026-07): delete on (score_date, factor_name)
                # alone, REGARDLESS of which group_name(s) appear in this
                # payload. Grouping the delete by df's (score_date, group_name)
                # pairs (the original Fix #5) no-ops whenever an owned factor's
                # entire group is absent from the payload - exactly the
                # multi-group-writer orphan class this parameter exists to
                # close. owned_factors is a per-caller exclusive-ownership
                # list, so this is safe: no other writer's factors can be
                # named in it.
                # ONE statement for all payload dates (a statement against the
                # cold tier costs seconds); the literal date array lets the
                # planner prune to the touched partitions.
                _log_replace_scope(
                    "upsert_daily_scores(owned_factors)",
                    "factors.daily_scores",
                    payload_dates,
                    len(owned_factors),
                    len(rows),
                )
                replace_deleted += _rowcount(
                    conn.execute(
                        text(
                            "DELETE FROM factors.daily_scores "
                            "WHERE score_date = ANY(CAST(:score_dates AS date[])) "
                            "AND factor_name = ANY(:factor_names)"
                        ),
                        {
                            "score_dates": _distinct_dates(payload_dates),
                            "factor_names": owned_factors,
                        },
                    )
                )
            else:
                # owned_factors=None: preserve the #160 per-(score_date,
                # group_name) scoping. Multiple assets can write into the
                # SAME group in sequence (one asset writes some factors, then
                # a second asset writes the rest), and a group-wide delete
                # made each writer clobber the others, leaving only the last
                # writer's factors (#160). Deleting per (group, df's factors)
                # still clears universe-shrink orphans (#108) for the factors
                # being rewritten. Single-writer groups are unaffected: their
                # df already contains the whole group's factors.
                # One statement keyed by the distinct (score_date, group_name,
                # factor_name) triples: the same rows the old per-(date, group)
                # loop deleted, without a cold-tier round trip per pair.
                # dropna: the old groupby skipped a NULL group_name, and a NaN
                # would make the text[] array literal mixed-type.
                triples = (
                    df[["score_date", "group_name", "factor_name"]]
                    .dropna(subset=["group_name"])
                    .drop_duplicates()
                )
                replace_deleted += _rowcount(
                    conn.execute(
                        text(
                            "DELETE FROM factors.daily_scores AS d USING unnest("
                            "CAST(:k_dates AS date[]), CAST(:k_groups AS text[]), "
                            "CAST(:k_factors AS text[])) "
                            "AS k(score_date, group_name, factor_name) "
                            "WHERE d.score_date = k.score_date AND d.group_name = k.group_name "
                            "AND d.factor_name = k.factor_name "
                            "AND d.score_date = ANY(CAST(:score_dates AS date[]))"
                        ),
                        {
                            "k_dates": [_as_date(v) for v in triples["score_date"]],
                            "k_groups": triples["group_name"].tolist(),
                            "k_factors": triples["factor_name"].tolist(),
                            "score_dates": _distinct_dates(payload_dates),
                        },
                    )
                )
        key_deleted, inserted = apply_staged_daily_scores(conn, staged, skip_key_delete=covered)
    _log.info(
        "upsert_daily_scores: run=%s campaign=%s dates=%d replace_deleted=%d "
        "key_deleted=%d inserted=%d",
        _write_run_id(),
        cid,
        len(payload_dates),
        replace_deleted,
        key_deleted,
        inserted,
    )
    # PG transaction committed above; mirror it into the columnar read stores.
    # campaign_id is stripped from the mirror frame: the flat-parquet and
    # Iceberg schemas are fixed and PG is the lineage authority for now.
    # Propagating the stamp into the mirrors is a deliberate follow-up (it needs
    # an Iceberg schema evolution on meridian.factors.daily_scores), not a
    # silent side effect of this change.
    mirror_df = pd.DataFrame(rows)
    if "campaign_id" in mirror_df.columns:
        mirror_df = mirror_df.drop(columns=["campaign_id"])
    mirror_scope = {"replace_date_range": replace_date_range} if replace_date_range else {}
    _dual_write_daily_scores(
        mirror_df, df, replace=replace, owned_factors=owned_factors, **mirror_scope
    )
    return len(df)


def _dual_write_daily_scores(
    sanitized_df: pd.DataFrame,
    source_df: pd.DataFrame,
    replace: bool,
    owned_factors: "list[str] | None",
    replace_date_range: "tuple[date, date] | None" = None,
) -> None:
    """Best-effort mirror of a committed factors.daily_scores write.

    Extracted so EVERY PG write path into factors.daily_scores mirrors
    identically. Both `upsert_daily_scores` (row-oriented, daily assets) and
    `copy_replace_daily_scores` (bulk COPY, burst promote) call this. A write
    path that skips it produces exactly the silent bypass Meridian#266 closed:
    the operator's preflight (`lakehouse.assert_dual_write_ready`) passes, the
    run reports success, and the mirror quietly falls behind PG forever.

    Two mirrors, both best-effort - PG is authoritative and neither may raise:

    1. Flat parquet (`meridian-parquet`), the STOCK_FACTORS subset read by the
       ML backfill / evidence engine. No-op when MERIDIAN_MINIO_* is unset
       (CI / local dev). Receives the SANITIZED rows (NaN/inf already coerced
       to None), not the raw df - passing the raw df let non-finite values
       reach the mirror even though PG never sees them.
    2. Iceberg (`meridian.factors.daily_scores`), gated behind
       MERIDIAN_ICEBERG_DUAL_WRITE (default ON since 2026-07-09).

    Best-effort/log-and-continue is right for in-cluster callers (step pods
    always carry lakehouse creds via the dagster-secrets k8s secret, so a skip
    means a transient fault, not missing config). Off-cluster BATCH callers
    (e.g. scripts/burst/promote_backfill.py) must not rely on the log line to
    notice a systemically missing credential - they call
    lakehouse.assert_dual_write_ready() once, up front, instead.
    """
    try:
        from meridian_pipeline import parquet_io

        parquet_io.append_factor_scores_partition(
            sanitized_df, sorted(set(sanitized_df["factor_name"]))
        )
    except Exception:
        pass  # dual-write is best-effort; the PG write already succeeded

    # Retain the automatic mirror until the separately approved checked cutover.
    try:
        from meridian_pipeline import lakehouse

        if lakehouse.iceberg_dual_write_enabled():
            try:
                scope = {"replace_date_range": replace_date_range} if replace_date_range else {}
                lakehouse.write_daily_scores(
                    source_df, replace=replace, owned_factors=owned_factors, **scope
                )
            except Exception as exc:
                _log.warning("iceberg dual-write skipped: %s", exc)
    except Exception:
        pass  # importing lakehouse must never break the PG write path


def copy_replace_daily_scores(
    engine,
    df: pd.DataFrame,
    owned_factors: list[str],
    campaign_id: "int | None" = None,
    allow_unwritten_owned_factors: bool = False,
) -> int:
    """Bulk-COPY delete-replace for a burst backfill promote (A2 throughput).

    Same scoped delete as upsert_daily_scores(owned_factors=...) - one DELETE of
    factor_name = ANY(owned_factors) over every payload score_date - but the INSERT uses psycopg2
    `COPY ... FROM STDIN` instead of chunked executemany, which is the throughput
    bottleneck on a multi-million-row promote (a multi-year run spent hours in
    row-oriented upserts). The delete-replace makes COPY safe: no surviving rows
    for the owned factors on each date, so there is no PK conflict to merge.

    Contract + scope:
      * owned_factors is REQUIRED (a plain COPY has no conflict handling, and the
        parent has no unique index to fall back on). It asserts the caller
        EXCLUSIVELY owns those factor names, exactly like the
        upsert_daily_scores owned_factors path. Do NOT use on the daily asset path,
        where several writers merge into shared groups and rely on the key-exact
        carry in upsert_daily_scores.
      * allow_unwritten_owned_factors (Meridian#378 parity with
        upsert_daily_scores, added after #378's call site broke this path): by
        default an owned factor with no rows in `df` on a date is REFUSED,
        because the scoped DELETE would remove it and insert nothing. Batch
        promoters (promote_backfill / promote_replay_out / incluster_bulk_rescore
        / gold_pg_sync) hold one scope for a whole campaign and write it one
        partition at a time, so a partition is legitimately a subset and they
        pass True. Before this parameter existed the default COPY path - which
        is the promote's default write path - raised TypeError on its first
        partition, so no promote had reached gold through it since #378.
      * computed_at IS NOW WRITTEN when the frame carries it (Meridian#294).
        It used to be omitted from the COPY column list so the table DEFAULT
        stamped now() - meaning the only provenance the table had recorded
        WRITE time, not COMPUTE time, on the one path where the two can be days
        apart (a burst kernel scores a partition, the promote lands it later).
        That defect is what forced write campaigns to be reconstructed by
        clustering computed_at, and the reconstruction was wrong. Now: if `df`
        has a `computed_at` column it is COPYed through as the compute stamp
        (scripts/burst/promote_backfill.py fills it from the staged object's
        S3 LastModified); if it does not, the column is omitted exactly as
        before and the DEFAULT applies. Delete-replace semantics are untouched
        either way - computed_at is not part of the PK and the DELETE runs
        first, so which rows exist is unchanged.
      * campaign_id (Meridian#294, optional) is COPYed as a constant extra
        field per row when a campaign is bound. Two constant CSV fields cost
        nothing measurable next to the COPY itself, and there is deliberately
        NO foreign key on the column, so no per-row referential trigger fires -
        the throughput reason this path exists is preserved.
      * The PG write is a single transaction (delete + copy commit together).
        Once it commits, this calls the SAME `_dual_write_daily_scores` helper
        `upsert_daily_scores` does, so swapping the promote path onto COPY does
        not silently drop the flat-parquet / Iceberg mirrors. That parity is
        load-bearing: promote_backfill.py runs
        `lakehouse.assert_dual_write_ready()` as a preflight, and a write path
        that skips the mirror would make that preflight pass while every
        promoted partition quietly missed Iceberg (the Meridian#266 bypass
        class). Do not add another daily_scores write path without it.

    Returns the number of rows written.
    """
    import csv
    import math as _math

    if df.empty:
        return 0
    _assert_df_score_dates_not_future(df)
    if not owned_factors:
        raise ValueError("copy_replace_daily_scores requires a non-empty owned_factors list")
    if not allow_unwritten_owned_factors:
        # #378 parity with upsert_daily_scores. The DELETE below is scoped to
        # (score_date, factor_name in owned_factors), so an owned name with no
        # rows in `df` on a date is a delete-WITHOUT-replace: that date loses
        # the factor in PG and in both mirrors and the run still succeeds.
        # Batch promoters hold one scope for a whole campaign and write it one
        # partition at a time, so a partition is legitimately a subset of the
        # scope and they opt out explicitly; every other caller gets the check.
        _assert_owned_factors_present(df, owned_factors, caller="copy_replace_daily_scores")

    cols = [
        "score_date",
        "ticker",
        "factor_name",
        "group_name",
        "raw_value",
        "z_score",
        "direction",
    ]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"copy_replace_daily_scores: df missing columns {missing}")

    # Meridian#432: a plain COPY has no ON CONFLICT, and the parent table soon
    # has no unique index to reject a bad payload either. The scoped DELETE only
    # clears owned factors, and a repeated key would be stored twice; both used
    # to surface as a unique violation on hot months and would now be silent
    # duplicates, so refuse them explicitly.
    foreign = sorted(set(df["factor_name"].unique()) - set(owned_factors))
    if foreign:
        raise ValueError(
            f"copy_replace_daily_scores: payload carries factors {foreign[:10]} that are not in "
            "owned_factors, so the scoped DELETE would not clear their existing rows and the "
            "COPY would duplicate them."
        )
    if df.duplicated(subset=["score_date", "ticker", "factor_name"]).any():
        raise ValueError(
            "copy_replace_daily_scores: payload repeats a (score_date, ticker, factor_name) key"
        )

    cid = resolve_campaign_id(engine, campaign_id, "factors.daily_scores")
    # Trailing lineage fields, appended in a fixed order to both the COPY
    # column list and every CSV row. When neither applies the buffer is
    # byte-identical to the pre-#294 one (locked by the parity test).
    copy_cols = list(cols)
    if cid is not None:
        copy_cols.append("campaign_id")
    trailing = [] if cid is None else [cid]
    computed_at_values = None
    if "computed_at" in df.columns:
        copy_cols.append("computed_at")
        stamps = pd.to_datetime(df["computed_at"], errors="coerce", utc=True)
        distinct = stamps.dropna().unique()
        if len(distinct) == 1 and not stamps.isna().any():
            # Overwhelmingly the common case: one compute stamp for the whole
            # partition. Render it ONCE and fold it into the constant trailing
            # fields - a per-row isoformat() over millions of rows measurably
            # slows the COPY (~19% on a 400k-row benchmark) and this path exists
            # because per-row work was the bottleneck.
            trailing.append(pd.Timestamp(distinct[0]).isoformat())
        else:
            # Mixed or partially-missing stamps: vectorised strftime rather than
            # a Python loop. NaT -> '' -> NULL under FORMAT csv, NULL ''.
            rendered = stamps.dt.strftime("%Y-%m-%dT%H:%M:%S.%f%z")
            computed_at_values = rendered.where(stamps.notna(), "").tolist()

    buf = io.StringIO()
    w = csv.writer(buf)
    rows = []
    n = 0
    for idx, r in enumerate(df[cols].itertuples(index=False, name=None)):
        sd, ticker, factor, group, raw_value, z_score, direction = r
        out = []
        for v in (raw_value, z_score):
            # NaN/inf must not reach float8 (mirror upsert_daily_scores Fix #2);
            # empty CSV field -> NULL under FORMAT csv, NULL ''.
            if v is None:
                out.append(None)
                continue
            try:
                fv = float(v)
                out.append(fv if _math.isfinite(fv) else None)
            except (TypeError, ValueError):
                out.append(None)
        w.writerow(
            [
                str(sd),
                ticker,
                factor,
                group,
                "" if out[0] is None else out[0],
                "" if out[1] is None else out[1],
                int(direction),
                *trailing,
                *(() if computed_at_values is None else (computed_at_values[idx],)),
            ]
        )
        # Same sanitized values the COPY carries, kept for the parquet mirror
        # so it can never diverge from what PG actually stored.
        rows.append(
            {
                "score_date": sd,
                "ticker": ticker,
                "factor_name": factor,
                "group_name": group,
                "raw_value": out[0],
                "z_score": out[1],
                "direction": int(direction),
            }
        )
        n += 1
    buf.seek(0)

    payload_dates = df["score_date"].drop_duplicates().tolist()
    raw = engine.raw_connection()
    deleted = 0
    try:
        cur = raw.cursor()
        # Meridian#432: the scoped DELETE below is this path's only duplicate
        # guard once the parent has no unique index, so serialise writers per
        # score_date first (sorted, transaction-scoped, deadlock-free).
        _lock_score_dates_cursor(cur, "factors.daily_scores", payload_dates)
        # ONE delete for every payload date (a statement against the cold tier
        # costs seconds); the literal date array lets the planner prune.
        _log_replace_scope(
            "copy_replace_daily_scores",
            "factors.daily_scores",
            payload_dates,
            len(owned_factors),
            n,
        )
        cur.execute(
            "DELETE FROM factors.daily_scores "
            "WHERE score_date = ANY(CAST(%s AS date[])) AND factor_name = ANY(%s)",
            (_distinct_dates(payload_dates), list(owned_factors)),
        )
        deleted += _rowcount(cur)
        cur.copy_expert(
            f"COPY factors.daily_scores ({', '.join(copy_cols)}) "
            "FROM STDIN WITH (FORMAT csv, NULL '')",
            buf,
        )
        raw.commit()
    except Exception:
        raw.rollback()
        raise
    finally:
        raw.close()
    _log.info(
        "copy_replace_daily_scores: run=%s campaign=%s dates=%d deleted=%d inserted=%d",
        _write_run_id(),
        cid,
        len(payload_dates),
        deleted,
        n,
    )

    # PG transaction committed above; mirror it exactly as upsert_daily_scores
    # does. replace=True because the COPY path is always a scoped
    # delete-replace over owned_factors.
    _dual_write_daily_scores(
        pd.DataFrame(rows), df, replace=True, owned_factors=list(owned_factors)
    )
    return n


_GROUP_KEY = ("score_date", "ticker", "group_name")


def upsert_group_scores(
    engine,
    df: pd.DataFrame,
    replace: bool = True,
    campaign_id: "int | None" = None,
    input_vintage: "dict | None" = None,
    input_campaign_rows: "pd.DataFrame | None" = None,
) -> int:
    """Expects: score_date, ticker, group_name, group_score.
    replace=True delete-replaces per (score_date, group_name) to clear orphans (#108).

    campaign_id (Meridian#294, optional): see upsert_daily_scores. group_scores
    is stamped because it is a SERVED derivative with its own writers (the
    composite asset, the GPU batch backfill, bulk reblends) - "which reblend
    produced this row" is a question that has been asked and could not be
    answered from daily_scores' lineage alone."""
    if df.empty:
        return 0
    _assert_df_score_dates_not_future(df)
    # Normalise NaN -> NULL before writing (Meridian#318). `_group_scores`
    # yields NaN for a group whose every non-excluded factor carries a NULL
    # z_score (e.g. a group whose non-excluded factors are all per-date
    # broadcast constants). `to_dict` keeps that as a float NaN and psycopg2
    # sends PostgreSQL NaN, which is NOT NULL - so those rows then read as
    # COVERED to any `group_score IS NOT NULL` baseline while pandas'
    # `.notna()` on the current partition reads them as UNCOVERED. That
    # disagreement makes the completeness gate permanently unsatisfiable for
    # such a group from the first date that lands NaN onward. many historical
    # rows are NULL; the NaN form was a live-path-only divergence.
    df = df.copy()
    df["group_score"] = df["group_score"].astype(object).where(df["group_score"].notna(), None)
    rows = df.to_dict(orient="records")
    cid = resolve_campaign_id(engine, campaign_id, "factors.group_scores")
    if input_vintage:
        from meridian_pipeline import lineage

        lineage.attach_input_vintage(engine, cid, input_vintage)
    if cid is not None:
        for r in rows:
            r["campaign_id"] = cid
    has_campaign_col = _campaign_column_available(engine, "factors.group_scores")
    rows, n_dup = _dedupe_keep_last(rows, _GROUP_KEY)
    if n_dup:
        _log.warning("upsert_group_scores: collapsed %d duplicate payload key(s), last wins", n_dup)
    payload_dates = df["score_date"].drop_duplicates().tolist()
    replace_deleted = key_deleted = 0
    # The old SET left campaign_id alone unless a campaign was bound: carry it
    # forward in that case.
    gcols = ["score_date", "ticker", "group_name", "group_score"]
    carry = ()
    if has_campaign_col:
        gcols.append("campaign_id")
        if cid is None:
            carry = ("campaign_id",)
        for r in rows:
            r["campaign_id"] = cid
    with engine.begin() as conn:
        # Meridian#432: stage the payload FIRST (temp table + one COPY, lock-free),
        # THEN take the locks, so they are held only for the deletes and the INSERT.
        staged = _stage_rows(
            conn,
            "factors.group_scores",
            rows,
            cols=gcols,
            key_cols=_GROUP_KEY,
            carry_cols=carry,
        )
        # No unique index on a cold-tier parent; serialise writers per
        # score_date before any delete (see _lock_score_dates).
        _lock_score_dates(conn, "factors.group_scores", payload_dates)
        if replace:
            pairs = df[["score_date", "group_name"]].drop_duplicates().to_dict(orient="records")
            for p in pairs:
                replace_deleted += _rowcount(
                    conn.execute(
                        text(
                            "DELETE FROM factors.group_scores "
                            "WHERE score_date = :score_date AND group_name = :group_name"
                        ),
                        p,
                    )
                )
        # Key-exact pass. With replace=True the (score_date, group_name) delete
        # above already removed every payload key (the key contains both
        # columns), so it is only needed for replace=False.
        key_deleted, _ins = _delete_insert_staged(conn, staged, key_delete=not replace)
        _replace_derived_surface_input_campaigns(
            conn,
            "group_scores",
            cid,
            df["score_date"].drop_duplicates().tolist(),
            input_campaign_rows,
        )
    _log.info(
        "upsert_group_scores: run=%s campaign=%s dates=%d replace_deleted=%d "
        "key_deleted=%d inserted=%d",
        _write_run_id(),
        cid,
        len(payload_dates),
        replace_deleted,
        key_deleted,
        len(rows),
    )
    return len(df)


# TODO(M14.5): composite/group Iceberg mirror pending lakehouse writer
def upsert_composite_scores(
    engine,
    df: pd.DataFrame,
    replace: bool = True,
    campaign_id: "int | None" = None,
    input_vintage: "dict | None" = None,
    input_campaign_rows: "pd.DataFrame | None" = None,
) -> int:
    """Expects: score_date, ticker, composite_score, quintile.
    replace=True delete-replaces per score_date (composite has no group) to clear
    orphan tickers from a shrunk universe on rerun (#108).

    campaign_id (Meridian#294, optional): see upsert_daily_scores. The composite
    is the API's headline served number and is re-blended independently of the
    factor rows it derives from (weights change, membership changes), so its
    provenance is genuinely NOT derivable from daily_scores' campaigns - it
    needs its own, whose input_vintage names the daily campaigns it consumed."""
    if df.empty:
        return 0
    _assert_df_score_dates_not_future(df)
    rows = df.to_dict(orient="records")
    cid = resolve_campaign_id(engine, campaign_id, "factors.composite_scores")
    if input_vintage:
        from meridian_pipeline import lineage

        lineage.attach_input_vintage(engine, cid, input_vintage)
    if cid is not None:
        for r in rows:
            r["campaign_id"] = cid
    # Hard NaN sanitize at the DB boundary: a non-finite quintile is rejected by
    # the smallint column ("smallint out of range") and wedges the whole
    # partition. pandas silently re-coerces a mixed int/None column back to
    # float64 (None -> NaN), so the asset-side conversion alone is not enough.
    # #329: write n_groups only if the column exists (probe) AND the caller
    # supplied it - keeps the write byte-identical to pre-#329 until the migration
    # lands and the asset ships the column together.
    write_ng = "n_groups" in df.columns and _composite_has_ngroups(engine)
    write_provenance = _composite_has_blend_provenance(engine)
    for r in rows:
        q = r.get("quintile")
        r["quintile"] = None if pd.isna(q) else int(q)
        cs = r.get("composite_score")
        if pd.isna(cs):
            r["composite_score"] = None
        if write_ng:
            ng = r.get("n_groups")
            r["n_groups"] = None if pd.isna(ng) else int(ng)
        if write_provenance:
            import json

            provenance = r.get("blend_provenance")
            r["blend_provenance"] = (
                json.dumps(provenance, allow_nan=False) if isinstance(provenance, dict) else None
            )
    with engine.begin() as conn:
        if replace:
            for sd in df["score_date"].drop_duplicates().tolist():
                conn.execute(
                    text("DELETE FROM factors.composite_scores WHERE score_date = :score_date"),
                    {"score_date": sd},
                )
        cols_sql, binds_sql, conflict_extra = _campaign_sql_fragments(cid)
        ng_cols = ", n_groups" if write_ng else ""
        ng_binds = ", :n_groups" if write_ng else ""
        ng_conflict = ", n_groups = EXCLUDED.n_groups" if write_ng else ""
        bp_cols = ", blend_provenance" if write_provenance else ""
        bp_binds = ", CAST(:blend_provenance AS jsonb)" if write_provenance else ""
        bp_conflict = ", blend_provenance = EXCLUDED.blend_provenance" if write_provenance else ""
        insert_sql = text(f"""
                INSERT INTO factors.composite_scores
                    (score_date, ticker, composite_score, quintile{ng_cols}{bp_cols}{cols_sql})
                VALUES (:score_date, :ticker, :composite_score, :quintile{ng_binds}{bp_binds}{binds_sql})
                ON CONFLICT (score_date, ticker)
                DO UPDATE SET composite_score = EXCLUDED.composite_score,
                              quintile = EXCLUDED.quintile,
                              computed_at = NOW(){ng_conflict}{bp_conflict}{conflict_extra}
            """)
        for i in range(0, len(rows), _UPSERT_CHUNK):
            conn.execute(insert_sql, rows[i : i + _UPSERT_CHUNK])
        _replace_derived_surface_input_campaigns(
            conn,
            "composite_scores",
            cid,
            df["score_date"].drop_duplicates().tolist(),
            input_campaign_rows,
        )
    return len(df)


def mark_request(engine, score_date: date, status: str, error_msg: str = None) -> None:
    """Update a compute_request status. Called by sensor/assets."""
    with engine.begin() as conn:
        conn.execute(
            text("""
            UPDATE factors.compute_requests
            SET status = :s, error_msg = :e
            WHERE score_date = :d
        """),
            {"d": score_date, "s": status, "e": error_msg},
        )


def submit_request(engine, score_date: date) -> None:
    """Submit a compute request (called by the API when data is missing)."""
    with engine.begin() as conn:
        conn.execute(
            text("""
            INSERT INTO factors.compute_requests (score_date, status)
            VALUES (:d, 'pending')
            ON CONFLICT (score_date) DO NOTHING
        """),
            {"d": score_date},
        )


# Windows longer than this many CALENDAR days short-circuit to a weekday
# count in count_trading_days_between instead of scanning a multi-year slice
# of ohlcv.prices. Every staleness gate that consumes the result tops out far
# below this, so past this bound the exact figure
# changes no decision - only the magnitude reported next to an already-failing
# check.
_CALENDAR_EXACT_MAX_DAYS = 90


def _weekday_count(start_exclusive: date, end_inclusive: date) -> int:
    """Mon-Fri count in (start_exclusive, end_inclusive]. Holiday-blind, so it
    is an UPPER BOUND on the trading-day distance. Only used past
    _CALENDAR_EXACT_MAX_DAYS, where any consumer has already failed its gate."""
    first = start_exclusive + timedelta(days=1)
    if end_inclusive < first:
        return 0
    span = (end_inclusive - first).days + 1
    full_weeks, remainder = divmod(span, 7)
    n = full_weeks * 5
    for i in range(remainder):
        if (first + timedelta(days=i)).weekday() < 5:
            n += 1
    return n


def count_trading_days_between(engine, start_exclusive: date, end_inclusive: date) -> int:
    """Trading-day distance in (start_exclusive, end_inclusive] (#215 staleness
    contract), measured against the REAL trading calendar in ohlcv.prices.

    #375: this used to count DISTINCT score_date rows in factors.group_scores,
    which made it blind to the exact failure it exists to catch. A missing
    group_scores date is both the hole AND absent from a calendar derived from
    group_scores itself, so a genuine one-trading-day hole scored 0 days of
    staleness and the API served the previous day's answer as if current. The
    calendar has to come from a source that is not the thing being checked.

    Uses the same predicate as is_trading_day (>100 equity daily rows on the
    day, guarding partial-ingest races) and the same sargable half-open
    ts_event range so the btree index on ts_event is usable.

    Returns 0 when start_exclusive >= end_inclusive.
    """
    if start_exclusive >= end_inclusive:
        return 0
    if (end_inclusive - start_exclusive).days > _CALENDAR_EXACT_MAX_DAYS:
        return _weekday_count(start_exclusive, end_inclusive)
    with engine.connect() as conn:
        row = conn.execute(
            text("""
            SELECT COUNT(*) FROM (
                SELECT ts_event::date AS d
                FROM ohlcv.prices
                WHERE ts_event >= :start_next AND ts_event < :end_next
                  AND security_type = 'equity' AND resolution = 'daily'
                GROUP BY 1
                HAVING COUNT(*) > 100
            ) t
            """),
            {
                "start_next": start_exclusive + timedelta(days=1),
                "end_next": end_inclusive + timedelta(days=1),
            },
        ).fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def count_trading_days_from_dates(engine, starts, end_inclusive: date) -> "dict[date, int]":
    """{start: count_trading_days_between(start, end_inclusive)} for every start
    in `starts`, from ONE calendar query.

    A per-ticker staleness check over a wide set (e.g. /risk/volatility with no
    `tickers`, where delisted names spread their last score_date over many
    distinct days) would otherwise issue one ohlcv.prices range scan per
    distinct date. The trading dates in (earliest exact start, end_inclusive]
    are fetched once with the same predicate and sargable range as
    count_trading_days_between, then each start is counted by bisect. Starts
    at or after `end_inclusive` score 0 and starts past _CALENDAR_EXACT_MAX_DAYS
    take the same holiday-blind weekday count, with no query, exactly as in
    count_trading_days_between, so the two always agree.
    """
    out: dict = {}
    exact = []
    for s in set(starts):
        if s >= end_inclusive:
            out[s] = 0
        elif (end_inclusive - s).days > _CALENDAR_EXACT_MAX_DAYS:
            out[s] = _weekday_count(s, end_inclusive)
        else:
            exact.append(s)
    if exact:
        with engine.connect() as conn:
            rows = conn.execute(
                text("""
                SELECT d FROM (
                    SELECT ts_event::date AS d
                    FROM ohlcv.prices
                    WHERE ts_event >= :start_next AND ts_event < :end_next
                      AND security_type = 'equity' AND resolution = 'daily'
                    GROUP BY 1
                    HAVING COUNT(*) > 100
                ) t
                ORDER BY d
                """),
                {
                    "start_next": min(exact) + timedelta(days=1),
                    "end_next": end_inclusive + timedelta(days=1),
                },
            ).fetchall()
        days = sorted(r[0] for r in rows)
        for s in exact:
            out[s] = len(days) - bisect.bisect_right(days, s)
    return out


def missing_trading_days(engine, lo: date, hi: date, payload_days: set) -> "list[date]":
    """Trading days in [lo, hi] INCLUSIVE that carry NO rows in `payload_days`.

    Backs Meridian#401's replace_date_range completeness guard in
    upsert_daily_scores: that branch deletes every owned-factor row across the
    whole declared interval, so an in-scope trading day absent from the
    payload has its rows removed with nothing inserted to replace them. A
    containment check cannot catch that, because a payload missing an interior
    date is still entirely inside the range.

    The calendar comes from ohlcv.prices and NEVER from the payload being
    validated: a payload silently missing an interior date is, by
    construction, also absent from any calendar built out of that same
    payload, so the hole would be invisible to the check. That is the trap
    #375 hit deriving a staleness calendar from factors.group_scores (see
    count_trading_days_between above).

    Asks about SUSPECTS rather than materialising the calendar. A weekend can
    never be a trading day, and a weekday present in the payload needs no
    answer, so the only dates worth a query are weekdays the payload does not
    cover - holidays, plus any genuine gap. Cost therefore scales with how
    suspicious the payload looks, not with how wide the range is, and a
    payload covering every weekday issues NO query at all.

    That shape was chosen on measurement, not taste. The range-scan form this
    replaces aggregated the whole equity daily universe to derive a calendar:
    ohlcv.prices indexes (ts_event DESC, resolution) but not security_type, so
    a multi-year range ran as a Parallel Seq Scan. Chunking it by year was
    measured too and bought nothing. Suspect-set lookup costs seconds at
    holiday scale, well under a second for a normal one-month promotion, and
    nothing for a complete payload.

    Exactness is load-bearing here, which is why this does not take
    count_trading_days_between's cap-and-approximate path: that gate trades
    exactness for speed on a hot path, but this function backs a REFUSAL, so
    a weekday approximation misclassifying a holiday as an expected trading
    day would block a legitimate promotion rather than merely misreport.
    """
    if lo > hi:
        return []
    suspects = []
    d = lo
    while d <= hi:
        if d.weekday() < 5 and d not in payload_days:
            suspects.append(d)
        d += timedelta(days=1)
    if not suspects:
        return []
    # CANARY. The `ts_event = ANY(...)` equality below is fast because it uses
    # idx_ohlcv_ts_res, but it only matches when daily equity rows are stamped
    # at midnight. That holds today (verified against production: all
    # daily/equity rows have ts_event::time = 00:00:00), but
    # tradable_universe_size's comment in this same file asserts the opposite
    # ("stamped at the session close (16:00Z)"), so the invariant is not
    # universally believed and nothing else here pins it. If it ever stops
    # holding, this query silently matches NOTHING and the caller reads that
    # as "no gaps" - a FALSE PASS on a delete path, which is precisely the
    # bug this guard exists to prevent. So send one date the payload itself
    # covers along with the suspects and require it back: a date the recompute
    # produced scores for is a trading day with the full cross-section, so its
    # absence means the equality stopped working, not that the calendar moved.
    canary = max(payload_days) if payload_days else None
    probe = suspects if canary is None else [*suspects, canary]
    started = time.monotonic()
    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT ts_event::date AS d
                FROM ohlcv.prices
                WHERE ts_event = ANY(:suspects)
                  AND security_type = 'equity' AND resolution = 'daily'
                GROUP BY 1
                HAVING COUNT(*) > 100
                ORDER BY 1
            """),
            {"suspects": probe},
        ).fetchall()
    found = [r[0] for r in rows]
    if canary is not None and canary not in found:
        raise ValueError(
            f"missing_trading_days: calendar probe failed - {canary} is covered by the "
            "payload but ohlcv.prices returned no daily equity cross-section for it on "
            "an exact ts_event match. That means this guard cannot see the calendar, so "
            "it would report ZERO gaps for every date and let replace_date_range delete "
            "in-scope dates with nothing to replace them (Meridian#401). Most likely "
            "cause: daily equity rows are no longer stamped at midnight, which would "
            "break the ts_event equality this query relies on. Do not bypass with "
            "allow_interior_gaps - fix the calendar predicate."
        )
    suspect_set = set(suspects)
    elapsed = time.monotonic() - started
    if elapsed > 5.0:
        _log.info(
            "missing_trading_days: %s..%s tested %d suspect weekday(s) in %.1fs",
            lo,
            hi,
            len(suspects),
            elapsed,
        )
    return [d for d in found if d in suspect_set]
