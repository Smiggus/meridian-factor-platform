"""
risk_metrics.py

PostgreSQL-backed risk metrics computation: volatility, ADV, beta, idio vol.

Two paths:
1. Read pre-computed metrics from `factors.risk_metrics` (populated by Dagster).
   This is the production path - fast, PIT-correct, served from cached snapshots.
   Resolved via "latest score_date <= as_of_date" (Meridian#408), matching the
   pattern already used for factor_covariance/group_scores (#215/#375) - a
   weekend, holiday or a request made before today's phase-2 completes still
   gets the canonical asset's own numbers from the prior trading day instead
   of falling through to on-the-fly compute.

2. Fall back to on-the-fly SQL computation for any ticker still missing after
   that lookup (a ticker never scored by the asset, or no risk_metrics row
   has ever been written). Meridian#408: this used to run on CALENDAR-day
   windows with minimum-observation thresholds up to an order of magnitude
   laxer than the canonical asset, and fired silently with nothing in the
   response distinguishing it from a canonical value. It now mirrors the
   asset's own trading-day windows and floors exactly (see
   dagster/meridian_pipeline/risk_metrics_asset.py):

      metric        window / min_obs         pre-#408 fallback (for contrast)
      vol_60d       60 trading days / >=55    90 calendar days / >=20 total
      vol_20d       20 trading days / >=18    20 calendar days / no real floor
      beta_60d      60 trading days / >=55    60 calendar days / no real floor
      beta_252d     252 trading days / >=240  380 calendar days / >=30
      idio_vol_60d  60 trading days / >=55    not computed (None)
      adv_20d/60d   20/60 trading days / >=18/55 (aligned to vol's own bars)
                                               20/60 calendar days / >=5

   "Trading day" windows are anchored to the shared market calendar (SPY's
   own print dates - SPY trades every session) rather than each ticker's own
   print dates, matching the asset's shared-date-index pivot: a thinly
   traded ticker's gaps reduce ITS OWN count against the window, they do not
   stretch the window itself further back to compensate.

   Every value returned by get_risk_metrics() carries `score_date` (the date
   the numbers actually reflect) and `source` (`"canonical"` or `"fallback"`)
   so a caller can tell the two apart - additive fields, the seven original
   metric keys are unchanged.

All computations are point-in-time: filter `ts_event <= as_of_date`.

Units: vol_60d, vol_20d and idio_vol_60d are DAILY rolling std of log returns
(NOT annualised - no sqrt(252)); typical values are ~0.01-0.05. This is the
canonical basis for `factors.risk_metrics` and differs intentionally from
`factors.risk_factor_scores`, whose vol factors are annualised then z-scored.
"""

from __future__ import annotations

from datetime import date as date_cls
from functools import lru_cache

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool

from api.pg_identity import API_APPLICATION_NAME, pg_connect_args
from api_db import engine_url


@lru_cache(maxsize=1)
def _engine():
    """Process-cached, pooled engine for marketdata PostgreSQL (M7).

    Switched from a per-call NullPool engine to a process-cached bounded
    QueuePool so repeated get_risk_metrics() calls reuse connections instead
    of opening a fresh TCP+auth handshake each time. Callers use pd.read_sql,
    which checks connections out of and back into the pool automatically -
    never dispose() this engine (that would tear down the shared pool)."""
    url = engine_url()
    return create_engine(
        url,
        connect_args=pg_connect_args(API_APPLICATION_NAME),
        poolclass=QueuePool,
        pool_size=5,
        max_overflow=10,
        pool_recycle=300,
        pool_pre_ping=True,
    )


def get_risk_metrics(
    tickers: list[str],
    as_of_date: date_cls,
) -> dict[str, dict[str, float]]:
    """
    Returns a dict mapping ticker -> {vol_60d, vol_20d, adv_20d, adv_60d,
    beta_252d, beta_60d, idio_vol_60d, score_date, source}.

    `score_date` (ISO string) and `source` (`"canonical"` or `"fallback"`)
    are ADDITIVE (Meridian#408) - the seven original numeric keys are
    unchanged in name and meaning.

    First tries pre-computed metrics from `factors.risk_metrics` via a
    "latest score_date <= as_of_date" lookup (Meridian#408 - previously an
    EXACT-date match, so a weekend/holiday/pre-phase-2 request fell straight
    through to on-the-fly compute even when a perfectly good canonical row
    existed a day or two earlier). Any ticker still missing after that -
    never scored by the asset, or the table has no rows at all yet - is
    filled in from the on-the-fly fallback, which mirrors the canonical
    asset's own trading-day windows and minimum-observation floors (see the
    module docstring).
    """
    engine = _engine()
    requested = list(tickers)
    result: dict[str, dict[str, float]] = {}

    # Try pre-computed first (separate try block - swallow missing table).
    # Resolve the date FIRST, then fetch rows for it, so partial coverage on
    # the resolved date (some but not all requested tickers) is visible to
    # the caller as `source: "canonical"` for the tickers that have a row,
    # falling back only for the ones that don't.
    precomputed_date = None
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT MAX(score_date) FROM factors.risk_metrics WHERE score_date <= :d"),
                {"d": as_of_date},
            ).fetchone()
        precomputed_date = row[0] if row and row[0] is not None else None
    except Exception:
        # Table doesn't exist yet OR query failed - fall through to compute
        precomputed_date = None

    if precomputed_date is not None:
        try:
            df = pd.read_sql(
                text("""
                SELECT ticker, vol_60d, vol_20d, adv_20d, adv_60d,
                       beta_252d, beta_60d, idio_vol_60d
                FROM factors.risk_metrics
                WHERE score_date = :d AND ticker = ANY(:tickers)
            """),
                engine,
                params={"d": precomputed_date, "tickers": requested},
            )
            if not df.empty:
                result = _df_to_dict(df, score_date=precomputed_date, source="canonical")
        except Exception:
            pass

    # Fall back to on-the-fly computation for whatever wasn't served above.
    # The engine is a shared cached pool (M7) - do NOT dispose() it;
    # pd.read_sql returns connections to the pool.
    missing = [t for t in requested if t not in result]
    if missing:
        try:
            fallback = _compute_on_the_fly(engine, missing, as_of_date)
        except Exception as e:
            print(f"[risk_metrics] on-the-fly compute failed: {e}")
            fallback = {}
        result.update(fallback)

    return result


def _df_to_dict(
    df: pd.DataFrame,
    *,
    score_date: date_cls | None = None,
    source: str | None = None,
) -> dict[str, dict[str, float]]:
    """Convert risk metrics DataFrame to dict keyed by ticker.

    `score_date` and `source` (Meridian#408, ADDITIVE) are stamped onto every
    ticker's entry when given, so a caller can tell a canonical value (from
    `factors.risk_metrics`, written by the Dagster asset under its own
    trading-day windows/min-obs) from a fallback value (on-the-fly, parity-
    aligned since #408 but computed from whatever data happens to be
    available rather than the asset's universe/hygiene filters). The seven
    numeric keys are unchanged - this only adds keys to each entry.
    """
    out = {}
    for _, row in df.iterrows():
        entry = {
            k: (None if pd.isna(row[k]) else float(row[k]))
            for k in [
                "vol_60d",
                "vol_20d",
                "adv_20d",
                "adv_60d",
                "beta_252d",
                "beta_60d",
                "idio_vol_60d",
            ]
        }
        if score_date is not None:
            entry["score_date"] = (
                score_date.isoformat() if hasattr(score_date, "isoformat") else str(score_date)
            )
        if source is not None:
            entry["source"] = source
        out[row["ticker"]] = entry
    return out


# Window sizes (in trading days) and minimum-observation floors, mirroring
# dagster/meridian_pipeline/risk_metrics_asset.py's `win60`/`win20`/`win252`
# slices and their `_window_stats(..., min_obs=...)` calls exactly
# (Meridian#408). Kept as named constants so the parity claim is checkable
# in one place instead of scattered magic numbers.
_WINDOW_20D = 20
_WINDOW_60D = 60
_WINDOW_252D = 252
_MIN_OBS_20D = 18
_MIN_OBS_60D = 55
_MIN_OBS_252D = 240


def _trading_day_calendar(engine, as_of_date: date_cls, n: int = 252) -> list:
    """Up to the `n` most recent distinct trading days on or before
    as_of_date, most-recent-first, using SPY's own print dates as the shared
    market calendar (SPY trades every session `ohlcv.prices` covers).

    This anchors fallback window boundaries to the SAME kind of shared date
    axis the canonical asset builds via `prices.pivot_table(index="d", ...)`
    (Meridian#408) - a thinly-traded ticker's own gaps reduce ITS count
    against a fixed window, they never stretch the window itself further
    back in time to compensate, which is what a naive per-ticker "last N
    rows this ticker has" window would silently do.
    """
    df = pd.read_sql(
        text("""
            SELECT DISTINCT ts_event::date AS d
            FROM ohlcv.prices
            WHERE ticker = 'SPY' AND resolution = 'daily' AND ts_event <= :d
            ORDER BY d DESC
            LIMIT :n
        """),
        engine,
        params={"d": as_of_date, "n": n},
    )
    return df["d"].tolist()


def _compute_on_the_fly(
    engine,
    tickers: list[str],
    as_of_date: date_cls,
) -> dict[str, dict[str, float]]:
    """
    Compute risk metrics from raw ohlcv.prices for tickers with no
    `factors.risk_metrics` coverage on or before as_of_date.

    Meridian#408: window boundaries and minimum-observation floors mirror
    the canonical asset exactly (see the module docstring table) instead of
    the previous calendar-day windows with far laxer thresholds. Returned
    entries carry `source="fallback"` (stamped by the caller, get_risk_
    metrics) so this is never confused with a canonical value.

    Returns are sourced from ret_1d (dividend-corrected, split-invariant,
    PIT-safe). Dollar volume uses raw_close * volume for a consistent
    notional (raw price * raw volume = actual traded dollar value).
    """
    tickers_list = list(tickers)
    if not tickers_list:
        return {}

    calendar = _trading_day_calendar(engine, as_of_date, n=_WINDOW_252D)
    if not calendar:
        # No market calendar at all on or before this date - nothing to
        # compute (e.g. as_of_date predates any OHLCV history).
        return {}

    d60 = calendar[min(_WINDOW_60D - 1, len(calendar) - 1)]
    d20 = calendar[min(_WINDOW_20D - 1, len(calendar) - 1)]
    d252 = calendar[-1]

    # Volatility (60d, 20d) from LN(1 + ret_1d), windows bounded by the
    # trading-day calendar above rather than a calendar-day INTERVAL. Counts
    # are fetched per window and the min_obs floor applied in pandas below -
    # HAVING can only gate on COUNT(*) for the whole GROUP BY row, which is
    # exactly how the old query let vol_20d through on the 90-day window's
    # count instead of its own.
    vol_df = pd.read_sql(
        text("""
        WITH returns AS (
            SELECT ticker, ts_event::date AS d,
                   LN(1.0 + ret_1d) AS r
            FROM ohlcv.prices
            WHERE resolution = 'daily' AND ts_event <= :d
              AND ts_event >= :d60
              AND ticker = ANY(:tickers)
              AND ret_1d IS NOT NULL
        )
        SELECT ticker,
               STDDEV(r) AS vol_60d,
               COUNT(r) AS n60,
               STDDEV(r) FILTER (WHERE d >= :d20) AS vol_20d,
               COUNT(r) FILTER (WHERE d >= :d20) AS n20
        FROM returns
        GROUP BY ticker
    """),
        engine,
        params={"d": as_of_date, "d60": d60, "d20": d20, "tickers": tickers_list},
    )
    vol_df["vol_60d"] = vol_df["vol_60d"].where(vol_df["n60"] >= _MIN_OBS_60D)
    vol_df["vol_20d"] = vol_df["vol_20d"].where(vol_df["n20"] >= _MIN_OBS_20D)
    vol_df = vol_df.drop(columns=["n60", "n20"])

    # ADV (20d, 60d) using raw_close * volume for consistent notional. Same
    # trading-day windows as vol above; no ret_1d requirement (a print with
    # volume but no return, e.g. a new listing's first bar, still counts).
    # min_obs aligned to vol's own floors at the same window length - the
    # canonical asset has no per-window ADV floor beyond the universe's
    # "60 daily bars before score_date" gate, so this is a deliberately
    # conservative choice for the fallback rather than a literal mirror.
    adv_df = pd.read_sql(
        text("""
        SELECT ticker,
               AVG(raw_close * volume) FILTER (WHERE ts_event >= :d20) AS adv_20d,
               COUNT(*) FILTER (WHERE ts_event >= :d20) AS n20,
               AVG(raw_close * volume) AS adv_60d,
               COUNT(*) AS n60
        FROM ohlcv.prices
        WHERE resolution = 'daily' AND ts_event <= :d
          AND ts_event >= :d60
          AND ticker = ANY(:tickers)
        GROUP BY ticker
    """),
        engine,
        params={"d": as_of_date, "d60": d60, "d20": d20, "tickers": tickers_list},
    )
    adv_df["adv_20d"] = adv_df["adv_20d"].where(adv_df["n20"] >= _MIN_OBS_20D)
    adv_df["adv_60d"] = adv_df["adv_60d"].where(adv_df["n60"] >= _MIN_OBS_60D)
    adv_df = adv_df.drop(columns=["n20", "n60"])

    # Beta (252d, 60d) + idio_vol_60d vs SPY using LN(1 + ret_1d). idio_vol_60d
    # uses the residual-variance identity Var(r - beta*m) = Var(r) - beta^2 *
    # Var(m), which is algebraically identical (same sum-of-squares algebra)
    # regardless of whether Var/Cov use the population (n) or sample (n-1)
    # normalisation, AS LONG AS beta and both variances use the SAME one.
    # beta_60d itself is normalisation-invariant (COVAR_POP/VAR_POP - the n
    # cancels in the ratio), but var_r60/var_m60 below are VAR_SAMP (n-1) on
    # purpose: the canonical asset's idio_vol (risk_metrics_asset.py
    # `_window_stats`) divides by (n-1), same as vol_60d's own STDDEV
    # (Postgres STDDEV = STDDEV_SAMP) - using VAR_POP here instead would
    # under-state idio_vol_60d by a systematic factor of sqrt((n-1)/n)
    # (~1% at min_obs=55) relative to both of those. Same 60d window and
    # min_obs as beta_60d, so it is no longer hardcoded to None.
    beta_df = pd.read_sql(
        text("""
        WITH stock_rets AS (
            SELECT ticker, ts_event::date AS d,
                   LN(1.0 + ret_1d) AS r
            FROM ohlcv.prices
            WHERE resolution = 'daily' AND ts_event <= :d
              AND ts_event >= :d252
              AND ticker = ANY(:tickers)
              AND ret_1d IS NOT NULL
        ),
        spy_rets AS (
            SELECT ts_event::date AS d,
                   LN(1.0 + ret_1d) AS m_r
            FROM ohlcv.prices
            WHERE ticker = 'SPY' AND resolution = 'daily'
              AND ts_event <= :d AND ts_event >= :d252
              AND ret_1d IS NOT NULL
        ),
        joined AS (
            SELECT s.ticker, s.d, s.r, m.m_r
            FROM stock_rets s
            JOIN spy_rets m ON s.d = m.d
        )
        SELECT ticker,
               COVAR_POP(r, m_r) / NULLIF(VAR_POP(m_r), 0) AS beta_252d,
               COUNT(*) AS n252,
               (COVAR_POP(r, m_r) FILTER (WHERE d >= :d60))
                 / NULLIF(VAR_POP(m_r) FILTER (WHERE d >= :d60), 0) AS beta_60d,
               VAR_SAMP(r) FILTER (WHERE d >= :d60) AS var_r60,
               VAR_SAMP(m_r) FILTER (WHERE d >= :d60) AS var_m60,
               COUNT(*) FILTER (WHERE d >= :d60) AS n60
        FROM joined
        GROUP BY ticker
    """),
        engine,
        params={"d": as_of_date, "d252": d252, "d60": d60, "tickers": tickers_list},
    )
    beta_df["beta_252d"] = beta_df["beta_252d"].where(beta_df["n252"] >= _MIN_OBS_252D)
    ok60 = beta_df["n60"] >= _MIN_OBS_60D
    beta_df["beta_60d"] = beta_df["beta_60d"].where(ok60)
    resid_var = (beta_df["var_r60"] - beta_df["beta_60d"] ** 2 * beta_df["var_m60"]).clip(lower=0)
    beta_df["idio_vol_60d"] = resid_var.pow(0.5).where(ok60)
    beta_df = beta_df.drop(columns=["n252", "n60", "var_r60", "var_m60"])

    # Merge by ticker
    merged = (
        vol_df.set_index("ticker")
        .join(adv_df.set_index("ticker"), how="outer")
        .join(beta_df.set_index("ticker"), how="outer")
    )
    merged.reset_index(inplace=True)

    return _df_to_dict(merged, score_date=as_of_date, source="fallback")
