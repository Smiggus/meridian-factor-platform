"""
factor_store.py

Reads factor scores from PostgreSQL (marketdata.factors schema).
Used by the Meridian API to serve point-in-time factor scores for:
  - Single-stock lookups (live trading: GET /scores?symbol=AAPL)
  - Historical backtesting (GET /scores?symbol=AAPL&date=2024-06-01)
  - Universe bulk queries (GET /universe/scores)

Also provides the submit_request() function to trigger Dagster
materialization for missing dates (API-driven on-demand scoring).

Connects to the marketdata database (PGHOST, port 5432).
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool

from api.pg_identity import API_APPLICATION_NAME, pg_connect_args
from api_db import engine_url

logger = logging.getLogger(__name__)

# Make the dagster/ subtree importable without installing it as a package.
# Same trick api/factor_catalogue.py and api/routers/risk.py already use;
# lakehouse.py itself only imports stdlib at module scope (pyiceberg is
# imported lazily inside its functions), so this import is always cheap and
# never requires pyiceberg to be installed (M14.4).
_DAGSTER_DIR = Path(__file__).resolve().parent / "dagster"
if str(_DAGSTER_DIR) not in sys.path:
    sys.path.insert(0, str(_DAGSTER_DIR))

from meridian_pipeline import lakehouse


def _iceberg_first(pg_read: Callable[[], pd.DataFrame], iceberg_read: Callable[[], pd.DataFrame]):
    """Iceberg-first / Postgres-fallback choke point for factors.daily_scores
    reads (M14.4). Both callables are zero-arg and must return a DataFrame in
    the exact shape the caller expects downstream.

    Gated by lakehouse.iceberg_read_primary_api() / MERIDIAN_ICEBERG_API_READ -
    an env var STRUCTURALLY SEPARATE from MERIDIAN_ICEBERG_READ_PRIMARY (the
    research-substrate switch parquet_io.py reads). This is on purpose: doc 18
    ("Superseded decisions") records that routing the serving API through
    Iceberg was explored and REJECTED (point-lookup regression vs PG; no
    OIDC path for downstream tenants). Do not set MERIDIAN_ICEBERG_API_READ in
    any deployment without a fresh decision superseding that rejection -
    docs/06-CRITICAL_RULES.md Absolute Rule 1 requires the API stay read-only against
    PostgreSQL.

    Behaviour is byte-for-byte identical to pure Postgres when
    MERIDIAN_ICEBERG_API_READ is unset/false (the default) - the Iceberg
    branch, and even the daily_scores_available() reachability probe, is never
    evaluated in that case. Any exception raised while reading Iceberg is
    logged and swallowed; the Postgres read always runs as the fallback.
    """
    if lakehouse.iceberg_read_primary_api() and lakehouse.daily_scores_available():
        try:
            return iceberg_read()
        except Exception as exc:
            logger.warning("factor_store: iceberg read failed, falling back to PG: %s", exc)
    return pg_read()


def _engine_url() -> str:
    return engine_url()


@lru_cache(maxsize=1)
def _engine():
    """Process-cached, pooled engine (audit batch D #1 - mirrors the M7
    pattern already used in api/routers/dcf.py, admin.py, and risk.py).
    factor_store used to build a fresh NullPool engine on every call, so
    every core read (get_stock_factors, get_universe_scores,
    get_scoreable_universe, get_regime, get_sentiment, ...) paid one full
    TCP+auth handshake per pd.read_sql - multiplied per request, and again
    per ticker for a QC universe sweep. A bounded QueuePool lets concurrent
    requests share pooled connections instead. Never dispose() this cached
    engine - pd.read_sql / engine.connect() check connections in and out of
    the pool automatically."""
    return create_engine(
        _engine_url(),
        connect_args=pg_connect_args(API_APPLICATION_NAME),
        poolclass=QueuePool,
        pool_size=5,
        max_overflow=10,
        pool_recycle=300,
        pool_pre_ping=True,
    )


# Single stock factor lookup

_ngroups_present: "bool | None" = None


def _composite_ngroups_present(engine) -> bool:
    """#329: cached probe - does factors.composite_scores carry the additive
    n_groups column on this DB? The column self-applies when composite_scores next
    runs (db.ensure_composite_ngroups_column), so the API may deploy before it
    exists; a False here just serves n_groups: null (merged != applied)."""
    global _ngroups_present
    if _ngroups_present is None:
        try:
            # Schema probe via a lightweight connection.execute, not pd.read_sql:
            # it runs once per process (cached) and must not sit in the per-request
            # read_sql sequence that callers/tests drive.
            with engine.connect() as conn:
                row = conn.execute(
                    text(
                        "SELECT 1 FROM information_schema.columns "
                        "WHERE table_schema='factors' AND table_name='composite_scores' "
                        "AND column_name='n_groups'"
                    )
                ).first()
            _ngroups_present = row is not None
        except Exception:
            return False
    return _ngroups_present


def get_stock_factors(ticker: str, score_date: date) -> Optional[dict]:
    """
    Returns full factor detail for one stock on one date.
    Used by GET /scores?symbol=AAPL&date=2024-06-01.

    Returns dict with:
        factors: {factor_name: {raw_value, z_score, direction, group_name}}
        group_scores: {group_name: score}
        composite_score: float
        quintile: int
    Or None if no data exists for this date.
    """
    engine = _engine()

    # Factor scores (M14.4: Iceberg-first, PG fallback - see _iceberg_first)
    _cols = ["factor_name", "group_name", "raw_value", "z_score", "direction"]
    daily = _iceberg_first(
        pg_read=lambda: pd.read_sql(
            text("""
            SELECT factor_name, group_name, raw_value, z_score, direction
            FROM factors.daily_scores
            WHERE score_date = :d AND ticker = :t
        """),
            engine,
            params={"d": score_date, "t": ticker},
        ),
        iceberg_read=lambda: lakehouse.read_daily_scores(
            score_date=score_date, tickers=[ticker], columns=_cols, as_pandas=True
        )[_cols],
    )

    if daily.empty:
        return None

    factors = {}
    for _, row in daily.iterrows():
        factors[row["factor_name"]] = {
            "raw": _safe(row["raw_value"]),
            "z_score": _safe(row["z_score"]),
            "direction": int(row["direction"]),
            "group": row["group_name"],
        }

    # Group scores
    groups = pd.read_sql(
        text("""
        SELECT group_name, group_score
        FROM factors.group_scores
        WHERE score_date = :d AND ticker = :t
    """),
        engine,
        params={"d": score_date, "t": ticker},
    )

    group_scores = {row["group_name"]: _safe(row["group_score"]) for _, row in groups.iterrows()}

    # Composite + quintile (+ n_groups additively, #329)
    ng_col = ", n_groups" if _composite_ngroups_present(engine) else ""
    comp = pd.read_sql(
        text(f"""
        SELECT composite_score, quintile{ng_col},
               to_jsonb(c)->'blend_provenance' AS blend_provenance
        FROM factors.composite_scores c
        WHERE score_date = :d AND ticker = :t
    """),
        engine,
        params={"d": score_date, "t": ticker},
    )

    composite = _safe(comp.iloc[0]["composite_score"]) if not comp.empty else None
    quintile = (
        int(comp.iloc[0]["quintile"])
        if not comp.empty and pd.notna(comp.iloc[0]["quintile"])
        else None
    )
    n_groups = (
        int(comp.iloc[0]["n_groups"])
        if not comp.empty and "n_groups" in comp.columns and pd.notna(comp.iloc[0]["n_groups"])
        else None
    )

    return {
        "factors": factors,
        "group_scores": group_scores,
        "composite_score": composite,
        "quintile": quintile,
        "n_groups": n_groups,
        "n_weighted_groups": _n_weighted_groups(
            comp.iloc[0].get("blend_provenance") if not comp.empty else None
        ),
        "blend_provenance": comp.iloc[0].get("blend_provenance") if not comp.empty else None,
    }


def _n_weighted_groups(provenance) -> Optional[int]:
    """RANK-003: positive-weight non-null blend contributors, read from the
    per-row blend_provenance jsonb. None for rows written before the field
    existed (not backfilled) or with no captured provenance."""
    if isinstance(provenance, str):
        try:
            provenance = json.loads(provenance)
        except ValueError:
            return None
    if not isinstance(provenance, dict):
        return None
    n = provenance.get("n_weighted_groups")
    return int(n) if isinstance(n, (int, float)) and not isinstance(n, bool) else None


def get_universe_scores(score_date: date) -> Optional[list[dict]]:
    """
    Returns composite + group scores for all tickers on a date.
    Used by GET /universe/scores?date=2024-06-01.

    Includes a `regime_factors` block per ticker (empty in this export).
    """
    engine = _engine()

    ng_col = ", c.n_groups" if _composite_ngroups_present(engine) else ""
    comp = pd.read_sql(
        text(f"""
        SELECT c.ticker, c.composite_score, c.quintile{ng_col},
               to_jsonb(c)->'blend_provenance' AS blend_provenance,
               g.group_name, g.group_score
        FROM factors.composite_scores c
        LEFT JOIN factors.group_scores g
            ON c.score_date = g.score_date AND c.ticker = g.ticker
        WHERE c.score_date = :d
        ORDER BY c.composite_score DESC
    """),
        engine,
        params={"d": score_date},
    )

    if comp.empty:
        return None

    has_ng = "n_groups" in comp.columns
    # Pivot group scores into columns
    records = {}
    for _, row in comp.iterrows():
        ticker = row["ticker"]
        if ticker not in records:
            records[ticker] = {
                "symbol": ticker,
                "composite_score": _safe(row["composite_score"]),
                "quintile": int(row["quintile"]) if pd.notna(row["quintile"]) else None,
                "n_groups": int(row["n_groups"]) if has_ng and pd.notna(row["n_groups"]) else None,
                "n_weighted_groups": _n_weighted_groups(row.get("blend_provenance")),
                "blend_provenance": row.get("blend_provenance"),
            }
        if pd.notna(row["group_name"]):
            records[ticker][row["group_name"]] = _safe(row["group_score"])

    # The response contract carries a per-ticker `regime_factors` block. This
    # export ships it empty: the regime factors are not published.
    for rec in records.values():
        rec["regime_factors"] = {}

    return sorted(records.values(), key=lambda x: x.get("composite_score") or 0, reverse=True)


def get_scoreable_universe(
    as_of: Optional[date] = None,
    excluded_sectors: Optional[list[str]] = None,
    *,
    allow_spx_fallback: bool = True,
) -> list[str]:
    """
    Returns the tradable universe from PostgreSQL (no FMP calls).

    The scoreable universe is the set of tickers with a row in
    factors.daily_scores on the given date, minus any ticker whose
    fundamentals.company_profile.sector is in excluded_sectors.

    Parameters
    ----------
    as_of : date | None
        PIT membership date. None uses the most recent date with scores.
    excluded_sectors : list[str] | None
        Sectors to exclude. Defaults to the configured EXCLUDED_SECTORS
        to preserve existing behaviour.
    allow_spx_fallback : bool
        When True (default - preserves historical behaviour), a date with
        zero rows in factors.daily_scores falls back to SPX membership
        (universes.constituents, PIT-correct). When False, return an empty
        list instead so the caller can 404 rather than silently serving SPX
        membership as "tickers with scores" for that date (audit batch D #3:
        api/routers/universe.py is the only remaining caller, and its
        contract documents this list as scored tickers, not index
        membership - a weekend/holiday date or a backfill gap otherwise
        serves 200 with ~500 SPX tickers instead of 404). Every fallback
        trigger is logged regardless of this flag.

    Returns
    -------
    Sorted list of ticker symbols. Empty list if no scores exist for
    the requested date (caller can fall back to FMP in that case).
    """
    if excluded_sectors is None:
        try:
            from config import EXCLUDED_SECTORS

            excluded_sectors = list(EXCLUDED_SECTORS)
        except ImportError:
            excluded_sectors = []

    engine = _engine()
    params: dict = {"excl": excluded_sectors}
    if as_of is None:
        # "latest date" requires a MAX(score_date) subquery - not cleanly
        # separable against Iceberg without an extra full-table scan, so
        # this branch always stays on Postgres (M14.4 scope decision).
        sql = """
            WITH latest AS (
                SELECT MAX(score_date) AS d FROM factors.daily_scores
            )
            SELECT DISTINCT f.ticker
            FROM factors.daily_scores f
            JOIN latest ON f.score_date = latest.d
            LEFT JOIN fundamentals.company_profile p USING (ticker)
            WHERE p.sector IS NULL OR NOT (p.sector = ANY(:excl))
            ORDER BY f.ticker
        """
        df = pd.read_sql(text(sql), engine, params=params)
    else:
        # M14.4: Iceberg-first, PG fallback for the daily_scores portion.
        # The sector-exclusion join has no Iceberg equivalent (sector data
        # lives only in Postgres fundamentals.company_profile), so the
        # Iceberg path pulls distinct tickers scored on `as_of` from
        # Iceberg, then applies the same sector exclusion as a small,
        # ticker-scoped Postgres follow-up query - same net filtering as
        # the single-query PG path, same output shape (a "ticker" column).
        params["d"] = as_of
        pg_sql = """
            SELECT DISTINCT f.ticker
            FROM factors.daily_scores f
            LEFT JOIN fundamentals.company_profile p USING (ticker)
            WHERE f.score_date = :d
              AND (p.sector IS NULL OR NOT (p.sector = ANY(:excl)))
            ORDER BY f.ticker
        """

        def _iceberg_scoreable_tickers() -> pd.DataFrame:
            ice = lakehouse.read_daily_scores(score_date=as_of, columns=["ticker"], as_pandas=True)
            all_tickers = sorted(ice["ticker"].dropna().unique().tolist())
            if not all_tickers or not excluded_sectors:
                return pd.DataFrame({"ticker": all_tickers})
            excl_df = pd.read_sql(
                text("""
                    SELECT ticker FROM fundamentals.company_profile
                    WHERE ticker = ANY(:t) AND sector = ANY(:excl)
                """),
                engine,
                params={"t": all_tickers, "excl": excluded_sectors},
            )
            excl_set = set(excl_df["ticker"].tolist())
            return pd.DataFrame({"ticker": [t for t in all_tickers if t not in excl_set]})

        df = _iceberg_first(
            pg_read=lambda: pd.read_sql(text(pg_sql), engine, params=params),
            iceberg_read=_iceberg_scoreable_tickers,
        )
    tickers = df["ticker"].tolist()
    if tickers:
        return tickers

    logger.warning(
        "get_scoreable_universe: factors.daily_scores has zero rows for as_of=%s (%s); %s",
        as_of.isoformat() if as_of is not None else "latest",
        "explicit date" if as_of is not None else "MAX(score_date)",
        "falling back to SPX membership (universes.constituents)"
        if allow_spx_fallback
        else "allow_spx_fallback=False, returning empty list",
    )
    if not allow_spx_fallback:
        return []

    # Fallback: SPX membership from universes.constituents (PIT-correct via
    # start_date/end_date). Used when factors.daily_scores has no rows for
    # the requested date - typically a brand-new deployment or a future
    # date before LDP's next materialisation.
    spx_params: dict = {"excl": excluded_sectors}
    if as_of is None:
        spx_sql = """
            SELECT DISTINCT c.ticker
            FROM universes.constituents c
            LEFT JOIN fundamentals.company_profile p USING (ticker)
            WHERE c.universe_id = 'SPX'
              AND c.end_date IS NULL
              AND (p.sector IS NULL OR NOT (p.sector = ANY(:excl)))
            ORDER BY c.ticker
        """
    else:
        spx_params["d"] = as_of
        spx_sql = """
            SELECT DISTINCT c.ticker
            FROM universes.constituents c
            LEFT JOIN fundamentals.company_profile p USING (ticker)
            WHERE c.universe_id = 'SPX'
              AND c.start_date <= :d
              AND (c.end_date IS NULL OR c.end_date > :d)
              AND (p.sector IS NULL OR NOT (p.sector = ANY(:excl)))
            ORDER BY c.ticker
        """
    df = pd.read_sql(text(spx_sql), engine, params=spx_params)
    return df["ticker"].tolist()


def has_scores(score_date: date) -> bool:
    """Check if factor scores exist for a given date in PostgreSQL."""
    engine = _engine()
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT 1 FROM factors.composite_scores WHERE score_date = :d LIMIT 1"),
            {"d": score_date},
        ).fetchone()
    return row is not None


def get_available_dates(limit: int = 30) -> list[date]:
    """Returns the most recent dates that have factor scores."""
    engine = _engine()
    df = pd.read_sql(
        text("""
        SELECT DISTINCT score_date FROM factors.composite_scores
        ORDER BY score_date DESC LIMIT :n
    """),
        engine,
        params={"n": limit},
    )
    return df["score_date"].tolist()


# On-demand compute request (triggers Dagster)


def submit_compute_request(score_date: date) -> bool:
    """
    Submit a request to compute factor scores for a missing date.
    The Dagster request_sensor polls this table every 30s and triggers
    materialization for pending dates.

    Returns True if a new request was submitted, False if already exists.
    """
    engine = _engine()
    with engine.begin() as conn:
        result = conn.execute(
            text("""
            INSERT INTO factors.compute_requests (score_date, status)
            VALUES (:d, 'pending')
            ON CONFLICT (score_date) DO NOTHING
            RETURNING id
        """),
            {"d": score_date},
        )
        return result.fetchone() is not None


def get_request_status(score_date: date) -> Optional[str]:
    """Returns the compute request status for a date (pending/computing/done/failed)."""
    engine = _engine()
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT status FROM factors.compute_requests WHERE score_date = :d"),
            {"d": score_date},
        ).fetchone()
    return row[0] if row else None


def _safe(val) -> Optional[float]:
    if val is None:
        return None
    try:
        f = float(val)
        import math

        return round(f, 4) if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None
