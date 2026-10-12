"""
lakehouse_parity.py - PG-vs-Iceberg parity monitor for factors.daily_scores
(Meridian M14.6).

The mirror-faithfulness check for the medallion architecture
(docs/18-ICEBERG_MIGRATION.md): PG `factors.*` is gold-serving-primary and
authoritative; the Iceberg `meridian` warehouse is the lab layer, including a
best-effort mirror of daily_scores. This module compares cheap per-date
aggregate stats (row/factor/ticker counts + z/raw checksums) computed
independently from PostgreSQL (authoritative) and from
meridian.factors.daily_scores (lakehouse.py, dual-written since M14.2) to
verify the lab mirror stays faithful to gold.

Import-light by design: this module imports pandas/sqlalchemy at module
scope (both are hard deps of the whole pipeline already) but only reaches
into `lakehouse` lazily, inside functions, so it never forces a pyiceberg
import (and the OIDC/MinIO env requirement that comes with it) just to be
imported. The Dagster asset in lakehouse_parity_asset.py is the only regular
caller; the CLI at the bottom of this module is for ad-hoc runs.

FINITE FILTER PARITY (read this before touching the SQL):
lakehouse._to_daily_scores_arrow nulls out non-finite raw_value/z_score
before writing to Iceberg (matches db.upsert_daily_scores's own NaN/inf
sanitisation - CRITICAL_RULES rule 9 lineage). So daily_scores_parity_stats
sums only the non-null (finite) values Iceberg actually stored. The PG query
below must apply the equivalent finite filter itself: PostgreSQL's own
NaN/Infinity float8 literals are compared with `= 'NaN'`/`= 'Infinity'`, and
`x = x` is false for NaN, so `raw_value = raw_value AND raw_value NOT IN
('Infinity', '-Infinity')` isolates exactly the finite population.
"""

import logging
from datetime import date, timedelta
from typing import Any

from sqlalchemy import text

logger = logging.getLogger(__name__)

_STATS_KEYS: tuple[str, ...] = ("rows", "factors", "tickers", "z_checksum", "raw_checksum")

# PG query: counts are exact (all rows, regardless of finiteness - matches
# Iceberg's arrow.num_rows / distinct counts, which are computed over ALL
# scanned rows since only the checksums null out non-finite values).
_PG_STATS_SQL = text("""
    SELECT
        COUNT(*) AS rows,
        COUNT(DISTINCT factor_name) AS factors,
        COUNT(DISTINCT ticker) AS tickers,
        SUM(z_score) FILTER (
            WHERE z_score = z_score
              AND z_score NOT IN ('Infinity'::float8, '-Infinity'::float8)
        ) AS z_checksum,
        SUM(raw_value) FILTER (
            WHERE raw_value = raw_value
              AND raw_value NOT IN ('Infinity'::float8, '-Infinity'::float8)
        ) AS raw_checksum
    FROM factors.daily_scores
    WHERE score_date = :d
""")


def pg_daily_scores_stats(engine, score_date: date) -> dict[str, Any]:
    """PostgreSQL-side counterpart to lakehouse.daily_scores_parity_stats.

    Returns the same {rows, factors, tickers, z_checksum, raw_checksum} shape,
    with checksums rounded to 6dp and summed over the finite population only
    so the two sides compare like-for-like.
    """
    with engine.connect() as conn:
        row = conn.execute(_PG_STATS_SQL, {"d": score_date}).fetchone()

    if row is None:
        return {"rows": 0, "factors": 0, "tickers": 0, "z_checksum": 0.0, "raw_checksum": 0.0}

    rows, factors, tickers, z_checksum, raw_checksum = row
    return {
        "rows": int(rows or 0),
        "factors": int(factors or 0),
        "tickers": int(tickers or 0),
        "z_checksum": round(float(z_checksum), 6) if z_checksum is not None else 0.0,
        "raw_checksum": round(float(raw_checksum), 6) if raw_checksum is not None else 0.0,
    }


_PG_MAX_DATE_SQL = text("""
    SELECT MAX(score_date) FROM factors.daily_scores WHERE score_date >= :floor
""")


def pg_daily_scores_max_date(engine, floor_date: date) -> "date | None":
    """Bounded MAX(score_date) over PostgreSQL factors.daily_scores (#415).

    score_date leads the PK so MAX(score_date) is cheap even unbounded, but
    the floor keeps this query's shape identical to the Iceberg side's
    partition-pruned scan (lakehouse.daily_scores_max_date) and guarantees
    neither side ever has to answer for the full table (docs/06-CRITICAL_RULES.md
    rule: never aggregate factors.daily_scores on an unindexed dimension -
    this one leads the PK, but bounding by date is still the cheaper, more
    explicit query and matches the paired Iceberg call).
    """
    with engine.connect() as conn:
        row = conn.execute(_PG_MAX_DATE_SQL, {"floor": floor_date}).fetchone()
    return row[0] if row and row[0] else None


def staleness_report(
    pg_max: "date | None",
    ice_max: "date | None",
    is_trading_day,
    max_lag_trading_days: int = 0,
) -> dict[str, Any]:
    """Pure staleness comparator (#415): independent of the per-date trailing
    window in parity_report, and independent of the failing dual-write path.

    A per-date count/checksum comparison over a fixed trailing window only
    ever looks at dates BOTH sides already claim to have. It cannot see a
    mirror that is perfectly consistent for every date it holds and simply
    has stopped receiving new ones - exactly the gap the sibling LDP checks
    have (their `ohlcv_tablewide` state: consistent history, silently absent
    newest days). This compares the two sides' MAX(score_date) directly.

    `is_trading_day` is a callable(date) -> bool, injected rather than an
    engine, so this function has no DB/Iceberg dependency and is unit-
    testable with fakes for both sides (see tests/test_lakehouse_parity.py).

    max_lag_trading_days is the number of trading days the mirror is allowed
    to trail PG's latest date before this reports not-ok. Default 0: the
    dual-write happens synchronously inside the same daily-pipeline step
    that writes PG (db.upsert_daily_scores), so once a date is the anchor's
    "latest" (already fully phase-2-complete in PG - see
    lakehouse_parity_asset.py), the mirror should already have it. Any lag
    is exactly the signature of #415 (a best-effort dual-write that silently
    took the log-and-continue path), not an expected race.
    """
    if pg_max is None:
        return {"ok": True, "pg_max": None, "ice_max": ice_max, "lag_trading_days": 0}

    if ice_max is None:
        return {"ok": False, "pg_max": pg_max, "ice_max": None, "lag_trading_days": None}

    if ice_max >= pg_max:
        return {"ok": True, "pg_max": pg_max, "ice_max": ice_max, "lag_trading_days": 0}

    lag = 0
    cursor = ice_max + timedelta(days=1)
    while cursor <= pg_max:
        if is_trading_day(cursor):
            lag += 1
        cursor += timedelta(days=1)

    return {
        "ok": lag <= max_lag_trading_days,
        "pg_max": pg_max,
        "ice_max": ice_max,
        "lag_trading_days": lag,
    }


def staleness_check(
    engine,
    pg_latest: date,
    lookback_days: int = 30,
    max_lag_trading_days: int = 0,
) -> dict[str, Any]:
    """Wrapper around staleness_report that fetches both sides' bounded
    MAX(score_date) (#415). `pg_latest` anchors the floor (lookback_days
    calendar days back from it) so both bounded scans - PG's indexed query
    and Iceberg's partition-pruned scan - share the same bounded window.
    30 calendar days is generous relative to the 5-trading-day window
    parity_report uses elsewhere in this module: staleness only needs to
    find the mirror's actual max date, not compare every date in between, so
    a wider floor costs nothing extra (Iceberg partition pruning + PG's PK)
    while giving room for a mirror that has fallen behind by more than a
    week without the check itself going blind.

    Never raises - a query failure on either side is reported as a
    not-ok finding with an `error` key, matching parity_report's contract.
    """
    from meridian_pipeline import db, lakehouse

    floor_date = pg_latest - timedelta(days=lookback_days)

    try:
        pg_max = pg_daily_scores_max_date(engine, floor_date)
    except Exception as exc:
        logger.warning("staleness_check: PG max(score_date) query failed: %s", exc)
        return {"ok": False, "error": f"pg_query_failed: {exc}"}

    try:
        ice_max = lakehouse.daily_scores_max_date(floor_date)
    except Exception as exc:
        logger.warning("staleness_check: Iceberg max(score_date) query failed: %s", exc)
        return {"ok": False, "error": f"iceberg_query_failed: {exc}"}

    return staleness_report(
        pg_max,
        ice_max,
        lambda d: db.is_trading_day(engine, d),
        max_lag_trading_days=max_lag_trading_days,
    )


def compare_stats(
    pg: dict[str, Any],
    ice: dict[str, Any],
    *,
    checksum_tol: float = 1e-3,
    checksum_rel_tol: float = 1e-9,
) -> dict[str, Any]:
    """Compare a PG stats dict against an Iceberg stats dict.

    rows/factors/tickers must match exactly. z_checksum/raw_checksum are
    compared with a SCALE-AWARE tolerance: float summation order differs
    between the two engines, and the drift is proportional to the magnitude
    of the sum (summing ~1e5 raw_value ~1e12 values accumulates ~N*eps*scale
    of reordering noise, which for raw market-cap-scale sums is O(0.1-1) in
    absolute terms - far above any fixed 1e-3 floor but ~1e-13 in relative
    terms). So a checksum passes when the difference is within
    max(checksum_tol, checksum_rel_tol * max(|pg|, |ice|)): the absolute floor
    catches near-zero sums (z_score is roughly cross-sectionally mean-zero),
    the relative band absorbs large-magnitude reordering while still flagging
    a genuinely divergent value set (a missing/corrupt row shifts the sum by
    far more than 1e-9 of its scale). Returns {ok, diffs} where diffs lists
    only the mismatched keys as {key: (pg, ice)}.
    """
    diffs: dict[str, tuple[Any, Any]] = {}

    for key in ("rows", "factors", "tickers"):
        if pg.get(key) != ice.get(key):
            diffs[key] = (pg.get(key), ice.get(key))

    for key in ("z_checksum", "raw_checksum"):
        pg_val = float(pg.get(key) or 0.0)
        ice_val = float(ice.get(key) or 0.0)
        tol = max(checksum_tol, checksum_rel_tol * max(abs(pg_val), abs(ice_val)))
        if abs(pg_val - ice_val) > tol:
            diffs[key] = (pg_val, ice_val)

    return {"ok": len(diffs) == 0, "diffs": diffs}


def parity_report(engine, dates: "list[date]", checksum_tol: float = 1e-3) -> list[dict[str, Any]]:
    """Run compare_stats for each date in `dates`. Returns a list of dicts:

        {date, ok, diffs, pg, ice}

    one per date, in the same order as `dates`. Never raises on a single
    date's failure - the caller (asset / CLI) decides pass/fail policy.
    """
    from meridian_pipeline import lakehouse

    out: list[dict[str, Any]] = []
    for d in dates:
        try:
            pg_stats = pg_daily_scores_stats(engine, d)
        except Exception as exc:
            logger.warning("parity_report [%s]: PG stats query failed: %s", d, exc)
            out.append({"date": d, "ok": False, "diffs": {"error": ("pg_query_failed", str(exc))}})
            continue

        try:
            ice_stats = lakehouse.daily_scores_parity_stats(d)
        except Exception as exc:
            logger.warning("parity_report [%s]: Iceberg stats query failed: %s", d, exc)
            out.append(
                {"date": d, "ok": False, "diffs": {"error": ("iceberg_query_failed", str(exc))}}
            )
            continue

        cmp = compare_stats(pg_stats, ice_stats, checksum_tol=checksum_tol)
        out.append(
            {"date": d, "ok": cmp["ok"], "diffs": cmp["diffs"], "pg": pg_stats, "ice": ice_stats}
        )

    return out


def rolling_spotcheck_dates(
    engine,
    floor_date: date,
    ceiling_date: date,
    n: int,
    anchor: date,
    bucket_search_days: int = 5,
) -> "list[date]":
    """Deterministically pick up to `n` trading days spread across the full
    [floor_date, ceiling_date] history for a low-frequency spot check (#420
    detection gap).

    #420's orphan rows sat undetected for many monthly partitions because
    `daily_scores_parity_check`'s trailing MERIDIAN_PARITY_WINDOW_DAYS window
    (default 5 trading days) only ever compares dates from the last week or
    two - it structurally cannot see a defect anywhere outside that window,
    and the #420 repair does not change that: the very next correctness fix
    that deletes rows from an untouched-since-last-sweep partition reproduces
    the same undetectable condition. This function is the cheap, bounded
    mechanism that reaches further back.

    Splits [floor_date, ceiling_date] into `n` equal-sized calendar-day
    buckets and probes one date inside each bucket, snapping forward to the
    nearest trading day (bounded by `bucket_search_days`, cheap - a handful
    of db.is_trading_day calls per bucket, same primitive
    trailing_trading_days already uses in a loop). The probe point inside
    each bucket is `anchor.toordinal() % bucket_size`, so as `anchor` (the
    date the check runs on) advances day by day, each bucket's probe point
    advances too, walking through that bucket's `bucket_size` calendar days
    without persisting a cursor: `anchor` (today's date) already IS the
    cursor.

    HONEST CAVEAT on "eventual coverage": `bucket_size = total_days // n` is
    recomputed fresh every call from `ceiling_date`'s current value, and
    `ceiling_date` advances by roughly one day per scheduled tick (it tracks
    the trailing-window boundary in the caller). That makes `total_days`,
    and therefore `bucket_size` and the bucket boundaries themselves, drift
    upward by one unit every `n` ticks or so - which resets the walk through
    whichever bucket was mid-sweep before it completes a full pass. This is
    NOT a guaranteed full sweep on a fixed cadence; it is a continuously
    drifting, approximately-uniform sample across history that gets denser
    as more of it is queried over many runs, and the drift rate itself slows
    as history grows (bucket_size grows with total_days, so it re-anchors
    less often in relative terms). Treat this as a best-effort broadening of
    coverage beyond the trailing window, not a proof that every date is
    checked within any specific number of days. A cursor-based sweep would
    give a tighter guarantee at the cost of persisted state; this function
    trades that guarantee for statelessness, which was judged an acceptable
    tradeoff for a monitoring signal (not a repair mechanism - see
    mirror_reconcile_asset.py for the thing that actually fixes what this
    finds).

    COST BOUND: exactly `n` calls to `daily_scores_parity_stats` /
    `pg_daily_scores_stats` (via `parity_report` on the returned dates) - the
    expensive per-date Iceberg scan this module's own docstring flags as the
    costly half of a comparison - plus up to `n * (bucket_search_days + 1)`
    cheap `db.is_trading_day` calls to snap each probe onto a real trading
    day. Both bounds are independent of how large [floor_date, ceiling_date]
    is: a 20-year history and a 2-year history cost the same per run. Set `n`
    via MERIDIAN_PARITY_SPOTCHECK_DATES (lakehouse_parity_asset.py).

    Returns dates in ascending order, deduplicated - a bucket that fails to
    land on any trading day within `bucket_search_days` is skipped, so the
    result can hold fewer than `n` dates.
    """
    from meridian_pipeline import db

    if n <= 0 or ceiling_date < floor_date:
        return []

    total_days = (ceiling_date - floor_date).days + 1
    bucket_size = max(1, total_days // n)
    offset = anchor.toordinal() % bucket_size

    out: "list[date]" = []
    for i in range(n):
        bucket_start = floor_date + timedelta(days=i * bucket_size)
        if bucket_start > ceiling_date:
            break
        probe = min(bucket_start + timedelta(days=offset), ceiling_date)
        d = probe
        scanned = 0
        found = False
        while d <= ceiling_date and scanned <= bucket_search_days:
            if db.is_trading_day(engine, d):
                found = True
                break
            d += timedelta(days=1)
            scanned += 1
        if found:
            out.append(d)

    return sorted(set(out))


def trailing_trading_days(
    engine, end_date: date, n: int, max_lookback_days: int = 60
) -> "list[date]":
    """Walk backwards from `end_date` collecting the last `n` trading days
    (inclusive of end_date), using db.is_trading_day. Bounded by
    `max_lookback_days` calendar days so a data gap can't spin this forever.

    Returns dates in ascending (chronological) order.
    """
    from meridian_pipeline import db

    found: list[date] = []
    cursor = end_date
    scanned = 0
    while len(found) < n and scanned < max_lookback_days:
        if db.is_trading_day(engine, cursor):
            found.append(cursor)
        cursor -= timedelta(days=1)
        scanned += 1
    return sorted(found)


def _cli() -> None:
    import argparse

    from meridian_pipeline import db

    parser = argparse.ArgumentParser(description="Ad-hoc PG-vs-Iceberg daily_scores parity check")
    parser.add_argument("--start", type=str, default=None, help="YYYY-MM-DD (inclusive)")
    parser.add_argument("--end", type=str, default=None, help="YYYY-MM-DD (inclusive)")
    parser.add_argument(
        "--days",
        type=int,
        default=5,
        help="If --start is omitted, check the last N trading days ending at --end "
        "(default: latest available score_date).",
    )
    parser.add_argument("--tol", type=float, default=1e-3, help="Checksum tolerance")
    args = parser.parse_args()

    engine = db.engine_from_env()

    if args.end:
        end_date = date.fromisoformat(args.end)
    else:
        end_date = db.get_latest_score_date(engine)
        if end_date is None:
            print("No score_date found in factors.daily_scores; nothing to check.")
            return

    if args.start:
        start_date = date.fromisoformat(args.start)
        dates = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]
        dates = [d for d in dates if db.is_trading_day(engine, d)]
    else:
        dates = trailing_trading_days(engine, end_date, args.days)

    report = parity_report(engine, dates, checksum_tol=args.tol)

    n_fail = sum(1 for r in report if not r["ok"])
    for r in report:
        status = "PASS" if r["ok"] else "FAIL"
        print(f"{r['date']}  {status}  diffs={r.get('diffs') or {}}")

    print(f"\n{len(report) - n_fail}/{len(report)} dates passed (tol={args.tol}).")
    if n_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    _cli()
