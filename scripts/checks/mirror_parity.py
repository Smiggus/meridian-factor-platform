#!/usr/bin/env python3
"""Detective check: flat parquet mirror vs canonical factors.daily_scores.

PG is gold (docs/18). The `meridian-parquet` flat mirror is what
`parquet_io.fetch_factor_scores_chunks` serves to the research read path. Its
duplicate-key precedence is recency-first (Meridian #275): the newest
`computed_at` wins, so a later `dualwrite__*` correction beats an earlier
materialised `data_*` file, and only UNSTAMPED rows fall back to the
`data_` tier then filename. (This docstring claimed the inverse until
Meridian #381.) The mirror can still diverge from PG - rows PG no longer
has are never pruned from `data_*` files until the year is re-materialised -
so this script measures that divergence on a random (score_date, factor)
sample.

For each sampled cell it reports:
    pg_rows / mirror_rows
    ghosts    - tickers present in the mirror but absent from PG
    missing   - tickers present in PG but absent from the mirror
    mismatch  - shared tickers whose z_score differs by > --tol
    max_abs_dz

READ-ONLY on both sides. Needs DB_* (or DATABASE_URL) and MERIDIAN_MINIO_*.

Usage:
    python scripts/checks/mirror_parity.py --samples 24 --seed 7
    python scripts/checks/mirror_parity.py --cell 2021-06-15:momentum_12_1

Exit 1 if any sampled cell has ghosts, missing tickers or mismatches.
"""

import argparse
import os
import random
import sys
from datetime import date, timedelta

import pandas as pd
from sqlalchemy import create_engine, text

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "dagster"))

from meridian_pipeline import parquet_io  # noqa: E402


def _engine():
    url = os.environ.get("DATABASE_URL")
    if not url:
        user = os.environ["DB_USER"]
        pwd = os.environ["DB_PASSWORD"]
        host = os.environ.get("DB_HOST", "db.internal")
        port = os.environ.get("DB_PORT", "5432")
        name = os.environ.get("DB_NAME_OVERRIDE", "marketdata")
        url = f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{name}"
    return create_engine(url, execution_options={"postgresql_readonly": True})


def _pg_cell(eng, d: date, factor: str) -> pd.Series:
    with eng.connect() as c:
        rows = c.execute(
            text(
                # The mirror contract (both materialize_factor_features and the
                # dual-write path) is z_score IS NOT NULL rows only, so the
                # market-level raw_value-only factors are absent from it by
                # design.
                "SELECT ticker, z_score FROM factors.daily_scores "
                "WHERE score_date = :d AND factor_name = :f AND z_score IS NOT NULL"
            ),
            {"d": d, "f": factor},
        ).fetchall()
    return pd.Series({r[0]: r[1] for r in rows}, dtype="float64")


def _mirror_cell(d: date, factor: str) -> pd.Series:
    parts = list(parquet_io.fetch_factor_scores_chunks([factor], d, d))
    if not parts:
        return pd.Series(dtype="float64")
    df = pd.concat(parts, ignore_index=True)
    df["score_date"] = pd.to_datetime(df["score_date"]).dt.date
    df = df[df["score_date"] == d]
    dup = df.duplicated(["ticker"]).sum()
    if dup:
        print(f"  !! mirror returned {dup} duplicate tickers for {d} {factor} (dedup failed)")
    return (
        df.drop_duplicates("ticker", keep="last").set_index("ticker")["z_score"].astype("float64")
    )


def _pick_cells(eng, n: int, seed: int, start: date, end: date) -> list[tuple[date, str]]:
    with eng.connect() as c:
        factors = [
            r[0]
            for r in c.execute(
                text(
                    "SELECT DISTINCT factor_name FROM factors.daily_scores "
                    "WHERE score_date = (SELECT MAX(score_date) FROM factors.daily_scores) "
                    "AND z_score IS NOT NULL"
                )
            ).fetchall()
        ]
        dates = [
            r[0]
            for r in c.execute(
                text(
                    "SELECT DISTINCT ts_event::date FROM ohlcv.prices "
                    "WHERE security_type='equity' AND resolution='daily' "
                    "AND ts_event >= :s AND ts_event <= :e"
                ),
                {"s": start, "e": end},
            ).fetchall()
        ]
    rng = random.Random(seed)
    return [(rng.choice(dates), rng.choice(factors)) for _ in range(n)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start", default="2009-01-01")
    ap.add_argument("--end", default=(date.today() - timedelta(days=3)).isoformat())
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--cell", action="append", default=[], help="YYYY-MM-DD:factor (repeatable)")
    ap.add_argument(
        "--duckdb-threads",
        type=int,
        default=4,
        help="cap DuckDB threads; the many-file glob otherwise floods Docker's "
        "embedded resolver with 'Could not resolve hostname'",
    )
    args = ap.parse_args()

    if not (parquet_io.MERIDIAN_MINIO_ACCESS_KEY and parquet_io.MERIDIAN_MINIO_SECRET_KEY):
        print("MERIDIAN_MINIO_* unset; cannot reach the mirror")
        return 2
    # NOT meridian_parquet_available(): that probe globs the whole
    # tree and is what falls over on a flaky resolver. Read errors surface per
    # cell below instead.
    parquet_io._meridian_con().execute(f"SET threads={args.duckdb_threads}")

    eng = _engine()
    # Explicit --cell entries are ADDED to the random sample, never a
    # replacement for it: a hand-picked list is a regression probe, the
    # sample is the gate.
    cells = _pick_cells(
        eng,
        args.samples,
        args.seed,
        date.fromisoformat(args.start),
        date.fromisoformat(args.end),
    )
    cells += [(date.fromisoformat(c.split(":")[0]), c.split(":")[1]) for c in args.cell or []]

    bad = 0
    absent = 0
    print(
        f"{'date':<11}{'factor':<34}{'pg':>7}{'mirror':>7}{'ghost':>6}{'miss':>6}{'diff':>6}  max|dz|"
    )
    for d, f in cells:
        pg = _pg_cell(eng, d, f)
        if pg.empty:
            # A factor can legitimately be absent on a sampled date (group-specific,
            # live-only, or an ML factor with z-scores only in recent years).
            # Both sides empty is "absent", not divergence; a populated mirror
            # against an empty PG cell is a ghost and fails.
            try:
                mi = _mirror_cell(d, f)
            except Exception as e:  # noqa: BLE001
                print(f"{d.isoformat():<11}{f:<34} mirror read failed: {type(e).__name__}")
                bad += 1
                continue
            if mi.empty:
                print(f"{d.isoformat():<11}{f:<34} absent on both sides (skipped)")
                absent += 1
            else:
                print(f"{d.isoformat():<11}{f:<34} {0:>7}{len(mi):>7}  GHOST CELL  <-- DIVERGED")
                bad += 1
            continue
        try:
            mi = _mirror_cell(d, f)
        except Exception as e:  # noqa: BLE001 - report and keep sampling
            print(
                f"{d.isoformat():<11}{f:<34} mirror read failed: {type(e).__name__}: {str(e)[:90]}"
            )
            bad += 1
            continue
        ghosts = mi.index.difference(pg.index)
        missing = pg.index.difference(mi.index)
        shared = pg.index.intersection(mi.index)
        dz = (pg.loc[shared] - mi.loc[shared]).abs()
        # NaN on both sides counts as agreement
        both_nan = pg.loc[shared].isna() & mi.loc[shared].isna()
        diff = int(((dz > args.tol) | (dz.isna() & ~both_nan)).sum())
        mx = float(dz.max()) if len(dz) else 0.0
        flag = "" if not (len(ghosts) or len(missing) or diff) else "  <-- DIVERGED"
        print(
            f"{d.isoformat():<11}{f:<34}{len(pg):>7}{len(mi):>7}{len(ghosts):>6}{len(missing):>6}{diff:>6}  {mx:.3g}{flag}"
        )
        bad += bool(flag)
    print(
        f"\n{bad}/{len(cells) - absent} populated cells diverged from PG ({absent} absent skipped)"
    )
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
