#!/usr/bin/env python3
"""Full-history audit of the Meridian #174 trading-day-only invariant.

The Dagster asset check (`meridian_pipeline.checks.trading_day_checks`) sweeps
a bounded trailing window on the daily path so it stays cheap. This script is
the unbounded counterpart: it audits every scored table over all of history,
prints offending dates and row counts by year, and emits the exact scoped
DELETE statements needed to restore the invariant.

READ-ONLY. This script never deletes anything. It prints the DELETEs for an
operator to review and run under their own authority.

Usage:
    python scripts/check_trading_day_invariant.py
    python scripts/check_trading_day_invariant.py --start 2018-01-01
    python scripts/check_trading_day_invariant.py --no-row-counts   # dates only, fast

Exit code 1 if any non-trading-day row is found, so it can be wired into CI or
a cron tripwire.
"""

import argparse
import os
import sys
from collections import defaultdict
from datetime import date

from sqlalchemy import create_engine, text

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dagster"))

from meridian_pipeline.checks.trading_day_checks import (  # noqa: E402
    _DISTINCT_DATES_SQL,
    _TRADING_DAY_MIN_EQUITY_ROWS,
    SCORED_TABLES,
    non_trading_dates,
)

# Full-history calendar. Unlike the check's windowed form this has no bounds,
# so it is a single grouped aggregate over ohlcv.prices (a few seconds).
_FULL_CALENDAR_SQL = f"""
SELECT ts_event::date AS d
FROM ohlcv.prices
WHERE security_type = 'equity' AND resolution = 'daily'
GROUP BY 1
HAVING COUNT(*) > {_TRADING_DAY_MIN_EQUITY_ROWS}
"""


def _engine():
    url = os.environ.get("DATABASE_URL")
    if not url:
        user = os.environ["DB_USER"]
        pwd = os.environ["DB_PASSWORD"]
        host = os.environ.get("DB_HOST", "db.internal")
        port = os.environ.get("DB_PORT", "5432")
        name = os.environ.get("DB_NAME_OVERRIDE", "marketdata")
        url = f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{name}"
    return create_engine(url)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2000-01-01")
    ap.add_argument("--end", default=date.today().isoformat())
    ap.add_argument(
        "--no-row-counts",
        action="store_true",
        help="report offending dates only; skip the per-date COUNT(*) pass",
    )
    args = ap.parse_args()
    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)

    engine = _engine()
    with engine.connect() as conn:
        calendar = {r[0] for r in conn.execute(text(_FULL_CALENDAR_SQL)).fetchall()}
    print(f"trading calendar: {len(calendar)} days, {min(calendar)} .. {max(calendar)}\n")

    weekend_in_cal = sorted(d for d in calendar if d.isoweekday() >= 6)
    if weekend_in_cal:
        print(
            f"!! CALENDAR BUG: {len(weekend_in_cal)} weekend days claimed as trading, "
            f"first {weekend_in_cal[0]}\n"
        )

    total_dirty = 0
    for table in SCORED_TABLES:
        with engine.connect() as conn:
            dates = [
                r[0]
                for r in conn.execute(
                    text(_DISTINCT_DATES_SQL.format(tbl=table)), {"s": start, "e": end}
                ).fetchall()
            ]
        offenders = non_trading_dates(dates, calendar)
        if not offenders:
            print(f"{table}: CLEAN ({len(dates)} dates)")
            continue

        total_dirty += len(offenders)
        by_year: dict[int, list[date]] = defaultdict(list)
        for d in offenders:
            by_year[d.year].append(d)

        print(f"\n{table}: {len(offenders)} non-trading dates of {len(dates)}")
        for yr in sorted(by_year):
            ds = by_year[yr]
            wknd = sum(1 for d in ds if d.isoweekday() >= 6)
            line = f"  {yr}: {len(ds):>4} dates ({wknd} weekend, {len(ds) - wknd} holiday)"
            if not args.no_row_counts:
                with engine.connect() as conn:
                    n = conn.execute(
                        text(f"SELECT COUNT(*) FROM {table} WHERE score_date = ANY(:ds)"),
                        {"ds": ds},
                    ).scalar()
                line += f", {n:>10} rows"
            print(line)

        print(f"  -- proposed (NOT executed) cleanup for {table}:")
        print(f"  DELETE FROM {table} t")
        print("   WHERE NOT EXISTS (SELECT 1 FROM ohlcv.prices p")
        print("                      WHERE p.security_type = 'equity'")
        print("                        AND p.resolution = 'daily'")
        print("                        AND p.ts_event >= t.score_date")
        print("                        AND p.ts_event <  t.score_date + 1")
        print(
            f"                      GROUP BY 1 HAVING COUNT(*) > {_TRADING_DAY_MIN_EQUITY_ROWS});"
        )

    print(f"\nTOTAL non-trading (table, date) pairs: {total_dirty}")
    return 1 if total_dirty else 0


if __name__ == "__main__":
    sys.exit(main())
