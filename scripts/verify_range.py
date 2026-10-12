#!/usr/bin/env python3
"""Operator CLI for Tier 2 of Meridian #285: bulk set-level range verification.

Same code path as the Dagster job (backfill_verification_job.py -> op ->
range_verification.verify_range) and the same code path the Tier 3
backfill_completion_sensor drives automatically - this script exists so an
operator can run the identical sweep by hand, ad hoc, without launching a
Dagster run.

READ-ONLY. Never writes to any factors.* table.

Usage:
    python scripts/verify_range.py --start 2018-01-01 --end 2018-06-30
    python scripts/verify_range.py --start 2018-01-01 --end 2018-06-30 --json

Exit code 1 if any check fails (any finding at all - every check is
currently advisory, so this exit code is for operator/CI visibility, not a
statement about whether the underlying data is unusable). The Dagster job
never fails its step on a finding - see range_verification.py's module
docstring.
"""

import argparse
import json
import os
import sys
from datetime import date

from sqlalchemy import create_engine

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dagster"))

from meridian_pipeline.checks.range_verification import verify_range  # noqa: E402


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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", required=True, help="ISO date, inclusive")
    ap.add_argument("--end", required=True, help="ISO date, inclusive")
    ap.add_argument("--json", action="store_true", help="emit the report as JSON")
    args = ap.parse_args()
    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)

    engine = _engine()
    try:
        report = verify_range(engine, start, end)
    finally:
        engine.dispose()

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(f"verify_range: {start} .. {end}\n")
        for check in report.checks:
            status = "OK  " if check.passed else "WARN"
            print(f"[{status}] {check.name}: {check.message}")
        n_ok = sum(1 for c in report.checks if c.passed)
        print(f"\n{n_ok}/{len(report.checks)} checks clean")

    return 0 if report.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
