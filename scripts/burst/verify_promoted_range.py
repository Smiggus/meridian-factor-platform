#!/usr/bin/env python3
"""Independently verify a promoted burst range against the staged payload.

Read-only against PostgreSQL: it loads the staged parquet from MinIO and compares
it to what gold holds, per date, for the run's owned factors. It does NOT read the
promote's own campaign rows, so it can certify dates whose campaign was killed
before it could record a verdict (see scripts/lineage_close_campaign.py, which
consumes this evidence).

The comparison is exact on the served payload: same (ticker, factor_name)
membership and z_score equal to < 1e-9.

    python scripts/burst/verify_promoted_range.py --run-id backfill-2018 \
        --start 2018-01-02 --end 2019-12-31 --all

It reads the staged parquet once and runs one indexed point query per date, so `--all`
issues one query for every date in the range.
"""

import argparse
import os
import sys

import pandas as pd
from sqlalchemy import text

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, "scripts", "burst"))

import promote_backfill as pb  # noqa: E402  (script-local sibling import)

TOL = 1e-9


def match_frames(staged, gold):
    """Exact payload equality: same (ticker, factor_name) key set, same z_score.

    Returns (membership_diff, max_abs_dz). `max_abs_dz` is inf when nothing joined,
    so a completely disjoint pair can never read as a zero-difference pass.
    """
    m = staged.merge(
        gold, on=["ticker", "factor_name"], how="outer", suffixes=("_st", "_gd"), indicator=True
    )
    both = m[m["_merge"] == "both"].dropna(subset=["z_score_st", "z_score_gd"])
    diff = len(m) - len(both)
    dz = float((both["z_score_st"] - both["z_score_gd"]).abs().max()) if len(both) else float("inf")
    return diff, dz


def compare(engine, s3, bucket, key, owned):
    """Return (date, staged_rows, gold_rows, membership_diff, max_abs_dz, ok)."""
    d = pb._date_of(key)
    staged = pb._load(s3, bucket, key, stamp_computed_at=False)[
        ["ticker", "factor_name", "z_score"]
    ]
    with engine.connect() as c:
        gold = pd.read_sql(
            text(
                "SELECT ticker, factor_name, z_score FROM factors.daily_scores "
                "WHERE score_date=:d AND factor_name = ANY(:f)"
            ),
            c,
            params={"d": d, "f": owned},
        )
    diff, dz = match_frames(staged, gold)
    return d, len(staged), len(gold), diff, dz, (diff == 0 and dz < TOL)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Independently verify a promoted burst range against the staged payload."
    )
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--start", default=None, help="only scores >= this (YYYY-MM-DD)")
    ap.add_argument("--end", default=None, help="only scores <= this (YYYY-MM-DD)")
    ap.add_argument(
        "--all",
        action="store_true",
        help="verify every staged date in the window (default: a 12-date sample)",
    )
    ap.add_argument(
        "--sample",
        type=int,
        default=12,
        help="dates to verify when --all is not given (spread + the last partition)",
    )
    ap.add_argument(
        "--max-failures",
        type=int,
        default=0,
        help="tolerate this many failing dates before exiting non-zero (default 0)",
    )
    args = ap.parse_args()

    bucket = os.environ.get("MERIDIAN_MINIO_BUCKET", "meridian-parquet")
    s3 = pb._s3()
    manifest = pb.load_manifest(s3, bucket, args.run_id)
    owned, src = pb.resolve_owned_factors(manifest, [], pb.OWNED_FACTORS)
    print(f"[verify] run_id={args.run_id} owned_factors={len(owned)} (scope source: {src})")

    keys = pb._staged_keys(s3, bucket, args.run_id)
    if not keys:
        print(f"[verify] ABORT: no staged scores under burst/{args.run_id}/scores/")
        return 2
    if args.start or args.end:
        lo, hi = args.start or "0000-00-00", args.end or "9999-99-99"
        keys = [k for k in keys if lo <= pb._date_of(k) <= hi]
    keys = sorted(keys, key=pb._date_of)
    dates = [pb._date_of(k) for k in keys]
    window = f"[{args.start or '-inf'}, {args.end or '+inf'}]"
    print(f"[verify] window={window} staged dates={len(dates)} ({dates[0]} .. {dates[-1]})")

    selected = keys if args.all else pb.sample_keys(keys, args.sample, include_last=True)
    print(f"[verify] checking {len(selected)} of {len(keys)} staged dates")

    engine = pb._gold_engine()
    failures = []
    total_rows = 0
    worst = 0.0
    for k in selected:
        d, n_st, n_gd, diff, dz, ok = compare(engine, s3, bucket, k, owned)
        total_rows += n_st
        if dz != float("inf"):
            worst = max(worst, dz)
        print(
            f"[verify] {d}: staged={n_st} gold={n_gd} membership_diff={diff} "
            f"max|dz|={dz:.2e} {'PASS' if ok else 'FAIL'}"
        )
        if not ok:
            failures.append(d)

    # Coverage: every staged date in the window must be present in gold. Point
    # lookups, one per date - the presence pattern is NOT monotonic, so a binary
    # search over it would be wrong (this range mixes dates that never had gold
    # with dates that were promoted, so "present" is not a prefix).
    absent = []
    failed = set(failures)
    with engine.connect() as c:
        for k in keys:
            d = pb._date_of(k)
            if d in failed:
                continue
            n = c.execute(
                text(
                    "SELECT count(DISTINCT factor_name) FROM factors.daily_scores "
                    "WHERE score_date=:d AND factor_name = ANY(:f)"
                ),
                {"d": d, "f": owned},
            ).scalar()
            if not n:
                absent.append(d)
    if absent:
        print(f"[verify] ABSENT from gold: {len(absent)} -> {absent[:10]}")

    print(
        f"[verify] SUMMARY dates_checked={len(selected)} rows_compared={total_rows} "
        f"max|dz|={worst:.2e} failures={len(failures)} absent_from_gold={len(absent)}"
    )
    bad = len(failures) + len(absent)
    if bad:
        print(f"[verify] FAILING dates: {failures[:20]}{' ...' if len(failures) > 20 else ''}")
        return 0 if bad <= args.max_failures else 2
    print("[verify] ALL STAGED DATES MATCH GOLD")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
