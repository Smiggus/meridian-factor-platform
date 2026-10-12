#!/usr/bin/env python3
"""replay_coverage.py - did a replay actually cover every trading day?

`--resume` in scripts/burst/replay_asset.py skips a date whose probe factor already
carries `campaign_id > 0`. That test cannot tell "written by THIS replay" from
"written by some older campaign", so a date already stamped by a previous,
pre-correction run is skipped and silently never re-scored.
That is not hypothetical: a replay can leave holes exactly this way, when dates
already stamped by earlier pre-correction campaigns are passed over by `--resume`.
They were
found only by checking coverage against the trading calendar afterwards, and the
lesson recorded was: always verify replay coverage against the trading calendar,
never trust `--resume` alone.

This script is that verification. It asks a different question from the probe:

    for every trading day in the window, does EVERY expected factor carry a row
    whose campaign_id is at or above this replay's own floor?

Three distinct failures, reported separately because they need different fixes:

  MISSING    the date has no row for a factor at all
             -> re-run that date
  STALE      the date has rows, but from a campaign BELOW the floor
             -> the `--resume` hole; re-run WITHOUT --resume or the probe skips it again
  PARTIAL    some expected factors are at/above the floor and others are not
             -> the date died mid-write; re-run that date

Exit 1 on any of the three, so it can gate the completion chain.

Usage:
  replay_coverage.py --start 2020-01-02 --end 2020-12-31 --min-campaign 100 \\
      --factors factor_a,factor_b,factor_c
"""

import argparse
import os
import sys
from collections import defaultdict

REPO = os.environ.get("MERIDIAN_REPO_ROOT") or os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, os.path.join(REPO, "dagster"))

from meridian_pipeline import db  # noqa: E402
from sqlalchemy import text  # noqa: E402

# Same predicate replay_asset uses to enumerate its work, so the two agree on what
# a trading day is. Anything else risks reporting a hole that was never in scope.
_CAL_SQL = text(
    """
    SELECT ts_event::date AS d
      FROM ohlcv.prices
     WHERE security_type = 'equity' AND resolution = 'daily'
       AND ts_event >= CAST(:s AS date) AND ts_event < (CAST(:e AS date) + 1)
     GROUP BY 1 HAVING COUNT(*) > 100
     ORDER BY 1
    """
)

_COV_SQL = text(
    """
    SELECT score_date, factor_name, MAX(campaign_id) AS max_campaign
      FROM factors.daily_scores
     WHERE score_date BETWEEN :s AND :e
       AND factor_name = ANY(:factors)
     GROUP BY 1, 2
    """
)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument(
        "--min-campaign",
        type=int,
        required=True,
        help="lowest campaign_id belonging to this replay; anything below it is STALE",
    )
    ap.add_argument("--factors", required=True, help="comma-separated expected factors")
    ap.add_argument("--max-report", type=int, default=25)
    args = ap.parse_args()

    factors = [f.strip() for f in args.factors.split(",") if f.strip()]
    if not factors:
        print("FAIL: no factors given - a check over zero factors would pass vacuously")
        return 1

    # Same construction as scripts/burst/replay_asset.py, so the check and the
    # thing it checks cannot disagree about which database they are talking to.
    eng = db._engine(
        pguser=os.environ.get("PGUSER", "dbuser"),
        pgpass=os.environ["PGPASS"],
        pghost=os.environ.get("PGHOST", "db.internal"),
        pgport=int(os.environ.get("PGPORT", 5432)),
        pgdb=os.environ.get("PGDB", "marketdata"),
    )
    with eng.connect() as conn:
        # Catch a typo'd factor name up front. Without this it surfaces as PARTIAL on
        # every date, which reads as "the replay died mid-write" on every date.
        known = {
            r[0]
            for r in conn.execute(
                text(
                    "SELECT DISTINCT factor_name FROM factors.daily_scores "
                    "WHERE factor_name = ANY(:f)"
                ),
                {"f": factors},
            )
        }
        unknown = [f for f in factors if f not in known]
        if unknown:
            print(
                f"FAIL: these factor names appear nowhere in factors.daily_scores: "
                f"{unknown}. Check the spelling; this is not a coverage hole."
            )
            return 1
        days = [r[0] for r in conn.execute(_CAL_SQL, {"s": args.start, "e": args.end})]
        rows = conn.execute(
            _COV_SQL, {"s": args.start, "e": args.end, "factors": factors}
        ).fetchall()

    # An empty calendar means the window or the source is wrong, not that coverage
    # is perfect. Zero days checked must never read as a pass.
    if not days:
        print(f"FAIL: 0 trading days found in {args.start}..{args.end} - not a result")
        return 1

    seen = defaultdict(dict)
    for d, f, c in rows:
        seen[d][f] = c if c is not None else -1

    missing, stale, partial = [], [], []
    for d in days:
        got = seen.get(d, {})
        absent = [f for f in factors if f not in got]
        below = [f for f in factors if f in got and got[f] < args.min_campaign]
        at_or_above = [f for f in factors if f in got and got[f] >= args.min_campaign]
        if not got:
            missing.append((d, "no rows for any expected factor"))
        elif absent and at_or_above:
            partial.append((d, f"absent={absent}"))
        elif absent:
            missing.append((d, f"absent={absent}"))
        elif below and at_or_above:
            partial.append((d, f"below floor={below}"))
        elif below:
            stale.append((d, f"max campaign {max(got[f] for f in below)} < {args.min_campaign}"))

    print(f"window       {args.start} .. {args.end}")
    print(f"trading days {len(days):,}")
    print(f"factors      {len(factors)}  {factors}")
    print(f"floor        campaign_id >= {args.min_campaign}")
    print()
    ok = len(days) - len(missing) - len(stale) - len(partial)
    print(f"  COVERED  {ok:,}")
    print(f"  MISSING  {len(missing):,}   (no row; re-run the date)")
    print(f"  STALE    {len(stale):,}   (older campaign; --resume WILL skip these again)")
    print(f"  PARTIAL  {len(partial):,}   (mixed; the date died mid-write)")

    for label, items in (("MISSING", missing), ("STALE", stale), ("PARTIAL", partial)):
        if not items:
            continue
        print(f"\n-- {label} --")
        for d, why in items[: args.max_report]:
            print(f"  {d}  {why}")
        if len(items) > args.max_report:
            print(f"  ... and {len(items) - args.max_report} more")

    if missing or stale or partial:
        kinds = [
            f"{n} {k}"
            for k, n in (
                ("MISSING", len(missing)),
                ("STALE", len(stale)),
                ("PARTIAL", len(partial)),
            )
            if n
        ]
        print(f"\nFAIL: coverage is incomplete - {', '.join(kinds)}.")
        if stale:
            print(
                "  STALE dates must be re-run WITHOUT --resume: the probe is "
                "campaign_id > 0, so it passes over them and they stay stale."
            )
        if missing or partial:
            print("  MISSING / PARTIAL dates can be re-run with --resume.")
        return 1
    print("\nOK: every trading day carries every expected factor at or above the floor.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
