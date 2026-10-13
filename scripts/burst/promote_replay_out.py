#!/usr/bin/env python3
"""promote_replay_out.py - upsert a burst run's output under the explicit campaign.

Meridian#356. Runs in-cluster or on a LAN box: reads
`s3://meridian-parquet/burst/out/<run_id>/`, verifies the run is complete, then
upserts every date under ONE explicit campaign so the whole pass has a single
auditable id regardless of how many box runs produced it.

The completeness check runs before anything is written. A box run that died
two thirds through ships a prefix that looks well formed (the files present are
valid parquet with correct columns), and promoting it would leave a third of
the range uncorrected, which shows up as a coverage hole days later, if at all.

Before any upsert it requires:

  * the manifest exists and declares `complete`
  * the set of date= directories matches the manifest's dates_written exactly,
    in both directions (a file the manifest does not know about is as
    suspicious as a manifest entry with no file)
  * per-date row counts match the manifest

The git SHA of the producing code is reported. Only then does it open the
campaign context and upsert.

It also refuses when the lakehouse mirror is configured but not ready
(lakehouse.assert_dual_write_ready), because the per-date mirror write is
best-effort and would otherwise drift silently after PG commits. Pass
`--no-iceberg` to accept a PG-only promote.
"""

import argparse
import contextlib
import json
import os
import sys

REPO = os.environ.get("MERIDIAN_REPO_ROOT") or os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, os.path.join(REPO, "dagster"))

from meridian_pipeline import db, lineage  # noqa: E402
from meridian_pipeline.pg_identity import (  # noqa: E402
    BURST_APPLICATION_NAME,
    adopt_default_application_name,
)


@contextlib.contextmanager
def bind_existing_campaign(campaign_id: int):
    """Stamp writes with an ALREADY-OPEN campaign.

    `lineage.open_campaign` mints a new campaign; this pass needs every row to
    land under the one explicit campaign opened before any mechanism was chosen
    (for a history remediation), so that a history pass split across box runs, or across
    mechanisms, still reads as one operation.

    `db.upsert_*` stamps `lineage.current_campaign_id()`, which resolves the
    ambient `_current` first - so binding a Campaign carrying the existing id is
    exactly the documented path, without a second campaign row.
    """
    tok = lineage._current.set(
        lineage.Campaign(
            campaign_id=campaign_id,
            writer=lineage.WRITER_MANUAL_SCRIPT,
            code_version=lineage.resolve_code_version(),
            run_id=None,
            input_vintage={},
        )
    )
    try:
        yield campaign_id
    finally:
        lineage._current.reset(tok)


def _duck():
    import duckdb

    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute(f"SET s3_endpoint='{os.environ['LDP_MINIO_ENDPOINT']}'")
    con.execute("SET s3_url_style='path'")
    con.execute("SET s3_use_ssl=true")
    con.execute(f"SET s3_access_key_id='{os.environ['LDP_MINIO_ACCESS_KEY']}'")
    con.execute(f"SET s3_secret_access_key='{os.environ['LDP_MINIO_SECRET_KEY']}'")
    return con


def verify(con, root: str) -> dict:
    """Return the manifest, or exit non-zero naming exactly what is wrong."""
    try:
        raw = con.execute(
            f"SELECT content FROM read_text('{root.rstrip('/')}/manifest.json')"
        ).fetchone()[0]
        man = json.loads(raw)
    except Exception as exc:
        raise SystemExit(f"FAIL: no readable manifest at {root}: {exc}")

    if man.get("complete") is not True:
        raise SystemExit(
            f"FAIL: manifest says complete={man.get('complete')}. The run did not "
            "finish its date range; promoting it would leave a silent coverage hole."
        )

    # str() on the hive partition value, on purpose. DuckDB autocasts a
    # `date=YYYY-MM-DD` partition column to DATE (it returned VARCHAR in older
    # versions), so `present` came back as datetime.date while `declared` is
    # ISO strings from the manifest JSON. The two sets then never intersect and
    # the gate reports EVERY date as both "not in manifest" and "without files".
    # Normalising both sides to ISO strings makes the comparison version-proof.
    present = {
        str(r[0])
        for r in con.execute(
            f"SELECT DISTINCT date FROM read_parquet('{root.rstrip('/')}/date=*/part.parquet', "
            "hive_partitioning=1)"
        ).fetchall()
    }
    declared = set(man["dates_written"])
    only_files = sorted(present - declared)
    only_manifest = sorted(declared - present)
    if only_files or only_manifest:
        raise SystemExit(
            "FAIL: manifest and files disagree.\n"
            f"  files not in manifest: {only_files[:10]}\n"
            f"  manifest without files: {only_manifest[:10]}"
        )

    # Same normalisation as `present`: the manifest keys per_date by ISO string.
    counts = {
        str(d): n
        for d, n in con.execute(
            f"SELECT date, COUNT(*) FROM read_parquet('{root.rstrip('/')}/date=*/part.parquet', "
            "hive_partitioning=1) GROUP BY 1"
        ).fetchall()
    }
    bad = [
        (d, int(counts.get(d, 0)), man["per_date"][d]["rows"])
        for d in declared
        if int(counts.get(d, 0)) != man["per_date"][d]["rows"]
    ]
    if bad:
        raise SystemExit(f"FAIL: row counts differ from the manifest for {bad[:10]}")
    return man


def assert_owned_covers_shipped(man, owned):
    """Refuse an --owned-factors list that does not cover what the box shipped.

    `--owned-factors` is free text, and it drives a DELETE: upsert_daily_scores
    removes `factor_name = ANY(owned_factors)` for each date before inserting.
    Passing one asset's list while promoting another asset's run would
    therefore delete every factor of the first asset on those dates, insert
    the second asset's rows, and report success.

    The reverse - owned but not shipped - is NOT an error: clearing a factor the
    asset no longer computes is exactly what owned_factors is for (#161).
    """
    shipped = sorted({f for v in man["per_date"].values() for f in v["factors"]})
    stray = [f for f in shipped if f not in owned]
    if stray:
        raise SystemExit(
            f"FAIL: the run shipped factors that --owned-factors does not cover: {stray}.\n"
            f"       asset={man.get('asset')} shipped={shipped}\n"
            f"       owned={sorted(owned)}\n"
            "       owned_factors drives a DELETE, so promoting with the wrong asset's "
            "list would wipe that asset's factors for these dates."
        )
    return shipped


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, help="s3://meridian-parquet/burst/out/<run_id>/")
    ap.add_argument("--campaign-id", type=int, required=True)
    ap.add_argument("--owned-factors", required=True, help="comma-separated")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--no-iceberg",
        action="store_true",
        help="explicitly accept a PG-only promote; disables the Iceberg "
        "dual-write and skips the readiness preflight. Mirror the promoted "
        "range afterward with scripts/load_daily_scores_iceberg.py --replace.",
    )
    args = ap.parse_args()

    con = _duck()
    man = verify(con, args.root)
    owned = [f.strip() for f in args.owned_factors.split(",") if f.strip()]

    assert_owned_covers_shipped(man, owned)

    print(f"run_id      {man['run_id']}")
    print(f"asset       {man['asset']}")
    print(f"git_sha     {man['git_sha']}")
    print(f"dates       {len(man['dates_written'])}")
    print(f"total_rows  {man['total_rows']:,}")
    print(f"campaign    {args.campaign_id}")
    print(f"owned       {owned}")
    if args.dry_run:
        print("\nDRY RUN - verification passed, nothing upserted.")
        return 0

    # db.upsert_daily_scores mirrors best-effort per date (log and continue), so
    # off-cluster a missing credential or unreachable catalog would commit PG
    # and silently drift the Iceberg mirror for the whole range. Fail here,
    # before the first write, unless the operator opts out explicitly.
    if args.no_iceberg:
        os.environ["MERIDIAN_ICEBERG_DUAL_WRITE"] = "false"
        print(
            "[promote] --no-iceberg: Iceberg dual-write disabled for this run. "
            "Remember to reconcile the promoted dates afterward."
        )
    else:
        from meridian_pipeline import lakehouse

        try:
            lakehouse.assert_dual_write_ready("promote_replay_out")
        except RuntimeError as exc:
            print(f"[promote] ABORT: {exc}")
            return 2

    eng = db._engine(
        pguser=os.environ.get("PGUSER", "dbuser"),
        pgpass=os.environ["PGPASS"],
        pghost=os.environ.get("PGHOST", "db.internal"),
        pgport=int(os.environ.get("PGPORT", 5432)),
        pgdb=os.environ.get("PGDB", "marketdata"),
        application_name=BURST_APPLICATION_NAME,
    )
    total = 0
    with bind_existing_campaign(args.campaign_id):
        for d in sorted(man["dates_written"]):
            df = con.execute(
                f"SELECT * EXCLUDE (date) FROM read_parquet("
                f"'{args.root.rstrip('/')}/date={d}/part.parquet', hive_partitioning=1)"
            ).df()
            n = db.upsert_daily_scores(
                eng,
                df,
                owned_factors=owned,
                # batch promoter: --owned-factors is the run's scope (already
                # checked to cover what shipped) and a single date may carry a
                # subset of it (#378).
                allow_unwritten_owned_factors=True,
            )
            total += int(n or 0)
            print(f"  {d}  {n:,} rows", flush=True)
    print(f"\npromoted {total:,} rows under campaign {args.campaign_id}")
    print(
        "Next: replay_coverage.py at the campaign floor - the manifest proves the RUN "
        "was complete, the coverage check proves the DATABASE is."
    )
    return 0


if __name__ == "__main__":
    adopt_default_application_name(BURST_APPLICATION_NAME)
    sys.exit(main())
