#!/usr/bin/env python3
"""
promote_backfill.py - land a burst backfill's staged scores into gold.

The CONTROL-plane promote step of the elastic burst pattern (proposal 07): reads
the per-partition score parquet a burst `backfill_core` run staged to MinIO and
delete-replaces it into PostgreSQL `factors.daily_scores`, scoped to the kernel's
owned factors so nothing else in gold is touched.

Safety, in order:
  0. Scope binding (#294): the owned-factor (delete) scope comes from the run's
     `manifest.json`, which the kernel derived from the groups it actually
     scored. A `--owned-factors` that disagrees with the manifest is refused,
     and the staged payload is checked against the scope both ways: no staged
     factor outside it (would leave stale rows), and no owned factor the
     campaign never reproduces on a date that HAS gold rows for it (would
     silently erase a factor - the !474 near-miss class). Runs staged before
     manifests existed keep the legacy behaviour behind a loud warning.
  0b. Hole check (#294): the kernel records every date each invocation
     undertook under `{run_root}/requested/`. Promote reconciles that union
     against the staged parquet and refuses an incomplete campaign - a dead
     fan-out worker used to leave a silent gap that promoted anyway.
  0c. Live-only guard (#380): some factors are owned by a group but CANNOT be
     reproduced for a historical date (an upstream source overwrites the fetch
     timestamp, so the scorer omits the column) - the live-written gold rows
     are the only PIT record. Any
     live-only factor left in the resolved delete scope is a hard abort: name it
     in --retain-gold (keep the rows) or in --allow-erasure (delete them on
     purpose). NOT overridable by --force-coverage, and `--allow-erasure all`
     does not cover it - a blanket permit must not reach an irreversible one.
  1. Coverage gate (default): the staged re-score must not LOSE coverage vs the
     gold it replaces (a re-backfill on fuller/corrected data should keep or grow
     ticker coverage). Aborts on a coverage regression unless --force-coverage.
  2. Delete-replace is scoped to (score_date, factor_name in owned_factors) via
     db.copy_replace_daily_scores(owned_factors=...) (or db.upsert_daily_scores
     under --row-upsert) - other factors/groups untouched, orphans cleared if a
     day's universe shrank.
  3. Post-promote verification: re-reads gold for a sample of promoted dates and
     asserts it matches the staged rows exactly (max|dz|==0, 0 membership diff).
     The sample ALWAYS includes the last promoted partition (#380): an evenly
     spaced `keys[::len//n][:n]` stops at index (n-1)*floor(len/n) and left the
     trailing ~len/n partitions - the newest, and the only ones a live-only
     factor occupies - unverified.

Each gate has exactly ONE override and overrides do not bleed (#380): coverage
-> --force-coverage, holes -> --allow-incomplete, erasure -> --allow-erasure,
scope violation -> none, live-only -> --retain-gold / an explicitly named
--allow-erasure. `--force` is a deprecated alias for --force-coverage and no
longer silences the hole check or the erasure gate.

Idempotent + resumable: re-running re-promotes (delete-replace), so a mid-run
death is safe to relaunch. Writes PG gold (the served store) via
db.copy_replace_daily_scores() by default (bulk COPY; ~10-20x the row-oriented
upsert on a multi-million-row promote) or db.upsert_daily_scores() under
--row-upsert. BOTH route through db._dual_write_daily_scores, so either way
each partition is best-effort dual-written into the meridian Iceberg mirror
when MERIDIAN_ICEBERG_DUAL_WRITE is enabled (default on) and lakehouse
credentials are present. The write-path choice is a throughput knob only; it
can never silently change whether the mirror is written (that would make the
assert_dual_write_ready preflight below a lie).

Iceberg credentials: LAKEKEEPER_M2M_CLIENT_ID/SECRET live in the infra
repo's .env, not this repo's .env - source it before running off-cluster,
alongside this repo's own
MERIDIAN_MINIO_ACCESS_KEY/SECRET_KEY. If those creds are absent, OR they are
present but the catalog is unreachable (Meridian#299: the default
MERIDIAN_LAKEHOUSE_ENDPOINT is an in-cluster DNS name that cannot resolve
off-cluster, so credentials alone are not enough - also export
MERIDIAN_LAKEHOUSE_ENDPOINT=https://lakehouse.example.com/catalog), this
script FAILS LOUDLY before writing anything (see assert_dual_write_ready in
meridian_pipeline.lakehouse, which preflights both credential presence AND
catalog reachability) rather than silently skipping the mirror on every
partition. Pass --no-iceberg (or set MERIDIAN_ICEBERG_OPTIONAL=1) to
explicitly accept a PG-only run; the promoted range can then be mirrored
afterward with scripts/load_daily_scores_iceberg.py --replace (see docs/17,
"Off-cluster batch dual-write").

    cd ~/Meridian && set -a && . .env && set +a
    python scripts/burst/promote_backfill.py --run-id bf2010-2013 [--dry-run] \
        [--retain-gold FACTORS] [--force-coverage] [--verify-sample 8] [--no-iceberg]
"""

import argparse
import io
import json
import os
import sys
import time

import boto3
import pyarrow.parquet as pq
from sqlalchemy import text

_REPO = os.environ.get("MERIDIAN_REPO_ROOT") or os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, os.path.join(_REPO, "dagster"))

from meridian_pipeline import db, lineage  # noqa: E402
from meridian_pipeline.partition_scoring import (  # noqa: E402
    FUNDAMENTALS_GROUPS,
    LIVE_ONLY_DEFAULT,
    owned_factors_for_groups,
)
from meridian_pipeline.pg_identity import (  # noqa: E402
    BURST_APPLICATION_NAME,
    adopt_default_application_name,
)

# Legacy default for pre-#294 runs with no manifest: score_partition's
# fundamentals scope. DERIVED from the group
# map rather than re-listed here - a second hand-maintained copy of an owned
# set is the delete-scope hazard itself. Runs staged by the current kernel
# carry a manifest and never reach this default.
OWNED_FACTORS = owned_factors_for_groups(FUNDAMENTALS_GROUPS)


def _s3():
    return boto3.client(
        "s3",
        endpoint_url="https://" + os.environ.get("MERIDIAN_MINIO_ENDPOINT", "s3.example.com"),
        aws_access_key_id=os.environ["MERIDIAN_MINIO_ACCESS_KEY"],
        aws_secret_access_key=os.environ["MERIDIAN_MINIO_SECRET_KEY"],
        region_name=os.environ.get("MERIDIAN_MINIO_REGION", "us-east-1"),
    )


def _gold_engine():
    return db._engine(
        os.environ["DB_USER"],
        os.environ["DB_PASSWORD"],
        os.environ["DB_HOST"],
        int(os.environ.get("DB_PORT", "5432")),
        os.environ.get("GOLD_DB", "marketdata"),
        application_name=BURST_APPLICATION_NAME,
    )


def load_manifest(s3, bucket, run_id):
    """The run's scope manifest, or None for a pre-#294 run."""
    try:
        body = s3.get_object(Bucket=bucket, Key=f"burst/{run_id}/manifest.json")["Body"].read()
    except Exception:  # noqa: BLE001 - absent key (ClientError shape varies by client)
        return None
    return json.loads(body)


def load_requested(s3, bucket, run_id):
    """Union of every date the kernel invocations undertook for this run.
    Empty set for a pre-#294 run (no reconciliation possible)."""
    out = set()
    for pg in s3.get_paginator("list_objects_v2").paginate(
        Bucket=bucket, Prefix=f"burst/{run_id}/requested/"
    ):
        for o in pg.get("Contents", []):
            if not o["Key"].endswith(".json"):
                continue
            body = s3.get_object(Bucket=bucket, Key=o["Key"])["Body"].read()
            out.update(json.loads(body).get("dates", []))
    return out


def resolve_owned_factors(manifest, cli_owned, default_owned):
    """Decide the delete scope, and say where it came from.

    Precedence is deliberately NOT "CLI wins": the manifest is what the kernel
    actually scored, and a hand-typed CLI list that disagrees with it is the
    exact shape of the !474 near-miss (a printed promote command carrying the
    fundamentals default against a value+momentum+risk payload would have wiped
    the other groups). Disagreement is an error, not an override.

    Returns (owned_factors, source). Raises ValueError on disagreement.
    """
    manifest_owned = sorted(manifest.get("owned_factors") or []) if manifest else []
    if manifest_owned:
        if cli_owned and sorted(cli_owned) != manifest_owned:
            raise ValueError(
                "--owned-factors disagrees with the run manifest.\n"
                f"  manifest ({len(manifest_owned)}): {','.join(manifest_owned)}\n"
                f"  --owned-factors ({len(cli_owned)}): {','.join(sorted(cli_owned))}\n"
                "The manifest is derived from the groups the kernel actually scored. "
                "Drop --owned-factors (or fix it) - promoting a scope the payload does "
                "not match DELETES factors nothing rewrites."
            )
        return manifest_owned, "manifest"
    if cli_owned:
        return sorted(cli_owned), "--owned-factors"
    return sorted(default_owned), "legacy default (fundamentals)"


def payload_scope_violations(staged_factors, owned_factors):
    """Staged factor names the delete scope does not cover. Any hit means the
    promote would INSERT rows it cannot delete-replace on the next run, leaving
    stale duplicates behind the scope's back."""
    owned = set(owned_factors)
    return sorted(f for f in set(staged_factors) if f not in owned)


def parse_retain_gold(spec):
    """Parse --retain-gold into a sorted factor list (empty when unset)."""
    if not spec:
        return []
    return sorted({x.strip() for x in spec.split(",") if x.strip()})


def drop_retained(df, retained):
    """Discard staged rows for carved-out factors so they are neither counted
    in coverage nor written. The gold rows for those factors are left alone."""
    if not retained or df is None or df.empty:
        return df
    return df[~df["factor_name"].isin(retained)].reset_index(drop=True)


def allowed_erasures(spec):
    """Parse --allow-erasure: None -> nothing allowed, 'all' -> everything,
    otherwise the explicit factor set. The explicit form is the safe one - it
    accepts the erasures you reasoned about without blanket-permitting the one
    you did not."""
    if not spec:
        return set()
    if spec.strip() == "all":
        return "all"
    return {x.strip() for x in spec.split(",") if x.strip()}


def live_only_in_scope(owned_factors, allow_erasure_spec, live_only=LIVE_ONLY_DEFAULT):
    """Live-only factors still inside the delete scope, and not signed off.

    `owned_factors` is the scope AFTER --retain-gold has carved its factors out,
    so a retained factor is by construction absent here. The only other sign-off
    is naming the factor EXPLICITLY in --allow-erasure. `--allow-erasure all` is
    deliberately not enough: it is the blunt permit an operator reaches for to
    clear a batch of reproducible erasures, and it must not silently authorise
    deleting the only PIT record of a served factor (#380).
    """
    allowed = allowed_erasures(allow_erasure_spec)
    named = set() if allowed == "all" else set(allowed)
    owned = set(owned_factors)
    return sorted(f for f in live_only if f in owned and f not in named)


# One aggregate, issued at most once per promote, and only for factors the
# coverage sample never witnessed (see the call site). Plan: the PK is
# (score_date, ticker, factor_name), so `score_date = ANY(:dates)` is a
# ScalarArrayOp index condition - one index scan per promoted date over that
# date's own subtree, with factor_name filtered inside the index tuple and no
# heap access needed (both projected columns are in the index). Nothing here
# scans outside the promoted range, and on a campaign that reproduces every
# owned factor - every well-formed one, including a multi-group promote - the
# query is not issued at all.
_GOLD_FACTOR_PRESENCE_SQL = """
SELECT factor_name, count(*) AS n_dates,
       min(score_date) AS first_date, max(score_date) AS last_date
FROM (
    SELECT DISTINCT score_date, factor_name
    FROM factors.daily_scores
    -- CAST is required: `dates` arrives as a list of ISO strings (from
    -- `_date_of(key)`), and psycopg2 binds it as text[], so comparing the date
    -- column against it directly raises `operator does not exist: date = text`
    -- on the first call (pinned by tests/test_promote_backfill_erasure_probe.py).
    -- The probe is skipped whenever every owned factor appears in the 12-date
    -- sample, which is why the uncast form survived: a promote that reproduces
    -- its whole scope never issues it. A load that legitimately does not
    -- (excluded window-sensitive factors, a panel whose coverage ends early)
    -- does, every time.
    WHERE score_date = ANY(CAST(:dates AS date[])) AND factor_name = ANY(:factors)
) g
GROUP BY 1
"""


def gold_factor_presence(eng, dates, factors):
    """{factor: (n_dates, first_date, last_date)} over EVERY promoted date.

    The date axis of the erasure gate is sampled (12 dates); this closes the
    FACTOR axis. A factor whose gold rows live entirely outside the sampled
    dates - a live-only factor starts late, in the trailing few percent of a multi-year range,
    while the last sampled index is 11*floor(n/12) - is invisible to the sample
    in both directions: gold has no rows on any sampled date, so nothing looks
    erased, and the promote then deletes those rows on the dates nobody looked
    at (#380).
    """
    if not dates or not factors:
        return {}
    with eng.connect() as c:
        rows = c.execute(
            text(_GOLD_FACTOR_PRESENCE_SQL),
            {"dates": list(dates), "factors": list(factors)},
        ).fetchall()
    return {r[0]: (r[1], r[2], r[3]) for r in rows}


def sample_keys(keys, n, include_last=False):
    """Evenly spaced sample of at most `n` keys.

    `keys[::len//n][:n]` stops at index (n-1)*floor(len/n), so unless len is a
    multiple of n it never looks at the trailing ~len/n partitions. That tail is
    the NEWEST slice of any promote and the only place a live-only factor has
    gold rows, so the post-promote verification pins the last partition into the
    sample (#380). The coverage sample keeps the plain spacing: changing which
    dates it compares would change the gate's verdict on an unrelated promote.
    """
    if not keys:
        return []
    n = max(1, int(n))
    out = keys[:: max(1, len(keys) // n)][:n]
    if include_last and keys[-1] not in out:
        out = ([] if n == 1 else out[: n - 1]) + [keys[-1]]
    return out


def reconcile_holes(requested, staged, lo=None, hi=None):
    """Dates the run undertook but never staged, restricted to the promote
    window. A non-empty result means the campaign is incomplete."""
    lo = lo or "0000-00-00"
    hi = hi or "9999-99-99"
    want = {d for d in requested if lo <= d <= hi}
    return sorted(want - set(staged))


def _staged_keys(s3, bucket, run_id):
    return sorted(
        o["Key"]
        for pg in s3.get_paginator("list_objects_v2").paginate(
            Bucket=bucket, Prefix=f"burst/{run_id}/scores/"
        )
        for o in pg.get("Contents", [])
        if o["Key"].endswith("scores.parquet")
    )


def _load(s3, bucket, key, stamp_computed_at: bool = True):
    """Read one staged partition, stamping COMPUTE time onto every row.

    Meridian#294: `factors.daily_scores.computed_at` used to record WRITE time
    on this path - `db.copy_replace_daily_scores` omitted the column so the
    table DEFAULT now() stamped it at promote time, which on a burst run is
    hours-to-days after the kernel actually computed the scores. That gap is
    precisely what made "cluster computed_at to recover the write campaign"
    both the only available method and a wrong one.

    The staged object's S3 LastModified IS the kernel's compute time (the
    kernel writes each partition once, immediately after scoring it), so it is
    the best compute stamp available without changing the kernel's output
    schema. If the object metadata is unreadable the column is simply omitted
    and the pre-#294 DEFAULT applies - never a reason to fail a promote.
    """
    obj = s3.get_object(Bucket=bucket, Key=key)
    df = pq.read_table(io.BytesIO(obj["Body"].read())).to_pandas()
    if stamp_computed_at and "computed_at" not in df.columns:
        staged_at = obj.get("LastModified")
        if staged_at is not None:
            df["computed_at"] = staged_at
    return df


def _staged_at_max(s3, bucket, prefix, keys):
    """Max staging (= compute) time over the promoted partitions.

    This is the campaign's input_vintage for the staged score set: the newest
    moment any promoted partition was produced by the burst kernel. One
    paginated listing rather than a HEAD per object, so it stays cheap on a
    1000+ partition run. Returns None if the listing fails - an unknown vintage
    is recorded as unknown, never guessed."""
    wanted = set(keys)
    latest = None
    try:
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                if obj["Key"] in wanted and (latest is None or obj["LastModified"] > latest):
                    latest = obj["LastModified"]
    except Exception as exc:  # pragma: no cover - best effort
        print(f"[promote] input_vintage: could not read staged object times ({exc})")
    return latest


def _date_of(key):
    return key.split("score_date=")[1].split("/")[0]


def main() -> int:
    ap = argparse.ArgumentParser(description="Promote a burst backfill's staged scores into gold.")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--dry-run", action="store_true", help="coverage-check + report, write nothing")
    ap.add_argument(
        "--force-coverage",
        action="store_true",
        help="promote even if the coverage gate sees a ticker-coverage regression "
        "(expected after a stricter hygiene guard - see the note the gate prints). Overrides the "
        "COVERAGE gate and nothing else.",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="DEPRECATED alias for --force-coverage. It used to silence the hole "
        "check and the erasure gate as well, so an operator reaching for the "
        "coverage override - which the tool itself can recommend - "
        "silently disarmed two unrelated gates (#380). It now maps to "
        "--force-coverage ONLY, so a runbook still carrying it fails safe on the "
        "others instead of passing them.",
    )
    ap.add_argument("--verify-sample", type=int, default=8, help="dates to re-verify post-promote")
    ap.add_argument(
        "--owned-factors",
        default=None,
        help="comma-separated factor_name list overriding the default fundamentals set "
        "(the fundamentals groups). Use for kernels that stage a different scope, "
        "e.g. rescore_ret1d_close.py's value+momentum+risk set. IGNORED-BY-DESIGN when "
        "the run has a manifest: it must then either be omitted or match it exactly.",
    )
    ap.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="promote even though some dates the run undertook were never staged "
        "(hole check). Only correct when the missing dates are deliberately abandoned.",
    )
    ap.add_argument(
        "--allow-erasure",
        default=None,
        metavar="FACTORS|all",
        help="accept that the delete-replace removes gold rows for owned factors the "
        "staged payload does not reproduce. Takes a comma-separated factor list (the "
        "safe form: only those factors may be erased) or 'all'. Correct when the "
        "erasure IS the intent, e.g. a deprecated factor or a factor whose "
        "inputs do not exist for the era being re-scored. Does NOT cover the "
        f"live-only factors {','.join(LIVE_ONLY_DEFAULT)} unless one is named "
        "explicitly - 'all' is refused for those.",
    )
    ap.add_argument(
        "--start",
        default=None,
        help="only promote staged score_dates >= this (YYYY-MM-DD). Lets N instances "
        "fan out over disjoint date ranges - delete-replace is scoped per "
        "(score_date, factor_name), so disjoint ranges never lock-contend.",
    )
    ap.add_argument(
        "--end", default=None, help="only promote staged score_dates <= this (YYYY-MM-DD)."
    )
    ap.add_argument(
        "--no-iceberg",
        action="store_true",
        help="explicitly accept a PG-only promote; disables the Iceberg "
        "dual-write and skips the credential preflight check. Mirror the "
        "promoted range afterward with scripts/load_daily_scores_iceberg.py "
        "--replace.",
    )
    ap.add_argument(
        "--retain-gold",
        default=None,
        metavar="FACTORS",
        help="comma-separated owned factors to CARVE OUT of the promote: they are "
        "dropped from the delete scope and their staged rows are discarded, so the "
        "existing gold rows survive untouched. For factors whose historical inputs "
        "are no longer PIT-reproducible (an upstream source overwrote the inputs: "
        "the live-written gold rows are the only PIT record). Must be a subset of the "
        "run's owned scope.",
    )
    ap.add_argument(
        "--row-upsert",
        action="store_true",
        help="use the row-oriented upsert instead of the default bulk COPY (fallback)",
    )
    args = ap.parse_args()

    force_coverage = args.force_coverage or args.force
    if args.force:
        print(
            "[promote] NOTE: --force is DEPRECATED and now means --force-coverage ONLY. "
            "It no longer silences the hole check (--allow-incomplete) or the erasure "
            "gate (--allow-erasure); see #380."
        )

    bucket = os.environ.get("MERIDIAN_MINIO_BUCKET", "meridian-parquet")
    s3 = _s3()

    cli_owned = (
        sorted(x.strip() for x in args.owned_factors.split(",") if x.strip())
        if args.owned_factors
        else []
    )
    manifest = load_manifest(s3, bucket, args.run_id)
    try:
        owned_factors, owned_src = resolve_owned_factors(manifest, cli_owned, OWNED_FACTORS)
    except ValueError as exc:
        print(f"[promote] ABORT: {exc}")
        return 2
    retained = set(parse_retain_gold(args.retain_gold))
    if retained:
        bad = sorted(retained - set(owned_factors))
        if bad:
            print(
                f"[promote] ABORT: --retain-gold names factor(s) outside the run's owned "
                f"scope: {bad}. Retaining a factor the run never owned is a no-op that "
                "hides a typo."
            )
            return 2
        owned_factors = [f for f in owned_factors if f not in retained]
        print(
            f"[promote] --retain-gold: {sorted(retained)} carved out of the delete scope; "
            f"their staged rows will be discarded. owned_factors now {len(owned_factors)}."
        )
    if manifest:
        print(
            f"[promote] manifest: groups={manifest.get('groups')} "
            f"owned_factors={len(owned_factors)} (scope source: {owned_src})"
        )
    else:
        print(
            f"[promote] WARNING: no manifest under burst/{args.run_id}/ - this is a "
            f"pre-#294 run. Delete scope comes from {owned_src} and is NOT verified "
            "against what was staged. Check it by hand before proceeding:\n"
            f"           {','.join(owned_factors)}"
        )

    # --- Live-only guard (#380) --------------------------------------------
    # Scope-level, so it fires before any listing, preflight or DB work, and it
    # does not depend on what the payload happens to hold: a live-only factor in
    # the delete scope is wrong even if this particular staged frame does carry
    # rows for it, because those rows cannot be the PIT record the deleted ones
    # were. No --force of any spelling reaches it.
    exposed = live_only_in_scope(owned_factors, args.allow_erasure)
    if exposed:
        print(
            f"[promote] ABORT: live-only factor(s) {exposed} are in the delete scope.\n"
            "  Their gold rows are the ONLY PIT record: the burst scorer cannot "
            "reproduce them for a historical date (an upstream source overwrites "
            "the fetch timestamp, so the column is simply absent from "
            "the staged payload), so the delete-replace would delete and never "
            "re-insert.\n"
            f"  Keep them:   --retain-gold {','.join(exposed)}   (carves them out of the "
            "scope; their staged rows, if any, are discarded)\n"
            f"  Erase them:  --allow-erasure {','.join(exposed)}   (only if losing the "
            "history IS the intent - it is not recoverable from a re-score)\n"
            "  --force / --force-coverage do NOT override this, and --allow-erasure all "
            "does not cover it."
        )
        return 2

    # Keep automatic mirroring and its preflight until approved checked cutover.
    if args.no_iceberg:
        os.environ["MERIDIAN_ICEBERG_DUAL_WRITE"] = "false"
        print(
            "[promote] --no-iceberg: Iceberg dual-write disabled for this run. "
            "Remember to reconcile the promoted dates afterward."
        )
    else:
        from meridian_pipeline import lakehouse

        try:
            lakehouse.assert_dual_write_ready("promote_backfill")
        except RuntimeError as exc:
            print(f"[promote] ABORT: {exc}")
            return 2

    eng = _gold_engine()
    keys = _staged_keys(s3, bucket, args.run_id)
    if args.start or args.end:
        lo = args.start or "0000-00-00"
        hi = args.end or "9999-99-99"
        keys = [k for k in keys if lo <= _date_of(k) <= hi]
    if not keys:
        print(f"no staged scores under burst/{args.run_id}/scores/ - nothing to promote")
        return 2
    rng = (
        f" range=[{args.start or '-inf'}, {args.end or '+inf'}]" if (args.start or args.end) else ""
    )
    print(f"[promote] {len(keys)} staged partitions | owned_factors={len(owned_factors)}{rng}")

    # --- Hole check: every date the run undertook must have been staged ---
    requested = load_requested(s3, bucket, args.run_id)
    if requested:
        holes = reconcile_holes(requested, [_date_of(k) for k in keys], args.start, args.end)
        want = len([d for d in requested if (args.start or "0") <= d <= (args.end or "z")])
        print(f"[promote] hole check: requested={want} staged={len(keys)} missing={len(holes)}")
        if holes:
            print(f"[promote] missing dates (first 20): {holes[:20]}")
            if not args.allow_incomplete:
                print(
                    "[promote] ABORT: campaign is INCOMPLETE. Re-run the kernel for the "
                    "missing dates (it is resumable - staged dates are skipped), or pass "
                    "--allow-incomplete if they are deliberately abandoned. --force does "
                    "NOT cover the hole check (#380)."
                )
                return 2
            print("[promote] proceeding with an incomplete campaign (override in effect)")
    else:
        print(
            "[promote] hole check SKIPPED: no requested-date records under "
            f"burst/{args.run_id}/requested/ (pre-#294 run). Staged-vs-discovered cannot "
            "be reconciled; verify the date coverage by hand."
        )

    # --- Coverage gate: sample a few dates, compare staged vs current gold coverage ---
    sample = sample_keys(keys, 12)
    regressions = 0
    scope_violations = set()
    erasures = {}
    staged_seen = set()
    for k in sample:
        d = _date_of(k)
        staged = drop_retained(_load(s3, bucket, k), retained)
        # Payload-vs-scope, both directions (see resolve_owned_factors).
        staged_factors = set(staged["factor_name"].unique())
        staged_seen.update(staged_factors)
        scope_violations.update(payload_scope_violations(staged_factors, owned_factors))
        with eng.connect() as c:
            gold_facs = {
                r[0]
                for r in c.execute(
                    text(
                        "SELECT DISTINCT factor_name FROM factors.daily_scores "
                        "WHERE score_date=:d AND factor_name = ANY(:f)"
                    ),
                    {"d": d, "f": owned_factors},
                ).fetchall()
            }
        gone = sorted(gold_facs - staged_factors)
        if gone:
            erasures[d] = gone
        with eng.connect() as c:
            old = c.execute(
                text(
                    "SELECT count(DISTINCT ticker) FROM factors.daily_scores "
                    "WHERE score_date=:d AND factor_name = ANY(:f)"
                ),
                {"d": d, "f": owned_factors},
            ).scalar()
        new = staged["ticker"].nunique()
        flag = " <-- REGRESSION" if new < old else ""
        if new < old:
            regressions += 1
        print(f"[coverage] {d}: gold {old} -> staged {new} tickers{flag}")
    if regressions and not force_coverage:
        print(
            f"[promote] ABORT: {regressions}/{len(sample)} sampled dates lose coverage. "
            "Use --force-coverage to override (it overrides THIS gate only)."
        )
        print(
            "[promote] NOTE: a re-score staged under a stricter hygiene guard than "
            "the gold it replaces can legitimately lose coverage. Inspect the "
            "dropped names; --force-coverage is correct only if they are "
            "expected."
        )
        return 2

    # --- Payload vs scope, direction 1: staged factor outside the delete scope ---
    if scope_violations:
        print(
            f"[promote] ABORT: staged payload carries {len(scope_violations)} factor(s) the "
            f"delete scope does not cover: {sorted(scope_violations)}. Those rows would be "
            "INSERTed without ever being delete-replaceable, leaving stale duplicates on the "
            "next promote. Fix the owned-factor derivation, do not --force this."
        )
        return 2

    # --- Payload vs scope, direction 2: owned factor the campaign does not rewrite ---
    # Two layers (#380). The 12-date sample above catches a per-date erasure on
    # the dates it happens to inspect. This second layer closes the FACTOR axis:
    # every owned factor the whole sample never witnessed is checked against gold
    # over EVERY promoted date, in one aggregate. Without it, a factor whose gold
    # rows sit entirely in the trailing n/12 the sample never reaches - the exact
    # shape of a live-written factor - is erased with both the
    # gate and the post-promote verify looking elsewhere.
    unseen = sorted(f for f in owned_factors if f not in staged_seen)
    tail_erasures = {}
    if unseen:
        all_dates = sorted({_date_of(k) for k in keys})
        print(
            f"[promote] exhaustive erasure check: {len(unseen)} owned factor(s) never "
            f"appeared in the {len(sample)}-date sample; probing gold over all "
            f"{len(all_dates)} promoted dates for {unseen}"
        )
        tail_erasures = gold_factor_presence(eng, all_dates, unseen)
        for f, (n_dates, first, last) in sorted(tail_erasures.items()):
            print(f"[erasure] {f}: gold on {n_dates} promoted date(s), {first}..{last}")

    if erasures or tail_erasures:
        n_fac = sorted({f for facs in erasures.values() for f in facs} | set(tail_erasures))
        if erasures:
            print(
                f"[promote] ERASURE WARNING: on {len(erasures)}/{len(sample)} sampled dates "
                f"the delete-replace would remove gold rows for owned factor(s) the staged "
                f"payload does not reproduce: "
                f"{sorted({f for facs in erasures.values() for f in facs})}"
            )
            for d, facs in list(erasures.items())[:5]:
                print(f"[erasure] {d}: {facs}")
        if tail_erasures:
            print(
                f"[promote] ERASURE WARNING (exhaustive): {sorted(tail_erasures)} have gold "
                "rows inside the promoted range and are reproduced on NO sampled date"
            )
        allowed = allowed_erasures(args.allow_erasure)
        unexpected = [] if allowed == "all" else [f for f in n_fac if f not in allowed]
        if unexpected:
            print(
                f"[promote] ABORT: refusing to erase {unexpected} - this campaign does not "
                "rewrite them. Either widen --groups on the re-score so those factors are "
                "recomputed, carve them out with --retain-gold if their gold rows must "
                "survive, or name them in --allow-erasure (or pass --allow-erasure all) if "
                "removing them IS the intent. --force does NOT cover the erasure gate (#380)."
            )
            return 2
        print(f"[promote] proceeding with erasures (allowed: {args.allow_erasure})")

    if args.dry_run:
        print("[promote] dry-run: gates OK, nothing written.")
        return 0

    # --- Promote: per-partition scoped delete-replace ---
    # Default to bulk COPY (A2: ~10-20x faster than the row-oriented upsert that
    # dominated the 2010-2013 promote at ~2.6h); --row-upsert falls back.
    write = db.upsert_daily_scores if args.row_upsert else db.copy_replace_daily_scores
    print(f"[promote] write path: {'row-upsert' if args.row_upsert else 'bulk COPY'}")

    # Meridian#294: everything this promote writes belongs to ONE campaign, so
    # a future "which promote landed these rows / what did it read / did
    # anything gate it" is a join instead of an archaeology exercise. The
    # campaign is bound ambiently, so both write paths pick it up with no
    # per-callsite change, and it closes with the verification verdict below.
    staged_at = _staged_at_max(s3, bucket, f"burst/{args.run_id}/scores/", keys)
    input_vintage = lineage.source_vintage(
        f"burst/{args.run_id}/scores",
        staged_at_max=staged_at,
        n_partitions=len(keys),
        owned_factors=len(owned_factors),
        start=args.start,
        end=args.end,
    )
    print(f"[promote] input_vintage: staged_at_max={staged_at}")

    with lineage.open_campaign(
        eng,
        lineage.WRITER_BURST_PROMOTE,
        run_id=args.run_id,
        input_vintage=input_vintage,
        detail={
            "write_path": "row-upsert" if args.row_upsert else "bulk-copy",
            "coverage_forced": bool(force_coverage and regressions),
            "iceberg_dual_write": not args.no_iceberg,
            # #294: record WHAT scope this campaign delete-replaced and where
            # that scope came from. Lineage answers "which promote landed these
            # rows"; without the scope it cannot answer "and what else could it
            # have deleted", which is the question the !474 near-miss raises.
            "groups": list(manifest.get("groups") or []) if manifest else None,
            "owned_factors_source": owned_src,
            "owned_factors": len(owned_factors),
            "erasures_allowed": args.allow_erasure,
            "retained_gold": sorted(retained) or None,
            "incomplete_allowed": bool(args.allow_incomplete),
        },
    ) as campaign:
        print(f"[promote] campaign_id={campaign.campaign_id}")
        t0 = time.time()
        tot = 0
        for i, k in enumerate(keys):
            tot += write(
                eng,
                drop_retained(_load(s3, bucket, k), retained),
                owned_factors=owned_factors,
                # batch promoter: one owned scope for the whole campaign,
                # upserted one staged partition at a time (#378).
                allow_unwritten_owned_factors=True,
            )
            if (i + 1) % 150 == 0 or i + 1 == len(keys):
                print(
                    f"[promote] {i + 1}/{len(keys)} partitions, {tot} rows, {time.time() - t0:.0f}s",
                    flush=True,
                )

        # --- Post-promote verification: gold == staged exactly for a sample ---
        import pandas as pd

        bad = 0
        for k in sample_keys(keys, args.verify_sample, include_last=True):
            d = _date_of(k)
            staged = drop_retained(_load(s3, bucket, k, stamp_computed_at=False), retained)[
                ["ticker", "factor_name", "z_score"]
            ]
            with eng.connect() as c:
                gold = pd.read_sql(
                    text(
                        "SELECT ticker, factor_name, z_score FROM factors.daily_scores "
                        "WHERE score_date=:d AND factor_name = ANY(:f)"
                    ),
                    c,
                    params={"d": d, "f": owned_factors},
                )
            m = staged.merge(
                gold,
                on=["ticker", "factor_name"],
                how="outer",
                suffixes=("_st", "_gd"),
                indicator=True,
            )
            both = m[m["_merge"] == "both"].dropna(subset=["z_score_st", "z_score_gd"])
            dz = (both["z_score_st"] - both["z_score_gd"]).abs()
            ok = (len(m) - len(both) == 0) and (dz.max() < 1e-9)
            print(
                f"[verify] {d}: matched={len(both)} membership_diff={len(m) - len(both)} max|dz|={dz.max():.1e} {'PASS' if ok else 'FAIL'}"
            )
            bad += 0 if ok else 1

        # The gate verdict is the point of recording a campaign at all: it is
        # what lets a row claim to be trustable rather than merely present.
        lineage.set_gate_result(
            lineage.GATE_PASS if bad == 0 else lineage.GATE_FAIL,
            verify_sample=args.verify_sample,
            verify_failures=bad,
            rows_written=tot,
            partitions=len(keys),
        )

    print(
        f"[promote] DONE: {tot} rows across {len(keys)} partitions in {time.time() - t0:.0f}s | verify failures={bad}"
    )
    return 0 if bad == 0 else 2


if __name__ == "__main__":
    adopt_default_application_name(BURST_APPLICATION_NAME)
    raise SystemExit(main())
