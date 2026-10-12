"""
gold_pg_sync_asset.py - gold_scores_pg_sync asset (Meridian#265).

One-way promotion of a FINALIZED Iceberg gold batch into PostgreSQL
factors.daily_scores (the authoritative, point-lookup-fast serving gold - see
docs/18-ICEBERG_MIGRATION.md section 6, decision ACCEPTED 2026-07-17).

Large recompute / backfill / research batches are computed on the cheap
columnar Iceberg substrate and written to meridian.factors.gold_promotion_scores
(lakehouse.write_gold_scores), which stamps the promotion SCOPE - the exclusive
owned_factors set + the batch's score_date range - onto the snapshot summary.
This asset reads a target snapshot (the current one, or a specific snapshot_id
passed via run config by iceberg_gold_sync_sensor), then delete-replaces exactly
those owned_factors in PG via db.upsert_daily_scores(..., owned_factors=<scope>).

Idempotent + scoped:
  * run_key on the sensor side is the snapshot_id, so a re-fire of the same
    snapshot dedupes at the Dagster level.
  * upsert_daily_scores(replace=True, owned_factors=<scope>) only ever deletes
    and rewrites the promoted factors for the promoted dates - a factor this
    batch does NOT own is never touched, so the daily pipeline's direct PG
    writes and a promotion batch coexist without clobbering each other.

PG stays authoritative-once-written: this is a SECOND write path (batch
promotion) alongside the daily pipeline's direct PG write, not a fallback
derived from Iceberg. The API is never pointed at Iceberg (docs/06-CRITICAL_RULES.md Absolute
Rule 1).

Do NOT use `from __future__ import annotations` in this file - it breaks
Dagster's AssetExecutionContext validation (a repo convention
for Dagster asset modules).
"""

from typing import Optional

from dagster import AssetExecutionContext, Config, Failure, MetadataValue, Output, asset
from meridian_pipeline import db, lakehouse, lineage
from meridian_pipeline.k8s_config import LIGHT_OP_TAGS


class GoldPgSyncConfig(Config):
    """Run config for gold_scores_pg_sync.

    snapshot_id: the Iceberg gold snapshot to promote. None (default) promotes
    the CURRENT snapshot of meridian.factors.gold_promotion_scores. The
    iceberg_gold_sync_sensor always sets this to the exact snapshot id it
    detected, so the promoted batch is the one the sensor fired for even if a
    newer snapshot has since landed.

    A run scoped to a specific snapshot_id FAILS (raises) rather than skipping
    when the gold table is unavailable, the snapshot is not found, or it
    carries no owned_factors scope: a skip is a SUCCESS run, and the sensor
    would acknowledge a snapshot that was never promoted. Only an unscoped
    manual run (snapshot_id None) skips clean.

    allow_interior_gaps (Meridian#401, default False): the snapshot's declared
    [score_date_min, score_date_max] scope is delete-replaced in full,
    including any in-scope trading day the payload carries no rows for - that
    date's existing PG rows are deleted and nothing replaces them. By default
    this asset REFUSES to promote when that happens, because nothing
    distinguishes a batch that silently dropped a date from one that legitimately
    computed an empty date. Set this to True only after a human has confirmed
    the missing date(s) are a deliberate, reviewed outcome of the recompute -
    this is exactly the manual first-launch moment #401 was filed for, so the
    default stays fail-closed and this must be stated explicitly at launch,
    not left on.
    """

    snapshot_id: Optional[int] = None
    allow_interior_gaps: bool = False


@asset(
    group_name="meridian_factors",
    op_tags=LIGHT_OP_TAGS,
    description=(
        "Promote a finalized Iceberg gold batch into PostgreSQL "
        "factors.daily_scores (Meridian#265). Reads meridian.factors."
        "gold_promotion_scores at the target snapshot (current, or a "
        "snapshot_id from run config set by iceberg_gold_sync_sensor), then "
        "delete-replaces exactly that snapshot's owned_factors scope in PG via "
        "db.upsert_daily_scores(owned_factors=...). Idempotent (run_key = "
        "snapshot_id) and scoped so it never touches an unowned factor. An "
        "unscoped run skips clean when the gold table is unpopulated/"
        "unreachable or the snapshot carries no owned_factors scope; a run "
        "scoped to a snapshot_id FAILS instead. PG stays authoritative; this is the "
        "batch promotion write path, not an Iceberg-derived serving fallback "
        "(docs/18-ICEBERG_MIGRATION.md section 6)."
    ),
)
def gold_scores_pg_sync(context: AssetExecutionContext, config: GoldPgSyncConfig) -> Output:
    if not lakehouse.gold_promotion_available():
        if config.snapshot_id is not None:
            raise Failure(
                description=(
                    f"gold_scores_pg_sync: snapshot {config.snapshot_id} was requested but "
                    "meridian.factors.gold_promotion_scores is not available - failing the "
                    "run instead of skipping so the snapshot is not acknowledged unpromoted."
                )
            )
        context.log.info(
            "gold_scores_pg_sync: meridian.factors.gold_promotion_scores not "
            "available yet - skipping (expected pre-cutover state; no recompute "
            "has written the gold promotion table)."
        )
        return Output(value=None, metadata={"skipped": MetadataValue.bool(True)})

    df, scope = lakehouse.read_gold_snapshot(snapshot_id=config.snapshot_id)
    snapshot_id = scope.get("snapshot_id")
    owned = scope.get("owned_factors") or []

    if not owned:
        if config.snapshot_id is not None:
            raise Failure(
                description=(
                    f"gold_scores_pg_sync: snapshot {config.snapshot_id} was not found or "
                    f"carries no owned_factors scope (rows={0 if df.empty else len(df)}) - "
                    "failing the run instead of skipping so the snapshot is not "
                    "acknowledged unpromoted."
                )
            )
        context.log.info(
            "gold_scores_pg_sync: nothing to promote for snapshot %s "
            "(rows=%d, owned_factors=%d) - skipping.",
            snapshot_id,
            0 if df.empty else len(df),
            len(owned),
        )
        return Output(
            value=None,
            metadata={
                "skipped": MetadataValue.bool(True),
                "snapshot_id": MetadataValue.text(str(snapshot_id)),
                "owned_factors": MetadataValue.int(len(owned)),
            },
        )

    # Snapshot scope, not the surviving payload dates, defines replacement.
    from datetime import date

    date_range = (
        date.fromisoformat(scope["score_date_min"]),
        date.fromisoformat(scope["score_date_max"]),
    )
    engine = db.engine_from_env()
    # Meridian#294: record the run id so this run's write campaign carries it.
    lineage.bind_dagster_run(context)
    n_rows = db.upsert_daily_scores(
        engine,
        df,
        replace=True,
        owned_factors=owned,
        # batch promoter: the snapshot's owned scope is the authority and may
        # name a factor the snapshot itself has no rows for (#378).
        allow_unwritten_owned_factors=True,
        replace_date_range=date_range,
        # #401: fail closed by default on an in-scope trading day with zero
        # payload rows. Only an explicit run-config opt-in bypasses it.
        allow_interior_gaps=config.allow_interior_gaps,
    )

    context.log.info(
        "gold_scores_pg_sync: promoted snapshot %s -> PostgreSQL "
        "factors.daily_scores: %d rows across %d owned factor(s) [%s .. %s].",
        snapshot_id,
        n_rows,
        len(owned),
        scope.get("score_date_min"),
        scope.get("score_date_max"),
    )

    return Output(
        value=None,
        metadata={
            "snapshot_id": MetadataValue.text(str(snapshot_id)),
            "rows_promoted": MetadataValue.int(int(n_rows)),
            "owned_factors_count": MetadataValue.int(len(owned)),
            "owned_factors": MetadataValue.text(", ".join(owned)),
            "score_date_min": MetadataValue.text(str(scope.get("score_date_min"))),
            "score_date_max": MetadataValue.text(str(scope.get("score_date_max"))),
            "promotion_source": MetadataValue.text(str(scope.get("source"))),
            "allow_interior_gaps": MetadataValue.bool(config.allow_interior_gaps),
        },
    )
