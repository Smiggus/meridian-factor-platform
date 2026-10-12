"""
jobs.py - named jobs for every schedule and sensor trigger in the
meridian-factors code location.

Every schedule and sensor in this code location previously declared its
target as a bare AssetSelection (e.g. @schedule(target=AssetSelection.keys(
"some_asset"))), which makes Dagster auto-generate an anonymous job
named __anonymous_asset_job_<schedule|sensor>. The UI then shows runs under
those opaque names and they cannot be addressed by name for backfills.

This module defines one named job per selection, used by:
  - schedules.py via @schedule(job=<job>)
  - sensors.py via @sensor(job=<job>)

It lives in its own module (not definitions.py) so sensors.py and
schedules.py can import the jobs without an import cycle (definitions.py
imports both).

Job names are stable identifiers: daily_factor_core_job,
daily_gpu_composite_job and gold_pg_sync_job predate this module and are
referenced by tests via defs.get_job_def(...) - keep those names exactly.
"""

from dagster import AssetSelection, define_asset_job
from meridian_pipeline.asset_selections import GPU_COMPOSITE_ASSETS, PHASE1_TRIGGER_ASSETS

# Two-phase daily jobs (Meridian#88 follow-up, 2026-05-24).
# Phase 1: CPU-only factor groups (CORE) plus the two canonical Component-D
# vol riders (PHASE1_TRIGGER_ASSETS). Phase 2: GPU-bound + composite.
# See asset_selections.py for the split rationale.
daily_factor_core_job = define_asset_job(
    name="daily_factor_core_job",
    selection=PHASE1_TRIGGER_ASSETS,
    description=(
        "Phase 1 daily factor scoring: the CPU-only factor group assets plus the "
        "canonical market_vol asset. Triggered by watermark_complete_sensor, "
        "stale_factor_sensor, request_sensor and ldp_ohlcv_upstream_sensor; "
        "core_complete_sensor fires daily_gpu_composite_job for the same "
        "partition once all CORE assets land (market_vol is not a gate)."
    ),
)

daily_gpu_composite_job = define_asset_job(
    name="daily_gpu_composite_job",
    selection=GPU_COMPOSITE_ASSETS,
    description=(
        "Phase 2 daily scoring: risk_metrics + composite + factor_group_scores. "
        "GPU-tagged via the assets' op_tags. Triggered by core_complete_sensor "
        "serialised cluster-wide by the gpu pool "
        "cap=1."
    ),
)

# Iceberg gold -> PG promotion (Meridian#265). Materializes the single
# gold_scores_pg_sync asset. iceberg_gold_sync_sensor targets this job
# directly; also launchable manually with a snapshot_id in run config.
gold_pg_sync_job = define_asset_job(
    name="gold_pg_sync_job",
    selection=AssetSelection.keys("gold_scores_pg_sync"),
    description=(
        "Promote a finalized Iceberg gold batch (meridian.factors."
        "gold_promotion_scores) into PostgreSQL factors.daily_scores, scoped "
        "to the target snapshot's owned_factors (Meridian#265). Fired per new "
        "snapshot by iceberg_gold_sync_sensor; also launchable manually with a "
        "snapshot_id in run config."
    ),
)

# Monthly whole-partition Iceberg mirror reconciliation for factors.daily_scores
# (Meridian #420 recurrence fix).
daily_scores_mirror_reconcile_job = define_asset_job(
    name="daily_scores_mirror_reconcile_job",
    selection=AssetSelection.keys("daily_scores_mirror_reconcile"),
    description=(
        "Monthly whole-partition Iceberg mirror reconciliation for "
        "factors.daily_scores (Meridian #420 recurrence fix). Runs via "
        "mirror_reconcile_schedule (shipped STOPPED - see that schedule's "
        "description). Whole-partition-replaces the trailing "
        "MERIDIAN_MIRROR_RECONCILE_WINDOW_MONTHS (default 24) calendar months "
        "so orphan rows left by any PG write path that bypasses "
        "db.upsert_daily_scores eventually clear from the mirror."
    ),
)
