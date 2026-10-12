"""
definitions.py - SHOWCASE code-location entry point.

The private repository registers a much larger asset graph here, with its
schedules and sensors. This entry point registers what the showcase ships:

  - the three data-integrity asset checks,
  - stand-ins for the withheld factor assets (same keys and partitions, so the
    jobs and sensors below resolve against a complete asset graph),
  - the gold sync and mirror reconcile assets (the lakehouse side of the write
    path) and the risk_metrics asset,
  - the alerting and orchestration sensors,
  - the jobs those sensors target, and the range-verification job.

`dagster code-server start -m meridian_pipeline.definitions` serves it.
"""

from dagster import Definitions, asset
from meridian_pipeline import jobs, sensors
from meridian_pipeline.asset_selections import CORE_FACTOR_ASSET_KEYS
from meridian_pipeline.backfill_verification_job import backfill_verification_job
from meridian_pipeline.checks import ALL_FACTOR_CHECKS
from meridian_pipeline.gold_pg_sync_asset import gold_scores_pg_sync
from meridian_pipeline.mirror_reconcile_asset import daily_scores_mirror_reconcile
from meridian_pipeline.partitions import DAILY_BACKFILL_POLICY, DAILY_PARTITIONS
from meridian_pipeline.risk_metrics_asset import risk_metrics

_WITHHELD_KEYS = (*CORE_FACTOR_ASSET_KEYS, "market_vol", "composite_scores", "factor_group_scores")


def _withheld(key: str):
    @asset(
        name=key,
        group_name="withheld",
        partitions_def=DAILY_PARTITIONS,
        backfill_policy=DAILY_BACKFILL_POLICY,
        description="Showcase stand-in: the factor computation behind this key is not published.",
    )
    def _stand_in():
        raise NotImplementedError(f"{key} is a showcase stand-in; the computation is withheld")

    return _stand_in


defs = Definitions(
    assets=[
        risk_metrics,
        gold_scores_pg_sync,
        daily_scores_mirror_reconcile,
        *[_withheld(k) for k in _WITHHELD_KEYS],
    ],
    asset_checks=ALL_FACTOR_CHECKS,
    jobs=[
        jobs.daily_factor_core_job,
        jobs.daily_gpu_composite_job,
        jobs.gold_pg_sync_job,
        jobs.daily_scores_mirror_reconcile_job,
        backfill_verification_job,
    ],
    sensors=[
        sensors.checked_score_mirror_sensor,
        sensors.watermark_complete_sensor,
        sensors.stale_factor_sensor,
        sensors.request_sensor,
        sensors.core_complete_sensor,
        sensors.ntfy_run_failure_sensor,
        sensors.asset_check_alert_sensor,
        sensors.ldp_ohlcv_upstream_sensor,
        sensors.composite_cache_warm_sensor,
        sensors.iceberg_gold_sync_sensor,
        sensors.backfill_completion_sensor,
    ],
)
