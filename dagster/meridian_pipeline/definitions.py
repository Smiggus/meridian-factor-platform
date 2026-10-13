"""Code-location entry point.

Registers the three data-integrity asset checks, the gold sync and mirror
reconcile assets, the risk_metrics asset, the sensors, the jobs those sensors
target and the range-verification job.

The factor assets are minimal placeholders for code not included here. They use
the same keys and partitions as the real ones, so the jobs and sensors resolve
against a complete asset graph, and they raise if materialised.

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

_PLACEHOLDER_KEYS = (
    *CORE_FACTOR_ASSET_KEYS,
    "market_vol",
    "composite_scores",
    "factor_group_scores",
)


def _placeholder(key: str):
    @asset(
        name=key,
        group_name="placeholder",
        partitions_def=DAILY_PARTITIONS,
        backfill_policy=DAILY_BACKFILL_POLICY,
        description="Placeholder for a factor computation that is not included in this copy.",
    )
    def _stand_in():
        raise NotImplementedError(f"{key} is a placeholder; its computation is not included")

    return _stand_in


defs = Definitions(
    assets=[
        risk_metrics,
        gold_scores_pg_sync,
        daily_scores_mirror_reconcile,
        *[_placeholder(k) for k in _PLACEHOLDER_KEYS],
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
