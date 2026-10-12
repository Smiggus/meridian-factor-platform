"""
partitions.py
-------------
Shared partition definitions for the Meridian Dagster code-server.

Extracted from `assets.py` and `risk_metrics_asset.py`, both of which used
to define their own copy of `DAILY_PARTITIONS`. Two copies meant the
`resolve_partition_start()` PG query ran twice on every run pod cold start
(once per import). Centralising here halves that.

If you add a new daily-partitioned asset, import `DAILY_PARTITIONS` from
this module rather than re-creating it. Test
`tests/test_partitions_module.py` enforces the single-instance invariant.
"""

from __future__ import annotations

from dagster import BackfillPolicy, DailyPartitionsDefinition
from meridian_pipeline.db import resolve_partition_start

# end_offset=1 makes the current UTC day a valid partition key. The
# post-close evaluation window of stale_factor_sensor (17:00 ET onwards)
# falls before 00:00 UTC rollover, so without this offset the sensor
# couldn't target today's ET date as a partition.
DAILY_PARTITIONS = DailyPartitionsDefinition(start_date=resolve_partition_start(), end_offset=1)

# Multi-partition backfill policy for the daily factor assets.
#
# Phase 2 of #62. The asset compute bodies were refactored in Phase 1 to
# iterate `context.partition_keys`; attaching this policy lets the backfill
# daemon batch up to 50 partitions per run pod instead of launching one
# pod per partition. For a 1000-partition rerun (2010-2013), that's
# ~20 pods - cuts cold-start overhead 50x vs naive 1-per-partition, keeps
# cluster pressure low (max 6 concurrent runs platform-wide), and limits
# the OOM blast radius to 50 partitions if a pod fails mid-batch.
#
# Bump from 10 -> 50 (Meridian#79, 2026-05-13) for the 2010-2013 backfill.
# Lower numbers caused too many pods to queue against the platform's
# code_location cap of 2; bigger batches mean fewer pods and less cluster
# churn. The trade-off is longer-running pods (~15-25 min each at 50);
# both ends of the spectrum hit the same total throughput because the
# tenant cap is the binding constraint, not per-pod latency.
#
# All daily assets MUST share this same instance: Dagster requires
# co-materialized assets to carry a compatible BackfillPolicy or it
# refuses to build the implicit asset job. Importing from one module
# enforces the lockstep.
#
# Daily schedule (single partition) is unaffected - the policy only
# activates during backfills with N > 1 partitions.
DAILY_BACKFILL_POLICY = BackfillPolicy.multi_run(max_partitions_per_run=50)
