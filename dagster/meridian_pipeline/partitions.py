"""Daily partition definitions shared by the Dagster assets.

Import `DAILY_PARTITIONS` from here instead of building another one. Creating it runs the
`resolve_partition_start()` query against PostgreSQL, so every extra copy is an extra query on
each run pod's cold start. `tests/test_partitions_module.py` checks that there is one instance.
"""

from __future__ import annotations

from dagster import BackfillPolicy, DailyPartitionsDefinition
from meridian_pipeline.db import resolve_partition_start

# end_offset=1 makes the current UTC day a valid partition key. stale_factor_sensor evaluates
# after the close (17:00 ET onwards), which is still before the 00:00 UTC rollover, and
# without the offset it could not target today's ET date.
DAILY_PARTITIONS = DailyPartitionsDefinition(start_date=resolve_partition_start(), end_offset=1)

# Backfill policy for the daily factor assets (#62). The asset bodies iterate
# `context.partition_keys`, so the backfill daemon can put up to 50 partitions into one run pod
# instead of starting a pod per partition. A 1000-partition rerun then takes about 20 pods.
#
# The limit was 10 until Meridian#79 (2026-05-13), raised for the 2010-2013 backfill. At 10 the
# runs queued behind the platform's cap of 2 code-location pods. Each pod now runs longer, about
# 15-25 minutes at 50, and one that dies mid-batch costs up to 50 partitions of rework. Total
# throughput barely moved because the tenant cap is what limits it.
#
# Every daily asset has to use this one instance. Dagster refuses to build the implicit asset
# job when co-materialized assets carry different BackfillPolicies.
#
# The daily schedule runs a single partition and never triggers the batching.
DAILY_BACKFILL_POLICY = BackfillPolicy.multi_run(max_partitions_per_run=50)
