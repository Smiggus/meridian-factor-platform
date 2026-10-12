"""
backfill_verification_job.py - Tier 2 of Meridian #285.

Wraps range_verification.verify_range in a single-op Dagster job so it can
be launched two ways:

  - by an operator, via the Dagster launchpad / GraphQL, with start_date /
    end_date in run config (same shape as the other config-driven backfill
    jobs in backfill_jobs.py / incluster_bulk_rescore.py);
  - automatically, by backfill_completion_sensor (sensors.py) once per
    finished backfill (Tier 3), which derives start/end from the completed
    backfill's partition range and fires a RunRequest with matching
    run_config.

READ-ONLY against the DB. ADVISORY ONLY: this job logs every finding at
WARN and always succeeds - see range_verification.py's module docstring for
why (owner decision, #285) and how a specific check can later be hardened to
hard-fail without a rewrite.
"""

from __future__ import annotations

from datetime import date

from dagster import Field, job, op
from meridian_pipeline import db
from meridian_pipeline.checks.range_verification import verify_range
from meridian_pipeline.k8s_config import LIGHT_OP_TAGS
from meridian_pipeline.notify import notify_failed_checks

_CONFIG_SCHEMA = {
    "start_date": Field(str, description="ISO date, inclusive - start of the range to verify."),
    "end_date": Field(str, description="ISO date, inclusive - end of the range to verify."),
}


@op(config_schema=_CONFIG_SCHEMA, tags=LIGHT_OP_TAGS)
def verify_range_op(context) -> dict:
    cfg = context.op_config
    start = date.fromisoformat(cfg["start_date"])
    end = date.fromisoformat(cfg["end_date"])

    engine = db.engine_from_env()
    try:
        report = verify_range(engine, start, end)
    finally:
        engine.dispose()

    findings = [c for c in report.checks if not c.passed]
    for check in findings:
        # Advisory only (owner decision, #285): log at WARN, never raise.
        context.log.warning(f"[verify_range] {check.name}: {check.message}")

    # #295 D3.4: verify_range findings were unreachable by ANY alert path -
    # the one set-level control in the repo could report a broken range into
    # a job log and nothing else. Fail-open, like every other notify caller.
    if findings:
        try:
            notify_failed_checks(
                "verify_range",
                [
                    {
                        "name": c.name,
                        "failed_partitions": [f"{start}..{end}"],
                        "detail": c.message,
                    }
                    for c in findings
                ],
                log=context.log,
            )
        except Exception as exc:  # noqa: BLE001 - alerting must never fail the run
            context.log.warning(f"[verify_range] ntfy alert failed: {exc}")

    context.log.info(
        f"verify_range: {start} .. {end}, "
        f"{sum(1 for c in report.checks if c.passed)}/{len(report.checks)} checks clean"
    )
    return report.to_dict()


@job(
    name="backfill_verification_job",
    description=(
        "Tier 2 of Meridian #285: bulk SET-LEVEL verification over a date "
        "range, run ONCE in a single pod (as opposed to a pod-per-partition "
        "asset check). Checks partition completeness, non-trading-day "
        "orphans, per-date row-count floors, universe/coverage drift, and "
        "cross-date composite-score distribution sanity across every "
        "SCORED_TABLES table. ADVISORY ONLY - logs findings at WARN, never "
        "fails the run. Launched manually (start_date/end_date in run "
        "config) or automatically by backfill_completion_sensor once per "
        "finished backfill (Tier 3, sensors.py)."
    ),
    tags={
        "kind": "verification",
        "group": "meridian_factors",
    },
)
def backfill_verification_job():
    verify_range_op()
