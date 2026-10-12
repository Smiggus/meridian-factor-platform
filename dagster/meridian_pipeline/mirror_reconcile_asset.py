"""Fail-closed checked mirror for both score tables (ships unscheduled).

The sensor supplies a committed group-score date; the nightly sweep enumerates
recent completed composite dates independently. Neither silently skips a missing
table or missing date. Production schedules remain stopped pending approval.
"""

import os
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from sqlalchemy import text

from dagster import AssetExecutionContext, Field, MetadataValue, Output, asset
from meridian_pipeline import checked_mirror, db
from meridian_pipeline.k8s_config import LIGHT_OP_TAGS

_DEFAULT_WINDOW_MONTHS = 24


def _sweep_end_date():
    # The sweep runs in the morning ET, after the preceding session's scoring.
    return datetime.now(ZoneInfo("America/New_York")).date() - timedelta(days=1)


def _window_months() -> int:
    """Legacy parity spot-check floor; not the nightly repair interval."""
    return int(os.environ.get("MERIDIAN_MIRROR_RECONCILE_WINDOW_MONTHS", _DEFAULT_WINDOW_MONTHS))


def _window_start(end_date: date, window_months: int) -> date:
    year, month = end_date.year, end_date.month - (window_months - 1)
    while month <= 0:
        month += 12
        year -= 1
    return date(year, month, 1)


@asset(
    group_name="meridian_factors",
    op_tags=LIGHT_OP_TAGS,
    config_schema={
        "score_date": Field(str, is_required=False),
        "score_dates": Field([str], is_required=False),
        "start_date": Field(str, is_required=False),
        "end_date": Field(str, is_required=False),
    },
    description="Checked, month-aligned PG to Iceberg repair for both score tables.",
)
def daily_scores_mirror_reconcile(context: AssetExecutionContext) -> Output:
    # Dagster's stored RUNNING state overrides STOPPED defaults. Require an
    # explicit rollout opt-in even for a previously enabled schedule.
    if os.environ.get("MERIDIAN_CHECKED_MIRROR_ENABLED", "false").lower() != "true":
        raise RuntimeError("checked mirror disabled pending approved production cutover")
    engine = db.engine_from_env()
    config = context.op_execution_context.op_config
    explicit = config.get("score_date")
    batch = config.get("score_dates")
    if sum(bool(config.get(key)) for key in ("score_date", "score_dates", "start_date")) > 1:
        raise ValueError("choose one mirror date scope")
    if batch:
        days = sorted({date.fromisoformat(day) for day in batch})
        explicit = True
        lo, latest = days[0], days[-1]
    elif explicit:
        lo = latest = date.fromisoformat(explicit)
        days = [latest]
    else:
        if bool(config.get("start_date")) != bool(config.get("end_date")):
            raise ValueError("start_date and end_date must be supplied together")
        if config.get("start_date"):
            lo = date.fromisoformat(config["start_date"])
            latest = date.fromisoformat(config["end_date"])
            if lo > latest:
                raise ValueError("invalid mirror date range")
        else:
            if db.get_latest_score_date(engine) is None:
                raise RuntimeError("no completed score date in PostgreSQL")
            latest = _sweep_end_date()
            lo = latest - timedelta(days=7)
        with engine.connect() as conn:
            days = (
                conn.execute(
                    text(
                        "SELECT DISTINCT score_date FROM factors.composite_scores "
                        "WHERE score_date BETWEEN :lo AND :hi ORDER BY score_date"
                    ),
                    {"lo": lo, "hi": latest},
                )
                .scalars()
                .all()
            )
    if not days:
        raise RuntimeError("no completed dates in reconciliation window")
    missing = set()
    if not explicit:
        expected = set()
        cursor = lo
        while cursor <= latest:
            if db.is_trading_day(engine, cursor):
                expected.add(cursor)
            cursor += timedelta(days=1)
        missing = expected - set(days)
        if missing:
            context.log.error(
                f"completed date inventory has missing trading days: {sorted(missing)}"
            )
    results = {}
    failures = []
    for day in days:
        with engine.connect() as conn:
            complete = conn.execute(
                text(
                    "SELECT EXISTS (SELECT 1 FROM factors.composite_scores WHERE score_date=:day) "
                    "AND EXISTS (SELECT 1 FROM factors.group_scores WHERE score_date=:day)"
                ),
                {"day": day},
            ).scalar_one()
        if not complete:
            failures.append(f"{day}: incomplete phase-2 scoring; refusing mirror")
            context.log.error(failures[-1])
            continue
        try:
            results[str(day)] = checked_mirror.reconcile_both(engine, day)
            context.log.info(f"{day}: checked mirror evidence {results[str(day)]}")
        except Exception as exc:
            failures.append(f"{day}: {exc}")
            context.log.exception(f"{day}: checked mirror failed")
    if missing or failures:
        raise RuntimeError(
            f"mirror sweep checked {len(results)} dates; missing trading days: {sorted(missing)}; "
            f"failures: {failures}"
        )
    return Output(
        value=None,
        metadata={
            "dates_checked": MetadataValue.int(len(days)),
            "evidence": MetadataValue.json(results),
        },
    )
