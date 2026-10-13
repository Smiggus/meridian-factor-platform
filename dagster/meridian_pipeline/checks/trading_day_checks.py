"""
Trading-day invariant asset check (meridian-factors code location).

Meridian #174 purged non-trading-day rows from ``factors.daily_scores`` and
declared a trading-day-only invariant on that table
(docs/07-DESIGN_DECISIONS.md). The invariant had no detective control, so it
silently broke again:

  * a derived-table backfill path wrote ``daily_scores`` rows on weekends and
    holidays. That path only grew its own trading-day gate afterwards, i.e.
    AFTER the damage.
  * a backfill discovered its partition set FROM gold (``SELECT DISTINCT
    score_date FROM factors.daily_scores``) rather than from a calendar, so it
    inherited the re-seeded non-trading dates and wrote more groups onto them.

The second case is the one that matters. A writer that derives its partition
set from gold reproduces whatever junk dates gold already carries, so a
one-off purge does not hold the invariant by itself.

The purge covered one table, so the sweep below checks every other
score-date-keyed output table as well.

``check_trading_day_invariant`` detects; it does not prevent. Preventing the
write is the job of the ``_skip_non_trading`` gate in ``assets.py``, and this
check asks whether anything got past it. It sweeps a bounded trailing window
(``_LOOKBACK_DAYS``, default 400 calendar days) of every scored table and
FAILs if any table holds a score_date that is not a trading day. The sweep is
bounded so the check stays cheap on the daily path;
``scripts/check_trading_day_invariant.py`` runs the same logic over full
history for audits and emits the corresponding DELETE statements.

The calendar is built with the same semantics as ``db.is_trading_day``:
a date is a trading day iff ``ohlcv.prices`` holds more than 100
``security_type = 'equity'`` rows for it at ``resolution = 'daily'``. That
definition yields the NYSE trading calendar: no weekend days, and the
non-weekend exclusions are the exchange holidays, including unscheduled
closures. The ``resolution = 'daily'`` filter (Meridian #275 Phase 0) guards
against ``ohlcv.prices`` co-mingling intraday bars (hourly/minute, all with
NULL ret_1d) that would otherwise count toward the >100 threshold and could
manufacture a phantom trading day; this is currently a no-op, but
correctness must not depend on that.

The calendar SQL is duplicated here rather than imported as a per-date
``db.is_trading_day`` call on purpose: the check needs a SET of trading days
in one round trip, and a per-date loop over a 400-day window would be 400
queries. The duplication is asserted by ``test_trading_day_checks.py``, which
pins this module's SQL predicate to the one in ``db.is_trading_day``, so the
two cannot drift silently.

Distinct score_dates per table are read with a loose index scan (recursive
skip-scan) instead of ``SELECT DISTINCT``. Every scored table has score_date
as the leading column of a btree index, so the skip-scan costs O(distinct
dates) index probes. That matters: ``factors.daily_scores`` is a very large
table and a plain ``SELECT DISTINCT score_date`` over a 400-day window scans
a very large number of index entries.

Pure helpers (DB-free, unit-tested):

    non_trading_dates(table_dates, calendar) -> sorted list[date]
    invariant_ok(offenders_by_table)         -> tuple[bool, str]
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Iterable, Mapping

from sqlalchemy import text

from dagster import (
    AssetCheckResult,
    AssetCheckSeverity,
    MetadataValue,
    asset_check,
)
from meridian_pipeline.checks.alerting import alerting_check
from meridian_pipeline.k8s_config import LIGHT_OP_TAGS
from meridian_pipeline.risk_metrics_asset import _get_engine

# Trailing window swept on the daily path. 400 calendar days comfortably
# covers a multi-month backfill landing behind the daily schedule while
# keeping the skip-scan to a few hundred index probes per table. Full-history
# auditing is the standalone script's job, not the daily check's.
_LOOKBACK_DAYS = 400

# The >100-equity-rows threshold that defines a trading day. Kept as a module
# constant so the test can pin it to db.is_trading_day's literal.
_TRADING_DAY_MIN_EQUITY_ROWS = 100

# Every table keyed by score_date that holds scored output. All of these have
# score_date as the leading column of a btree index, so
# the skip-scan below is index-driven on each.
SCORED_TABLES: tuple[str, ...] = (
    "factors.daily_scores",
    "factors.group_scores",
    "factors.composite_scores",
    "factors.risk_metrics",
    "factors.etf_features",
    "factors.specific_returns",
    "factors.specific_risk",
    "factors.factor_returns",
    "factors.factor_covariance",
    # Meridian#379: a derived table with one row per date (PK score_date, so
    # score_date IS the btree leading column). Added because its asset had no
    # trading-day gate at all until #379 - a contiguous-range backfill seeded
    # one junk row here per non-trading date alongside the junk it seeded in
    # daily_scores. Included in the sweep so the preventive gate now on that
    # asset has a matching detective control.
    "factors.regime_snapshots",
)

# Loose index scan (skip-scan) for the distinct score_dates a table holds in a
# window. Table name is interpolated because PostgreSQL has no bind parameter
# for identifiers; every value passed is from SCORED_TABLES, never user input.
_DISTINCT_DATES_SQL = """
WITH RECURSIVE t AS (
    SELECT MIN(score_date) AS score_date
    FROM {tbl}
    WHERE score_date >= :s AND score_date <= :e
    UNION ALL
    SELECT (
        SELECT MIN(x.score_date) FROM {tbl} x
        WHERE x.score_date > t.score_date AND x.score_date <= :e
    )
    FROM t WHERE t.score_date IS NOT NULL
)
SELECT score_date FROM t WHERE score_date IS NOT NULL
"""

# Mirrors db.is_trading_day exactly: >100 daily equity rows on the date. Uses
# the same sargable half-open range form so it can use the ts_event btree
# index (see the perf note in db.is_trading_day).
_CALENDAR_SQL = """
SELECT ts_event::date AS d
FROM ohlcv.prices
WHERE ts_event >= :s AND ts_event < :e_next AND security_type = 'equity'
  AND resolution = 'daily'
GROUP BY 1
HAVING COUNT(*) > :min_rows
"""


# Pure decision helpers (DB-free, unit-tested)


def non_trading_dates(table_dates: Iterable[date], calendar: Iterable[date]) -> list[date]:
    """Dates present in a scored table that are not in the trading calendar."""
    cal = set(calendar)
    return sorted(d for d in set(table_dates) if d not in cal)


def invariant_ok(offenders_by_table: Mapping[str, list[date]]) -> tuple[bool, str]:
    """Return (passed, human-readable reason) for a per-table offender map.

    ANY non-trading-day row in ANY scored table is a failure. There is no
    tolerance band: a weekend score is never legitimate, so a threshold here
    would only serve to hide the thing the check exists to find.
    """
    dirty = {t: ds for t, ds in offenders_by_table.items() if ds}
    if not dirty:
        return True, "all scored tables trading-day-clean in window"
    parts = [
        f"{t}: {len(ds)} dates (first {ds[0]}, last {ds[-1]})" for t, ds in sorted(dirty.items())
    ]
    return False, "non-trading-day rows present - " + "; ".join(parts)


def load_trading_calendar(engine, start: date, end: date) -> set[date]:
    """Trading days in [start, end], per db.is_trading_day's definition."""
    with engine.connect() as conn:
        rows = conn.execute(
            text(_CALENDAR_SQL),
            {
                "s": start,
                "e_next": end + timedelta(days=1),
                "min_rows": _TRADING_DAY_MIN_EQUITY_ROWS,
            },
        ).fetchall()
    return {r[0] for r in rows}


def load_table_dates(engine, table: str, start: date, end: date) -> list[date]:
    """Distinct score_dates a scored table holds in [start, end]."""
    if table not in SCORED_TABLES:
        raise ValueError(f"{table!r} is not a known scored table")
    with engine.connect() as conn:
        rows = conn.execute(
            text(_DISTINCT_DATES_SQL.format(tbl=table)), {"s": start, "e": end}
        ).fetchall()
    return [r[0] for r in rows]


def scan_window(engine, start: date, end: date) -> dict[str, list[date]]:
    """Per-table non-trading score_dates in [start, end]."""
    calendar = load_trading_calendar(engine, start, end)
    out: dict[str, list[date]] = {}
    for table in SCORED_TABLES:
        out[table] = non_trading_dates(load_table_dates(engine, table, start, end), calendar)
    return out


@asset_check(
    asset="composite_scores",
    description=(
        "Detective control for the Meridian #174 trading-day-only invariant. "
        "Sweeps the trailing 400 calendar days of every scored table "
        "(daily_scores, group_scores, composite_scores, risk_metrics, "
        "etf_features, specific_returns, specific_risk, factor_returns, "
        "factor_covariance, regime_snapshots) and FAILs if any of them holds "
        "a score_date that "
        "is not a trading day per the canonical db.is_trading_day definition "
        "(>100 daily equity rows in ohlcv.prices). Non-blocking: the contamination "
        "it detects is historical rows written by some other path, so halting "
        "today's composite would punish the wrong partition. Zero tolerance - "
        "a weekend or holiday score is never legitimate."
    ),
    blocking=False,
    op_tags=LIGHT_OP_TAGS,
)
@alerting_check("composite_scores")
def check_trading_day_invariant(context) -> AssetCheckResult:
    kr = context.partition_key_range
    end_date = date.fromisoformat(kr.end)
    start_date = end_date - timedelta(days=_LOOKBACK_DAYS)
    engine = _get_engine()

    offenders = scan_window(engine, start_date, end_date)
    passed, reason = invariant_ok(offenders)

    dirty = {t: [d.isoformat() for d in ds] for t, ds in offenders.items() if ds}
    if not passed:
        context.log.error(f"[trading_day_invariant] {reason}")

    return AssetCheckResult(
        passed=passed,
        severity=AssetCheckSeverity.ERROR,
        metadata={
            "window": MetadataValue.text(f"{start_date} .. {end_date}"),
            "tables_scanned": MetadataValue.int(len(SCORED_TABLES)),
            "dirty_table_count": MetadataValue.int(len(dirty)),
            "offending_dates": MetadataValue.json(dirty),
            "reason": MetadataValue.text(reason),
        },
    )
