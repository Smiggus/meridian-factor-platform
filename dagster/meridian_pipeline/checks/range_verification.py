"""
range_verification.py
----------------------
Tier 2 of Meridian #285: bulk SET-LEVEL verification over a whole backfill
date range, run ONCE in a single pod.

Context (#285): per-partition asset checks cost a k8s pod each on a
multi-partition backfill (much of the cost is pure pod-startup overhead).
Cheap PER-DATE checks (e.g. does this one date have
a trading-day-valid score_date) are being moved in-step into the scoring
asset itself so they no longer fan out a pod per partition - see
trading_day_checks.py / factor_checks.py / data_quality_checks.py.

This module is the complement: truths that are only meaningful across a
whole RANGE of dates - completeness, thin days, coverage drift, cross-date
distribution shape - and would be nonsensical (or needlessly expensive) to
ask per-partition. `verify_range(engine, start, end)` runs once over the
whole range in a single pod/process and returns a structured
VerificationReport; it does not print-only.

Checks:

  1. partition_completeness - every trading day in [start, end] (canonical
     calendar) is present in every SCORED_TABLES table.
  2. non_trading_orphans    - no scored row sits on a non-trading date, in
     any SCORED_TABLES table.
     (1) and (2) reuse meridian_pipeline.checks.trading_day_checks's
     load_trading_calendar / load_table_dates / non_trading_dates so the
     canonical calendar predicate (db.is_trading_day's >100-equity-rows
     definition) is never duplicated here - see that module's docstring for
     why it is legitimately duplicated ONCE (as a set query) relative to
     db.is_trading_day (a per-date query), and why a third copy would be
     one duplication too many.
  3. row_count_floors  - dates present in factors.daily_scores but with a
     row count suspiciously thin vs. the range's own median.
  4. universe_drift     - day-over-day swings in factors.daily_scores'
     distinct-ticker count outside a tolerance band.
  5. distribution_sanity - dates whose factors.composite_scores per-date
     mean sits far outside the range's own mean-of-means (a coarse
     range-level control; NOT a substitute for the per-partition z-score
     drift check in data_quality_checks.py, which compares one partition to
     its immediate predecessor).

ADVISORY ONLY (owner decision, #285), AND NOW ALERTING (#295 D3/D5).
verify_range NEVER raises and backfill_verification_job NEVER fails its step
because of a finding. What changed on 2026-08-03: findings now reach ntfy via
notify_failed_checks, because a set-level control whose only output is a log
line in a job nobody watches is not a control (design doc P4).

The `blocking` field and the `blocking_failed` property were REMOVED in the
same change (design doc D5: "wire it or remove it"). Every check shipped
advisory, so `blocking_failed` was documented as "always empty today" and the
one branch that read it could never differ from its else-branch: a
forward-compat hook that has never been exercised reads as coverage that does
not exist, which is precisely the false-assurance class this program is
deleting. Hardening one check to hard-fail later is a change at the call site
either way.

Shared code path: this module backs BOTH scripts/verify_range.py (operator
CLI) and backfill_verification_job.py (the Dagster op the sensor fires) -
same function call, so "the sensor found X" and "an operator ran this by
hand and found X" can never disagree.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from statistics import mean, median, pstdev
from typing import Any

from sqlalchemy import text

from meridian_pipeline.checks.trading_day_checks import (
    SCORED_TABLES,
    load_table_dates,
    load_trading_calendar,
    non_trading_dates,
)


@dataclass
class CheckResult:
    """One named check's outcome. Advisory: see the module docstring."""

    name: str
    passed: bool
    severity: str  # "warn" | "error" - informational, nothing here ever raises
    message: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationReport:
    start: date
    end: date
    checks: list[CheckResult]

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "all_passed": self.all_passed,
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "severity": c.severity,
                    "message": c.message,
                    "metadata": c.metadata,
                }
                for c in self.checks
            ],
        }


# Pure decision helpers (DB-free, unit-tested)


def missing_dates(calendar: set[date], present: set[date]) -> list[date]:
    """Calendar trading days with no row at all in a scored table."""
    return sorted(calendar - present)


def thin_dates(counts: dict[date, int], factor: float = 0.5) -> list[date]:
    """Dates whose row count is below `factor` * the range's own median
    (computed over dates with a nonzero count). Needs at least 3 nonzero
    dates to form a meaningful median; returns [] below that (too small a
    range to judge "thin" against)."""
    nonzero = [n for n in counts.values() if n > 0]
    if len(nonzero) < 3:
        return []
    floor = median(nonzero) * factor
    return sorted(d for d, n in counts.items() if 0 < n < floor)


def day_over_day_drift(
    counts: dict[date, int], pct_threshold: float = 0.35
) -> list[tuple[date, date, float]]:
    """Day-over-day fractional changes between consecutive PRESENT dates in
    the series (not necessarily adjacent calendar days - gaps are reported
    separately by missing_dates) exceeding pct_threshold. Returns
    (prev_date, date, pct_change) triples, sorted by date."""
    dates = sorted(counts)
    out: list[tuple[date, date, float]] = []
    for prev, cur in zip(dates, dates[1:]):
        p, c = counts[prev], counts[cur]
        if p <= 0:
            continue
        pct = (c - p) / p
        if abs(pct) > pct_threshold:
            out.append((prev, cur, round(pct, 4)))
    return out


def distribution_outliers(
    daily_means: dict[date, float], z_thresh: float = 4.0
) -> list[tuple[date, float]]:
    """Dates whose per-date mean sits more than z_thresh population-stddevs
    from the range's own mean-of-means. Needs at least 5 dates to form a
    meaningful spread; returns [] below that or when the range has zero
    variance (e.g. every date identical - nothing to flag)."""
    values = list(daily_means.values())
    if len(values) < 5:
        return []
    m = mean(values)
    sd = pstdev(values)
    if sd == 0:
        return []
    return sorted(
        ((d, round(v, 6)) for d, v in daily_means.items() if abs(v - m) > z_thresh * sd),
        key=lambda t: t[0],
    )


def load_row_counts(engine, table: str, start: date, end: date) -> dict[date, int]:
    """Per-date row count for a scored table in [start, end]."""
    if table not in SCORED_TABLES:
        raise ValueError(f"{table!r} is not a known scored table")
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                f"SELECT score_date, COUNT(*) FROM {table} "
                "WHERE score_date BETWEEN :s AND :e GROUP BY score_date"
            ),
            {"s": start, "e": end},
        ).fetchall()
    return {r[0]: r[1] for r in rows}


def load_universe_counts(engine, start: date, end: date) -> dict[date, int]:
    """Per-date distinct-ticker count on factors.daily_scores in [start, end]."""
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT score_date, COUNT(DISTINCT ticker) FROM factors.daily_scores "
                "WHERE score_date BETWEEN :s AND :e GROUP BY score_date"
            ),
            {"s": start, "e": end},
        ).fetchall()
    return {r[0]: r[1] for r in rows}


def load_composite_means(engine, start: date, end: date) -> dict[date, float]:
    """Per-date mean composite_score on factors.composite_scores in [start, end]."""
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT score_date, AVG(composite_score) FROM factors.composite_scores "
                "WHERE score_date BETWEEN :s AND :e AND composite_score IS NOT NULL "
                "GROUP BY score_date"
            ),
            {"s": start, "e": end},
        ).fetchall()
    return {r[0]: float(r[1]) for r in rows if r[1] is not None}


# verify_range - the Tier 2 entry point


def verify_range(engine, start: date, end: date) -> VerificationReport:
    """Run every set-level check over [start, end] and return a
    VerificationReport. Never raises on a failing check - see the module
    docstring's "ADVISORY ONLY" section. Genuine infrastructure errors
    (a DB connection failure, an unknown table) still propagate normally;
    only *findings* are advisory, not the ability to run the sweep at all.
    """
    checks: list[CheckResult] = []

    calendar = load_trading_calendar(engine, start, end)

    # --- 1 & 2: partition completeness + non-trading orphans, one pass per
    # table so both checks share the single load_table_dates round trip.
    missing_by_table: dict[str, list[date]] = {}
    orphans_by_table: dict[str, list[date]] = {}
    for table in SCORED_TABLES:
        table_dates = load_table_dates(engine, table, start, end)
        table_dates_set = set(table_dates)
        missing = missing_dates(calendar, table_dates_set)
        if missing:
            missing_by_table[table] = missing
        orphans = non_trading_dates(table_dates, calendar)
        if orphans:
            orphans_by_table[table] = orphans

    checks.append(
        CheckResult(
            name="partition_completeness",
            passed=not missing_by_table,
            severity="warn",
            message=(
                "all scored tables carry every trading day in range"
                if not missing_by_table
                else "missing trading-day partitions - "
                + "; ".join(
                    f"{t}: {len(ds)} missing (first {ds[0]}, last {ds[-1]})"
                    for t, ds in sorted(missing_by_table.items())
                )
            ),
            metadata={
                "trading_days_in_range": len(calendar),
                "missing_by_table": {
                    t: [d.isoformat() for d in ds] for t, ds in missing_by_table.items()
                },
            },
        )
    )

    checks.append(
        CheckResult(
            name="non_trading_orphans",
            passed=not orphans_by_table,
            severity="warn",
            message=(
                "no scored rows on non-trading dates"
                if not orphans_by_table
                else "non-trading-day rows present - "
                + "; ".join(
                    f"{t}: {len(ds)} dates (first {ds[0]}, last {ds[-1]})"
                    for t, ds in sorted(orphans_by_table.items())
                )
            ),
            metadata={
                "orphans_by_table": {
                    t: [d.isoformat() for d in ds] for t, ds in orphans_by_table.items()
                },
            },
        )
    )

    # --- 3: row-count floors, evaluated per table against that table's own
    # in-range median (a thin daily_scores day and a thin risk_metrics day
    # are not comparable, so each table gets its own floor).
    thin_by_table: dict[str, list[date]] = {}
    for table in SCORED_TABLES:
        counts = load_row_counts(engine, table, start, end)
        thin = thin_dates(counts)
        if thin:
            thin_by_table[table] = thin

    checks.append(
        CheckResult(
            name="row_count_floors",
            passed=not thin_by_table,
            severity="warn",
            message=(
                "no dates fall below half the table's in-range median row count"
                if not thin_by_table
                else "thin dates present - "
                + "; ".join(
                    f"{t}: {len(ds)} dates (first {ds[0]}, last {ds[-1]})"
                    for t, ds in sorted(thin_by_table.items())
                )
            ),
            metadata={
                "thin_by_table": {
                    t: [d.isoformat() for d in ds] for t, ds in thin_by_table.items()
                },
            },
        )
    )

    # --- 4: universe/coverage drift on factors.daily_scores.
    universe_counts = load_universe_counts(engine, start, end)
    drift = day_over_day_drift(universe_counts)
    checks.append(
        CheckResult(
            name="universe_drift",
            passed=not drift,
            severity="warn",
            message=(
                "no day-over-day universe swing exceeds 35%"
                if not drift
                else f"{len(drift)} day-over-day universe swing(s) exceed 35%"
            ),
            metadata={
                "drift": [
                    {"from": p.isoformat(), "to": c.isoformat(), "pct_change": pct}
                    for p, c, pct in drift
                ],
            },
        )
    )

    # --- 5: cross-date distribution sanity on factors.composite_scores.
    composite_means = load_composite_means(engine, start, end)
    outliers = distribution_outliers(composite_means)
    checks.append(
        CheckResult(
            name="distribution_sanity",
            passed=not outliers,
            severity="warn",
            message=(
                "no date's composite_score mean is a range-level outlier"
                if not outliers
                else f"{len(outliers)} date(s) with an outlier composite_score mean"
            ),
            metadata={
                "outliers": [{"date": d.isoformat(), "mean": v} for d, v in outliers],
            },
        )
    )

    return VerificationReport(start=start, end=end, checks=checks)
