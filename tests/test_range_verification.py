"""
Tests for checks.range_verification (Meridian #285 Tier 2).

Two layers:

  * the pure decision helpers (missing_dates, thin_dates, day_over_day_drift,
    distribution_outliers), which are DB-free;
  * verify_range end-to-end against a small SYNTHETIC in-memory dataset that
    deliberately includes a known non-trading date, driven through a fake
    engine so no database is touched.

The fake engine is deliberately thin: it only has to answer the handful of
statements range_verification issues. It is NOT a SQL implementation - each
query is recognised by a distinctive fragment and answered from the synthetic
fixture, so the test asserts the CHECK LOGIC, not the SQL.

Dagster is only partially installed on the slim CI image and
range_verification imports trading_day_checks (which imports Dagster asset
symbols), so the module skips there via a hasattr check (NOT importorskip -
see a repo convention for slim CI).
"""

from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path

import pytest

_DAGSTER_DIR = Path(__file__).resolve().parents[1] / "dagster"
if str(_DAGSTER_DIR) not in sys.path:
    sys.path.insert(0, str(_DAGSTER_DIR))

os.environ.setdefault("PGPASS", "test-pg-pass")
os.environ.setdefault("PARTITION_START_DATE", "2010-01-02")


def _dagster_complete() -> bool:
    try:
        import dagster
    except ImportError:
        return False
    return all(hasattr(dagster, n) for n in ("asset_check", "AssetCheckResult"))


if not _dagster_complete():
    pytest.skip(
        "Dagster not fully installed (slim CI test image)",
        allow_module_level=True,
    )

from meridian_pipeline.checks.range_verification import (  # noqa: E402
    CheckResult,
    VerificationReport,
    day_over_day_drift,
    distribution_outliers,
    missing_dates,
    thin_dates,
    verify_range,
)
from meridian_pipeline.checks.trading_day_checks import SCORED_TABLES  # noqa: E402

# ---------------------------------------------------------------------------
# Synthetic fixture
# ---------------------------------------------------------------------------

# 2020-01-04 is a SATURDAY: a known non-trading date. The calendar below
# omits it while the scored tables carry rows for it, so the orphan check
# has something real to find.
_NON_TRADING = date(2020, 1, 4)
_TRADING_DAYS = [
    date(2020, 1, 2),
    date(2020, 1, 3),
    date(2020, 1, 6),
    date(2020, 1, 7),
    date(2020, 1, 8),
    date(2020, 1, 9),
]


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return list(self._rows)


class _FakeConn:
    def __init__(self, fixture):
        self._f = fixture

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, stmt, params=None):
        sql = str(stmt)
        if "ohlcv.prices" in sql:
            return _FakeResult([(d,) for d in self._f["calendar"]])
        if "WITH RECURSIVE" in sql:
            # load_table_dates formats the table name into the SQL text.
            table = next(t for t in SCORED_TABLES if t in sql)
            return _FakeResult([(d,) for d in sorted(self._f["dates"].get(table, []))])
        if "COUNT(DISTINCT ticker)" in sql:
            return _FakeResult(sorted(self._f["universe"].items()))
        if "AVG(composite_score)" in sql:
            return _FakeResult(sorted(self._f["composite_means"].items()))
        if "COUNT(*)" in sql:
            table = next(t for t in SCORED_TABLES if t in sql)
            return _FakeResult(sorted(self._f["counts"].get(table, {}).items()))
        raise AssertionError(f"unexpected SQL in test: {sql!r}")


class _FakeEngine:
    def __init__(self, fixture):
        self._f = fixture

    def connect(self):
        return _FakeConn(self._f)

    def dispose(self):
        pass


def _fixture(**overrides):
    all_dates = _TRADING_DAYS
    f = {
        "calendar": list(_TRADING_DAYS),
        "dates": {t: list(all_dates) for t in SCORED_TABLES},
        "counts": {t: {d: 3000 for d in all_dates} for t in SCORED_TABLES},
        "universe": {d: 3000 for d in all_dates},
        "composite_means": {d: 0.01 for d in all_dates},
    }
    f.update(overrides)
    return f


def _check(report: VerificationReport, name: str) -> CheckResult:
    return next(c for c in report.checks if c.name == name)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_missing_dates():
    cal = {date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 6)}
    assert missing_dates(cal, {date(2020, 1, 2)}) == [date(2020, 1, 3), date(2020, 1, 6)]
    assert missing_dates(cal, cal) == []


def test_thin_dates_needs_three_points():
    # Below three nonzero dates there is no meaningful median to judge against.
    assert thin_dates({date(2020, 1, 2): 1, date(2020, 1, 3): 5000}) == []


def test_thin_dates_flags_below_half_median():
    counts = {
        date(2020, 1, 2): 3000,
        date(2020, 1, 3): 3100,
        date(2020, 1, 6): 2900,
        date(2020, 1, 7): 40,  # thin
    }
    assert thin_dates(counts) == [date(2020, 1, 7)]


def test_day_over_day_drift():
    counts = {
        date(2020, 1, 2): 1000,
        date(2020, 1, 3): 1050,  # +5%, fine
        date(2020, 1, 6): 300,  # -71%, flagged
    }
    drift = day_over_day_drift(counts)
    assert len(drift) == 1
    prev, cur, pct = drift[0]
    assert (prev, cur) == (date(2020, 1, 3), date(2020, 1, 6))
    assert pct < -0.5


def test_distribution_outliers():
    means = {d: 0.01 for d in _TRADING_DAYS}
    assert distribution_outliers(means) == []  # zero variance, nothing to flag
    means[_TRADING_DAYS[-1]] = 9.0
    flagged = [d for d, _ in distribution_outliers(means, z_thresh=1.5)]
    assert flagged == [_TRADING_DAYS[-1]]


# ---------------------------------------------------------------------------
# verify_range end to end
# ---------------------------------------------------------------------------


def test_verify_range_clean_dataset():
    report = verify_range(_FakeEngine(_fixture()), date(2020, 1, 2), date(2020, 1, 9))
    assert report.all_passed, [c.message for c in report.checks if not c.passed]
    assert {c.name for c in report.checks} == {
        "partition_completeness",
        "non_trading_orphans",
        "row_count_floors",
        "universe_drift",
        "distribution_sanity",
    }


def test_verify_range_detects_non_trading_orphan():
    """A row on 2020-01-04 (a Saturday) must be flagged as an orphan."""
    dates = {t: list(_TRADING_DAYS) for t in SCORED_TABLES}
    dates["factors.daily_scores"] = list(_TRADING_DAYS) + [_NON_TRADING]
    report = verify_range(_FakeEngine(_fixture(dates=dates)), date(2020, 1, 2), date(2020, 1, 9))
    orphans = _check(report, "non_trading_orphans")
    assert not orphans.passed
    assert orphans.metadata["orphans_by_table"]["factors.daily_scores"] == ["2020-01-04"]
    # Completeness is a separate axis and stays clean: every trading day is
    # still present, an extra junk date does not make one missing.
    assert _check(report, "partition_completeness").passed


def test_verify_range_detects_missing_partition():
    dates = {t: list(_TRADING_DAYS) for t in SCORED_TABLES}
    dates["factors.composite_scores"] = [d for d in _TRADING_DAYS if d != date(2020, 1, 7)]
    report = verify_range(_FakeEngine(_fixture(dates=dates)), date(2020, 1, 2), date(2020, 1, 9))
    completeness = _check(report, "partition_completeness")
    assert not completeness.passed
    assert completeness.metadata["missing_by_table"]["factors.composite_scores"] == ["2020-01-07"]


def test_verify_range_detects_thin_day_and_universe_drift():
    counts = {t: {d: 3000 for d in _TRADING_DAYS} for t in SCORED_TABLES}
    counts["factors.daily_scores"][date(2020, 1, 8)] = 20
    universe = {d: 3000 for d in _TRADING_DAYS}
    universe[date(2020, 1, 8)] = 20
    report = verify_range(
        _FakeEngine(_fixture(counts=counts, universe=universe)),
        date(2020, 1, 2),
        date(2020, 1, 9),
    )
    assert not _check(report, "row_count_floors").passed
    assert _check(report, "row_count_floors").metadata["thin_by_table"]["factors.daily_scores"] == [
        "2020-01-08"
    ]
    assert not _check(report, "universe_drift").passed


def test_findings_are_advisory_not_blocking():
    """Owner decision (#285): every check ships advisory. verify_range must
    not raise on a finding so the job can log, alert, and succeed.

    The `blocking` field and `blocking_failed` were removed in #295 D5: they
    were documented "always empty today" and the single branch reading them
    could never differ from its else-branch, i.e. coverage that did not
    exist. Findings now reach an operator through notify_failed_checks
    instead of a hard-fail hook nobody ever wired.
    """
    dates = {t: list(_TRADING_DAYS) for t in SCORED_TABLES}
    dates["factors.daily_scores"] = list(_TRADING_DAYS) + [_NON_TRADING]
    report = verify_range(_FakeEngine(_fixture(dates=dates)), date(2020, 1, 2), date(2020, 1, 9))
    assert not report.all_passed
    assert not hasattr(report, "blocking_failed")
    assert all(not hasattr(c, "blocking") for c in report.checks)
    assert all(hasattr(c, "severity") for c in report.checks)


def test_verify_range_findings_reach_the_alert_path():
    """A set-level control whose only output is a job log is not a control
    (#295 D3.4). The op must hand its findings to notify_failed_checks."""
    src = (
        Path(__file__).resolve().parents[1]
        / "dagster"
        / "meridian_pipeline"
        / "backfill_verification_job.py"
    ).read_text()
    assert "from meridian_pipeline.notify import notify_failed_checks" in src
    assert "notify_failed_checks(" in src


def test_report_to_dict_is_json_shaped():
    import json

    report = verify_range(_FakeEngine(_fixture()), date(2020, 1, 2), date(2020, 1, 9))
    payload = json.loads(json.dumps(report.to_dict()))
    assert payload["start"] == "2020-01-02"
    assert payload["end"] == "2020-01-09"
    assert payload["all_passed"] is True
    assert len(payload["checks"]) == 5


def test_calendar_is_the_shared_definition_not_a_new_one():
    """#280's root cause was a calendar definition drifting between call
    sites. range_verification must reuse the shared helpers, never define
    its own predicate."""
    import inspect

    from meridian_pipeline.checks import range_verification

    src = inspect.getsource(range_verification)
    assert "ohlcv.prices" not in src, "range_verification must not carry its own calendar SQL"
    assert "load_trading_calendar" in src
