"""
Tests for the standalone-check alert wrapper (Meridian #295 D3).

Two properties this must have, and one it must not lose:
  * a FAILED check reaches ntfy (the gap that lets a universe
    collapse go unnoticed),
  * an ERRORED check reaches ntfy through a DIFFERENT message, because
    "I checked and it is bad" and "I could not check" are different facts,
  * and neither path may ever raise into the run.
"""

import os
import sys
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
    return all(hasattr(dagster, n) for n in ("AssetCheckResult", "AssetCheckSeverity"))


if not _dagster_complete():
    pytest.skip("Dagster not fully installed (slim CI test image)", allow_module_level=True)

from meridian_pipeline.checks import alerting  # noqa: E402

from dagster import AssetCheckResult, AssetCheckSeverity, MetadataValue  # noqa: E402


class _Log:
    def __init__(self):
        self.warnings = []

    def warning(self, msg):
        self.warnings.append(msg)


class _KeyRange:
    start = "2026-03-02"
    end = "2026-03-02"


class _Ctx:
    def __init__(self):
        self.log = _Log()
        self.partition_key_range = _KeyRange()


@pytest.fixture
def sent(monkeypatch):
    calls = []
    monkeypatch.setattr(
        alerting, "notify_failed_checks", lambda *a, **k: calls.append(("failed", a, k)) or True
    )
    monkeypatch.setattr(
        alerting, "notify_errored_check", lambda *a, **k: calls.append(("errored", a, k)) or True
    )
    return calls


def test_passing_check_is_silent(sent):
    @alerting.alerting_check("composite_scores")
    def check_thing(context):
        return AssetCheckResult(passed=True, severity=AssetCheckSeverity.WARN)

    assert check_thing(_Ctx()).passed is True
    assert sent == []


def test_failing_check_alerts_as_failed(sent):
    @alerting.alerting_check("composite_scores")
    def check_thing(context):
        return AssetCheckResult(
            passed=False,
            severity=AssetCheckSeverity.WARN,
            metadata={"note": MetadataValue.text("9 groups lost")},
        )

    result = check_thing(_Ctx())
    assert result.passed is False
    assert [c[0] for c in sent] == ["failed"]
    asset_name, failures = sent[0][1][0], sent[0][1][1]
    assert asset_name == "composite_scores"
    assert failures[0]["name"] == "check_thing"
    assert failures[0]["failed_partitions"] == ["2026-03-02"]
    assert "9 groups lost" in failures[0]["detail"]


def test_raising_check_alerts_as_ERRORED_and_does_not_fail_the_run(sent):
    """The distinction that matters: a broken detector is missing coverage,
    not a data finding, and must not be rendered as one."""

    @alerting.alerting_check("composite_scores")
    def check_thing(context):
        raise ValueError("column daily_scores.grp does not exist")

    result = check_thing(_Ctx())
    assert result.passed is False
    assert result.metadata["outcome"].text == "errored"
    assert [c[0] for c in sent] == ["errored"]
    assert sent[0][1][1] == "check_thing"


def test_blocking_check_reraises_and_leaves_the_page_to_the_run_failure_sensor(sent):
    """A blocking check must still fail its step - silence there would be a
    false green. The failed step makes ntfy_run_failure_sensor page, so the
    wrapper must not also page: one incident, one alert."""

    @alerting.alerting_check("momentum_scores", swallow_errors=False)
    def check_thing(context):
        raise ValueError("boom")

    with pytest.raises(ValueError):
        check_thing(_Ctx())
    assert sent == []


def test_passing_result_is_stamped_passed(sent):
    @alerting.alerting_check("composite_scores")
    def check_thing(context):
        return AssetCheckResult(passed=True, metadata={"rows": MetadataValue.int(3)})

    result = check_thing(_Ctx())
    assert result.metadata["outcome"].text == "passed"
    assert result.metadata["rows"].value == 3


def test_failing_result_is_stamped_failed(sent):
    @alerting.alerting_check("composite_scores")
    def check_thing(context):
        return AssetCheckResult(passed=False, severity=AssetCheckSeverity.WARN)

    result = check_thing(_Ctx())
    assert result.metadata["outcome"].text == "failed"
    assert [c[0] for c in sent] == ["failed"]


def test_outcome_set_by_the_check_is_kept(sent):
    @alerting.alerting_check("composite_scores")
    def check_thing(context):
        return AssetCheckResult(passed=True, metadata={"outcome": MetadataValue.text("custom")})

    assert check_thing(_Ctx()).metadata["outcome"].text == "custom"


def test_stamping_also_applies_when_blocking(sent):
    @alerting.alerting_check("momentum_scores", swallow_errors=False)
    def check_thing(context):
        return AssetCheckResult(passed=True)

    assert check_thing(_Ctx()).metadata["outcome"].text == "passed"


def test_alert_transport_failure_never_breaks_the_check(monkeypatch):
    def _explode(*a, **k):
        raise RuntimeError("ntfy is down")

    monkeypatch.setattr(alerting, "notify_failed_checks", _explode)

    @alerting.alerting_check("composite_scores")
    def check_thing(context):
        return AssetCheckResult(passed=False, severity=AssetCheckSeverity.WARN)

    ctx = _Ctx()
    assert check_thing(ctx).passed is False
    assert any("ntfy alert failed" in w for w in ctx.log.warnings)


def test_decorator_preserves_the_check_name():
    """@asset_check reads __name__ to derive the check key; losing it renames
    every wrapped check and orphans its UI history."""

    @alerting.alerting_check("composite_scores")
    def check_named_thing(context):
        return AssetCheckResult(passed=True)

    assert check_named_thing.__name__ == "check_named_thing"
