"""
Unit tests for data_quality_checks pure helper functions (Meridian #144).

DB-free: all tests exercise the pass/fail decision helpers that accept plain
Python values, not Dagster contexts or SQLAlchemy engines.

The same Dagster-availability guard used in test_risk_factor_scores.py lines
26-38 is applied here so the suite gracefully degrades in the slim CI image.
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
    return all(
        hasattr(dagster, n)
        for n in ("asset", "AssetExecutionContext", "AssetMaterialization", "Output")
    )


if not _dagster_complete():
    pytest.skip("Dagster not fully installed (slim CI test image)", allow_module_level=True)

from meridian_pipeline.checks.data_quality_checks import (  # noqa: E402
    _COVERAGE_MIN_TICKERS,
    _LIQUID_UNSCORED_MAX_TICKERS,
    _PINNED_FLOOR_MAX_PCT,
    _PINNED_FLOOR_VALUE,
    _UNIVERSE_DRIFT_MAX_PCT,
    _ZSCORE_MEAN_MAX_ABS,
    _ZSCORE_STD_HIGH,
    _ZSCORE_STD_LOW,
    _latest_expected_trading_day,
    _warn_on_error,
    composite_freshness_ok,
    coverage_ok,
    liquid_unscored_ok,
    pinned_at_floor_ok,
    split_bias_ok,
    universe_drift_ok,
    zscore_drift_ok,
)

# ---------------------------------------------------------------------------
# coverage_ok
# ---------------------------------------------------------------------------


def test_coverage_ok_passes_at_threshold():
    assert coverage_ok(_COVERAGE_MIN_TICKERS) is True


def test_coverage_ok_passes_above_threshold():
    assert coverage_ok(4_000) is True


def test_coverage_ok_fails_below_threshold():
    assert coverage_ok(_COVERAGE_MIN_TICKERS - 1) is False


def test_coverage_ok_fails_zero():
    assert coverage_ok(0) is False


# ---------------------------------------------------------------------------
# zscore_drift_ok
# ---------------------------------------------------------------------------


def test_zscore_drift_ok_passes_zero_mean_unit_std():
    ok, msg = zscore_drift_ok(0.0, 1.0)
    assert ok is True
    assert msg == ""


def test_zscore_drift_ok_passes_near_zero_mean():
    ok, msg = zscore_drift_ok(_ZSCORE_MEAN_MAX_ABS, 1.0)
    assert ok is True


def test_zscore_drift_ok_fails_mean_too_large():
    ok, msg = zscore_drift_ok(_ZSCORE_MEAN_MAX_ABS + 0.01, 1.0)
    assert ok is False
    assert "mean=" in msg


def test_zscore_drift_ok_fails_negative_mean_too_large():
    ok, msg = zscore_drift_ok(-(_ZSCORE_MEAN_MAX_ABS + 0.01), 1.0)
    assert ok is False
    assert "mean=" in msg


def test_zscore_drift_ok_fails_std_too_low():
    ok, msg = zscore_drift_ok(0.0, _ZSCORE_STD_LOW - 0.01)
    assert ok is False
    assert "std=" in msg


def test_zscore_drift_ok_fails_std_too_high():
    ok, msg = zscore_drift_ok(0.0, _ZSCORE_STD_HIGH + 0.01)
    assert ok is False
    assert "std=" in msg


def test_zscore_drift_ok_passes_at_std_boundaries():
    ok_low, _ = zscore_drift_ok(0.0, _ZSCORE_STD_LOW)
    ok_high, _ = zscore_drift_ok(0.0, _ZSCORE_STD_HIGH)
    assert ok_low is True
    assert ok_high is True


# ---------------------------------------------------------------------------
# pinned_at_floor_ok (added 2026-08-01)
#
# The gap this closes: a [-1, 1] range bound on example_level_a is passed by a
# value of -0.99999. These tests pin both the gap and the thresholds, on
# synthetic cross-sections.
# ---------------------------------------------------------------------------


def test_pinned_value_passes_split_bias_but_not_pinned_check():
    """The regression that motivated this check: -0.99999 is INSIDE the
    split-bias range, so only the share-based check can see it."""
    assert split_bias_ok("example_level_a", -0.99999)[0] is True
    # 5% of the (synthetic) cross-section pinned there.
    ok, msg = pinned_at_floor_ok("example_level_a", 100, 2000)
    assert ok is False
    assert "example_level_a" in msg


def test_pinned_at_floor_ok_healthy_share_passes():
    # Synthetic healthy share: 5/2000 = 0.25%, well under the threshold.
    ok, msg = pinned_at_floor_ok("example_level_a", 5, 2000)
    assert ok is True
    assert msg == ""


def test_pinned_at_floor_ok_broken_share_fails():
    # Synthetic broken state: 100/2000 = 5.00%.
    ok, msg = pinned_at_floor_ok("example_level_a", 100, 2000)
    assert ok is False
    assert "5.00%" in msg


def test_pinned_at_floor_ok_modest_broken_share_fails():
    # A modest corrupted share must still trip the threshold, not just the
    # worst day.
    ok, _ = pinned_at_floor_ok("example_level_a", 353, 10_000)
    assert ok is False


def test_pinned_at_floor_ok_covid_crash_rate_passes():
    # [redacted: internal research note]
    # A tiny pinned share must not trip the check.
    ok, _ = pinned_at_floor_ok("example_level_b", 1, 3000)
    assert ok is True


def test_pinned_at_floor_ok_at_threshold_passes():
    # Strictly-greater comparison: exactly at the threshold is not a failure.
    pct = _PINNED_FLOOR_MAX_PCT["example_level_a"]
    ok, _ = pinned_at_floor_ok("example_level_a", int(pct * 100), 10_000)
    assert ok is True


def test_pinned_at_floor_ok_unknown_factor_passes():
    ok, msg = pinned_at_floor_ok("pe_ratio", 999, 1000)
    assert ok is True
    assert msg == ""


def test_pinned_at_floor_ok_empty_cross_section_passes():
    assert pinned_at_floor_ok("example_level_a", 0, 0)[0] is True


def test_pinned_floor_value_is_inside_split_bias_range():
    """Documents the relationship between the two checks: the floor probe
    must sit INSIDE the range check, or it is redundant with it."""
    assert -1.0 < _PINNED_FLOOR_VALUE < 0.0
    for factor in _PINNED_FLOOR_MAX_PCT:
        assert split_bias_ok(factor, _PINNED_FLOOR_VALUE - 0.009)[0] is True


# ---------------------------------------------------------------------------
# split_bias_ok
# ---------------------------------------------------------------------------


def test_split_bias_no_longer_covers_the_ath_factors():
    """#295 D5: the range legs for the pinned-at-floor factors were DELETED.

    They bounded a quantity that is algebraically confined to [-1, 0] to
    [-1, 1], so no corruption of any magnitude could violate them. The tests that
    used to live here asserted the same identity from the other side
    (-1.5 is out of range, 1.5 is out of range) and were green for exactly
    the same vacuous reason: neither value is one the defect can produce.
    check_dq_pinned_at_floor owns this defect class now, on a statistic the
    corruption actually moves. See tests/test_check_assertion_sensitivity.py.
    """
    from meridian_pipeline.checks.data_quality_checks import _SPLIT_BIAS_BOUNDS

    assert "example_level_a" not in _SPLIT_BIAS_BOUNDS
    assert "example_level_b" not in _SPLIT_BIAS_BOUNDS
    # Unknown factors pass by design, so the corrupted value now passes here
    # and is caught by pinned_at_floor_ok instead.
    assert split_bias_ok("example_level_a", -0.99999)[0] is True


def test_split_bias_ok_example_ratio_a_in_range():
    ok, msg = split_bias_ok("example_ratio_a", 1.5)
    assert ok is True


def test_split_bias_ok_example_ratio_a_at_upper_bound():
    # Bound is 75.0; 20.0 is well within range.
    ok, msg = split_bias_ok("example_ratio_a", 20.0)
    assert ok is True


def test_split_bias_ok_example_ratio_a_normal_spike():
    # A genuine volume spike of 50x is within the sanity bound.
    ok, msg = split_bias_ok("example_ratio_a", 50.0)
    assert ok is True


def test_split_bias_ok_example_ratio_a_above_bound():
    # Values above 75.0 indicate a pathological split-bias regression.
    ok, msg = split_bias_ok("example_ratio_a", 150.0)
    assert ok is False
    assert "example_ratio_a" in msg


def test_split_bias_ok_example_ratio_a_negative():
    ok, msg = split_bias_ok("example_ratio_a", -1.0)
    assert ok is False


def test_split_bias_ok_unknown_factor_passes():
    ok, msg = split_bias_ok("made_up_factor", 9_999.0)
    assert ok is True
    assert msg == ""


# ---------------------------------------------------------------------------
# universe_drift_ok
# ---------------------------------------------------------------------------


def test_universe_drift_ok_passes_no_change():
    ok, msg = universe_drift_ok(4_000, 4_000)
    assert ok is True
    assert msg == ""


def test_universe_drift_ok_passes_small_drift():
    ok, msg = universe_drift_ok(4_100, 4_000)
    assert ok is True


def test_universe_drift_ok_passes_at_threshold():
    # Exactly at _UNIVERSE_DRIFT_MAX_PCT% - should pass (threshold is exclusive >).
    prev = 4_000
    curr = int(prev * (1 + _UNIVERSE_DRIFT_MAX_PCT / 100.0))
    ok, msg = universe_drift_ok(curr, prev)
    assert ok is True


def test_universe_drift_ok_fails_large_drop():
    # 50% drop - well above the threshold.
    ok, msg = universe_drift_ok(2_000, 4_000)
    assert ok is False
    assert "%" in msg


def test_universe_drift_ok_fails_large_spike():
    ok, msg = universe_drift_ok(8_000, 4_000)
    assert ok is False
    assert "%" in msg


def test_universe_drift_ok_passes_when_prev_zero():
    # No prior data available - cannot compute drift, skip gracefully.
    ok, msg = universe_drift_ok(4_000, 0)
    assert ok is True


def test_universe_drift_ok_message_contains_counts():
    ok, msg = universe_drift_ok(1_000, 4_000)
    assert not ok
    assert "curr=1000" in msg or "1000" in msg
    assert "prev=4000" in msg or "4000" in msg


# ---------------------------------------------------------------------------
# _latest_expected_trading_day
# ---------------------------------------------------------------------------


def test_latest_expected_trading_day_weekday_unchanged():
    from datetime import date

    # Monday 2026-06-22
    d = date(2026, 6, 22)
    assert _latest_expected_trading_day(d) == d


def test_latest_expected_trading_day_saturday_rolls_to_friday():
    from datetime import date

    sat = date(2026, 6, 20)  # Saturday
    assert _latest_expected_trading_day(sat) == date(2026, 6, 19)  # Friday


def test_latest_expected_trading_day_sunday_rolls_to_friday():
    from datetime import date

    sun = date(2026, 6, 21)  # Sunday
    assert _latest_expected_trading_day(sun) == date(2026, 6, 19)  # Friday


# ---------------------------------------------------------------------------
# composite_freshness_ok
# ---------------------------------------------------------------------------


def test_composite_freshness_ok_same_day():
    from datetime import date

    d = date(2026, 6, 20)
    ok, msg = composite_freshness_ok(d, d)
    assert ok is True
    assert msg == ""


def test_composite_freshness_ok_one_day_lag_passes():
    from datetime import date

    ok, msg = composite_freshness_ok(date(2026, 6, 19), date(2026, 6, 20), max_lag_days=1)
    assert ok is True


def test_composite_freshness_ok_two_day_lag_fails_with_default():
    from datetime import date

    ok, msg = composite_freshness_ok(date(2026, 6, 18), date(2026, 6, 20), max_lag_days=1)
    assert ok is False
    assert "2" in msg  # lag in message


def test_composite_freshness_ok_message_contains_dates():
    from datetime import date

    ok, msg = composite_freshness_ok(date(2026, 6, 15), date(2026, 6, 20))
    assert not ok
    assert "2026-06-15" in msg
    assert "2026-06-20" in msg


# ---------------------------------------------------------------------------
# _warn_on_error decorator
# ---------------------------------------------------------------------------


def test_warn_on_error_returns_warn_result_on_exception():
    """A wrapped function that raises must return WARN AssetCheckResult, not re-raise."""
    from dagster import AssetCheckResult, AssetCheckSeverity

    def _always_raises(context):
        raise RuntimeError("boom")

    wrapped = _warn_on_error(_always_raises)
    result = wrapped(context=None)  # context=None is fine; the body raises before using it

    assert isinstance(result, AssetCheckResult)
    assert result.passed is False
    assert result.severity == AssetCheckSeverity.WARN


def test_warn_on_error_passes_through_on_success():
    """A wrapped function that succeeds returns its result unchanged."""
    from dagster import AssetCheckResult, AssetCheckSeverity

    expected = AssetCheckResult(passed=True, severity=AssetCheckSeverity.WARN)

    def _succeeds(context):
        return expected

    wrapped = _warn_on_error(_succeeds)
    result = wrapped(context=None)

    assert result is expected


def test_warn_on_error_preserves_function_name():
    """functools.wraps must preserve __name__ for Dagster check registration."""

    def check_something(context):
        pass

    wrapped = _warn_on_error(check_something)
    assert wrapped.__name__ == "check_something"


# ---------------------------------------------------------------------------
# _era_gate (historical-era downgrade for older-era vendor data)
# ---------------------------------------------------------------------------


def test_era_gate_historical_only_violations_pass_with_warn():
    from meridian_pipeline.checks.ret1d_checks import _era_gate

    from dagster import AssetCheckSeverity

    passed, sev = _era_gate([{"score_date": "2013-11-04", "reasons": "max=321>2.0"}])
    assert passed is True
    assert sev == AssetCheckSeverity.WARN


def test_era_gate_modern_violation_still_fails_with_error():
    from meridian_pipeline.checks.ret1d_checks import _era_gate

    from dagster import AssetCheckSeverity

    passed, sev = _era_gate(
        [
            {"score_date": "2013-11-04", "reasons": "max=321>2.0"},
            {"score_date": "2026-06-29", "reasons": "max=18>2.0"},
        ]
    )
    assert passed is False
    assert sev == AssetCheckSeverity.ERROR


def test_era_gate_clean_input_passes_with_error_severity_unused():
    from meridian_pipeline.checks.ret1d_checks import _era_gate

    from dagster import AssetCheckSeverity

    passed, sev = _era_gate([])
    assert passed is True
    assert sev == AssetCheckSeverity.ERROR


# ---------------------------------------------------------------------------
# liquid_unscored_ok (reused-ticker hygiene)
#
# The contradiction-shaped counterpart to universe_drift_ok. Drift fires on
# any large count change, including entirely legitimate bulk ingests of
# delisted tickers; this one fires only when names that genuinely TRADED
# on the score date are missing from the cross-section, which no legitimate
# universe change produces.
# ---------------------------------------------------------------------------


def test_liquid_unscored_ok_passes_when_none_missing():
    ok, msg = liquid_unscored_ok(0)
    assert ok is True
    assert msg == ""


def test_liquid_unscored_ok_passes_at_threshold():
    # Threshold is exclusive (>), so exactly at the tolerance still passes -
    # a handful of liquid non-universe names (fresh IPOs, ADRs) is normal.
    ok, _ = liquid_unscored_ok(_LIQUID_UNSCORED_MAX_TICKERS)
    assert ok is True


def test_liquid_unscored_ok_fails_on_a_large_count():
    # A large count of actively-trading names wrongly excluded by a
    # reused-ticker delisting bug.
    ok, msg = liquid_unscored_ok(_LIQUID_UNSCORED_MAX_TICKERS * 5)
    assert ok is False
    assert str(_LIQUID_UNSCORED_MAX_TICKERS * 5) in msg


def test_liquid_unscored_ok_fails_one_past_threshold():
    ok, _ = liquid_unscored_ok(_LIQUID_UNSCORED_MAX_TICKERS + 1)
    assert ok is False
