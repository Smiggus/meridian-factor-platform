"""Tests for the factor-continuity check (Meridian#298).

The defect this guards against: an upstream price gap removes tickers from
the scoreable universe. Broad factors lose part of their cross-section; a
narrow factor can lose all of it and be silently omitted, because every
existing check inspects VALUES and none inspected presence.

DB-free: exercises the pure decision helper (factor_continuity_ok) against
plain Python values, same idiom as tests/test_dq_checks.py. The same
Dagster-availability guard used there is applied here so the suite gracefully
degrades in the slim CI image (importing data_quality_checks pulls in
meridian_pipeline.checks.factor_checks, which needs a complete dagster
install).
"""

import sys
from pathlib import Path

import pytest

_DAGSTER_DIR = Path(__file__).resolve().parents[1] / "dagster"
if str(_DAGSTER_DIR) not in sys.path:
    sys.path.insert(0, str(_DAGSTER_DIR))


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
    _CONTINUITY_MIN_FRACTION,
    _CONTINUITY_MIN_MEDIAN,
    factor_continuity_ok,
)


def test_zero_rows_is_always_reported():
    """A narrow factor goes from some rows to none and is silently dropped."""
    ok, msg = factor_continuity_ok("narrow_factor", 0, 240)
    assert not ok
    assert "ZERO" in msg
    # The message must point upstream - the factor's own code was fine.
    assert "universe" in msg


def test_narrow_factor_collapse_is_caught():
    """A 300-name factor falling to 100 is a 67% loss and must fire.

    This is the case a single global row-count threshold misses: 100 rows looks
    unremarkable next to the broad factors in the same partition.
    """
    ok, msg = factor_continuity_ok("narrow_factor", 100, 300)
    assert not ok
    assert "33%" in msg


def test_broad_factor_ordinary_breathing_does_not_fire():
    """A modest cut from a universe event is ordinary breathing.

    Real cross-sections move with the universe. A check that fires on this gets
    muted, and a muted check is how a gap goes unseen for months.
    """
    ok, msg = factor_continuity_ok("broad_factor", 2640, 3000)
    assert ok, msg


def test_boundary_is_inclusive_of_the_threshold():
    """Exactly at the threshold passes; a hair under fires."""
    med = 1000
    at = int(med * _CONTINUITY_MIN_FRACTION)
    assert factor_continuity_ok("f", at, med)[0]
    assert not factor_continuity_ok("f", at - 1, med)[0]


def test_unseen_factor_is_not_reported():
    """No history means nothing to compare against - not a failure."""
    assert factor_continuity_ok("brand_new", 0, 0)[0]
    assert factor_continuity_ok("brand_new", 5, 0)[0]


def test_tiny_factor_ratio_is_suppressed_but_zero_is_not():
    """Below the median floor a ratio is noise, but vanishing still counts.

    A factor with a trailing median of 3 rows swinging to 1 is not evidence of
    anything. The same factor going to 0 still is.
    """
    tiny = _CONTINUITY_MIN_MEDIAN - 1
    assert factor_continuity_ok("sparse", 1, tiny)[0]
    ok, msg = factor_continuity_ok("sparse", 0, tiny)
    assert not ok and "ZERO" in msg


@pytest.mark.parametrize("median", [10, 250, 2900])
def test_full_coverage_never_fires(median):
    assert factor_continuity_ok("f", median, median)[0]
