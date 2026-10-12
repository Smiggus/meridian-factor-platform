"""The promoted-range verifier compares the served payload exactly.

`match_frames` is the whole correctness claim of scripts/burst/verify_promoted_range.py:
"gold equals the staged payload". These tests pin the two ways that claim can go
quietly wrong - a ticker present on one side only (membership diff), and a pair that
does not join at all (which must NOT read as zero difference).
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

_REPO = Path(__file__).resolve().parents[1]
for _p in (_REPO, _REPO / "dagster", _REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

pytest.importorskip("boto3", reason="slim CI does not install boto3")

from burst import verify_promoted_range as vpr  # noqa: E402


def _frame(rows):
    return pd.DataFrame(rows, columns=["ticker", "factor_name", "z_score"])


def test_identical_frames_pass_exactly():
    f = _frame([("AAA", "mom", 1.5), ("BBB", "mom", -0.5)])
    diff, dz = vpr.match_frames(f.copy(), f.copy())
    assert diff == 0
    assert dz == 0.0
    assert dz < vpr.TOL


def test_missing_ticker_is_a_membership_diff_not_a_pass():
    staged = _frame([("AAA", "mom", 1.5), ("BBB", "mom", -0.5)])
    gold = _frame([("AAA", "mom", 1.5)])
    diff, dz = vpr.match_frames(staged, gold)
    assert diff == 1
    # dz alone is 0 here, which is exactly why the caller must AND the two.
    assert dz == 0.0


def test_disjoint_frames_cannot_read_as_zero_difference():
    staged = _frame([("AAA", "mom", 1.5)])
    gold = _frame([("ZZZ", "mom", 1.5)])
    diff, dz = vpr.match_frames(staged, gold)
    assert diff == 2  # both sides unmatched
    assert dz == float("inf")


def test_null_z_on_one_side_is_a_membership_diff():
    staged = _frame([("AAA", "mom", 1.5)])
    gold = _frame([("AAA", "mom", None)])
    diff, dz = vpr.match_frames(staged, gold)
    assert diff == 1
    assert dz == float("inf")


def test_a_real_value_difference_is_visible_above_the_tolerance():
    staged = _frame([("AAA", "mom", 1.5)])
    gold = _frame([("AAA", "mom", 1.5 + 10 * vpr.TOL)])
    diff, dz = vpr.match_frames(staged, gold)
    assert diff == 0
    assert dz > vpr.TOL
