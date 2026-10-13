"""The checked mirror must catch divergences aggregate checksums miss."""

import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

_DAGSTER_DIR = Path(__file__).resolve().parents[1] / "dagster"
if str(_DAGSTER_DIR) not in sys.path:
    sys.path.insert(0, str(_DAGSTER_DIR))
from meridian_pipeline.mirror_digest import content_evidence

COLUMNS = (
    "score_date",
    "ticker",
    "factor_name",
    "group_name",
    "raw_value",
    "z_score",
    "direction",
    "computed_at",
)
KEY = ("score_date", "ticker", "factor_name")
NOW = datetime(2026, 9, 24, 12, tzinfo=timezone.utc)
A = (date(2026, 9, 24), "ABC", "quality", "quality", 1.0, 0.5, 1, NOW)
B = (date(2026, 9, 24), "XYZ", "quality", "quality", 2.0, -0.5, -1, NOW)


def evidence(rows):
    return content_evidence(rows, COLUMNS, KEY)


def test_permutation_and_timezone_preserve_digest():
    shifted = A[:-1] + (NOW.astimezone(timezone(timedelta(hours=2))),)
    assert evidence([A, B]) == evidence([B, shifted])
    assert evidence([A, B])["identities"] == 1


@pytest.mark.parametrize(
    "changed",
    [
        A[:4] + (2.0, A[5], A[6], A[7]),
        A[:5] + (-0.5, A[6], A[7]),
        A[:6] + (-1, A[7]),
        A[:-1] + (NOW + timedelta(seconds=1),),
        (A[0], "WRONG", *A[2:]),
    ],
)
def test_same_count_correction_is_detected(changed):
    assert evidence([A, B])["digest"] != evidence([changed, B])["digest"]


def test_missing_extra_and_empty_dates_are_distinct():
    assert evidence([])["rows"] == 0
    assert evidence([])["digest"] != evidence([A])["digest"]
    assert evidence([A])["digest"] != evidence([A, B])["digest"]


def test_duplicate_key_fails_closed():
    with pytest.raises(ValueError, match="duplicate"):
        evidence([A, A[:4] + (9.0, *A[5:])])


def test_identities_key_counts_distinct_labels_not_rows():
    other = (A[0], "ABC", "value", "value", 3.0, 0.1, 1, NOW)
    assert evidence([A, B, other])["rows"] == 3
    assert evidence([A, B, other])["identities"] == 2
    no_label = content_evidence([(A[0], "ABC", 1.0)], ("score_date", "ticker", "x"), ("ticker",))
    assert no_label["identities"] == 0


def test_group_identity_and_null_finite_normalization():
    cols = ("score_date", "ticker", "group_name", "group_score")
    key = ("score_date", "ticker", "group_name")
    row = (A[0], A[1], "quality", float("inf"))
    null_row = (A[0], A[1], "quality", None)
    assert content_evidence([row], cols, key) == content_evidence([null_row], cols, key)
    assert content_evidence([null_row], cols, key)["identities"] == 1


def test_evidence_uses_disk_scratch_and_cleans_up(tmp_path, monkeypatch):
    monkeypatch.setenv("MERIDIAN_MIRROR_SCRATCH_DIR", str(tmp_path))

    def rows():
        yield A
        assert list(tmp_path.iterdir()), "evidence must spill identities and hashes to disk"
        yield B

    assert evidence(rows()) == evidence([B, A])
    assert not list(tmp_path.iterdir())
