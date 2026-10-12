"""CRSP code 100 means "still trading" and must never delist a name (#325).

Guards an ABSENCE-shaped input: when a delisting flag is wrong, names vanish
from the factor surface silently and nothing errors. A wrong flag drops live
names from every scored date.
"""

import re
import sys
from pathlib import Path

_DAGSTER_DIR = Path(__file__).resolve().parents[1] / "dagster"
if str(_DAGSTER_DIR) not in sys.path:
    sys.path.insert(0, str(_DAGSTER_DIR))

from meridian_pipeline import ohlcv_hygiene as oh  # noqa: E402

_SRC = (_DAGSTER_DIR / "meridian_pipeline" / "ohlcv_hygiene.py").read_text()


def test_code_100_is_declared_non_delisting():
    assert "crsp_dlstcd_100" in oh.NON_DELISTING_REASONS


def test_the_set_is_immutable():
    """A mutable default here would let one caller silently widen the filter."""
    assert isinstance(oh.NON_DELISTING_REASONS, frozenset)


def test_query_actually_applies_the_exclusion():
    """The constant is worthless if the SQL does not reference it."""
    body = _SRC[_SRC.index("def fetch_delisted_tickers(") :]
    body = body[: body.index("\ndef ")]
    assert "non_delisting" in body, "fetch_delisted_tickers does not bind the exclusion param"
    assert re.search(r"delisting_reason\s*=\s*ANY\(:non_delisting\)", body), (
        "the exclusion is not applied against delisting_reason in the WHERE clause"
    )
    assert "NOT (" in body, "the predicate must EXCLUDE these reasons, not select them"


def test_exclusion_tolerates_null_reason():
    """Rows with a real delisting_date but NULL reason must still delist.

    Postgres `x = ANY(...)` is NULL for NULL x, and `NOT NULL` is NULL, which is
    not TRUE - so without an explicit IS NULL branch every legacy row with no
    reason recorded would stop delisting. That would turn a fix for
    over-delisting into a silent under-delisting bug.
    """
    body = _SRC[_SRC.index("def fetch_delisted_tickers(") :]
    body = body[: body.index("\ndef ")]
    assert "delisting_reason IS NULL" in body


def test_real_codes_are_not_excluded():
    """Only genuine non-delisting markers belong here. Real delisting
    codes must stay out of the set; excluding them would hide real delistings."""
    for real in ("01", "10", "03", "07", "crsp_dlstcd_584", "crsp_dlstcd_231"):
        assert real not in oh.NON_DELISTING_REASONS


def test_the_cutoff_pattern_is_documented_in_the_rationale():
    """The comment must keep describing the shared cut-off date pattern, so a
    future reader can re-verify rather than trust the assertion."""
    assert "IDENTICAL" in _SRC
