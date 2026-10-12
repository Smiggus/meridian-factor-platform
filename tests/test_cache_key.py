"""A research cache must MISS on a parameter change, not answer the wrong question.

A fetch cache keyed on a FIXED filename while its function takes varying parameters
returns whatever was cached last. The failure is silent and the output is plausible,
because it IS a real result - to a different question. These tests pin the property
that makes that impossible: every content-determining parameter appears in the path.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from _cache_key import keyed_path  # noqa: E402


def test_same_parameters_give_the_same_path(tmp_path):
    a = keyed_path(tmp_path, "z", anchors=["a", "b"], source="pg")
    b = keyed_path(tmp_path, "z", anchors=["a", "b"], source="pg")
    assert a == b


def test_order_does_not_matter(tmp_path):
    a = keyed_path(tmp_path, "z", anchors=["a", "b"])
    b = keyed_path(tmp_path, "z", anchors=["b", "a"])
    assert a == b, "a set of anchors is the same request whatever order it arrives in"


class TestParameterChangeIsAMiss:
    def test_extra_element_changes_path(self, tmp_path):
        a = keyed_path(tmp_path, "z", anchors=["a", "b"])
        b = keyed_path(tmp_path, "z", anchors=["a", "b", "c"])
        assert a != b

    def test_different_element_changes_path(self, tmp_path):
        """The dangerous case: same COUNT, different content."""
        a = keyed_path(tmp_path, "z", anchors=["a", "b"])
        b = keyed_path(tmp_path, "z", anchors=["a", "c"])
        assert a != b

    def test_scalar_change_changes_path(self, tmp_path):
        assert keyed_path(tmp_path, "z", start="2020-01-01") != keyed_path(
            tmp_path, "z", start="2021-01-01"
        )

    def test_added_parameter_changes_path(self, tmp_path):
        assert keyed_path(tmp_path, "z", start="2020-01-01") != keyed_path(
            tmp_path, "z", start="2020-01-01", end="2025-12-31"
        )

    def test_source_change_changes_path(self, tmp_path):
        """PG vs mirror answer differently; they must not share a cache."""
        assert keyed_path(tmp_path, "z", source="pg") != keyed_path(tmp_path, "z", source="mirror")


def test_count_is_visible_in_the_name(tmp_path):
    """A wrong element count should be readable without decoding the digest."""
    p = keyed_path(tmp_path, "z", anchors=[f"d{i}" for i in range(94)])
    assert "anchors-94-" in p.name


def test_directory_is_created(tmp_path):
    p = keyed_path(tmp_path / "nested" / "deep", "z", start="x")
    assert p.parent.exists()
