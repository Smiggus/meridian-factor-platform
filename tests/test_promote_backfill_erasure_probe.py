"""The promote's exhaustive erasure probe must cast its date array.

`gold_factor_presence` (scripts/burst/promote_backfill.py) passes the ISO date
strings produced by `_date_of(key)` into `score_date = ANY(:dates)`. psycopg2 binds
a Python list of str as text[], and PostgreSQL has no `date = text` operator, so the
uncorrected form raises `operator does not exist: date = text` on the first call.

It stayed latent because the probe is only issued when an owned factor never appears
in the 12-date coverage sample - a promote that reproduces its whole scope skips it
entirely (see the comment above the SQL). Any load that legitimately does not
reproduce a factor on a sampled date issues it: a bulk load that excludes
window-sensitive factors from staging by design, or a panel whose coverage
ends early, leaves factors out of a late sample. The probe raised before the
gate could report.
"""

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
for _p in (_REPO, _REPO / "dagster", _REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# scripts/burst/promote_backfill.py imports boto3 at module level and the slim CI
# job does not install it, so the whole module skips there rather than erroring at
# collection (same guard as tests/test_burst_ingest_window_guard.py).
pytest.importorskip("boto3", reason="slim CI does not install boto3")

from burst import promote_backfill as promote  # noqa: E402


def test_presence_probe_casts_the_date_array():
    sql = promote._GOLD_FACTOR_PRESENCE_SQL
    assert "CAST(:dates AS date[])" in sql
    # The bare form is the bug: pin its absence so a later edit cannot drop the cast
    # back to the shape that raises.
    assert "score_date = ANY(:dates)" not in sql
    assert "factor_name = ANY(:factors)" in sql


def test_presence_probe_binds_strings():
    """The cast is required precisely because the caller passes ISO strings, so the
    parameter shape is part of the contract under test."""
    import inspect

    src = inspect.getsource(promote.gold_factor_presence)
    assert '"dates": list(dates)' in src


def test_presence_probe_short_circuits_on_empty_input():
    assert promote.gold_factor_presence(None, [], ["example_factor"]) == {}
    assert promote.gold_factor_presence(None, ["2026-08-03"], []) == {}
