"""The bulk-COPY promote path must enforce #378's delete-without-replace guard.

Meridian#378 added ``allow_unwritten_owned_factors`` to
``db.upsert_daily_scores`` and to its call sites, including
``scripts/burst/promote_backfill.py``'s per-partition write. The other
``daily_scores`` write path, ``db.copy_replace_daily_scores`` - which is the
promote's DEFAULT path (bulk COPY, ~10-20x the row-oriented upsert) - never
gained the parameter, so the promote raised
``TypeError: copy_replace_daily_scores() got an unexpected keyword argument
'allow_unwritten_owned_factors'`` on its FIRST partition and no promote could
reach gold through the default path. The assert here is the shape that matters:
default refuses, explicit opt-out is required, and the error names the function
that actually refused.
"""

from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

_DAGSTER_DIR = Path(__file__).resolve().parents[1] / "dagster"
if str(_DAGSTER_DIR) not in sys.path:
    sys.path.insert(0, str(_DAGSTER_DIR))

os.environ.setdefault("PARTITION_START_DATE", "2010-01-02")
os.environ.setdefault("PGPASS", "test-pg-pass")


def _sqlalchemy_complete() -> bool:
    try:
        import sqlalchemy  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _sqlalchemy_complete(),
    reason="SQLAlchemy not installed (slim CI test image)",
)


def _mock_copy_engine():
    """Engine whose raw_connection() records the DELETEs and the COPY payload."""
    cur = MagicMock()
    raw = MagicMock()
    raw.cursor.return_value = cur
    engine = MagicMock()
    engine.raw_connection.return_value = raw
    return engine, raw, cur


def _df(n_rows: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "score_date": [date(2026, 5, 1)] * n_rows,
            "ticker": [f"T{i}" for i in range(n_rows)],
            "factor_name": ["earnings_yield"] * n_rows,
            "group_name": ["value"] * n_rows,
            "raw_value": [0.1 * i for i in range(n_rows)],
            "z_score": [0.05 * i for i in range(n_rows)],
            "direction": [1] * n_rows,
        }
    )


# The payload carries earnings_yield only: OWNED is a superset, i.e. exactly the
# batch-promoter shape (one scope for a whole campaign, written one partition at
# a time).
OWNED = ["earnings_yield", "book_to_market"]


def test_copy_path_refuses_unwritten_owned_factor_by_default():
    """No opt-out means the delete-replace must not silently drop a factor."""
    from meridian_pipeline import db

    engine, _raw, _cur = _mock_copy_engine()

    with pytest.raises(ValueError) as exc:
        db.copy_replace_daily_scores(engine, _df(), owned_factors=OWNED)

    message = str(exc.value)
    assert "copy_replace_daily_scores" in message, (
        "the refusal must name the write path that refused; naming "
        "upsert_daily_scores sends the operator to the wrong call site"
    )
    assert "book_to_market" in message
    assert "delete-without-replace" in message
    # Refused before the transaction opened: nothing was deleted or copied.
    assert not engine.raw_connection.called


def test_copy_path_accepts_the_promote_kwarg_and_writes():
    """The exact call shape scripts/burst/promote_backfill.py makes."""
    from meridian_pipeline import db

    engine, raw, _cur = _mock_copy_engine()

    with patch("meridian_pipeline.db._dual_write_daily_scores"):
        n = db.copy_replace_daily_scores(
            engine, _df(), owned_factors=OWNED, allow_unwritten_owned_factors=True
        )

    assert n == 3
    raw.commit.assert_called_once()


def test_upsert_refusal_still_names_upsert():
    """The shared guard's default caller label did not move."""
    from meridian_pipeline import db

    with pytest.raises(ValueError) as exc:
        db.upsert_daily_scores(MagicMock(), _df(), owned_factors=OWNED)

    assert str(exc.value).startswith("upsert_daily_scores:")
