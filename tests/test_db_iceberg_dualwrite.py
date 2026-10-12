"""
Tests for M14.2: best-effort Iceberg dual-write in db.upsert_daily_scores.

Mirrors the existing flat-parquet dual-write pattern (test_parquet_dual_write_
receives_sanitized_rows in test_db_upserts.py): gated behind
lakehouse.iceberg_dual_write_enabled(), must never affect the PG write or
propagate an exception from lakehouse.write_daily_scores.
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


def _mock_engine_capture():
    captured = []
    conn = MagicMock()

    def execute(stmt, params=None):
        captured.append({"sql": str(stmt), "params": params})
        return MagicMock()  # .fetchall() -> [] for the #432 key-exact DELETE

    conn.execute.side_effect = execute
    engine = MagicMock()
    engine.begin.return_value.__enter__.return_value = conn
    engine.begin.return_value.__exit__.return_value = False
    return engine, captured


def _make_daily_scores_df(n_rows: int = 3) -> pd.DataFrame:
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


def test_iceberg_dual_write_not_called_when_flag_off():
    """Default (flag off) behaviour is byte-identical to today: lakehouse.
    write_daily_scores must not be invoked at all."""
    from meridian_pipeline import db

    engine, _captured = _mock_engine_capture()
    df = _make_daily_scores_df()

    with (
        patch("meridian_pipeline.lakehouse.iceberg_dual_write_enabled", return_value=False),
        patch("meridian_pipeline.lakehouse.write_daily_scores") as mock_write,
    ):
        n = db.upsert_daily_scores(engine, df)

    assert n == len(df)
    mock_write.assert_not_called()


def test_iceberg_write_retained_until_approved_cutover():
    """Deploying stopped reconcilers must not freeze the existing mirror."""
    from meridian_pipeline import db

    engine, _captured = _mock_engine_capture()
    df = _make_daily_scores_df()
    owned = ["earnings_yield", "book_to_market"]

    with (
        patch("meridian_pipeline.lakehouse.iceberg_dual_write_enabled", return_value=True),
        patch("meridian_pipeline.lakehouse.write_daily_scores") as mock_write,
    ):
        db.upsert_daily_scores(engine, df, replace=False, owned_factors=owned)

    mock_write.assert_called_once()


def test_iceberg_dual_write_exception_does_not_propagate():
    """A raise inside lakehouse.write_daily_scores must be swallowed - the PG
    upsert already succeeded and must not be affected by an Iceberg failure."""
    from meridian_pipeline import db

    engine, _captured = _mock_engine_capture()
    df = _make_daily_scores_df()

    with (
        patch("meridian_pipeline.lakehouse.iceberg_dual_write_enabled", return_value=True),
        patch(
            "meridian_pipeline.lakehouse.write_daily_scores",
            side_effect=RuntimeError("iceberg unreachable"),
        ),
    ):
        n = db.upsert_daily_scores(engine, df)  # must not raise

    assert n == len(df)


def test_iceberg_dual_write_skipped_when_enabled_check_raises():
    """Even a raise from iceberg_dual_write_enabled() itself (e.g. missing
    pyiceberg / env parsing error) must not propagate out of upsert_daily_scores -
    it is caught by the outer best-effort guard around the whole Iceberg block."""
    from meridian_pipeline import db

    engine, _captured = _mock_engine_capture()
    df = _make_daily_scores_df()

    with patch(
        "meridian_pipeline.lakehouse.iceberg_dual_write_enabled",
        side_effect=ImportError("no pyiceberg"),
    ):
        n = db.upsert_daily_scores(engine, df)  # must not raise

    assert n == len(df)
