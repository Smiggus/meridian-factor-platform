"""Regression: the bulk-COPY promote path must mirror like the upsert path.

!407 makes db.copy_replace_daily_scores the DEFAULT write path for
scripts/burst/promote_backfill.py. promote_backfill runs
lakehouse.assert_dual_write_ready() as a preflight, so a COPY path that does
not dual-write would make the preflight pass while every promoted partition
silently missed the Iceberg / flat-parquet mirrors - the same silent-bypass
class as Meridian#266.

These tests fail on a copy_replace_daily_scores that skips
_dual_write_daily_scores.
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


OWNED = ["earnings_yield", "book_to_market"]


def test_copy_path_keeps_iceberg_until_checked_cutover():
    """Bulk COPY must keep the existing automatic mirror until cutover."""
    from meridian_pipeline import db

    engine, raw, _cur = _mock_copy_engine()
    df = _df()

    with (
        patch("meridian_pipeline.lakehouse.iceberg_dual_write_enabled", return_value=True),
        patch("meridian_pipeline.lakehouse.write_daily_scores") as mock_write,
        patch("meridian_pipeline.parquet_io.append_factor_scores_partition"),
    ):
        n = db.copy_replace_daily_scores(
            engine, df, owned_factors=OWNED, allow_unwritten_owned_factors=True
        )

    assert n == len(df)
    raw.commit.assert_called_once()
    mock_write.assert_called_once()


def test_copy_path_writes_flat_parquet_mirror_with_sanitized_rows():
    """The flat-parquet mirror must fire too, and must receive the same
    NaN/inf-sanitized values the COPY buffer carried (never the raw df)."""
    from meridian_pipeline import db

    engine, _raw, _cur = _mock_copy_engine()
    df = _df(2)
    df.loc[0, "z_score"] = float("nan")
    df.loc[1, "raw_value"] = float("inf")

    with (
        patch("meridian_pipeline.lakehouse.iceberg_dual_write_enabled", return_value=False),
        patch("meridian_pipeline.parquet_io.append_factor_scores_partition") as mock_pq,
    ):
        db.copy_replace_daily_scores(
            engine, df, owned_factors=OWNED, allow_unwritten_owned_factors=True
        )

    mock_pq.assert_called_once(), "COPY promote silently skipped the parquet mirror"
    passed = mock_pq.call_args[0][0]
    assert isinstance(passed, pd.DataFrame)
    assert list(passed["ticker"]) == list(df["ticker"])
    # inf is coerced away (it would never reach PG float8); NaN stays NaN once
    # pandas re-boxes the None into a float column, same as the upsert path.
    import numpy as np

    assert not np.isinf(passed["raw_value"].astype(float)).any()
    assert pd.isna(passed["z_score"].iloc[0])
    assert pd.isna(passed["raw_value"].iloc[1])


def test_copy_path_mirror_failure_does_not_propagate():
    """PG is authoritative: a mirror fault must not fail a committed promote."""
    from meridian_pipeline import db

    engine, _raw, _cur = _mock_copy_engine()
    df = _df()

    with (
        patch("meridian_pipeline.lakehouse.iceberg_dual_write_enabled", return_value=True),
        patch(
            "meridian_pipeline.lakehouse.write_daily_scores",
            side_effect=RuntimeError("iceberg unreachable"),
        ),
        patch(
            "meridian_pipeline.parquet_io.append_factor_scores_partition",
            side_effect=RuntimeError("minio unreachable"),
        ),
    ):
        n = db.copy_replace_daily_scores(
            engine, df, owned_factors=OWNED, allow_unwritten_owned_factors=True
        )  # must not raise

    assert n == len(df)


def test_copy_path_does_not_mirror_when_pg_write_fails():
    """A rolled-back COPY must never leave the mirrors ahead of PG."""
    from meridian_pipeline import db

    engine, raw, cur = _mock_copy_engine()
    cur.copy_expert.side_effect = RuntimeError("copy blew up")
    df = _df()

    with (
        patch("meridian_pipeline.lakehouse.iceberg_dual_write_enabled", return_value=True),
        patch("meridian_pipeline.lakehouse.write_daily_scores") as mock_write,
        patch("meridian_pipeline.parquet_io.append_factor_scores_partition") as mock_pq,
        pytest.raises(RuntimeError),
    ):
        db.copy_replace_daily_scores(
            engine, df, owned_factors=OWNED, allow_unwritten_owned_factors=True
        )

    raw.rollback.assert_called_once()
    mock_write.assert_not_called()
    mock_pq.assert_not_called()


def test_copy_and_upsert_mirror_identically():
    """Both daily_scores write paths route through the same helper, so neither
    can drift into a silent-bypass. Guards future write paths too."""
    from meridian_pipeline import db

    df = _df()

    copy_engine, _raw, _cur = _mock_copy_engine()
    upsert_engine = MagicMock()
    upsert_engine.begin.return_value.__enter__.return_value = MagicMock()
    upsert_engine.begin.return_value.__exit__.return_value = False

    with patch("meridian_pipeline.db._dual_write_daily_scores") as mock_dw:
        db.copy_replace_daily_scores(
            copy_engine, df, owned_factors=OWNED, allow_unwritten_owned_factors=True
        )
        copy_call = mock_dw.call_args

        mock_dw.reset_mock()
        # OWNED is a superset of this fixture's factors (the canonical-superset
        # writer shape), so opt out of the #378 delete-without-replace check -
        # what is under test here is mirror parity, not delete scope.
        db.upsert_daily_scores(
            upsert_engine,
            df,
            replace=True,
            owned_factors=OWNED,
            allow_unwritten_owned_factors=True,
        )
        upsert_call = mock_dw.call_args

    assert copy_call is not None and upsert_call is not None
    assert copy_call.kwargs["replace"] == upsert_call.kwargs["replace"]
    assert copy_call.kwargs["owned_factors"] == upsert_call.kwargs["owned_factors"]
    pd.testing.assert_frame_equal(
        copy_call.args[0].reset_index(drop=True),
        upsert_call.args[0].reset_index(drop=True),
        check_dtype=False,
    )
