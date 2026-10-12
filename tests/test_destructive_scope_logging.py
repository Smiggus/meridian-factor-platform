"""QoL 12 (Meridian#468): destructive replace deletes log their scope first.

Before replace_date_range / owned_factors deletes (PG) and the Iceberg
overwrite filters (daily scores) run, an INFO line names the
table, the date span, the factor count and the expected row bound, so a delete
that is far wider than its payload is visible before the commit.
"""

import logging
import sys
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dagster"))
from meridian_pipeline import db, lakehouse

LO, HI = date(2025, 9, 1), date(2025, 9, 5)


def _df(dates=(LO, HI), factors=("roe",)):
    return pd.DataFrame(
        [
            dict(
                score_date=d,
                ticker="A",
                factor_name=f,
                group_name="quality",
                raw_value=0.1,
                z_score=0.5,
                direction=1,
            )
            for d in dates
            for f in factors
        ]
    )


def _scope_records(caplog):
    return [r for r in caplog.records if "about to" in r.getMessage()]


def _conn_that_sees_log_at_delete(caplog, seen):
    conn = MagicMock()

    def execute(stmt, params=None):
        if str(stmt).startswith("DELETE"):
            seen.append(len(_scope_records(caplog)))
        res = MagicMock()
        res.rowcount = 0
        res.fetchall.return_value = []
        return res

    conn.execute.side_effect = execute
    return conn


def _engine(conn):
    eng = MagicMock()
    eng.begin.return_value.__enter__.return_value = conn
    eng.begin.return_value.__exit__.return_value = False
    return eng


def test_replace_date_range_logs_scope_before_the_delete(caplog):
    seen = []
    conn = _conn_that_sees_log_at_delete(caplog, seen)
    with (
        caplog.at_level(logging.INFO, logger=db._log.name),
        patch.object(db, "missing_trading_days", return_value=[]),
        patch.object(db, "resolve_campaign_id", return_value=None),
        patch.object(db, "_dual_write_daily_scores"),
    ):
        db.upsert_daily_scores(
            _engine(conn),
            _df(),
            replace=True,
            owned_factors=["roe", "pe"],
            allow_unwritten_owned_factors=True,
            replace_date_range=(LO, HI),
        )
    recs = _scope_records(caplog)
    assert len(recs) == 1
    msg = recs[0].getMessage()
    assert "table=factors.daily_scores" in msg
    assert "dates=2025-09-01..2025-09-05" in msg
    assert "span_days=5" in msg and "owned_factors=2" in msg
    assert "payload_rows=2" in msg and "key_space_days_x_factors=10" in msg
    assert seen and all(n == 1 for n in seen)  # logged before the DELETE ran


def test_owned_factors_delete_logs_scope(caplog):
    conn = _conn_that_sees_log_at_delete(caplog, [])
    with (
        caplog.at_level(logging.INFO, logger=db._log.name),
        patch.object(db, "resolve_campaign_id", return_value=None),
        patch.object(db, "_dual_write_daily_scores"),
    ):
        db.upsert_daily_scores(_engine(conn), _df(), replace=True, owned_factors=["roe"])
    (rec,) = _scope_records(caplog)
    assert "upsert_daily_scores(owned_factors)" in rec.getMessage()
    assert "payload_dates=2" in rec.getMessage() and "owned_factors=1" in rec.getMessage()


def test_non_destructive_write_logs_no_scope(caplog):
    conn = _conn_that_sees_log_at_delete(caplog, [])
    with (
        caplog.at_level(logging.INFO, logger=db._log.name),
        patch.object(db, "resolve_campaign_id", return_value=None),
        patch.object(db, "_dual_write_daily_scores"),
    ):
        db.upsert_daily_scores(_engine(conn), _df(), replace=False)
    assert _scope_records(caplog) == []


def test_copy_replace_logs_scope(caplog):
    cur = MagicMock()
    cur.rowcount = 0
    raw = MagicMock()
    raw.cursor.return_value = cur
    eng = MagicMock()
    eng.raw_connection.return_value = raw
    with (
        caplog.at_level(logging.INFO, logger=db._log.name),
        patch.object(db, "resolve_campaign_id", return_value=None),
        patch.object(db, "_campaign_column_available", return_value=False),
        patch.object(db, "_dual_write_daily_scores"),
    ):
        db.copy_replace_daily_scores(eng, _df(), owned_factors=["roe"])
    (rec,) = _scope_records(caplog)
    assert "copy_replace_daily_scores" in rec.getMessage()
    assert "payload_rows=2" in rec.getMessage()


def test_iceberg_range_overwrite_logs_scope_before_commit(caplog):
    pytest.importorskip("pyiceberg")
    table = MagicMock()
    seen = []
    table.overwrite.side_effect = lambda *a, **k: seen.append(len(_scope_records(caplog)))
    with (
        caplog.at_level(logging.INFO, logger=lakehouse.logger.name),
        patch.object(lakehouse, "ensure_daily_scores_table", return_value=table),
    ):
        lakehouse.write_daily_scores(_df(), owned_factors=["roe"], replace_date_range=(LO, HI))
    (rec,) = _scope_records(caplog)
    msg = rec.getMessage()
    assert "table=factors.daily_scores" in msg and "span_days=5" in msg
    assert "owned_factors=1" in msg and "payload_rows=2" in msg
    assert seen == [1]


def test_iceberg_owned_factors_overwrite_logs_scope(caplog):
    pytest.importorskip("pyiceberg")
    table = MagicMock()
    with (
        caplog.at_level(logging.INFO, logger=lakehouse.logger.name),
        patch.object(lakehouse, "ensure_daily_scores_table", return_value=table),
    ):
        lakehouse.write_daily_scores(_df(), owned_factors=["roe"])
    (rec,) = _scope_records(caplog)
    assert "payload_dates=2" in rec.getMessage()
