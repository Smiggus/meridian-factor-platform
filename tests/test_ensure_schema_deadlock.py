"""
Tests for the ensure_schema deadlock fix (Meridian, 2026-07-24).

Concurrent backfill runs can deadlock
(psycopg2.errors.DeadlockDetected, pgcode 40P01) inside ensure_schema's DDL
bootstrap. The fix has two layers:

  1. _table_and_columns_exist() fast path - the ensure_*_table helpers skip
     DDL entirely (no AccessExclusiveLock) once the schema is already
     current.
  2. _execute_ddl_with_deadlock_retry() - retries a deadlocked DDL
     transaction with backoff instead of propagating the failure.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_DAGSTER_DIR = Path(__file__).resolve().parents[1] / "dagster"
if str(_DAGSTER_DIR) not in sys.path:
    sys.path.insert(0, str(_DAGSTER_DIR))

os.environ.setdefault("PARTITION_START_DATE", "2010-01-02")
os.environ.setdefault("PGPASS", "test-pg-pass")


class _FakeDeadlockOrig(Exception):
    pgcode = "40P01"


def _make_deadlock_error():
    from sqlalchemy.exc import OperationalError

    return OperationalError("statement", {}, _FakeDeadlockOrig("deadlock detected"))


def test_is_deadlock_error_matches_pgcode():
    from meridian_pipeline import db

    assert db._is_deadlock_error(_make_deadlock_error())


def test_is_deadlock_error_false_for_other_errors():
    from meridian_pipeline import db
    from sqlalchemy.exc import OperationalError

    other = OperationalError("statement", {}, Exception("connection refused"))
    assert not db._is_deadlock_error(other)


def test_execute_ddl_retries_then_succeeds(monkeypatch):
    """A deadlock on the first attempt is retried and succeeds on the second,
    without raising and without sleeping for real."""
    from meridian_pipeline import db

    monkeypatch.setattr(db.time, "sleep", lambda _seconds: None)

    calls = {"n": 0}

    class _FakeConn:
        def execute(self, *_args, **_kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise _make_deadlock_error()

    class _FakeBeginCtx:
        def __enter__(self):
            return _FakeConn()

        def __exit__(self, *exc_info):
            return False

    class _FakeEngine:
        def begin(self):
            return _FakeBeginCtx()

    db._execute_ddl_with_deadlock_retry(_FakeEngine(), "CREATE TABLE IF NOT EXISTS x (y int);")

    assert calls["n"] == 2


def test_execute_ddl_gives_up_after_max_retries(monkeypatch):
    """A non-deadlock or persistently-deadlocking error still raises -
    retry must not silently swallow real failures."""
    from meridian_pipeline import db

    monkeypatch.setattr(db.time, "sleep", lambda _seconds: None)

    class _FakeConn:
        def execute(self, *_args, **_kwargs):
            raise _make_deadlock_error()

    class _FakeBeginCtx:
        def __enter__(self):
            return _FakeConn()

        def __exit__(self, *exc_info):
            return False

    class _FakeEngine:
        def begin(self):
            return _FakeBeginCtx()

    from sqlalchemy.exc import OperationalError

    with pytest.raises(OperationalError):
        db._execute_ddl_with_deadlock_retry(_FakeEngine(), "CREATE TABLE IF NOT EXISTS x (y int);")
