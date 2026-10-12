"""
Tests for scripts/burst/burst_common.py's ingest-window wiring:
`BurstConfig.from_env` must actually reach `refuse_if_ingest_window_open`
(scripts/burst/ingest_window.py), the one construction call every burst
driver makes before doing any read-heavy work.

Background: LDP's nightly OHLCV ingest window rewrites/repairs ohlcv.prices,
and a read-heavy campaign launched during it can read a half-rewritten
history. tests/test_ingest_window_guard.py pins the guard's own logic
(window parsing, boundary behaviour, overrides) against ingest_window.py
directly and needs no burst dependencies. This module additionally needs
burst_common - and therefore boto3 (via BurstConfig.s3_client / imports) -
so it is skipped under CI's slim `test` job, which deliberately does not
install boto3 (root requirements.txt / requirements-test.txt), and only
runs under `test-dagster`, whose image pulls boto3 in transitively via
dagster-aws[s3].
"""

import contextlib
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("boto3")

_BURST_DIR = Path(__file__).resolve().parents[1] / "scripts" / "burst"
if str(_BURST_DIR) not in sys.path:
    sys.path.insert(0, str(_BURST_DIR))

import burst_common  # noqa: E402
import ingest_window  # noqa: E402


class _FrozenDatetime(datetime):
    """Subclasses the real datetime so strptime (used by _parse_ingest_window)
    keeps working; only `.now()` is pinned to a fixed instant."""

    _now: "datetime"

    @classmethod
    def now(cls, tz=None):
        return cls._now


@contextlib.contextmanager
def _frozen_at(hour: int, minute: int, second: int = 0):
    # The guard's `datetime.now()` call resolves against ingest_window's
    # module globals (that is where refuse_if_ingest_window_open is
    # defined), not burst_common's, even though burst_common imports and
    # calls the same function object - so freeze it there.
    fixed = _FrozenDatetime(2026, 8, 14, hour, minute, second, tzinfo=timezone.utc)
    _FrozenDatetime._now = fixed
    orig = ingest_window.datetime
    ingest_window.datetime = _FrozenDatetime
    try:
        yield
    finally:
        ingest_window.datetime = orig


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(ingest_window._LDP_INGEST_WINDOW_ENV, raising=False)
    monkeypatch.delenv(ingest_window._LDP_INGEST_WINDOW_FORCE_ENV, raising=False)


def _minimal_burst_env(monkeypatch):
    monkeypatch.setenv("DB_HOST", "example.invalid")
    monkeypatch.setenv("DB_USER", "dbuser")
    monkeypatch.setenv("DB_PASSWORD", "test-pass")
    monkeypatch.setenv("MERIDIAN_MINIO_ACCESS_KEY", "test-access")
    monkeypatch.setenv("MERIDIAN_MINIO_SECRET_KEY", "test-secret")


def test_from_env_refuses_inside_window(monkeypatch):
    _minimal_burst_env(monkeypatch)
    with _frozen_at(22, 30):
        with pytest.raises(RuntimeError):
            burst_common.BurstConfig.from_env("test-run")


def test_from_env_force_ingest_window_flag_overrides(monkeypatch):
    _minimal_burst_env(monkeypatch)
    with _frozen_at(22, 30):
        cfg = burst_common.BurstConfig.from_env("test-run", force_ingest_window=True)
    assert cfg.run_id == "test-run"


def test_from_env_succeeds_outside_window(monkeypatch):
    _minimal_burst_env(monkeypatch)
    with _frozen_at(12, 0):
        cfg = burst_common.BurstConfig.from_env("test-run")
    assert cfg.run_id == "test-run"


# ---------------------------------------------------------------------------
# refuse_if_ingest_incomplete - the watermark preflight itself
# ---------------------------------------------------------------------------


class _FakeCursor:
    def __init__(self, row):
        self._row = row

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        pass

    def fetchone(self):
        return self._row


class _FakeConn:
    def __init__(self, row=None, raise_on_execute=False):
        self._row = row
        self._raise = raise_on_execute

    def cursor(self):
        if self._raise:
            raise RuntimeError('relation "ohlcv.ingest_watermark" does not exist')
        return _FakeCursor(self._row)

    def close(self):
        pass


def test_refuse_if_ingest_incomplete_complete_row_returns_true():
    conn = _FakeConn(row=(datetime(2026, 8, 14).date(), True, 0.998, 1996, 2000))
    assert burst_common.refuse_if_ingest_incomplete(conn) is True


def test_refuse_if_ingest_incomplete_incomplete_row_raises():
    conn = _FakeConn(row=(datetime(2026, 8, 14).date(), False, 0.40, 800, 2000))
    with pytest.raises(RuntimeError) as exc_info:
        burst_common.refuse_if_ingest_incomplete(conn)
    msg = str(exc_info.value)
    assert "2026-08-14" in msg
    assert "40.0%" in msg


def test_refuse_if_ingest_incomplete_empty_table_returns_false():
    conn = _FakeConn(row=None)
    assert burst_common.refuse_if_ingest_incomplete(conn) is False


def test_refuse_if_ingest_incomplete_unreachable_returns_false():
    conn = _FakeConn(raise_on_execute=True)
    assert burst_common.refuse_if_ingest_incomplete(conn) is False


def test_refuse_if_ingest_incomplete_force_kwarg_short_circuits():
    conn = _FakeConn(raise_on_execute=True)  # would blow up if actually queried
    assert burst_common.refuse_if_ingest_incomplete(conn, force=True) is True


def test_refuse_if_ingest_incomplete_force_env_short_circuits(monkeypatch):
    monkeypatch.setenv(burst_common._INGEST_SAFETY_FORCE_ENV, "1")
    conn = _FakeConn(raise_on_execute=True)
    assert burst_common.refuse_if_ingest_incomplete(conn) is True


# ---------------------------------------------------------------------------
# from_env: watermark preferred, clock window as the fallback
# ---------------------------------------------------------------------------


def test_from_env_prefers_complete_watermark_even_inside_clock_window(monkeypatch):
    """The core cutover behaviour: a complete watermark must let a campaign
    start even during the interim clock window - the watermark is a
    strictly cleaner signal and supersedes the wall-clock guess."""
    _minimal_burst_env(monkeypatch)
    complete_row = (datetime(2026, 8, 14).date(), True, 0.998, 1996, 2000)
    with (
        _frozen_at(22, 30),  # inside the default clock window
        patch("psycopg2.connect", return_value=_FakeConn(row=complete_row)),
    ):
        cfg = burst_common.BurstConfig.from_env("test-run")
    assert cfg.run_id == "test-run"


def test_from_env_raises_on_incomplete_watermark_even_outside_clock_window(monkeypatch):
    """An incomplete watermark must refuse even OUTSIDE the clock window -
    the watermark is authoritative once it is usable."""
    _minimal_burst_env(monkeypatch)
    incomplete_row = (datetime(2026, 8, 14).date(), False, 0.40, 800, 2000)
    with (
        _frozen_at(12, 0),  # outside the default clock window
        patch("psycopg2.connect", return_value=_FakeConn(row=incomplete_row)),
    ):
        with pytest.raises(RuntimeError) as exc_info:
            burst_common.BurstConfig.from_env("test-run")
    assert "ingest watermark" in str(exc_info.value)


def test_from_env_falls_back_to_clock_window_when_watermark_unreachable(monkeypatch):
    """Unreachable watermark (e.g. connection failure) must not silently
    mean "safe" - it falls back to the clock-window guard, which still
    refuses when we're inside the window."""
    _minimal_burst_env(monkeypatch)
    with (
        _frozen_at(22, 30),
        patch("psycopg2.connect", side_effect=RuntimeError("connection refused")),
    ):
        with pytest.raises(RuntimeError) as exc_info:
            burst_common.BurstConfig.from_env("test-run")
    assert "ingest window" in str(exc_info.value)


def test_from_env_falls_back_to_clock_window_when_watermark_empty(monkeypatch):
    """Empty watermark table (reachable, zero rows) must also fall back to
    the clock window, not be treated as "safe"."""
    _minimal_burst_env(monkeypatch)
    with (
        _frozen_at(22, 30),
        patch("psycopg2.connect", return_value=_FakeConn(row=None)),
    ):
        with pytest.raises(RuntimeError) as exc_info:
            burst_common.BurstConfig.from_env("test-run")
    assert "ingest window" in str(exc_info.value)


def test_from_env_falls_back_and_succeeds_when_watermark_empty_outside_window(monkeypatch):
    _minimal_burst_env(monkeypatch)
    with (
        _frozen_at(12, 0),
        patch("psycopg2.connect", return_value=_FakeConn(row=None)),
    ):
        cfg = burst_common.BurstConfig.from_env("test-run")
    assert cfg.run_id == "test-run"
