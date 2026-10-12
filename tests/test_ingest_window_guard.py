"""
Tests for scripts/burst/ingest_window.py's ingest-window campaign-start guard.

Background: LDP's nightly OHLCV ingest window rewrites/repairs ohlcv.prices.
`refuse_if_ingest_window_open` is a preflight that refuses to
START a new burst campaign while the window is open (a read-heavy campaign
launched during the window can read a half-rewritten history); it is wired into
`burst_common.BurstConfig.from_env`, the one construction call every driver
under scripts/burst/ makes before doing any read-heavy work.

The guard itself lives in ingest_window.py - a pure-stdlib module (time
parsing, env var reads, a datetime comparison) split out of burst_common.py
so it can be imported and unit-tested without burst_common's boto3
dependency, which CI's slim `test` job does not install (see
requirements.txt / requirements-test.txt; boto3 only arrives transitively
via dagster-aws[s3] in the test-dagster job). This module tests
ingest_window.py directly; tests/test_burst_ingest_window_wiring.py covers
the BurstConfig.from_env wiring, which does need burst_common/boto3.

The default window bound (21:45-01:15 UTC) is illustrative and configurable.
These tests pin the boundary behaviour and the two override paths (explicit
flag, env var), not the specific numbers chosen for the default.
"""

import contextlib
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

_BURST_DIR = Path(__file__).resolve().parents[1] / "scripts" / "burst"
if str(_BURST_DIR) not in sys.path:
    sys.path.insert(0, str(_BURST_DIR))

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


# ---------------------------------------------------------------------------
# _parse_ingest_window / _time_in_window
# ---------------------------------------------------------------------------


def test_parse_default_window():
    start, end = ingest_window._parse_ingest_window(ingest_window._LDP_INGEST_WINDOW_UTC_DEFAULT)
    assert (start.hour, start.minute) == (21, 45)
    assert (end.hour, end.minute) == (1, 15)


def test_parse_invalid_window_raises():
    with pytest.raises(ValueError):
        ingest_window._parse_ingest_window("not-a-window")


def test_time_in_window_wraps_midnight():
    start, end = ingest_window._parse_ingest_window("21:45-01:15")
    from datetime import time

    assert ingest_window._time_in_window(time(22, 0), start, end) is True
    assert ingest_window._time_in_window(time(0, 30), start, end) is True
    assert ingest_window._time_in_window(time(12, 0), start, end) is False
    # boundaries: start inclusive, end exclusive
    assert ingest_window._time_in_window(time(21, 45), start, end) is True
    assert ingest_window._time_in_window(time(21, 44, 59), start, end) is False
    assert ingest_window._time_in_window(time(1, 15), start, end) is False
    assert ingest_window._time_in_window(time(1, 14, 59), start, end) is True


def test_time_in_window_non_wrapping():
    start, end = ingest_window._parse_ingest_window("09:00-10:00")
    from datetime import time

    assert ingest_window._time_in_window(time(9, 30), start, end) is True
    assert ingest_window._time_in_window(time(8, 59), start, end) is False
    assert ingest_window._time_in_window(time(10, 0), start, end) is False


# ---------------------------------------------------------------------------
# refuse_if_ingest_window_open
# ---------------------------------------------------------------------------


def test_refuses_inside_the_default_window():
    """The core case: launching mid-ingest must be refused loudly."""
    with _frozen_at(22, 30):
        with pytest.raises(RuntimeError) as exc_info:
            ingest_window.refuse_if_ingest_window_open()
    msg = str(exc_info.value)
    assert "21:45-01:15" in msg
    assert "22:30" in msg


def test_permits_outside_the_default_window():
    with _frozen_at(12, 0):
        ingest_window.refuse_if_ingest_window_open()  # must not raise


def test_permits_just_before_window_opens():
    with _frozen_at(21, 44, 59):
        ingest_window.refuse_if_ingest_window_open()


def test_refuses_at_window_open_boundary():
    with _frozen_at(21, 45, 0):
        with pytest.raises(RuntimeError):
            ingest_window.refuse_if_ingest_window_open()


def test_permits_at_window_close_boundary():
    """End boundary is exclusive - 01:15 UTC exactly is already clear."""
    with _frozen_at(1, 15, 0):
        ingest_window.refuse_if_ingest_window_open()


def test_force_kwarg_overrides_inside_window():
    with _frozen_at(22, 30):
        ingest_window.refuse_if_ingest_window_open(force=True)  # must not raise


def test_env_var_override_inside_window(monkeypatch):
    monkeypatch.setenv(ingest_window._LDP_INGEST_WINDOW_FORCE_ENV, "1")
    with _frozen_at(22, 30):
        ingest_window.refuse_if_ingest_window_open()  # must not raise


def test_env_var_override_accepts_true_yes(monkeypatch):
    for val in ("true", "True", "YES", "yes"):
        monkeypatch.setenv(ingest_window._LDP_INGEST_WINDOW_FORCE_ENV, val)
        with _frozen_at(22, 30):
            ingest_window.refuse_if_ingest_window_open()


def test_env_var_override_rejects_falsy_values(monkeypatch):
    monkeypatch.setenv(ingest_window._LDP_INGEST_WINDOW_FORCE_ENV, "0")
    with _frozen_at(22, 30):
        with pytest.raises(RuntimeError):
            ingest_window.refuse_if_ingest_window_open()


def test_custom_window_env_var_is_honoured(monkeypatch):
    monkeypatch.setenv(ingest_window._LDP_INGEST_WINDOW_ENV, "05:00-06:00")
    with _frozen_at(5, 30):
        with pytest.raises(RuntimeError):
            ingest_window.refuse_if_ingest_window_open()
    with _frozen_at(22, 30):
        ingest_window.refuse_if_ingest_window_open()  # outside the custom window
