"""
The test suite must never post to the operator's real ntfy topic.

Background: `notify.ntfy_url()` resolves the endpoint at call time and defaults
to the live homelab topic. Nothing isolated it, so every CI run of a notifying
code path sent a real push notification. In one 24h window the topic was
flooded with CI alerts and the few real ones were drowned out; many came from a
fixture named `_always_raises` in tests/test_dq_checks.py - a test asserting that a raising
check degrades gracefully, which it does, while paging a phone each time.

Two independent defences, tested independently, because either alone can be
removed by someone who does not know about the other.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest import mock

import pytest

_DAGSTER_DIR = Path(__file__).resolve().parents[1] / "dagster"
if str(_DAGSTER_DIR) not in sys.path:
    sys.path.insert(0, str(_DAGSTER_DIR))

from meridian_pipeline import notify  # noqa: E402


def test_conftest_points_ntfy_at_a_blackhole():
    """Defence 1: the session env override is in place and is not the default."""
    url = notify.ntfy_url()
    assert url != notify.DEFAULT_NTFY_URL
    assert "127.0.0.1" in url


def test_send_ntfy_refuses_under_pytest_even_with_a_live_url():
    """
    Defence 2: the guard holds even if the env override is gone.

    Forces the LIVE default back and asserts no HTTP request is attempted.
    """
    with mock.patch.dict(os.environ, {"MERIDIAN_NTFY_URL": notify.DEFAULT_NTFY_URL}):
        with mock.patch("urllib.request.urlopen") as urlopen:
            sent = notify.send_ntfy("body", title="should not leave the box")
    assert sent is False
    urlopen.assert_not_called()


def test_guard_detects_pytest():
    assert notify._under_pytest() is True


def test_guard_is_off_outside_pytest():
    """The guard must not suppress production alerts - it keys only on pytest."""
    clean = {k: v for k, v in os.environ.items() if not k.startswith("PYTEST_")}
    with mock.patch.dict(os.environ, clean, clear=True):
        assert notify._under_pytest() is False


@pytest.mark.parametrize(
    "fn,args",
    [
        ("notify_failed_checks", ("composite_scores", [{"name": "c", "detail": "d"}])),
        ("notify_errored_check", ("composite_scores", "c", RuntimeError("boom"))),
    ],
)
def test_public_notifiers_send_nothing_under_pytest(fn, args):
    """The two callers real assets use, exercised end to end."""
    with mock.patch("urllib.request.urlopen") as urlopen:
        assert getattr(notify, fn)(*args) is False
    urlopen.assert_not_called()
