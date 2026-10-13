"""
Unit tests for meridian_pipeline.notify - the shared, fail-open ntfy
transport (2026-08-01).

Motivation: the nine Tier-1 detective checks inlined into composite_scores
(#285) are non-blocking by design and had NO notification path, so
a failure they caught would go unread.
`notify_failed_checks` is that path.

DB-free and network-free: `urllib.request.urlopen` is monkeypatched
throughout. Note this module imports NOTHING beyond the stdlib on purpose -
notify.py used to go through `requests`, which the slim CI `test` image does
not ship, so these tests were uncollectable exactly where an alerting path
most needs coverage.
"""

import sys
from pathlib import Path

_DAGSTER_DIR = Path(__file__).resolve().parents[1] / "dagster"
if str(_DAGSTER_DIR) not in sys.path:
    sys.path.insert(0, str(_DAGSTER_DIR))

import urllib.error  # noqa: E402
import urllib.request  # noqa: E402

import pytest  # noqa: E402
from meridian_pipeline import notify  # noqa: E402


@pytest.fixture(autouse=True)
def _allow_real_transport(monkeypatch):
    """These tests ARE the transport's tests, so they opt out of the pytest guard.

    send_ntfy no-ops under pytest by default: notify.ntfy_url() resolves at call
    time from an env var whose default is the live homelab topic, so before that
    guard existed the suite paged the operator's real phone (CI runs drowned
    out the few real alerts). This module is the one place that legitimately needs
    the POST path to execute, so it opts in explicitly - and urllib is
    monkeypatched throughout, so nothing leaves the box.
    """
    monkeypatch.setenv(notify.NTFY_TEST_TRANSPORT_ENV, "1")


class _FakeResponse:
    """urlopen returns a context manager; send_ntfy only needs it to close."""

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class _Recorder:
    """Stand-in for urllib.request.urlopen that records the call, flattening
    the Request object back into the fields the assertions care about."""

    def __init__(self, raise_exc=None):
        self.calls = []
        self.raise_exc = raise_exc

    def __call__(self, request, timeout=None):
        self.calls.append(
            {
                "url": request.full_url,
                "method": request.get_method(),
                "data": request.data,
                # urllib title-cases header keys (Title/Priority/Tags/Click all
                # survive that unchanged); normalise so assertions read plainly.
                "headers": {k.title(): v for k, v in request.headers.items()},
                "timeout": timeout,
            }
        )
        if self.raise_exc is not None:
            raise self.raise_exc
        return _FakeResponse()


class _Log:
    def __init__(self):
        self.warnings = []

    def warning(self, msg):
        self.warnings.append(msg)


# ─────────────────────────────────────────────
# send_ntfy
# ─────────────────────────────────────────────


def test_send_ntfy_posts_message_and_headers(monkeypatch):
    rec = _Recorder()
    monkeypatch.setattr(urllib.request, "urlopen", rec)
    monkeypatch.setenv("MERIDIAN_NTFY_URL", "http://ntfy.test/topic")

    ok = notify.send_ntfy("body text", title="T", priority="high", tags="warning")

    assert ok is True
    assert len(rec.calls) == 1
    call = rec.calls[0]
    assert call["url"] == "http://ntfy.test/topic"
    assert call["data"] == b"body text"
    assert call["headers"]["Title"] == "T"
    assert call["headers"]["Priority"] == "high"
    assert call["headers"]["Tags"] == "warning"
    assert call["timeout"] == notify.NTFY_TIMEOUT_SECONDS


def test_send_ntfy_url_read_at_call_time(monkeypatch):
    rec = _Recorder()
    monkeypatch.setattr(urllib.request, "urlopen", rec)
    monkeypatch.delenv("MERIDIAN_NTFY_URL", raising=False)
    notify.send_ntfy("m", title="T")
    assert rec.calls[0]["url"] == notify.DEFAULT_NTFY_URL

    monkeypatch.setenv("MERIDIAN_NTFY_URL", "http://elsewhere/t")
    notify.send_ntfy("m", title="T")
    assert rec.calls[1]["url"] == "http://elsewhere/t"


def test_send_ntfy_fails_open_on_transport_error(monkeypatch):
    """A broken/unreachable ntfy server must never raise into the caller -
    an observability signal must not become an outage."""
    rec = _Recorder(raise_exc=urllib.error.URLError("boom"))
    monkeypatch.setattr(urllib.request, "urlopen", rec)
    log = _Log()

    ok = notify.send_ntfy("m", title="T", log=log)

    assert ok is False
    assert len(log.warnings) == 1
    assert "ntfy notify failed" in log.warnings[0]


def test_send_ntfy_uses_post(monkeypatch):
    """urllib defaults to GET when no method is given; ntfy needs POST."""
    rec = _Recorder()
    monkeypatch.setattr(urllib.request, "urlopen", rec)
    notify.send_ntfy("m", title="T")
    assert rec.calls[0]["method"] == "POST"


def test_send_ntfy_fails_open_on_http_error(monkeypatch):
    """urlopen (unlike requests.post) raises on a 4xx/5xx. A rejecting ntfy
    server must still be fail-open, just logged rather than counted as
    delivered."""
    err = urllib.error.HTTPError("http://ntfy.test/t", 503, "unavailable", {}, None)
    rec = _Recorder(raise_exc=err)
    monkeypatch.setattr(urllib.request, "urlopen", rec)
    log = _Log()

    assert notify.send_ntfy("m", title="T", log=log) is False
    assert log.warnings


def test_send_ntfy_fails_open_on_bad_url(monkeypatch):
    """A malformed MERIDIAN_NTFY_URL raises ValueError out of Request(), not
    URLError - it must not escape either."""
    monkeypatch.setenv("MERIDIAN_NTFY_URL", "not-a-url")
    log = _Log()
    assert notify.send_ntfy("m", title="T", log=log) is False


def test_send_ntfy_omits_empty_optional_headers(monkeypatch):
    rec = _Recorder()
    monkeypatch.setattr(urllib.request, "urlopen", rec)
    notify.send_ntfy("m", title="T")
    headers = rec.calls[0]["headers"]
    assert "Tags" not in headers
    assert "Click" not in headers


# ─────────────────────────────────────────────
# notify_failed_checks
# ─────────────────────────────────────────────


def test_notify_failed_checks_no_op_when_nothing_failed(monkeypatch):
    """The happy path must be silent - callers invoke this unconditionally
    on every materialisation, so a clean run must not page anyone."""
    rec = _Recorder()
    monkeypatch.setattr(urllib.request, "urlopen", rec)

    assert notify.notify_failed_checks("composite_scores", []) is False
    assert rec.calls == []


def test_notify_failed_checks_alerts_on_universe_drift(monkeypatch):
    """A universe collapse replayed: check_dq_universe_drift fails and
    an alert goes out naming the check and the offending partition."""
    rec = _Recorder()
    monkeypatch.setattr(urllib.request, "urlopen", rec)

    ok = notify.notify_failed_checks(
        "composite_scores",
        [
            {
                "name": "check_dq_universe_drift",
                "failed_partitions": ["2026-01-15"],
                "detail": {
                    "curr": 2000,
                    "prev": 3500,
                    "detail": "universe drifted 42.9% (curr=2000, prev=3500); threshold=20%",
                },
            }
        ],
    )

    assert ok is True
    assert len(rec.calls) == 1
    body = rec.calls[0]["data"].decode("utf-8")
    assert "check_dq_universe_drift" in body
    assert "2026-01-15" in body
    assert "2000" in body and "3500" in body
    assert rec.calls[0]["headers"]["Title"] == "Meridian: composite_scores check failure"
    # Priority is DEFAULT, not high, when drift is the only failing check
    # (#295 / design doc D5: "keep, demote - do not page on it alone"). A
    # pure count delta can be a true positive for unrelated reasons, most of
    # the drop being correct, so it is context for
    # check_dq_liquid_unscored rather than a standalone alarm. The alert
    # still goes out, and still names the check and the partition.
    assert rec.calls[0]["headers"]["Priority"] == "default"


def test_notify_failed_checks_high_priority_when_drift_is_not_alone(monkeypatch):
    """The demotion above must not swallow a real batch: as soon as anything
    other than universe drift fails, the alert is high priority again."""
    rec = _Recorder()
    monkeypatch.setattr(urllib.request, "urlopen", rec)

    notify.notify_failed_checks(
        "composite_scores",
        [
            {
                "name": "check_dq_universe_drift",
                "failed_partitions": ["2026-01-15"],
                "detail": None,
            },
            {
                "name": "check_dq_liquid_unscored",
                "failed_partitions": ["2026-01-15"],
                "detail": None,
            },
        ],
    )

    assert rec.calls[0]["headers"]["Priority"] == "high"


def test_notify_failed_checks_one_alert_for_many_checks(monkeypatch):
    """Several failing checks in one materialisation produce ONE alert, not
    one per check - alert fatigue is how signals get ignored."""
    rec = _Recorder()
    monkeypatch.setattr(urllib.request, "urlopen", rec)

    notify.notify_failed_checks(
        "composite_scores",
        [
            {"name": "check_dq_row_coverage", "failed_partitions": ["2026-01-15"], "detail": None},
            {
                "name": "check_dq_universe_drift",
                "failed_partitions": ["2026-01-15"],
                "detail": None,
            },
        ],
    )

    assert len(rec.calls) == 1
    body = rec.calls[0]["data"].decode("utf-8")
    assert "2 detective check(s) FAILED" in body
    assert "check_dq_row_coverage" in body
    assert "check_dq_universe_drift" in body


def test_notify_failed_checks_truncates_partition_list(monkeypatch):
    """A long backfill can fail a check on dozens of partitions; the message
    stays readable."""
    rec = _Recorder()
    monkeypatch.setattr(urllib.request, "urlopen", rec)

    parts = [f"2026-06-{d:02d}" for d in range(1, 21)]
    notify.notify_failed_checks(
        "composite_scores",
        [{"name": "check_dq_row_coverage", "failed_partitions": parts, "detail": None}],
    )

    body = rec.calls[0]["data"].decode("utf-8")
    assert "20 partition(s)" in body
    assert "..." in body


def test_notify_failed_checks_fails_open(monkeypatch):
    rec = _Recorder(raise_exc=TimeoutError("slow"))
    monkeypatch.setattr(urllib.request, "urlopen", rec)
    log = _Log()

    ok = notify.notify_failed_checks(
        "composite_scores",
        [{"name": "check_dq_universe_drift", "failed_partitions": ["2026-01-15"], "detail": None}],
        log=log,
    )

    assert ok is False
    assert log.warnings


# ─────────────────────────────────────────────
# Wiring guards: the alert must actually be reachable from the two call
# sites, otherwise the transport above is dead code.
#
# These READ the source files rather than importing them.
# Importing meridian_pipeline.assets or .sensors drags in dagster, polars,
# and (in sensors' case) `requests` - none of which the slim CI `test` image
# has - which would put these guards behind a skip on the very job where
# they are cheapest to run. A path check plus a substring assertion needs
# none of that.
# ─────────────────────────────────────────────

_PIPELINE_DIR = _DAGSTER_DIR / "meridian_pipeline"


def test_run_failure_sensor_uses_shared_transport():
    """The run-failure sensor keeps paging, now through the shared
    transport, so URL/timeout/fail-open semantics live in one place."""
    src = (_PIPELINE_DIR / "sensors.py").read_text()
    assert "from meridian_pipeline.notify import send_ntfy" in src
    assert "send_ntfy(" in src
    # The inline POST it replaced must be gone, or the two paths can drift.
    assert "_NTFY_URL" not in src
