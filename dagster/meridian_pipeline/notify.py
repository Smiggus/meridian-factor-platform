"""
notify.py - one shared, fail-open ntfy transport for the meridian-factors
code location.

Before this module the only notification path in the repo was
`sensors.ntfy_run_failure_sensor`, which fires exclusively on a run
transitioning to FAILURE. Detective signals that deliberately do NOT fail
the run - the nine Tier-1 checks inlined into `composite_scores` (#285),
all of which are recorded via `_record` inside a broad `except` so they can
never wedge daily scoring - therefore had no way to reach an operator at
all. Without a notification path such a failure would go unseen.

This module is deliberately tiny: the POST that `ntfy_run_failure_sensor`
already performed, lifted out so a non-failing code path can reuse it.
Design invariants:

  * FAIL-OPEN, always. Every transport error is swallowed and reported as a
    False return value. A broken ntfy server must never fail an asset, wedge
    the daemon, or turn an observability signal into an outage.
  * STDLIB ONLY (urllib.request, not requests). The payload is one
    fire-and-forget plain-HTTP POST to a LAN host - no session, no retries,
    no proxy config, no TLS - so `requests` bought nothing here, and the
    slim CI `test` image does not ship it: importing it at module level made
    this module and its tests uncollectable on that job. An alerting path
    with no CI coverage is precisely the failure mode this module exists to
    fix, so it must be importable on EVERY image.
  * No Dagster imports. Callable from sensors, assets, and unit tests alike.
  * URL from MERIDIAN_NTFY_URL, read at CALL time (not import time) so a
    test can monkeypatch the env and so the topic can move without a code
    change or image rebuild.
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request

# Default targets the homelab ntfy "alerts" topic over the LAN
# (server-to-server, no proxy/cert). Same default the run-failure sensor has
# used since it shipped.
DEFAULT_NTFY_URL = "http://gpu.internal/alerts"

# Transport timeout, seconds. Short on purpose: a notification is never worth
# stalling an asset or a sensor tick for.
NTFY_TIMEOUT_SECONDS = 5

# Checks whose failure is real but AMBIGUOUS on its own, so a notification
# containing nothing else is sent at default rather than high priority.
# Deliberately tiny and explicit - see notify_failed_checks.
_CONTEXT_ONLY_CHECKS = frozenset({"check_dq_universe_drift"})


def ntfy_url() -> str:
    """Resolve the ntfy endpoint at call time (see module docstring)."""
    return os.environ.get("MERIDIAN_NTFY_URL", DEFAULT_NTFY_URL)


# Explicit opt-in for the handful of tests whose SUBJECT is the transport
# itself (tests/test_notify.py). Everything else stays denied by default.
# An opt-in is the right shape here: a test that wants to exercise the POST
# has to say so, and no test acquires that power by accident.
NTFY_TEST_TRANSPORT_ENV = "MERIDIAN_NTFY_ALLOW_TEST_TRANSPORT"


def _under_pytest() -> bool:
    """True while a pytest test is executing and has not opted in.

    pytest sets PYTEST_CURRENT_TEST for the duration of each test, and
    PYTEST_VERSION since 8.x for the whole session. Either is sufficient.
    """
    if os.environ.get(NTFY_TEST_TRANSPORT_ENV):
        return False
    return "PYTEST_CURRENT_TEST" in os.environ or "PYTEST_VERSION" in os.environ


def send_ntfy(
    message: str,
    title: str,
    priority: str = "default",
    tags: str = "",
    click: str = "",
    log=None,
) -> bool:
    """POST one ntfy notification. Returns True on success, False if the
    transport failed (never raises).

    `log` is an optional Dagster logger; when supplied, transport failures
    are logged as warnings so the miss is at least visible in the run log.

    NO-OPS UNDER PYTEST. The endpoint is resolved from an env var with a
    LIVE default, so any test that exercises a notifying code path posts to
    the operator's real phone. That is not hypothetical: CI test runs flooded
    the alert topic and drowned the few real alerts, including from a fixture
    literally named `_always_raises` in tests/test_dq_checks.py. Volume tracked
    CI activity, not production health.

    The guard lives HERE, not only in conftest, on purpose. A conftest
    override protects the tests that happen to run under that conftest; this
    protects every caller, including a test invoked from another directory,
    a doctest, or a future test file that forgets. Alerting is exactly the
    subsystem where the fail-safe must not be opt-in.
    """
    if _under_pytest():
        # Deliberately silent. `log` is whatever the caller passed - Dagster
        # loggers, the suite's own doubles - and a notifier that crashes its
        # caller on a logging API mismatch is worse than one that says nothing.
        return False

    headers = {"Title": title, "Priority": priority}
    if tags:
        headers["Tags"] = tags
    if click:
        headers["Click"] = click

    try:
        # Request() must be built INSIDE the try: a malformed
        # MERIDIAN_NTFY_URL raises ValueError right here, before any I/O.
        request = urllib.request.Request(  # noqa: S310 - fixed http:// LAN endpoint
            ntfy_url(),
            data=message.encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=NTFY_TIMEOUT_SECONDS):  # noqa: S310
            pass
        return True
    except (urllib.error.URLError, OSError, ValueError) as exc:
        # URLError and socket timeouts are both OSError subclasses; ValueError
        # covers a malformed MERIDIAN_NTFY_URL. urlopen (unlike requests.post)
        # also raises HTTPError - a URLError subclass - on a 4xx/5xx from the
        # ntfy server, so a rejected notification is now logged rather than
        # silently counted as delivered. Still fail-open either way.
        if log is not None:
            log.warning(f"ntfy notify failed ({title}): {exc}")
        return False


def notify_failed_checks(
    asset_name: str,
    failures: list[dict],
    log=None,
) -> bool:
    """Alert on non-blocking detective checks that FAILED.

    `failures` is a list of `{"name", "failed_partitions", "detail"}` dicts,
    one per failing check (the shape `composite_scores` accumulates in its
    `check_state`). No-ops (returns False) when the list is empty, so callers
    can invoke it unconditionally.

    Kept non-blocking on purpose. The repo's convention is that these checks
    "must never fail the run" - a data-quality WARN is not a reason to halt
    factor scoring and starve every downstream consumer of the day's scores.
    What was missing was not enforcement but VISIBILITY, so the fix is a
    notification, not a gate.
    """
    if not failures:
        return False

    lines = [f"{asset_name}: {len(failures)} detective check(s) FAILED", ""]
    # Ambiguous-by-design signals do not page on their own (design doc D5).
    # check_dq_universe_drift is a pure count delta: a large drop can have
    # several unrelated causes, some of them CORRECT (for example a bulk
    # ingest of delisted tickers). It is context for
    # check_dq_liquid_unscored - the contradiction - not a standalone alarm.
    # A notification consisting only of drift therefore goes out at default
    # priority; anything else in the batch restores high.
    only_context = all(f.get("name") in _CONTEXT_ONLY_CHECKS for f in failures)
    priority = "default" if only_context else "high"
    for f in failures:
        parts = sorted(f.get("failed_partitions") or [])
        where = f"{len(parts)} partition(s)"
        if parts:
            where += f": {', '.join(parts[:5])}"
            if len(parts) > 5:
                where += ", ..."
        lines.append(f"- {f['name']} ({where})")
        detail = f.get("detail")
        if detail:
            lines.append(f"    {str(detail)[:200]}")

    return send_ntfy(
        "\n".join(lines),
        title=f"Meridian: {asset_name} check failure",
        priority=priority,
        tags="warning",
        log=log,
    )


def notify_errored_check(
    asset_name: str,
    check_name: str,
    exc: BaseException,
    log=None,
) -> bool:
    """Alert on a check that could NOT be evaluated because it RAISED.

    Deliberately a separate call, a separate title and a separate tag from
    `notify_failed_checks`. Conflating the two is how a broken detector hides:
    `_warn_on_error` returns `passed=False, severity=WARN` on an exception,
    which renders identically to a genuine finding, so a check that has been
    raising for a month looks exactly like a check that has been failing for a
    month - and neither looks like what it actually is, which is coverage that
    silently went away. See checks/alerting.py.
    """
    body = "\n".join(
        [
            f"{asset_name}.{check_name} could NOT be evaluated (the check raised).",
            "",
            f"{type(exc).__name__}: {exc}"[:400],
            "",
            "This is NOT a data finding - it is missing coverage. The defect this",
            "check watches for is currently unmonitored.",
        ]
    )
    return send_ntfy(
        body,
        title=f"Meridian: {asset_name}.{check_name} ERRORED",
        priority="high",
        tags="rotating_light",
        log=log,
    )
