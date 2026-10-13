"""One ntfy transport for the meridian-factors code location.

The only notification path used to be `sensors.ntfy_run_failure_sensor`, which fires when a run
goes to FAILURE. Checks that don't fail the run, such as the nine Tier-1 checks inlined into
`composite_scores` (#285), which record through `_record` inside a broad `except` so they can't
wedge daily scoring, had no way to reach an operator. This module is the POST the sensor already
made, moved out so other code can use it.

It is fail-open. Every transport error is swallowed and comes back as a False return value, so a
dead ntfy server can't fail an asset or stall the daemon. It uses urllib and not requests: the
payload is one plain-HTTP POST to a LAN host, and the slim CI test image has no requests, which
used to make this module and its tests uncollectable there. It imports nothing from Dagster, so
sensors, assets and unit tests can all call it. The URL comes from MERIDIAN_NTFY_URL, read at
call time so a test can patch the environment and the topic can move without an image rebuild.
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request

# The LAN ntfy alerts topic, which is also the run-failure sensor default.
DEFAULT_NTFY_URL = "http://gpu.internal/alerts"

# Seconds. A notification isn't worth stalling a sensor tick.
NTFY_TIMEOUT_SECONDS = 5

# Checks whose failure is real but ambiguous alone. A batch made up only of these goes out at
# default priority; see notify_failed_checks.
_CONTEXT_ONLY_CHECKS = frozenset({"check_dq_universe_drift"})


def ntfy_url() -> str:
    """Resolve the ntfy endpoint at call time (see module docstring)."""
    return os.environ.get("MERIDIAN_NTFY_URL", DEFAULT_NTFY_URL)


# Set by the few tests whose subject is the transport itself (tests/test_notify.py).
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
    """POST one ntfy notification. Returns True on success, False if the transport failed.
    It never raises.

    `log` is an optional Dagster logger. When given, transport failures are logged as
    warnings so the miss shows in the run log.

    Under pytest this does nothing and returns False. The endpoint comes from an env var
    with a live default, and CI runs that reached a notifying code path posted to the
    operator's phone until the alert topic was unreadable. One source was a fixture called
    `_always_raises` in tests/test_dq_checks.py. The check lives here rather than in
    conftest so that it also holds for a test run from another directory.
    """
    if _under_pytest():
        # Silent. `log` is whatever the caller passed - Dagster
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
    """Alert on non-blocking detective checks that failed.

    `failures` is a list of `{"name", "failed_partitions", "detail"}` dicts, one per failing
    check (the shape `composite_scores` accumulates in its `check_state`). It returns False
    without sending when the list is empty, so callers can call it unconditionally.

    The checks stay non-blocking, since halting scoring over a data-quality WARN would hold
    back the day's scores for every consumer. They only needed somebody to see the failure.
    """
    if not failures:
        return False

    lines = [f"{asset_name}: {len(failures)} detective check(s) FAILED", ""]
    # check_dq_universe_drift is a plain count delta. A large drop can have harmless causes,
    # a bulk ingest of delisted tickers for one, so on its own it goes out at default
    # priority. It is mainly context for check_dq_liquid_unscored. Anything else in the
    # batch puts the priority back to high (design doc D5).
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
    """Alert on a check that could not be evaluated because it raised.

    This has its own call, title and tag, separate from `notify_failed_checks`.
    `_warn_on_error` returns `passed=False, severity=WARN` on an exception, which looks the
    same as a real finding. See checks/alerting.py.
    """
    body = "\n".join(
        [
            f"{asset_name}.{check_name} could NOT be evaluated (the check raised).",
            "",
            f"{type(exc).__name__}: {exc}"[:400],
            "",
            "No data problem was found. The check itself broke, so what it watches for",
            "is currently unchecked.",
        ]
    )
    return send_ntfy(
        body,
        title=f"Meridian: {asset_name}.{check_name} ERRORED",
        priority="high",
        tags="rotating_light",
        log=log,
    )
