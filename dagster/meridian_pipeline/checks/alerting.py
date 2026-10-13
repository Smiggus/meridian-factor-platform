"""Alerting wrapper for standalone asset checks (Meridian #295).

notify.py started with two callers: the run-failure sensor and the aggregator for the checks
inlined onto `composite_scores`. Every standalone `@asset_check` pod (factor coverage, the
trading-day invariant, the market_vol checks and the rest) could detect a problem and then tell
nobody.

A wrapped check reports one of three outcomes, stamped into the result metadata as `outcome`:

    passed  - evaluated, clean
    failed  - evaluated, the data is bad          (ntfy, priority high)
    errored - could not be evaluated: the check   (ntfy, priority high,
              itself raised                        distinct title and tag)

A check that has raised for a month looks the same as one that has failed for a month, so an
`errored` check gets its own alert, title and tag.

The wrapper must not fail the run. With `swallow_errors=True`, the default, an exception becomes
a WARN result, as in `data_quality_checks._warn_on_error`. Blocking checks pass
`swallow_errors=False`: the exception propagates and fails the step, the run-failure sensor does
the alerting, and swallowing the error there would give a false green.
"""

from __future__ import annotations

import functools

from dagster import AssetCheckResult, AssetCheckSeverity, MetadataValue
from meridian_pipeline.notify import notify_errored_check, notify_failed_checks


def _partitions_from_context(context) -> list[str]:
    try:
        kr = context.partition_key_range
        return [kr.start] if kr.start == kr.end else [f"{kr.start}..{kr.end}"]
    except Exception:  # noqa: BLE001 - unpartitioned check, or no context
        return []


def alerting_check(asset_name: str, *, swallow_errors: bool = True):
    """Decorate an `@asset_check` body so failures and errors reach ntfy.

    Apply directly UNDER `@asset_check` so the decorator sees the raw function
    and `@asset_check` sees the preserved name (functools.wraps), the same
    ordering rule as `_warn_on_error`.
    """

    def decorate(fn):
        check_name = fn.__name__

        @functools.wraps(fn)
        def wrapper(context):
            try:
                result = fn(context)
            except Exception as exc:  # noqa: BLE001 - see module docstring
                if not swallow_errors:
                    try:
                        notify_errored_check(
                            asset_name, check_name, exc, log=getattr(context, "log", None)
                        )
                    except Exception:  # noqa: BLE001 - alerting never masks the error
                        pass
                    raise
                try:
                    notify_errored_check(
                        asset_name, check_name, exc, log=getattr(context, "log", None)
                    )
                except Exception:  # noqa: BLE001 - alerting must never fail the run
                    pass
                return AssetCheckResult(
                    passed=False,
                    severity=AssetCheckSeverity.WARN,
                    metadata={
                        "outcome": MetadataValue.text("errored"),
                        "error": MetadataValue.text(f"{type(exc).__name__}: {exc}"[:500]),
                        "note": MetadataValue.text(
                            "check RAISED - this is not a data finding, it is missing coverage"
                        ),
                    },
                )

            try:
                if result is not None and not result.passed:
                    notify_failed_checks(
                        asset_name,
                        [
                            {
                                "name": check_name,
                                "failed_partitions": _partitions_from_context(context),
                                "detail": _detail_from(result),
                            }
                        ],
                        log=getattr(context, "log", None),
                    )
            except Exception as exc:  # noqa: BLE001 - alerting must never fail the run
                log = getattr(context, "log", None)
                if log is not None:
                    log.warning(f"[{check_name}] ntfy alert failed: {exc}")
            return result

        return wrapper

    return decorate


def _detail_from(result: AssetCheckResult) -> str:
    """Best-effort one-line detail for the notification body."""
    md = result.metadata or {}
    for key in ("note", "summary", "detail", "blocking_violations", "groups_lost"):
        if key in md:
            return f"{key}={_plain(md[key])}"
    if md:
        k = next(iter(md))
        return f"{k}={_plain(md[k])}"
    return "see the Dagster UI for detail"


def _plain(value):
    """Unwrap a Dagster MetadataValue to something str()-able."""
    for attr in ("value", "text", "data"):
        if hasattr(value, attr):
            return getattr(value, attr)
    return value
