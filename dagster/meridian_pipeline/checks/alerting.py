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

The wrapper adds `outcome` to a returned result unless the check already set its own `outcome`
key.

With `swallow_errors=True`, the default, an exception becomes a WARN result, as in
`data_quality_checks._warn_on_error`, so the run does not fail, and the wrapper sends the
`errored` alert itself. Blocking checks pass `swallow_errors=False`: the exception propagates and
fails the step, and the run-failure sensor (`ntfy_run_failure_sensor`) does the alerting. The
wrapper sends nothing in that case, so an erroring blocking check pages once, not twice.
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
                    # The step fails, so ntfy_run_failure_sensor pages. Alerting
                    # here as well would send the same incident twice.
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
            return _with_outcome(result)

        return wrapper

    return decorate


def _with_outcome(result):
    """Return `result` with `outcome` set to passed or failed, if it has none."""
    if result is None or not hasattr(result, "_replace"):
        return result
    metadata = dict(result.metadata or {})
    if "outcome" in metadata:
        return result
    metadata["outcome"] = MetadataValue.text("passed" if result.passed else "failed")
    return result._replace(metadata=metadata)


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
