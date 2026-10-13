"""LDP nightly-ingest-window preflight for burst campaigns.

It lives apart from burst_common.py so it can be imported without boto3, psycopg2 and pyarrow.
The check needs only os and datetime, and CI's slim `test` job has none of the burst
dependencies, so a guard inside burst_common could not be tested there. burst_common imports
`refuse_if_ingest_window_open` and calls it from `BurstConfig.from_env`.
"""

import os
from datetime import datetime, time, timezone

# LDP's nightly OHLCV ingest rewrites and repairs `ohlcv.prices` for a while each night, and a
# read-heavy burst campaign that starts in that window can read a half-repaired table. This
# check refuses to start one while the window is open. A campaign already running is left
# alone; db.assert_cross_section_single_dated catches a mid-rewrite read per partition.
#
# The window is a clock range that covers the ingest with some headroom.
#
# The clock window was the first guard. LDP's completion watermark (ohlcv.ingest_watermark) is
# the better signal and is checked by burst_common.py's `_refuse_if_unsafe_to_start`, which
# every driver's `BurstConfig.from_env` calls. The window remains as the fallback when the
# watermark is unreachable or empty, because a missing watermark must not read as safe.
_LDP_INGEST_WINDOW_ENV = "LDP_INGEST_WINDOW_UTC"
_LDP_INGEST_WINDOW_UTC_DEFAULT = "21:45-01:15"

# Escape hatch, an env var because the drivers already read their config from the environment.
# Set it once you have checked that ingest finished early, or when the box does not need
# same-night freshness from ohlcv.prices.
_LDP_INGEST_WINDOW_FORCE_ENV = "MERIDIAN_BURST_FORCE_INGEST_WINDOW"


def _parse_ingest_window(spec: str) -> tuple[time, time]:
    """Parse an 'HH:MM-HH:MM' (UTC) window spec into two time objects."""
    try:
        start_s, end_s = spec.split("-")
        start = datetime.strptime(start_s.strip(), "%H:%M").time()
        end = datetime.strptime(end_s.strip(), "%H:%M").time()
    except ValueError as exc:
        raise ValueError(
            f"invalid {_LDP_INGEST_WINDOW_ENV}={spec!r}, expected 'HH:MM-HH:MM' (UTC, "
            f"24-hour clock)"
        ) from exc
    return start, end


def _time_in_window(now: time, start: time, end: time) -> bool:
    """True if `now` falls in [start, end). Handles a window that wraps past
    midnight (e.g. the default window) as well as one that doesn't."""
    if start <= end:
        return start <= now < end
    return now >= start or now < end


def refuse_if_ingest_window_open(*, force: bool = False) -> None:
    """Refuse to start a burst campaign while LDP's nightly OHLCV ingest window is open.

    `BurstConfig.from_env` calls this, so every driver under scripts/burst/ is covered.
    `force=True` or MERIDIAN_BURST_FORCE_INGEST_WINDOW=1 skips the check, and
    LDP_INGEST_WINDOW_UTC=HH:MM-HH:MM sets the window. Raises RuntimeError inside the
    window.
    """
    if force:
        return
    if os.environ.get(_LDP_INGEST_WINDOW_FORCE_ENV, "").strip().lower() in ("1", "true", "yes"):
        return
    spec = os.environ.get(_LDP_INGEST_WINDOW_ENV, _LDP_INGEST_WINDOW_UTC_DEFAULT)
    start, end = _parse_ingest_window(spec)
    now = datetime.now(timezone.utc).time()
    if _time_in_window(now, start, end):
        raise RuntimeError(
            f"Refusing to start a burst campaign: current UTC time "
            f"{now.strftime('%H:%M:%S')} falls inside LDP's nightly OHLCV ingest window "
            f"({spec} UTC). Reads of ohlcv.prices in this window can see a partially "
            f"rewritten table (see the comment above _LDP_INGEST_WINDOW_UTC_DEFAULT). "
            f"Wait until the window closes, or set {_LDP_INGEST_WINDOW_FORCE_ENV}=1 if "
            f"you have verified that tonight's ingest has already finished."
        )
