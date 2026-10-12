"""
ingest_window.py - LDP nightly-ingest-window preflight for burst campaigns.

Split out of burst_common.py (2026-08-15) so this guard can be imported
without pulling in burst_common's boto3/psycopg2/pyarrow dependency set. It
is pure stdlib (os + datetime) on purpose: the check is time parsing, env
var reads, and a datetime comparison, none of which needs S3 or PG access.
A guard nobody can import without S3 credentials is a guard nobody can
reuse or unit-test cheaply, and it is exactly what CI's slim `test` job
(no boto3 - see requirements.txt / requirements-test.txt, which deliberately
carry no burst/S3 deps) needs to be able to exercise directly.

burst_common.py imports `refuse_if_ingest_window_open` from here and wires
it into `BurstConfig.from_env`, so behaviour for every burst driver is
unchanged - this is a pure move, not a behaviour change.
"""

import os
from datetime import datetime, time, timezone

# ─────────────────────────────────────────────────────────────────────────────
# Ingest-window preflight (read-during-write hardening, campaign START only)
# ─────────────────────────────────────────────────────────────────────────────
#
# LDP's nightly OHLCV ingest window rewrites/repairs `ohlcv.prices` for a
# period each night. A read-heavy burst campaign LAUNCHED during that window
# reads history while the upstream is concurrently rewriting it, and can pick
# up an inconsistent snapshot mid-repair. This preflight refuses to START a new campaign while
# the window is open. It does NOT gate an already-running campaign (a batch
# that overruns into the next night's window is a different problem - see
# db.assert_cross_section_single_dated for the per-partition guard that
# catches a mid-rewrite READ regardless of why it happened) or any
# per-partition compute - campaign START only.
#
# Window derivation: the window is a configurable clock range meant to cover
# the upstream ingest's start and finish with headroom. Size it from your own
# observed ingest durations, not from a growth model. The default below is
# illustrative.
#
# This constant was an INTERIM guard. LDP's completion watermark
# (ohlcv.ingest_watermark) is now live and is PREFERRED - see
# burst_common.py's `_refuse_if_unsafe_to_start` (above
# `_INGEST_WATERMARK_QUERY`), which every driver's `BurstConfig.from_env`
# already routes through. This clock window remains wired in as the
# fallback for when the watermark is unreachable or empty - a missing
# watermark must not silently mean "safe to run" - so it still needs to
# stay correct, just no longer needs to be the primary signal.
_LDP_INGEST_WINDOW_ENV = "LDP_INGEST_WINDOW_UTC"
_LDP_INGEST_WINDOW_UTC_DEFAULT = "21:45-01:15"

# Operator escape hatch (env var, not a CLI flag, since every driver in this
# package already sources its config from the environment via
# BurstConfig.from_env - see the module docstring). Set to override the
# refusal when you have verified LDP's ingest already finished early, or
# when the box's own use of ohlcv.prices does not depend on same-night
# freshness.
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
    """Refuse to START a burst rescore/backfill campaign while LDP's nightly
    OHLCV ingest window is open.

    Campaign-START-only: does not gate an already-running campaign or any
    per-partition compute (see the module-level comment above
    `_LDP_INGEST_WINDOW_UTC_DEFAULT` for why, and for the window's
    derivation). Called from `burst_common.BurstConfig.from_env`, the one
    construction call every driver under scripts/burst/ makes before doing
    any read-heavy work, so wiring it there covers every driver without
    touching each script individually.

    Override: pass `force=True` (programmatic) or set the env var
    MERIDIAN_BURST_FORCE_INGEST_WINDOW=1/true (operator escape hatch) - use
    this only when you have independently verified LDP's ingest has already
    finished for the night, or that this particular campaign does not
    depend on same-night freshness. The window itself is configurable via
    LDP_INGEST_WINDOW_UTC=HH:MM-HH:MM (UTC).

    Raises:
        RuntimeError: current UTC time falls inside the ingest window and no
            override was given.
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
            f"({spec} UTC). Reading ohlcv.prices history during this window risks reading "
            f"a partially-rewritten table (see the module comment above "
            f"_LDP_INGEST_WINDOW_UTC_DEFAULT). Wait until the "
            f"window closes, or "
            f"set {_LDP_INGEST_WINDOW_FORCE_ENV}=1 to override if you have verified "
            f"ingest has already finished tonight."
        )
