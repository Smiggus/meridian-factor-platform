"""
sensors.py - Dagster sensors for Meridian factor scoring.

This export ships eleven of the sensors (the list below names the ones that
drive the daily two-phase run; see each section's own comment banner for the
rest):

0. watermark_complete_sensor - PRIMARY trigger for daily phase-1 scoring
   (Meridian #286, 2026-08-18). Polls LDP's completion watermark
   (ohlcv.ingest_watermark) every 5 minutes and fires phase 1
   (CORE_FACTOR_ASSETS) the moment a trade_date is actually marked
   complete. Replaces the old `daily_factor_scoring` cron schedule
   (removed from schedules.py), which fired at a fixed 17:00 ET sized
   against an expected LDP finish time, which fires before the date's
   watermark clears whenever ingest runs late and then fails
   assert_cross_section_single_dated's guard. See the sensor's own
   docstring for the full cutover rationale and cursor design.

1. stale_factor_sensor - Runs every 30 minutes. Safety net for
   watermark_complete_sensor: if no rows exist in factors.daily_scores for
   the latest ET trading day AND db.trigger_cross_section_ready says the
   date is ready (Meridian #395: watermark-first, falling back to the same
   coverage-ratio predicate the write-side guard applies rather than
   assuming ready whenever the watermark has no row yet), fires phase 1
   (CORE_FACTOR_ASSETS). Phase 2 then auto-fires via core_complete_sensor.

2. request_sensor - Runs every 30 seconds. Polls factors.compute_requests
   for pending dates (submitted by the API when a consumer requests a date
   that isn't in the DB). Triggers phase 1 (CORE) for each pending date;
   phase 2 follows via core_complete_sensor when phase 1 lands.
   Flow: GET /scores?date=X -> API writes to compute_requests -> sensor fires
   phase 1 -> core_complete_sensor fires phase 2 -> composite lands -> next
   API retry serves the data.

3. core_complete_sensor - sensor watching all CORE_FACTOR_ASSETS.
   When all CORE assets have new materialisations for the same partition,
   fires GPU_COMPOSITE_ASSETS for that partition. Phase-1 to phase-2
   bridge for LIVE continuity only: a recent (within
   CORE_COMPLETE_MAX_PARTITION_AGE_DAYS, default 10) phase-1 partition
   landing auto-fires its phase-2 run (serialised by GPU cap=1). Historical
   partitions (e.g. a CORE backfill re-scoring old years) are age-gated out
   and dropped from tracked state without firing - historical phase-2 is
   launched deliberately via meridian_backfill_job, not this sensor
   (Meridian #99 follow-up, 2026-07-04/05: two backfill waves each fired a
   high-priority phase-2 RunRequest per backfilled historical partition,
   starving the gpu pool).

4. ldp_ohlcv_upstream_sensor - Event-driven cross-code-location sensor.
   Watches LDP's ohlcv_prices asset materialisations; when fresh upstream
   OHLCV activity is detected, fires phase-1 CORE_FACTOR_ASSETS for the
   most recent closed ET trading day. Ships STOPPED in code - see the
   cutover plan in the sensor's docstring - but a sensor's default_status only
   seeds its initial state, so it can end up running regardless; it now also consults
   db.trigger_cross_section_ready before firing, same as stale_factor_sensor.

5. ntfy_run_failure_sensor - Fires on every run transition to FAILURE.
   POSTs an ntfy alert to the alerts topic. Fail-open: transport
   errors are logged, never raised.
5b. asset_check_alert_sensor - Pages the same ntfy topic when an allowlisted
   asset check (sensors._ALERTED_CHECKS) evaluates failed, including WARN
   severity on a successful run (Meridian#427). Once per (check, partition).
6. composite_cache_warm_sensor - Watches composite_scores materialisations.
   After composite lands for a partition, calls POST /admin/cache/warm?date=X
   on the Meridian API to evict stale cache entries for that date. This closes
   the write->serve gap where the API cache would otherwise keep serving old
   scores until TTL expiry. Fail-soft: HTTP errors are logged and swallowed.
   Shipped STOPPED; enable via Dagster UI once the API is live.

"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime, timedelta, timezone
from typing import NamedTuple
from zoneinfo import ZoneInfo

import requests

# Meridian #285 Tier 3: backfill-completion polling. Dagster 1.13.4 exposes no
# native backfill-completion hook (run_status_sensor fires per RUN, so a
# 296-run backfill would fire 296 times), so the sensor reads backfill state
# straight off the instance via these types. Verified present in 1.13.4.
from dagster._core.execution.backfill import BulkActionsFilter, BulkActionStatus

from dagster import (
    AssetKey,
    DagsterEventType,
    DagsterRunStatus,
    DefaultSensorStatus,
    RunFailureSensorContext,
    RunRequest,
    RunsFilter,
    SensorEvaluationContext,
    SkipReason,
    run_failure_sensor,
    sensor,
)
from meridian_pipeline import db
from meridian_pipeline.asset_selections import (
    ALL_FACTOR_ASSET_KEYS,
    CORE_FACTOR_ASSET_KEYS,
)
from meridian_pipeline.backfill_verification_job import backfill_verification_job
from meridian_pipeline.jobs import (
    daily_factor_core_job,
    daily_gpu_composite_job,
    daily_scores_mirror_reconcile_job,
    gold_pg_sync_job,
)
from meridian_pipeline.notify import send_ntfy


@sensor(
    job=daily_scores_mirror_reconcile_job,
    minimum_interval_seconds=60,
    default_status=DefaultSensorStatus.STOPPED,
    description="After committed group scoring, reconcile both PG score tables; stopped until cutover approval.",
)
def checked_score_mirror_sensor(context: SensorEvaluationContext):
    """Dispatch only committed group scoring; historical promotions need explicit audits."""
    cursors = json.loads(context.cursor) if context.cursor else {}
    for source in ("factor_group_scores",):
        key = AssetKey([source])
        if source not in cursors:
            latest = context.instance.fetch_materializations(key, limit=1, ascending=False)
            if latest.cursor:
                cursors[source] = latest.cursor
            continue
        result = context.instance.fetch_materializations(
            key, cursor=cursors[source], limit=50, ascending=True
        )
        months = {}
        today = _today_utc()
        for record in result.records:
            if not record.partition_key:
                continue
            day = date.fromisoformat(record.partition_key)
            if not 0 <= (today - day).days <= 10:
                continue
            month = day.strftime("%Y-%m")
            batch = months.setdefault(month, {"dates": set(), "storage_id": 0})
            batch["dates"].add(record.partition_key)
            batch["storage_id"] = max(batch["storage_id"], record.storage_id)
        for month, batch in sorted(months.items()):
            config = {"score_dates": sorted(batch["dates"])}
            yield RunRequest(
                run_key=f"checked-mirror-{month}-{batch['storage_id']}",
                run_config={"ops": {"daily_scores_mirror_reconcile": {"config": config}}},
            )
        if result.cursor:
            cursors[source] = result.cursor
    context.update_cursor(json.dumps(cursors, sort_keys=True))


# Cap the in-progress partitions tracked by core_complete_sensor so cursor
# growth is bounded if upstream materialisations stall mid-partition for an
# extended period. ~100 days of trading is ~5 months; well past any normal
# in-flight backfill window.
_MAX_TRACKED_PARTITIONS = 100

# Per-tick per-asset fetch cap. Picks up bursts (e.g. a 30-partition backfill
# all landing within one tick) without spiralling memory on the daemon side.
_MAX_RECORDS_PER_ASSET_PER_TICK = 200

# A partition that has been tracked (partially materialised) longer than this
# without all CORE assets landing is reaped from partitions_seen. Keyed on
# wall-clock first-seen time, NOT partition date, so a legitimately in-flight
# backfill - where all CORE assets for a partition land within one phase-1 run,
# minutes apart - is never dropped, while a partial that can never complete
# (a single-asset CORE backfill, or an interrupted run) is cleared within the
# window instead of pinning a tracker slot forever (Meridian #99).
_PARTIAL_PARTITION_TTL_SECONDS = 6 * 60 * 60  # 6h

# core_complete_sensor's job is LIVE phase-2 continuity only: bridging a
# just-landed phase-1 partition to its phase-2 GPU run within the same
# evening. Historical phase-2 (re-scoring old partitions during a CORE
# backfill, e.g. the 2013-2014 rerun) is launched deliberately via
# meridian_backfill_job's union job selection - it must NOT also be fanned
# out per-partition by this sensor. Two backfill waves (66, then 100+ runs)
# each serialised their own GPU RunRequest for every backfilled historical
# partition, starving the gpu pool (cap=1) at high priority. The #99 cursor
# TTL does not help here: it reaps never-completing partials, but a backfill
# partition DOES complete (all CORE assets land for it), it is just old.
# This age gate is keyed on PARTITION DATE (not wall-clock first-seen like
# the TTL above), evaluated against the current tick's date in UTC - cheaper
# than the TTL and immune to it (a backfill can complete a partition within
# seconds of it being first seen, well inside the TTL window).
_DEFAULT_MAX_PARTITION_AGE_DAYS = 10


def _max_partition_age_days() -> int:
    """Env-overridable so an operator can widen/narrow the live window
    without a code change. Read per-tick (not cached at import time) so a
    changed env var takes effect on the daemon's next sensor evaluation."""
    return int(
        os.environ.get("CORE_COMPLETE_MAX_PARTITION_AGE_DAYS", _DEFAULT_MAX_PARTITION_AGE_DAYS)
    )


def _partition_age_days(partition_key: str, today: date) -> int | None:
    """Calendar-day age of an ISO ``YYYY-MM-DD`` partition key relative to
    ``today``. Returns None (never raises) if the key doesn't parse as a
    date, so a malformed/non-date partition key is left untouched by the age
    gate rather than treated as ancient."""
    try:
        pk_date = date.fromisoformat(partition_key)
    except ValueError:
        return None
    return (today - pk_date).days


# Market timezone. All date/time reasoning in this module anchors on ET so
# the sensor behaves correctly regardless of container TZ or DST.
ET = ZoneInfo("America/New_York")


def _now_epoch() -> float:
    """Wall-clock epoch seconds. Wrapped in a module function so tests can
    monkeypatch it to exercise partition TTL expiry deterministically."""
    return datetime.now(tz=ET).timestamp()


def _today_utc() -> date:
    """Current UTC calendar date. Wrapped in a module function so tests can
    monkeypatch it to exercise the partition age gate deterministically."""
    return datetime.now(timezone.utc).date()


def _run_covers_core(instance, run_id: str, core_set: set[str], cache: dict[str, bool]) -> bool:
    """Whether a materialisation's originating run targeted the full CORE set.

    A run that explicitly selected only a strict subset of CORE (e.g. a
    single-asset backfill) can never complete a partition,
    so tracking its materialisations only pollutes partitions_seen (Meridian
    #99: a cancelled single-asset backfill left dead partial entries). Such
    runs are skipped here; the TTL GC is the backstop, this is the guard at the
    source.

    Deliberately conservative - it must NEVER block a legitimate phase-1 run
    from firing phase 2 (that was the #99 outage). asset_selection is None for a
    whole-job run (the daily schedule and the union backfill job, verified in
    prod) and an empty frozenset for some in-process paths; both are treated as
    "covers CORE". Only a NON-EMPTY selection that omits a CORE member is
    skipped. Any lookup failure is permissive (counts). Cached per run_id since
    a backfill batch shares one run across many records.
    """
    if run_id in cache:
        return cache[run_id]
    covers = True
    try:
        record = instance.get_run_record_by_id(run_id)
        selection = record.dagster_run.asset_selection if record else None
        if selection:  # non-empty only; None / empty frozenset => whole job
            selected = {k.to_user_string() for k in selection}
            covers = core_set <= selected
    except Exception:
        covers = True
    cache[run_id] = covers
    return covers


# Sensor gate: only evaluate after 17:00 ET, which is 1h past the 16:00 ET
# close. This matches the daily_factor_scoring schedule (17:00 ET Mon-Fri)
# and avoids churning against partitions for sessions that haven't ended.
_POST_CLOSE_HOUR = 17
_POST_CLOSE_MINUTE = 0


def _latest_trading_day_et(now_et: datetime) -> date:
    """
    Most recent COMPLETED trading day in ET (calendar/weekday check only).
      - Weekday after close: today.
      - Weekday before close: yesterday (with weekend rollback).
      - Weekend: the prior Friday.
    Does NOT check US market holidays; callers pass the result to
    _ensure_trading_day() which does a DB-backed holiday check.
    """
    d = now_et.date()
    after_close = now_et.hour > _POST_CLOSE_HOUR or (
        now_et.hour == _POST_CLOSE_HOUR and now_et.minute >= _POST_CLOSE_MINUTE
    )
    if now_et.weekday() < 5 and after_close:
        return d
    # Roll back to the most recent weekday strictly before today.
    d -= timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def _ensure_trading_day(engine, d: date, max_lookback: int = 7) -> date:
    """Roll d back to the most recent date that was actually a trading day.

    Handles US market holidays that fall on weekdays (Memorial Day, etc.).
    Limits lookback to max_lookback days so a DB outage doesn't loop forever.
    """
    for _ in range(max_lookback):
        if db.is_trading_day(engine, d):
            return d
        d -= timedelta(days=1)
        while d.weekday() >= 5:
            d -= timedelta(days=1)
    return d


# Module-level engine cache. The Dagster daemon evaluates sensors
# sequentially in a single thread, so this lazy-init does not need a lock.
# Caching here eliminates ~120 PG connection-opens/hour from request_sensor's
# 30s tick (closes part of #65 Phase C).
_ENGINE = None


def _get_engine():
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = db._engine(
            pguser=os.environ.get("PGUSER", "dbuser"),
            pgpass=os.environ["PGPASS"],
            pghost=os.environ.get("PGHOST", "db.internal"),
            pgport=int(os.environ.get("PGPORT", 5432)),
            pgdb=os.environ.get("PGDB", "marketdata"),
        )
    return _ENGINE


# ─────────────────────────────────────────────
# Cross-sensor partition dedup (Meridian #358)
# ─────────────────────────────────────────────

# Statuses that count as "a run for this partition is already in flight".
# NOT_STARTED is included alongside the QUEUED/STARTING/STARTED trio because
# a run another sensor's RunRequest just created can sit briefly before the
# run queue daemon picks it up - excluding it would leave that window
# uncovered. SUCCESS/FAILURE/CANCELED are deliberately excluded: a date that
# failed or finished once must still be retriable by a later tick.
_PARTITION_IN_FLIGHT_STATUSES = [
    DagsterRunStatus.NOT_STARTED,
    DagsterRunStatus.QUEUED,
    DagsterRunStatus.STARTING,
    DagsterRunStatus.STARTED,
]


def _partition_in_flight(instance, job_name: str, partition_date) -> bool:
    """True if a run of ``job_name`` for ``partition_date`` is already
    NOT_STARTED/QUEUED/STARTING/STARTED, regardless of what launched it.

    Meridian #358: watermark_complete_sensor, ldp_ohlcv_upstream_sensor,
    stale_factor_sensor and request_sensor all target daily_factor_core_job
    for the same date and each only dedups against its OWN run_key, so two
    different sensors can both launch the same partition. Meridian#395 made these
    sensors DEFER instead of fire-and-fail when the cross-section isn't
    ready yet - which removes an accidental collision-avoidance mechanism
    (a burned date-scoped run_key from a failed early fire used to stop
    that same sensor from firing again later) and makes several sensors
    become ready and fire within the same few-minute window MORE likely,
    not less. This helper is the shared pre-flight check every
    daily_factor_core_job sensor now calls before yielding a RunRequest.

    Deliberately scoped by job_name with NO sensor-name allowlist, unlike
    _sensor_run_in_flight below. #358's failure mode is sensors that are not
    on each other's radar racing over the same job - a check keyed to a
    named list of sensors would require every future sensor added to
    daily_factor_core_job to remember to enrol itself, reproducing the exact
    coupling gap this fix exists to close. Filtering purely on
    RunsFilter(job_name=...) catches ANY in-flight run of the job for the
    partition, including one launched by a sensor not in any allowlist, a
    schedule, or an operator's manual launch from the UI.

    Not merged into _sensor_run_in_flight: that helper backs
    group_scores_gap_sensor's check, which spans TWO DIFFERENT jobs
    (daily_gpu_composite_job's sensors via _PHASE2_TRIGGERS, and this
    sensor's own group_scores_repair_job via _GROUP_SCORES_TRIGGERS) in one
    call - a single job_name filter cannot express an OR-across-jobs query,
    so that caller still needs the sensor-name form. These are two helpers
    answering two different questions ("did any of these named sensors
    fire" vs "is this job already running for this partition"), not two
    competing answers to the same one.

    Residual race (cannot be closed from a sensor): two ticks - from the
    same sensor or two different ones - can evaluate concurrently, both
    observe nothing in flight via this check, and both fire. Dagster does
    not expose an atomic check-and-launch primitive to a sensor; this
    helper shrinks the window (bounded by tick cadence and instance storage
    read latency) without closing it. See the #358 MR description for a
    stronger primitive (a job-level run concurrency limit) considered and
    not adopted here.
    """
    runs = instance.get_runs(
        filters=RunsFilter(
            job_name=job_name,
            statuses=_PARTITION_IN_FLIGHT_STATUSES,
            tags={"dagster/partition": str(partition_date)},
        ),
        limit=1,
    )
    return len(runs) > 0


# ─────────────────────────────────────────────
# Watermark-driven daily scoring trigger (Meridian #286)
# ─────────────────────────────────────────────

# Per-tick cap on newly-complete trade_dates pulled from the watermark. LDP
# marks at most one date complete per ordinary night; a generous cap absorbs
# a multi-day catch-up (e.g. the watermark itself was down for a stretch, or
# this sensor was STOPPED for a while) without unbounded work in one tick.
_WATERMARK_MAX_DATES_PER_TICK = 30


@sensor(
    minimum_interval_seconds=5 * 60,  # every 5 minutes
    job=daily_factor_core_job,
    default_status=DefaultSensorStatus.RUNNING,
    description=(
        "PRIMARY trigger for daily phase-1 factor scoring (Meridian #286), "
        "replacing the daily_factor_scoring cron schedule (removed from "
        "schedules.py - see its module docstring for the incident this "
        "closes). That schedule fired at a fixed wall-clock time, sized "
        "against an expected LDP finish time, so whenever the nightly ingest "
        "ran late the cron fired before "
        "ohlcv.ingest_watermark marked the date complete and "
        "assert_cross_section_single_dated correctly refused to score it. "
        "This is a wall-clock-constant failure class. The fix is "
        "to make the trigger dependency-driven instead of "
        "clock-scheduled.\n\n"
        "Polls ohlcv.ingest_watermark for trade_dates newly marked complete "
        "since the sensor's own high-water mark, and fires phase 1 "
        "(PHASE1_TRIGGER_ASSETS = CORE + market_vol) for each "
        "the moment it appears - there is no "
        "wall-clock assumption left to go stale as LDP's finish time "
        "drifts further.\n\n"
        "First tick seeds the high-water mark to the CURRENT latest "
        "complete date without firing (same seed-without-fire precedent as "
        "core_complete_sensor's newly-added-asset handling and "
        "backfill_completion_sensor's first tick), so enabling this sensor "
        "does not stampede one run per historical complete date already "
        "sitting in the watermark.\n\n"
        "Empty/unreachable watermark: a table with no complete rows yet "
        "(it went from zero rows to live population within the current "
        "week - see db.latest_complete_trade_date's docstring) and a "
        "transient connection/relation error both result in a quiet no-op "
        "tick, logged distinctly. Neither is treated as 'fire freely' "
        "(nothing is known-complete, so firing would recreate the exact "
        "premature-score race this sensor exists to close) nor as a "
        "permanent stop (every tick re-checks from the same high-water "
        "mark, so scoring resumes automatically the moment the watermark "
        "answers again - no cursor state here can wedge open the way "
        "core_complete_sensor's bootstrap once did in #99).\n\n"
        "Cursor format (JSON): {high_water_date: <ISO date string|null>}. "
        "run_key=f'watermark_{date}' gives Dagster-level dedup as a second "
        "line of defence if a tick's cursor commit is ever lost.\n\n"
        "stale_factor_sensor (below) remains registered as the secondary "
        "safety net (e.g. this sensor being STOPPED, or a watermark write "
        "path outage) and now also consults the watermark before firing, "
        "so it cannot dedup-lock itself out of a date the way it would "
        "have if it fired ahead of completion under the old regime."
    ),
)
def watermark_complete_sensor(context: SensorEvaluationContext):
    """Fire phase-1 CORE_FACTOR_ASSETS the moment LDP marks a trade_date
    complete in ohlcv.ingest_watermark, instead of on a fixed-clock cron.

    State machine (JSON cursor: {high_water_date}):
      1. No cursor yet (first tick / fresh deploy): seed high_water_date to
         db.latest_complete_trade_date() WITHOUT firing, so a backlog of
         already-complete historical dates does not stampede one run each.
         If the watermark has no complete rows at all yet, leave the cursor
         empty and retry next tick.
      2. Otherwise: pull trade_dates newly complete since high_water_date
         (db.get_new_complete_trade_dates), fire one RunRequest per date
         (skipping non-trading-day rows as a defensive belt in case the
         watermark ever marks a weekend/holiday complete), and advance
         high_water_date to the latest date seen.

    Any exception reading the watermark (connection error, missing
    table/schema) is caught here, logged, and the tick returns without
    touching the cursor - the next tick retries from the same high-water
    mark, so an outage delays firing but never loses a date or fires
    spuriously.
    """
    cursor = json.loads(context.cursor) if context.cursor else {}
    high_water: str | None = cursor.get("high_water_date")

    engine = _get_engine()

    if high_water is None:
        try:
            latest = db.latest_complete_trade_date(engine)
        except Exception as exc:  # noqa: BLE001
            context.log.warning(
                f"watermark_complete_sensor: ohlcv.ingest_watermark unreachable on "
                f"seed tick ({exc}) - will retry next tick, no state changed"
            )
            return
        if latest is None:
            context.log.info(
                "watermark_complete_sensor: ohlcv.ingest_watermark has no complete "
                "rows yet - nothing to seed to, will retry next tick"
            )
            return
        context.update_cursor(json.dumps({"high_water_date": latest.isoformat()}))
        context.log.info(
            f"watermark_complete_sensor: seeded high-water mark to {latest} "
            f"without firing (first tick / new deploy)"
        )
        return

    since = date.fromisoformat(high_water)
    try:
        new_dates = db.get_new_complete_trade_dates(
            engine, since, limit=_WATERMARK_MAX_DATES_PER_TICK
        )
    except Exception as exc:  # noqa: BLE001
        context.log.warning(
            f"watermark_complete_sensor: ohlcv.ingest_watermark unreachable ({exc}) "
            f"- will retry next tick, high-water mark unchanged at {high_water}"
        )
        return

    if not new_dates:
        return  # quiet tick: nothing newly complete yet

    fired = 0
    for d in new_dates:
        if not db.is_trading_day(engine, d):
            # Defensive belt: the watermark should only ever mark trading
            # days complete, but a weekend/holiday row must not fire a
            # useless run. The high-water mark still advances past it below.
            context.log.info(
                f"watermark_complete_sensor: {d} marked complete but is not a "
                f"trading day per ohlcv.prices - skipping without firing"
            )
            continue
        if _partition_in_flight(context.instance, daily_factor_core_job.name, d):
            # Meridian #358: another sensor already has a daily_factor_core_job
            # run in flight for this date (e.g. request_sensor beat the
            # watermark to it). The high-water mark still advances past it
            # below - the run already in flight will produce the scores.
            context.log.info(
                f"watermark_complete_sensor: {d} already has a daily_factor_core_job "
                f"run in flight - skipping to avoid a duplicate"
            )
            continue
        context.log.info(
            f"watermark_complete_sensor: {d} marked complete by LDP - triggering phase 1"
        )
        yield RunRequest(
            run_key=f"watermark_{d}",
            partition_key=d.isoformat(),
            tags={"dagster/priority": "50", "priority_class": "high"},
        )
        fired += 1

    high_water = max(new_dates).isoformat()
    context.update_cursor(json.dumps({"high_water_date": high_water}))
    if fired:
        context.log.info(
            f"watermark_complete_sensor: fired {fired} run(s), high-water mark now {high_water}"
        )


# Minimum gap between two fires for the SAME date. The lake export lands ~40
# minutes after the PG watermark, so a shorter retry just burns a library_heavy
# pod on the same stale substrate.
_QUEUE_RETRY_MINUTES = 60

# Maximum requests per work item (risk-model date, gold snapshot) before the
# sensor stops re-requesting it. A date that has failed this many times is not
# flaky, it is broken, and each further attempt only burns a pod and spams the
# run-failure pager (ntfy_run_failure_sensor already pages on every failed run,
# so no new notifier is added). The item is NEVER dropped from the work list -
# dropping it is the silent-gap failure the work list exists to prevent - it is
# reported at ERROR (at most hourly) until an operator fixes the cause and
# launches it by hand or clears it.
_SENSOR_MAX_ATTEMPTS = 3

# Re-alarm interval for a date that is past _QUEUE_PENDING_ALERT_HOURS.
# Without it a stuck date logs ERROR every tick, ~288 a day, indefinitely, which
# trains everybody to ignore it.
_QUEUE_ALERT_REPEAT_MINUTES = 60


def _queue_now() -> datetime:
    """Wall clock for this sensor's retry spacing and stuck-date escalation.
    A named seam so tests can pin it without monkeypatching `datetime`."""
    return datetime.now(timezone.utc)


def _queue_pending_hours(entry: dict, now: datetime) -> float:
    try:
        return (now - datetime.fromisoformat(entry["first_seen"])).total_seconds() / 3600.0
    except Exception:  # noqa: BLE001
        return 0.0


def _queue_alert_due(entry: dict, now: datetime) -> bool:
    """True when a stuck date may alarm again (at most hourly)."""
    last = entry.get("last_alerted")
    if not last:
        return True
    try:
        return (
            now - datetime.fromisoformat(last)
        ).total_seconds() >= _QUEUE_ALERT_REPEAT_MINUTES * 60
    except Exception:  # noqa: BLE001
        return True


def _queue_due(entry: dict, now: datetime) -> bool:
    """True when this pending date may be (re)fired: never fired, or the retry
    delay has elapsed since the last fire."""
    last = entry.get("last_fired")
    if not last:
        return True
    try:
        return (now - datetime.fromisoformat(last)).total_seconds() >= _QUEUE_RETRY_MINUTES * 60
    except Exception:  # noqa: BLE001
        return True


# ─────────────────────────────────────────────
# Stale factor sensor
# ─────────────────────────────────────────────


@sensor(
    minimum_interval_seconds=30 * 60,  # every 30 minutes
    description=(
        "Safety net for watermark_complete_sensor (Meridian #286). Checks "
        "whether any factor rows exist for the most recently closed ET "
        "trading day and, if not, consults db.trigger_cross_section_ready for "
        "that date before firing phase 1 (PHASE1_TRIGGER_ASSETS = CORE + vol): "
        "a not-ready answer defers (no run - the primary sensor will fire once "
        "the date clears), a ready answer proceeds. This keeps the backstop "
        "from re-introducing the clock-vs-watermark race the primary sensor "
        "exists to close - firing on a date that isn't actually ready would "
        "fail the same assert_cross_section_single_dated guard, and because "
        "this sensor's run_key is date-scoped (not tick-scoped), Dagster's own "
        "run_key dedup would then block EVERY future tick from retrying that "
        "date. Phase 2 then auto-fires via core_complete_sensor.\n\n"
        "Meridian#395 (2026-09): before this, an explicit watermark-NOT-complete "
        "answer deferred but an UNKNOWN answer (unreachable table, or simply no "
        "row yet for this date) proceeded unconditionally - and because LDP "
        "only writes the watermark row once a date's ingest actually finishes, "
        "'no row yet' during the evening ingest window is "
        "indistinguishable from 'watermark subsystem down' and may well "
        "mean 'still ingesting'. That let this sensor fire straight into the "
        "guard whenever LDP's ingest ran past this sensor's "
        "post-close gate (a fixed-clock delay cannot absorb a variable "
        "finish time). "
        "db.trigger_cross_section_ready still checks the watermark first, but "
        "when it is unknown it now falls back to the SAME coverage-ratio "
        "predicate the guard applies (queried directly against ohlcv.prices, "
        "so a broken watermark table specifically does not block a night where "
        "ingest actually finished normally) instead of assuming ready.\n\n"
        "The complementary phase-1-OK-but-phase-2-missing case (e.g. the "
        "gpu_asset RetryPolicy exhausts) is covered by "
        "stale_composite_sensor (Meridian #99 follow-up)."
    ),
    job=daily_factor_core_job,
)
def stale_factor_sensor(context: SensorEvaluationContext):
    now_et = datetime.now(ET)

    # Skip weekends - no new trading day to score. Monday 17:00 ET will pick
    # up Friday via _latest_trading_day_et's rollback. US market holidays that
    # fall on weekdays are handled downstream by _ensure_trading_day().
    if now_et.weekday() >= 5:
        return

    # Pre-close gate. Running before 17:00 ET would either target today
    # (partition doesn't exist yet in UTC-anchored daily partitions) or
    # target yesterday before watermark_complete_sensor has had a realistic
    # chance to fill it - either way, a useless tick.
    if now_et.hour < _POST_CLOSE_HOUR or (
        now_et.hour == _POST_CLOSE_HOUR and now_et.minute < _POST_CLOSE_MINUTE
    ):
        return

    engine = _get_engine()
    target_date = _ensure_trading_day(engine, _latest_trading_day_et(now_et))

    if db.has_scores_for_date(engine, target_date):
        return

    # Consult the shared readiness predicate before firing (Meridian #395).
    # Not-ready (watermark explicitly incomplete, OR watermark unknown AND the
    # coverage-ratio fallback also looks mid-ingest) defers - firing anyway
    # would hit the same assert_cross_section_single_dated guard that
    # motivated this whole cutover, and because run_key is date-scoped, a
    # failed run here would dedup-lock every later tick out of ever retrying
    # this date.
    ready, reason = db.trigger_cross_section_ready(engine, target_date)
    if not ready:
        context.log.info(f"stale_factor_sensor: {target_date} not ready ({reason}) - deferring")
        return

    # Meridian #358: another sensor (e.g. ldp_ohlcv_upstream_sensor,
    # watermark_complete_sensor) may already have a daily_factor_core_job run
    # in flight for this date - this sensor's own run_key would not catch
    # that, since it is scoped to ITS OWN prior fires only.
    if _partition_in_flight(context.instance, daily_factor_core_job.name, target_date):
        context.log.info(
            f"stale_factor_sensor: {target_date} already has a daily_factor_core_job "
            f"run in flight - skipping to avoid a duplicate"
        )
        return

    context.log.info(f"No scores for {target_date} (ET trading day) - triggering materialization")
    # High priority + numeric 50: mirrors watermark_complete_sensor. This
    # sensor is the safety net for the primary trigger, so it must preempt
    # backfill (priority 0) on the gpu pool. Without the bump, the backstop
    # sits behind the same queue that already starved the old schedule (see
    # schedules.py's module docstring for that history).
    yield RunRequest(
        run_key=f"stale_{target_date}",
        partition_key=str(target_date),
        tags={"dagster/priority": "50", "priority_class": "high"},
    )


# ─────────────────────────────────────────────
# API request sensor
# ─────────────────────────────────────────────


@sensor(
    minimum_interval_seconds=30,  # every 30 seconds - fast response to API requests
    description=(
        "Polls factors.compute_requests for pending dates. Triggers phase 1 "
        "phase 1 (CORE + vol, PHASE1_TRIGGER_ASSETS) for each pending date; phase 2 "
        "follows via "
        "core_complete_sensor when phase 1 lands. API consumers poll on a "
        "Retry-After loop; composite arrives ~phase-1 + phase-2 wallclock "
        "after the request."
    ),
    job=daily_factor_core_job,
)
def request_sensor(context: SensorEvaluationContext):
    """
    When the Meridian API gets a request for a date that's not in the DB,
    it writes a row to factors.compute_requests. This sensor picks up
    pending requests and triggers Dagster materialization for those dates.

    Flow:
    1. Consumer calls GET /scores?date=2024-06-01
    2. API checks factors.composite_scores - not found
    3. API writes to factors.compute_requests (score_date=2024-06-01, status='pending')
    4. API returns 202 Accepted with Retry-After: 120
    5. This sensor picks up the pending request within 30s
    6. Sensor triggers materialization for 2024-06-01 partition
    7. Assets compute and write to DB
    8. Consumer retries GET /scores?date=2024-06-01 - data now available
    """
    engine = _get_engine()
    pending = db.get_pending_requests(engine)

    for score_date in pending:
        # Meridian #358: another sensor (watermark_complete_sensor,
        # stale_factor_sensor, ldp_ohlcv_upstream_sensor) may already have a
        # daily_factor_core_job run in flight for this date. Leave the
        # request 'pending' rather than marking it 'computing' - the
        # composite_scores asset marks it 'done' unconditionally once the
        # in-flight run lands, regardless of which sensor launched it, so a
        # later tick simply re-checks rather than the request getting stuck.
        if _partition_in_flight(context.instance, daily_factor_core_job.name, score_date):
            context.log.info(
                f"API-requested: {score_date} already has a daily_factor_core_job "
                f"run in flight - leaving pending, skipping to avoid a duplicate"
            )
            continue
        context.log.info(f"API-requested: {score_date} - triggering materialization")
        db.mark_request(engine, score_date, "computing")
        # API-initiated requests are interactive: a consumer is waiting on
        # a 202 + Retry-After loop. Run at the same priority as the daily
        # schedule (50/high) so the request lands in one queue cycle even
        # when a backfill is in flight.
        yield RunRequest(
            run_key=f"request_{score_date}",
            partition_key=str(score_date),
            tags={"dagster/priority": "50", "priority_class": "high"},
        )


# ─────────────────────────────────────────────
# Phase-1 → Phase-2 bridge sensor
# ─────────────────────────────────────────────


@sensor(
    job=daily_gpu_composite_job,
    minimum_interval_seconds=30,
    default_status=DefaultSensorStatus.RUNNING,
    description=(
        "Bridges phase 1 (CORE_FACTOR_ASSETS) to phase 2 (GPU_COMPOSITE_ASSETS). "
        "Fires GPU_COMPOSITE_ASSETS for any partition where all CORE assets have "
        "materialised since the sensor's last seen storage_id per asset. Replaces "
        "the single-job model where phase 1 + phase 2 ran in one run pod, so a "
        "GPU asset fault could cascade across all assets. Backfill-correct: a "
        "phase-1 backfill of N partitions auto-emits N phase-2 RunRequests as each "
        "partition completes (serialised by the cluster-wide GPU cap=1).\n\n"
        "Cursor format (JSON): {asset_cursors: {asset: opaque_storage_cursor}, "
        "partitions_seen: {partition: [asset, ...]}, partition_first_seen: "
        "{partition: epoch}, partition_run_ids: {partition: {asset: [run_id, ...]}}}. "
        "Run keys include the upstream run so a fresh rescore can fire again. "
        "A CORE asset missing from asset_cursors is seeded "
        "to its latest record without processing history (handles both first "
        "tick and an asset newly added to CORE), and partials older than "
        "_PARTIAL_PARTITION_TTL_SECONDS are GC'd so a single-asset or cancelled "
        "backfill cannot wedge the sensor (Meridian #99). Replaces "
        "an earlier multi_asset_sensor variant which exhausted Dagster's "
        "25-event trailing-unconsumed buffer on partial-partition ticks.\n\n"
        "Age-gated to LIVE partitions only: any partition dated more than "
        "CORE_COMPLETE_MAX_PARTITION_AGE_DAYS (env, default 10) calendar days "
        "before the current tick's UTC date is dropped from partitions_seen "
        "without firing, even if it completed the full CORE set. Historical "
        "phase-2 (a CORE backfill re-scoring old partitions) is launched "
        "deliberately via meridian_backfill_job; this sensor's job is bridging "
        "phase 1 to phase 2 for the LIVE evening only. Two backfill waves "
        "(2026-07, 66 then 100+ runs) fired one high-priority GPU RunRequest "
        "per backfilled historical partition each, starving the gpu pool "
        "(cap=1) - the #99 TTL doesn't catch this because a backfill partition "
        "genuinely completes, it's just old."
    ),
)
def core_complete_sensor(context: SensorEvaluationContext):
    """Fire phase 2 per partition once all CORE assets have materialised.

    Polls the event log for new ASSET_MATERIALIZATION records per CORE asset
    (using fetch_materializations with a storage_id cursor), accumulates
    per-partition asset coverage in the sensor's JSON cursor, and emits one
    phase-2 RunRequest per partition that has hit all CORE assets. The
    cursor is the durable state - no reliance on Dagster's per-asset
    trailing-unconsumed-events buffer (the cap that broke the earlier
    multi_asset_sensor variant: DagsterInvariantViolationError 'You have
    reached the maximum number of trailing unconsumed events (25)').

    Before firing, any tracked partition older than
    CORE_COMPLETE_MAX_PARTITION_AGE_DAYS is dropped without a RunRequest -
    this sensor is LIVE-continuity only; historical phase-2 is launched
    deliberately via meridian_backfill_job (see _partition_age_days).
    """
    cursor = json.loads(context.cursor) if context.cursor else {}
    # Per-asset forward cursors. Stored as OPAQUE STRINGS as returned by
    # fetch_materializations - the API encodes a base64 JSON like
    # {"type": "STORAGE_ID", "value": N} that must be passed back verbatim
    # to advance the iterator. Casting to int() blows up on the base64
    # blob (Meridian!121 hotfix).
    asset_cursors: dict[str, str] = cursor.get("asset_cursors", {})
    partitions_seen: dict[str, list[str]] = cursor.get("partitions_seen", {})
    # Explicit materializations and implicit Output events from one upstream
    # run are one generation. Retain interleaved runs until each has full
    # coverage; a late event from an older run must not erase newer progress.
    partition_run_ids: dict[str, dict[str, list[str]]] = cursor.get("partition_run_ids", {})
    # Wall-clock epoch when each tracked partition was first observed (its
    # first CORE-asset materialisation). Drives the TTL GC below so partials
    # that never complete cannot accumulate (Meridian #99).
    partition_first_seen: dict[str, float] = cursor.get("partition_first_seen", {})

    now_ts = _now_epoch()
    core_set = set(CORE_FACTOR_ASSET_KEYS)
    # run_id -> "did this run cover the full CORE set" decisions, cached so a
    # backfill batch (one run, many records) costs a single run lookup.
    run_cover_cache: dict[str, bool] = {}
    new_records_total = 0
    for asset_key_str in CORE_FACTOR_ASSET_KEYS:
        # Seed any CORE asset that has no cursor yet, then SKIP it for this
        # tick. This covers two cases with one branch:
        #   - first tick ever (no asset has a cursor), and
        #   - an asset newly ADDED to CORE_FACTOR_ASSET_KEYS after the cursor
        #     was already initialised (e.g. a later-added asset).
        # We initialise it to its LATEST record WITHOUT processing history, so
        # the sensor only fires for materialisations strictly after this point.
        # Without this, a newly-added asset is fetched from cursor=None and
        # walks its entire backfilled history forward; until that crawl reaches
        # the present no current partition shows all CORE assets and phase-2
        # stops firing entirely. That is exactly what wedged the sensor on
        # 2026-05-28 (Meridian #99) - the old all-or-nothing `if not
        # asset_cursors` bootstrap did NOT handle a single newly-added asset.
        if asset_key_str not in asset_cursors:
            latest = context.instance.fetch_materializations(
                AssetKey(asset_key_str), limit=1, ascending=False
            )
            # Truthy guard: an asset that has never materialised returns no
            # cursor. Leave it unseeded; the regular fetch below passes
            # cursor=None and walks an empty history (nothing exists yet),
            # which is safe and cheap, and we seed it for real once it
            # produces its first record.
            if latest.cursor:
                asset_cursors[asset_key_str] = latest.cursor
                context.log.info(
                    f"core_complete_sensor: seeded newly-tracked CORE asset "
                    f"{asset_key_str!r} to its latest record without processing "
                    f"history"
                )
                # Cursor now sits at the latest record - nothing to accumulate.
                continue

        asset_cursor = asset_cursors.get(asset_key_str)
        # ascending=True so we walk forward in storage_id order and advance
        # the cursor to the last record we actually saw (otherwise a burst
        # > limit could leave a gap on subsequent ticks).
        result = context.instance.fetch_materializations(
            AssetKey(asset_key_str),
            cursor=asset_cursor,
            limit=_MAX_RECORDS_PER_ASSET_PER_TICK,
            ascending=True,
        )
        for record in result.records:
            partition_key = record.partition_key
            if not partition_key or not record.run_id:
                # Non-partitioned materialisation shouldn't happen for any
                # CORE asset (all DailyPartitionsDefinition-backed), but
                # guard against rogue manual materialisations.
                continue
            if not _run_covers_core(context.instance, record.run_id, core_set, run_cover_cache):
                # Materialisation came from a partial backfill that does not
                # cover the full CORE set (e.g. a single-asset
                # backfill). It can never complete a partition, so don't track
                # it. The cursor still advances below so we never re-see it.
                continue
            assets_for_partition = partitions_seen.setdefault(partition_key, [])
            source_runs = partition_run_ids.setdefault(partition_key, {}).setdefault(
                asset_key_str, []
            )
            if record.run_id not in source_runs:
                source_runs.append(record.run_id)
            if asset_key_str not in assets_for_partition:
                assets_for_partition.append(asset_key_str)
            partition_first_seen.setdefault(partition_key, now_ts)
            new_records_total += 1
        # Always advance the cursor to the API's forward cursor so we never
        # re-process a record on the next tick, even when the partition
        # isn't yet complete. The cursor is an opaque string - pass through
        # without parsing. fetch_materializations advances the cursor only
        # when it returns records; on an empty batch (no new materialisations
        # for the asset) result.cursor may be None or unchanged, so guard.
        if result.cursor:
            asset_cursors[asset_key_str] = result.cursor

    # Age gate: drop any tracked partition older than the live window before
    # deciding what's ready to fire. This runs BEFORE the ready-partition scan
    # so an old, fully-completed partition (e.g. a CORE backfill re-scoring
    # 2013-2014) never emits a phase-2 RunRequest - only meridian_backfill_job
    # is allowed to launch historical phase-2. Applies uniformly to entries
    # accumulated this tick and any pre-existing entries carried in the cursor
    # from before this gate was deployed.
    today_utc = _today_utc()
    max_age_days = _max_partition_age_days()
    age_dropped = [
        pk
        for pk in partitions_seen
        if (age := _partition_age_days(pk, today_utc)) is not None and age > max_age_days
    ]
    for pk in age_dropped:
        del partitions_seen[pk]
        partition_first_seen.pop(pk, None)
    if age_dropped:
        context.log.info(
            f"core_complete_sensor: age-gated {len(age_dropped)} partition(s) older than "
            f"{max_age_days}d - dropped without firing phase 2 (historical phase-2 is "
            f"launched deliberately via meridian_backfill_job)"
        )

    # Decide which partitions are ready to fire phase 2 (core_set defined above).
    ready_partitions = []
    for pk, seen in sorted(partitions_seen.items()):
        if not core_set <= set(seen):
            continue
        sources = partition_run_ids.get(pk, {})
        if core_set <= sources.keys():
            complete_runs = set.intersection(*(set(sources[k]) for k in core_set))
            ready_partitions.extend((pk, run_id) for run_id in sorted(complete_runs))
        elif len({run_id for runs in sources.values() for run_id in runs}) <= 1:
            # Pre-upgrade partial cursors do not contain source identities.
            # Finish that legacy coverage once using the old deduplication key.
            ready_partitions.append((pk, None))

    for pk, generation in ready_partitions:
        yield RunRequest(
            # Legacy entries without full source identities retain the old key.
            run_key=f"core_complete_{pk}_{generation}" if generation else f"core_complete_{pk}",
            partition_key=pk,
            tags={
                "dagster/priority": "50",
                "priority_class": "high",
                **({"meridian/upstream_run_id": generation} if generation else {}),
            },
        )
        # Consume only this generation; preserve interleaved incomplete runs.
        sources = partition_run_ids.get(pk, {})
        remaining = (
            {
                asset_key: [run_id for run_id in runs if run_id != generation]
                for asset_key, runs in sources.items()
            }
            if generation
            else {}
        )
        remaining = {k: v for k, v in remaining.items() if v}
        if remaining:
            partition_run_ids[pk] = remaining
            partitions_seen[pk] = list(remaining)
            partition_first_seen[pk] = now_ts
        else:
            partition_run_ids.pop(pk, None)
            partitions_seen.pop(pk, None)
            partition_first_seen.pop(pk, None)

    # TTL GC: reap partials tracked longer than the TTL without completing.
    # Keyed on wall-clock first-seen, NOT partition date, so an in-flight
    # backfill (all CORE assets for a partition land within one phase-1 run)
    # is never dropped, while a partial that can never complete is cleared
    # instead of pinning a tracker slot forever. This is the unbounded state
    # that polluted the sensor (Meridian #99): dead partials left by a
    # cancelled single-asset backfill.
    stale = [
        pk
        for pk in list(partitions_seen)
        if (now_ts - partition_first_seen.get(pk, now_ts)) > _PARTIAL_PARTITION_TTL_SECONDS
    ]
    for pk in stale:
        del partitions_seen[pk]
        partition_first_seen.pop(pk, None)
    if stale:
        context.log.info(
            f"core_complete_sensor: GC'd {len(stale)} stale partial partition(s) "
            f"tracked > {_PARTIAL_PARTITION_TTL_SECONDS}s without completing"
        )

    # Hard backstop. If partitions somehow accumulate faster than the TTL
    # reaps them, drop the oldest. ISO date keys sort chronologically so this
    # retains the most-recent in-flight set.
    if len(partitions_seen) > _MAX_TRACKED_PARTITIONS:
        keep = set(sorted(partitions_seen.keys())[-_MAX_TRACKED_PARTITIONS:])
        partitions_seen = {k: v for k, v in partitions_seen.items() if k in keep}

    # Keep partition_first_seen in lockstep so it cannot leak entries for
    # partitions that have been fired or dropped.
    partition_first_seen = {k: v for k, v in partition_first_seen.items() if k in partitions_seen}

    context.update_cursor(
        json.dumps(
            {
                "asset_cursors": asset_cursors,
                "partitions_seen": partitions_seen,
                "partition_first_seen": partition_first_seen,
                "partition_run_ids": {
                    k: v for k, v in partition_run_ids.items() if k in partitions_seen
                },
            }
        )
    )

    if ready_partitions or new_records_total:
        context.log.info(
            f"core_complete_sensor: saw {new_records_total} new materialisation(s), "
            f"fired {len(ready_partitions)} phase-2 run(s), "
            f"{len(partitions_seen)} partition(s) still in-flight"
        )


# The ntfy transport itself lives in meridian_pipeline.notify (shared with
# the non-blocking detective-check alert emitted by composite_scores, so the
# topic/URL/fail-open semantics are defined in exactly one place).


@run_failure_sensor(
    name="ntfy_run_failure_sensor",
    description=(
        "Fires on every run transition to FAILURE. POSTs an ntfy alert to the "
        "alerts topic with job name, run id prefix, and the first 300 "
        "chars of the failure event message. Fail-open: ntfy transport errors "
        "are logged, never raised, so a broken notifier can't wedge the daemon."
    ),
    default_status=DefaultSensorStatus.RUNNING,
)
def ntfy_run_failure_sensor(context: RunFailureSensorContext) -> None:
    run = context.dagster_run
    job = run.job_name
    event_msg = (context.failure_event.message or "")[:300]

    lines = ["Run FAILED", f"  job: {job}", f"  run_id: {run.run_id[:12]}"]
    if event_msg:
        lines.append(f"  failure: {event_msg}")

    send_ntfy(
        "\n".join(lines),
        title=f"Meridian: {job} FAILED",
        priority="urgent",
        tags="rotating_light",
        click=f"https://dagster.example.com/runs/{run.run_id}",
        log=context.log,
    )


# ─────────────────────────────────────────────
# Asset-check alert sensor (Meridian#427)
# ─────────────────────────────────────────────
#
# Why #427 exists: a staleness check can be deliberately WARN-and-write, so
# the run SUCCEEDS and ntfy_run_failure_sensor (which only fires on run
# FAILURE) never sees it. This sensor closes
# that class: it reads asset-check evaluations for a short explicit allowlist
# and pages the owner through the same send_ntfy transport.


class AlertedCheck(NamedTuple):
    """One allowlisted asset check. ``rows_key`` names the check-result
    metadata entry (a JSON list of dicts) carrying one row per affected
    partition; each row MUST have a ``partition`` key, every other key is
    rendered ``k=v`` into the page. A failed evaluation with no such rows is
    paged once per evaluation instead."""

    asset: str
    check: str
    rows_key: str | None = None
    label: str = ""


# ADD A CHECK = ADD ONE LINE HERE. The literal below equals
# risk_model_asset._GROUP_SCORES_GATE_CHECK_NAME (a test pins the equality;
# the literal avoids importing the heavy risk asset module into sensors.py).
_ALERTED_CHECKS: tuple[AlertedCheck, ...] = (
    AlertedCheck(
        "example_asset",
        "check_example_asset_gate",
        "stale_detail",
        "example staleness",
    ),
)

# Evaluations pulled per tick, and individual pages sent per tick. A burst
# (a backfill with many stale dates) drains over several ticks rather than
# holding one evaluation open on 5s ntfy timeouts.
_ALERT_MAX_EVALS_PER_TICK = 50
_ALERT_MAX_PAGES_PER_TICK = 5
# Dedupe memory bound (oldest dropped). An evicted key can only re-page if its
# evaluation is re-read, which the monotone storage cursor prevents.
_ALERT_MAX_PAGED_KEYS = 500
# First tick (no cursor): look back this many hours for evaluations instead of
# crawling all history or silently skipping a WARN that just landed.
_ALERT_BOOTSTRAP_LOOKBACK_HOURS = 72
# Partitions older than this are logged, not paged (a historical backfill that
# trips the check is not a live-continuity problem). Mirrors core_complete.
_ALERT_MAX_PARTITION_AGE_DAYS = 14


def _metadata_plain(mv):
    """Unwrap a Dagster MetadataValue to a plain Python value."""
    for attr in ("data", "value"):
        if hasattr(mv, attr):
            return getattr(mv, attr)
    return mv


def _alert_rows(spec: AlertedCheck, metadata, eval_id: int) -> list[dict]:
    """Per-partition rows for one failed evaluation (see AlertedCheck)."""
    rows = None
    if spec.rows_key and spec.rows_key in (metadata or {}):
        raw = _metadata_plain(metadata[spec.rows_key])
        if isinstance(raw, list):
            rows = [r for r in raw if isinstance(r, dict) and r.get("partition")]
    if rows:
        return rows
    return [{"partition": f"eval-{eval_id}"}]


@sensor(
    name="asset_check_alert_sensor",
    minimum_interval_seconds=120,
    default_status=DefaultSensorStatus.RUNNING,
    description=(
        "Pages the owner (ntfy alerts, same transport as "
        "ntfy_run_failure_sensor) when an allowlisted asset check evaluates "
        "FAILED, including WARN-severity failures that leave the run "
        "successful (Meridian#427: WARN-severity failures leave the run green "
        "and can go unseen). Allowlist: "
        "sensors._ALERTED_CHECKS (one line per check). Pages once per "
        "(check, partition); passing checks never page. Regular @sensor "
        "(never multi_asset_sensor, rule 13).\n\n"
        'Cursor (JSON): {"after_storage_id": <int|null>, '
        '"paged": {"<asset>/<check>/<partition>": <epoch>}}. The storage cursor advances only past '
        "evaluations whose pages all sent, so an ntfy outage retries on the "
        "next tick without re-paging what already went out. Partitions older "
        f"than {_ALERT_MAX_PARTITION_AGE_DAYS} days are logged, not paged."
    ),
)
def asset_check_alert_sensor(context: SensorEvaluationContext):
    """Page once per (check, partition) for failed allowlisted checks."""
    from dagster import DagsterEventType, EventRecordsFilter

    cursor = json.loads(context.cursor) if context.cursor else {}
    after_id = cursor.get("after_storage_id")
    paged: dict[str, float] = dict(cursor.get("paged") or {})
    now = _now_epoch()
    today = _today_utc()
    allow = {(c.asset, c.check): c for c in _ALERTED_CHECKS}
    pages_sent = 0
    blocked = False

    # NOTE: EventRecordsFilter(asset_key=...) matches NO asset-check evaluation
    # events (verified on Dagster 1.13), so read every evaluation and match the
    # allowlist in Python. Evaluations are low-volume (a handful per run).
    flt_kwargs: dict = {"event_type": DagsterEventType.ASSET_CHECK_EVALUATION}
    if after_id is not None:
        flt_kwargs["after_cursor"] = after_id
    else:
        flt_kwargs["after_timestamp"] = now - _ALERT_BOOTSTRAP_LOOKBACK_HOURS * 3600
    records = context.instance.get_event_records(
        EventRecordsFilter(**flt_kwargs),
        limit=_ALERT_MAX_EVALS_PER_TICK,
        ascending=True,
    )
    last_ok = after_id
    for rec in records:
        entry = rec.event_log_entry
        data = entry.dagster_event.event_specific_data if entry.dagster_event else None
        spec = allow.get((data.asset_key.to_user_string(), data.check_name)) if data else None
        if spec is None or data.passed:
            last_ok = rec.storage_id
            continue
        cursor_key = f"{spec.asset}/{spec.check}"
        rows = _alert_rows(spec, data.metadata, rec.storage_id)
        severity = getattr(getattr(data, "severity", None), "value", "WARN")
        for row in rows:
            part = str(row["partition"])
            key = f"{cursor_key}/{part}"
            if key in paged:
                continue
            age = _partition_age_days(part, today)
            if age is not None and age > _ALERT_MAX_PARTITION_AGE_DAYS:
                context.log.info(
                    f"asset_check_alert_sensor: {key} is {age}d old "
                    f"(> {_ALERT_MAX_PARTITION_AGE_DAYS}d) - not paged"
                )
                paged[key] = now
                continue
            if pages_sent >= _ALERT_MAX_PAGES_PER_TICK:
                blocked = True
                break
            extra = {k: v for k, v in row.items() if k != "partition"}
            lines = [
                f"{spec.label or spec.check} {severity}",
                f"  asset: {spec.asset}",
                f"  check: {spec.check}",
                f"  partition: {part}",
            ] + [f"  {k}: {v}" for k, v in extra.items()]
            ok = send_ntfy(
                "\n".join(lines),
                title=f"Meridian: {spec.label or spec.check} {severity} {part}",
                priority="urgent" if severity == "ERROR" else "high",
                tags="warning",
                click=f"https://dagster.example.com/runs/{entry.run_id}",
                log=context.log,
            )
            if not ok:
                # Transport down: leave this evaluation unconsumed so the next
                # tick retries it (already-paged keys dedupe the rest).
                blocked = True
                break
            pages_sent += 1
            paged[key] = now
            context.log.info(
                f"asset_check_alert_sensor: paged {key} severity={severity} "
                + " ".join(f"{k}={v}" for k, v in extra.items())
            )
        if blocked:
            break
        last_ok = rec.storage_id

    if len(paged) > _ALERT_MAX_PAGED_KEYS:
        keep = sorted(paged.items(), key=lambda kv: kv[1])[-_ALERT_MAX_PAGED_KEYS:]
        paged = dict(keep)
    new_cursor = {"after_storage_id": last_ok, "paged": paged}
    context.update_cursor(json.dumps(new_cursor, sort_keys=True))
    if pages_sent:
        context.log.info(f"asset_check_alert_sensor: paged {pages_sent} new (check, partition)")


# ─────────────────────────────────────────────
# LDP upstream OHLCV sensor
# ─────────────────────────────────────────────

# LDP's ohlcv_prices asset is ticker-partitioned (one partition per ticker).
# Meridian's CORE assets are date-partitioned. There is NO 1:1 partition map.
# The sensor strategy: when ANY new ohlcv_prices materialisation is seen
# (i.e. LDP wrote fresh OHLCV for at least one ticker), resolve the most
# recent closed ET trading day and fire one phase-1 RunRequest for it,
# deduped via a per-date run_key. This is conservative: a single fresh
# ticker-partition is enough signal that the day's OHLCV landed.
#
# Refining to "wait until N% of the universe has materialised before firing"
# is a follow-up design question - it requires knowing the expected ticker
# count per day (universe size) and tracking which tickers have landed.
# That complexity is deferred; the current approach mirrors how
# stale_factor_sensor reasons about the trading day and fires at the same
# priority (50/high).
_LDP_OHLCV_ASSET_KEY = AssetKey(["ohlcv_prices"])

# Per-tick fetch cap for LDP materialisation records. LDP is ticker-partitioned
# so a full daily run can emit thousands of records (one per ticker). We only
# need to know that AT LEAST ONE landed; limit=200 is sufficient signal while
# bounding memory per tick. The cursor still advances past all 200 so the next
# tick starts from where we left off; if the batch exceeds the cap the sensor
# will see the remainder next tick - still correct, just slightly lagged.
_MAX_LDP_RECORDS_PER_TICK = 200


@sensor(
    minimum_interval_seconds=5 * 60,  # every 5 minutes
    default_status=DefaultSensorStatus.STOPPED,
    job=daily_factor_core_job,
    description=(
        "SUPERSEDED (Meridian #286, 2026-08-18): watermark_complete_sensor now "
        "fills the role this sensor's cutover plan below describes, using a "
        "strictly better signal - LDP's own completion watermark "
        "(ohlcv.ingest_watermark, live 2026-08-15, postdates this sensor) "
        "instead of inferring completion from the first ticker materialising. "
        "'a single new ohlcv_prices materialisation is sufficient signal' "
        "(this sensor's own design note below) is exactly the kind of "
        "inference the watermark exists to replace with a direct answer. "
        "Left registered and STOPPED rather than deleted, in case the "
        "watermark table itself ever needs a fallback trigger - do not flip "
        "this to RUNNING without re-reading watermark_complete_sensor's "
        "docstring first, both target the same CORE_FACTOR_ASSETS and would "
        "double-fire.\n\n"
        "Original description follows unchanged.\n\n"
        "Event-driven cross-code-location sensor. Watches LDP's ohlcv_prices "
        "asset materialisations and fires Meridian phase-1 (CORE + vol, "
        "PHASE1_TRIGGER_ASSETS) "
        "when fresh upstream OHLCV data lands - as a more responsive alternative "
        "to the timer-based stale_factor_sensor.\n\n"
        "Partition mismatch: LDP ohlcv_prices is TICKER-partitioned; Meridian "
        "CORE assets are DATE-partitioned. There is no 1:1 partition map. The "
        "sensor resolves the most recent closed ET trading day using the same "
        "_latest_trading_day_et / _ensure_trading_day helpers as stale_factor_sensor, "
        "then fires one RunRequest for that date partition deduped via a per-date "
        "run_key. A single new ohlcv_prices materialisation is treated as sufficient "
        "signal that the day's OHLCV batch is landing.\n\n"
        "Cutover plan (replacing stale_factor_sensor):\n"
        "  1. Enable this sensor (DefaultSensorStatus.RUNNING) on a test window.\n"
        "  2. Confirm it fires within 5 min of LDP's ohlcv_prices landing on "
        "     multiple consecutive trading days.\n"
        "  3. Once stable (>= 5 consecutive days), disable stale_factor_sensor "
        "     and set this sensor to DefaultSensorStatus.RUNNING permanently.\n"
        "  4. Keep stale_factor_sensor registered as a last-resort backstop, or "
        "     remove it via a separate MR once confidence is high.\n\n"
        "Follow-up design question: refine to 'wait for N% of the ticker universe "
        "to materialise before firing' rather than reacting to the first ticker "
        "landing. This requires tracking per-ticker materialisation counts and "
        "knowing the expected universe size per day - deferred to a future issue.\n\n"
        "Ships STOPPED (DefaultSensorStatus.STOPPED). Do NOT remove or disable "
        "stale_factor_sensor until this sensor has been validated in production.\n\n"
        "Meridian#395: default_status only seeds a sensor's initial state; it "
        "does not overwrite a status someone has since toggled in the instance, "
        "so this sensor can end up running despite default_status=STOPPED here. "
        "That is an operational matter to fix by stopping it via the UI/GraphQL, "
        "not something a code change can reach - but the sensor now also "
        "consults db.trigger_cross_section_ready before firing (same gate as "
        "stale_factor_sensor) so that, running or not, it can no longer fire a "
        "run that is certain to fail."
    ),
)
def ldp_ohlcv_upstream_sensor(context: SensorEvaluationContext):
    """Fire phase-1 (CORE + vol, PHASE1_TRIGGER_ASSETS) when LDP's ohlcv_prices lands.

    Reads the Dagster event log for new ohlcv_prices ASSET_MATERIALIZATION
    records using fetch_materializations with a storage_id cursor (the same
    pattern as core_complete_sensor - never uses multi_asset_sensor which
    hits Dagster's 25-event trailing-unconsumed cap).

    Cursor format (JSON): {"ohlcv_cursor": <opaque_str_or_null>}.

    When fresh materialisations are seen since the cursor, resolves the most
    recent closed ET trading day (weekday + post-17:00 ET gate + DB holiday
    check) and yields ONE RunRequest for that date partition at priority
    50/high. The per-date run_key prevents duplicate runs if the sensor ticks
    again before the partition is fully scored.

    The 17:00 ET gate and weekday guard mirror stale_factor_sensor:
      - Before market close the day's partition does not yet exist in
        Dagster's UTC-anchored DailyPartitionsDefinition.
      - Weekend ticks still advance the ohlcv cursor (to stay current) but
        yield no RunRequest.
    """
    cursor_obj: dict = json.loads(context.cursor) if context.cursor else {}
    ohlcv_cursor: str | None = cursor_obj.get("ohlcv_cursor")

    result = context.instance.fetch_materializations(
        _LDP_OHLCV_ASSET_KEY,
        cursor=ohlcv_cursor,
        limit=_MAX_LDP_RECORDS_PER_TICK,
        ascending=True,
    )

    new_records = len(result.records)

    # Always advance the cursor so we never re-process seen records, even
    # when we decide not to fire (e.g. pre-close or weekend).
    if result.cursor:
        ohlcv_cursor = result.cursor

    context.update_cursor(json.dumps({"ohlcv_cursor": ohlcv_cursor}))

    if new_records == 0:
        return  # nothing new from LDP; nothing to do

    context.log.info(
        f"ldp_ohlcv_upstream_sensor: {new_records} new ohlcv_prices materialisation(s) detected"
    )

    now_et = datetime.now(ET)

    # Weekend guard: LDP may ingest or re-partition on weekends; we have no
    # date partition to target until the next trading day closes.
    if now_et.weekday() >= 5:
        context.log.info("ldp_ohlcv_upstream_sensor: weekend tick - cursor advanced, no RunRequest")
        return

    # Pre-close gate: the UTC-anchored DailyPartitionsDefinition creates
    # today's partition only after midnight UTC, which is typically well
    # before 17:00 ET. However, scoring before 17:00 ET targets a session
    # that has not yet closed; the resulting scores would be based on
    # incomplete intraday OHLCV data. Align with the daily schedule gate.
    if now_et.hour < _POST_CLOSE_HOUR or (
        now_et.hour == _POST_CLOSE_HOUR and now_et.minute < _POST_CLOSE_MINUTE
    ):
        context.log.info(
            f"ldp_ohlcv_upstream_sensor: pre-close ({now_et.strftime('%H:%M')} ET) "
            f"- cursor advanced, no RunRequest"
        )
        return

    engine = _get_engine()
    target_date = _ensure_trading_day(engine, _latest_trading_day_et(now_et))

    # Readiness gate (Meridian #395): this sensor's whole premise - "a single
    # new ohlcv_prices materialisation is sufficient signal" - is exactly the
    # premature-trigger inference watermark_complete_sensor was built to
    # replace (see the SUPERSEDED note above). default_status=STOPPED only
    # seeds the sensor's initial state, so it can end up enabled in a live
    # instance regardless of the code's stated intent. Consult the same shared
    # predicate stale_factor_sensor
    # uses so that, running or not, this sensor can no longer fire a run that
    # is certain to fail.
    ready, reason = db.trigger_cross_section_ready(engine, target_date)
    if not ready:
        context.log.info(
            f"ldp_ohlcv_upstream_sensor: {target_date} not ready ({reason}) - deferring"
        )
        return

    # Meridian #358: another sensor (watermark_complete_sensor,
    # stale_factor_sensor, request_sensor) may already have a
    # daily_factor_core_job run in flight for this date - this sensor's own
    # per-date run_key only dedupes against ITS OWN prior fires.
    if _partition_in_flight(context.instance, daily_factor_core_job.name, target_date):
        context.log.info(
            f"ldp_ohlcv_upstream_sensor: {target_date} already has a "
            f"daily_factor_core_job run in flight - skipping to avoid a duplicate"
        )
        return

    context.log.info(
        f"ldp_ohlcv_upstream_sensor: triggering phase-1 (CORE + vol) for {target_date}"
    )
    # Priority 50/high mirrors the daily schedule and stale_factor_sensor so
    # LDP-triggered runs preempt historical backfills on the gpu pool. The
    # per-date run_key deduplicates across ticks: if the sensor fires multiple
    # times in the same evening (LDP writes trickling in) only the first
    # RunRequest for a given date is executed; subsequent ones are no-ops.
    yield RunRequest(
        run_key=f"ldp_ohlcv_{target_date}",
        partition_key=str(target_date),
        tags={"dagster/priority": "50", "priority_class": "high"},
    )


# Composite -> API cache warm sensor
# ─────────────────────────────────────────────

_logger = logging.getLogger(__name__)

#: In-cluster default; override via MERIDIAN_API_BASE_URL env var.
_DEFAULT_API_BASE_URL = "http://meridian-api:8100"


@sensor(
    minimum_interval_seconds=60,
    default_status=DefaultSensorStatus.STOPPED,
    description=(
        "Watches composite_scores materialisations. For each new partition date "
        "seen, calls POST /admin/cache/warm?date=<date> on the Meridian API so "
        "stale in-process cache entries are evicted before the first consumer "
        "request arrives. Closes the write->serve gap (Meridian #145).\n\n"
        "Fail-soft: HTTP errors are logged and never re-raised so a transient "
        "API hiccup cannot break the sensor tick.\n\n"
        "Enable via the Dagster UI (sensor is STOPPED by default so it does not "
        "make live HTTP calls during local development or CI). Requires the "
        "MERIDIAN_API_TOKEN env var to hold a valid JWT Bearer token. "
        "MERIDIAN_API_BASE_URL may be set to override the in-cluster default "
        "(http://meridian-api:8100)."
    ),
)
def composite_cache_warm_sensor(context: SensorEvaluationContext):
    """Invalidate the API score cache for each newly materialised composite partition.

    Cursor format (JSON string): the opaque forward cursor returned by the last
    call to fetch_materializations for composite_scores. Passed back verbatim on
    the next tick to advance the iterator without re-processing old records.
    """
    cursor = context.cursor  # opaque string or None

    result = context.instance.fetch_materializations(
        AssetKey(["composite_scores"]),
        cursor=cursor,
        limit=50,
        ascending=True,
    )

    dates_warmed: list[str] = []
    for record in result.records:
        partition_key = record.partition_key
        if not partition_key:
            continue
        _call_warm_api(context, partition_key)
        dates_warmed.append(partition_key)

    if result.cursor:
        context.update_cursor(result.cursor)

    if dates_warmed:
        context.log.info(
            "composite_cache_warm_sensor: warmed API cache for %d date(s): %s",
            len(dates_warmed),
            ", ".join(sorted(dates_warmed)[-5:]),  # log last 5 to keep output terse
        )
    else:
        yield SkipReason("no new composite_scores materialisations")
        return


def _call_warm_api(context: SensorEvaluationContext, date_str: str) -> None:
    """POST /admin/cache/warm?date=<date> to the Meridian API.

    All exceptions are caught and logged so a transient HTTP error or
    misconfigured env var never crashes the sensor tick.
    """
    base_url = os.environ.get("MERIDIAN_API_BASE_URL", _DEFAULT_API_BASE_URL).rstrip("/")
    token = os.environ.get("MERIDIAN_API_TOKEN", "")
    url = f"{base_url}/admin/cache/warm"
    try:
        resp = requests.post(
            url,
            params={"date": date_str},
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
        )
        if resp.ok:
            context.log.info(
                "composite_cache_warm_sensor: warmed cache for %s -> %s",
                date_str,
                resp.json(),
            )
        else:
            _logger.warning(
                "composite_cache_warm_sensor: warm call for %s returned HTTP %s: %s",
                date_str,
                resp.status_code,
                resp.text[:200],
            )
    except Exception as exc:  # noqa: BLE001
        _logger.warning(
            "composite_cache_warm_sensor: warm call for %s failed: %s",
            date_str,
            exc,
        )


# ─────────────────────────────────────────────
# Iceberg gold -> PG promotion sensor (Meridian#265)
# ─────────────────────────────────────────────

# Targets the named gold_pg_sync_job. That job lives in jobs.py, not
# definitions.py, precisely so this module can import it without the cycle
# definitions.py would create (definitions.py imports sensors.py). Same job
# object serves manual/GraphQL launches (Meridian #314).


_GOLD_SNAPSHOT_TAG = "meridian/gold_snapshot_id"


def _gold_sync_request(snapshot_id: int, run_key: str) -> RunRequest:
    """RunRequest promoting exactly one gold snapshot. The snapshot tag is how a
    later tick finds this snapshot's runs (the run_key changes on a retry)."""
    return RunRequest(
        run_key=run_key,
        run_config={"ops": {"gold_scores_pg_sync": {"config": {"snapshot_id": snapshot_id}}}},
        tags={
            "dagster/priority": "50",
            "priority_class": "high",
            _GOLD_SNAPSHOT_TAG: str(snapshot_id),
        },
    )


# A promotion run QUEUED/STARTED longer than this is logged at ERROR and alerted
# (rate-limited by last_alerted); it is never cancelled.
_GOLD_STUCK_HOURS = 6


def _notify_stuck(context, title: str, message: str) -> None:
    """ntfy for a stuck sensor state. The CALLER owns the hourly last_alerted gate.
    Fail-open (send_ntfy never raises)."""
    send_ntfy(
        message,
        title=title,
        priority="high",
        tags="warning",
        click="https://dagster.example.com/runs",
        log=context.log,
    )


def _gold_run_promoted(instance, run) -> bool:
    """True unless the SUCCESS run's gold_scores_pg_sync materialization says it
    was skipped. A skipped run promoted nothing and must not acknowledge the
    snapshot. If the event log cannot be read the answer is True: the asset
    raises (not skips) for a snapshot-scoped run, so SUCCESS already implies a
    promotion and this check is defence in depth for hand-launched runs."""
    try:
        for entry in instance.all_logs(run.run_id, of_type=DagsterEventType.ASSET_MATERIALIZATION):
            mat = entry.dagster_event.event_specific_data.materialization
            skipped = (mat.metadata or {}).get("skipped")
            if skipped is not None and getattr(skipped, "value", None) is True:
                return False
    except Exception:  # noqa: BLE001
        return True
    return True


def _gold_snapshot_run_state(instance, snapshot_id: int) -> str:
    """'success' | 'in_flight' | 'failed' for the gold_pg_sync_job runs of one
    snapshot, found by tag. 'failed' covers failure, cancellation and no run at
    all (a RunRequest that never produced a run is retried like a failed one).
    A SUCCESS run whose materialization says it was skipped does not count."""
    runs = instance.get_runs(
        filters=RunsFilter(
            job_name=gold_pg_sync_job.name, tags={_GOLD_SNAPSHOT_TAG: str(snapshot_id)}
        ),
        limit=50,
    )
    statuses = {r.status for r in runs}
    if any(r.status == DagsterRunStatus.SUCCESS and _gold_run_promoted(instance, r) for r in runs):
        return "success"
    if statuses & set(_PARTITION_IN_FLIGHT_STATUSES):
        return "in_flight"
    return "failed"


@sensor(
    minimum_interval_seconds=60,
    # STOPPED-first (Meridian#265 review): the first promotion is a brand-new,
    # high-blast-radius write path into serving gold. Keep the sensor OFF until
    # the companion write path (a recompute wired through lakehouse.write_gold_
    # scores) lands AND the inaugural promotion has been validated via a manual
    # gold_pg_sync_job launch; then flip to RUNNING for unattended operation.
    # No-op cost today: nothing writes gold_promotion_scores yet, so RUNNING
    # would gain nothing while risking an unattended first promotion later.
    default_status=DefaultSensorStatus.STOPPED,
    job=gold_pg_sync_job,
    description=(
        "Promotes finalized Iceberg gold batches into PostgreSQL "
        "factors.daily_scores (Meridian#265, docs/18-ICEBERG_MIGRATION.md "
        "section 6, decision ACCEPTED 2026-07-17). Polls the current snapshot "
        "id of meridian.factors.gold_promotion_scores; when it changes vs the "
        "sensor's JSON cursor (last-synced snapshot id), fires ONE RunRequest "
        "for gold_scores_pg_sync passing the detected snapshot_id via run "
        "config, so the sync promotes exactly that snapshot's owned_factors "
        "scope even if a newer batch lands before the run starts.\n\n"
        "run_key = snapshot_id makes re-fires idempotent (an unchanged "
        "snapshot never re-promotes; a redeploy that resets the cursor "
        "dedupes against the already-run run_key at the Dagster level). "
        "Skips clean (no cursor change, no RunRequest) when the gold "
        "promotion table is unpopulated/unreachable - the expected "
        "pre-cutover state, since no recompute writes the gold table yet "
        "(that companion wiring is a follow-on, see the MR). Fail-soft: any "
        "catalog error yields a no-op tick, never a raise.\n\n"
        "This is a ONE-WAY Iceberg-gold -> PG promotion for batch/recompute/"
        "research outputs; PG stays authoritative and the daily LIVE pipeline "
        "still writes PG directly. Both write paths are scoped by "
        "owned_factors so neither clobbers the other's factors. Cursor format "
        '(JSON): {"last_snapshot_id": <int|null>, "queued": <null|{snapshot_id, '
        "first_seen, attempts, last_fired, gave_up}>}. last_snapshot_id advances "
        "only when the gold_scores_pg_sync run for that snapshot SUCCEEDED (runs "
        "are found by tag); a failed run is re-requested under a new run_key "
        "suffix at most hourly and at most 3 times, then logged at ERROR (hourly). "
        "A failed snapshot is never dropped for a newer one: newer snapshots wait behind it."
    ),
)
def iceberg_gold_sync_sensor(context: SensorEvaluationContext):
    """Fire gold_scores_pg_sync when the Iceberg gold promotion table gets a
    new snapshot, and acknowledge it only once the run SUCCEEDED (FND-004).

    State machine (JSON cursor {last_snapshot_id, queued}); last_snapshot_id is
    the last snapshot whose promotion run succeeded, queued is the one snapshot
    in progress ({snapshot_id, first_seen, attempts, last_fired, gave_up}):
      1. Read the current snapshot id of gold_promotion_scores (None if the
         table is unreachable/empty - no-op tick, cursor unchanged).
      2. If a snapshot is queued, inspect its runs (by the snapshot tag):
         a SUCCESS run retires it (last_snapshot_id advances); a run in flight
         means wait; otherwise it failed (or vanished) and, once the retry
         spacing has elapsed, it is re-requested under a fresh run_key suffix
         (run_key dedup is permanent), at most _SENSOR_MAX_ATTEMPTS times, then
         logged at ERROR (re-alerted hourly) and left queued. It is never
         abandoned for a newer snapshot: each snapshot is promoted by its own
         owned_factors scope, so dropping it would silently lose those factors.
      3. With nothing queued, a current snapshot different from
         last_snapshot_id is requested (run_key=str(snapshot_id)) and queued.
         Promotion is serial, so a newer snapshot waits behind a failed one.
    """
    from meridian_pipeline import lakehouse

    cursor = json.loads(context.cursor) if context.cursor else {}
    last_snapshot_id = cursor.get("last_snapshot_id")
    queued = cursor.get("queued") or None
    now = _queue_now()

    current = lakehouse.gold_promotion_current_snapshot_id()
    if current is None:
        # Table unreachable/empty - nothing to promote. Leave the cursor as-is.
        return

    def _save() -> None:
        context.update_cursor(json.dumps({"last_snapshot_id": last_snapshot_id, "queued": queued}))

    if queued:
        qid = int(queued["snapshot_id"])
        try:
            state = _gold_snapshot_run_state(context.instance, qid)
        except Exception as exc:  # noqa: BLE001
            context.log.warning(
                f"iceberg_gold_sync_sensor: could not read runs for snapshot {qid} "
                f"({type(exc).__name__}: {str(exc)[:200]}) - leaving it queued, retrying next tick"
            )
            return
        if state == "success":
            context.log.info(
                f"iceberg_gold_sync_sensor: snapshot {qid} promoted after "
                f"{queued.get('attempts', 0)} attempt(s) - acknowledged"
            )
            last_snapshot_id, queued = qid, None
        elif state == "in_flight":
            # Wait for it, but a run stuck QUEUED/STARTED for hours is invisible to
            # the failure pager: log ERROR + ntfy (hourly gate), never cancel.
            age_h = _queue_pending_hours(
                {"first_seen": queued.get("last_fired") or queued.get("first_seen")}, now
            )
            if age_h > _GOLD_STUCK_HOURS and _queue_alert_due(queued, now):
                msg = (
                    f"iceberg_gold_sync_sensor: gold promotion run for snapshot {qid} has been "
                    f"QUEUED/STARTED for {age_h:.1f}h (threshold {_GOLD_STUCK_HOURS}h). Not "
                    "cancelling it; check the run and the run queue."
                )
                context.log.error(msg)
                _notify_stuck(context, "Meridian: gold promotion run stuck", msg)
                queued["last_alerted"] = now.isoformat()
                _save()
            return
        else:
            # The queued snapshot's run did not succeed (or vanished). It is NEVER
            # dropped for a newer snapshot: read_gold_snapshot reads each snapshot
            # by its OWN scope (owned_factors + date range) and the table is
            # cumulative, so snapshots carry different factor scopes and a newer
            # one does not cover an older one's factors. Promotion is serial: the
            # failed snapshot is retried first and any newer snapshot waits.
            attempt = int(queued.get("attempts", 0))
            held = f"; newer snapshot {current} is held behind it" if current != qid else ""
            if attempt >= _SENSOR_MAX_ATTEMPTS:
                if not queued.get("gave_up") or _queue_alert_due(queued, now):
                    msg = (
                        f"iceberg_gold_sync_sensor: gold promotion of snapshot {qid} has failed "
                        f"{attempt} attempt(s) and is NOT being re-requested (cap "
                        f"{_SENSOR_MAX_ATTEMPTS}){held}. Fix the cause, then launch "
                        "gold_pg_sync_job by hand with config snapshot_id and the run tag "
                        f"{_GOLD_SNAPSHOT_TAG}={qid} (the tag is how a success acknowledges it "
                        "and releases the held snapshots). Re-alerts hourly."
                    )
                    context.log.error(msg)
                    _notify_stuck(context, "Meridian: gold promotion retries exhausted", msg)
                    queued["gave_up"] = True
                    queued["last_alerted"] = now.isoformat()
                    _save()
                return
            if not _queue_due(queued, now):
                return
            context.log.warning(
                f"iceberg_gold_sync_sensor: RETRYING snapshot {qid} (attempt {attempt + 1}){held} "
                "- the previous gold_scores_pg_sync run did not succeed"
            )
            yield _gold_sync_request(qid, f"{qid}_a{int(now.timestamp() // 60)}")
            queued["attempts"] = attempt + 1
            queued["last_fired"] = now.isoformat()
            _save()
            return

    if current == last_snapshot_id:
        if queued is None and cursor.get("queued"):
            _save()  # a queued item was retired above
        return  # no new promotion snapshot

    context.log.info(
        "iceberg_gold_sync_sensor: gold promotion snapshot changed "
        f"{last_snapshot_id} -> {current} - firing gold_scores_pg_sync"
    )
    yield _gold_sync_request(current, str(current))
    queued = {
        "snapshot_id": current,
        "first_seen": now.isoformat(),
        "attempts": 1,
        "last_fired": now.isoformat(),
    }
    _save()


# ─────────────────────────────────────────────
# Meridian #285 Tier 3: backfill-completion sensor
# ─────────────────────────────────────────────

# Max processed backfill ids retained in the cursor. This is the HARD bound
# the #99 post-mortem demands: core_complete_sensor wedged because its cursor
# state grew without a ceiling and its bootstrap was all-or-nothing, which
# silently stopped phase-2 firing for EVERY partition. A processed-id list is
# exactly that shape of state, so it is capped here and paired with a
# monotone timestamp high-water mark: an id that ages out of the list is
# still never re-fired, because its backfill_timestamp is older than the
# high-water mark. Neither mechanism alone is sufficient (the list alone is
# unbounded; the timestamp alone races on equal/near-equal timestamps within
# one tick), so both are kept.
_MAX_PROCESSED_BACKFILL_IDS = 200

# Per-tick cap on backfills pulled from the instance. Terminal backfills are
# a low-rate event (a handful per day at most); this only bounds the blast
# radius of a first tick against an instance with a long backfill history.
_MAX_BACKFILLS_PER_TICK = 50


def _meridian_backfill(backfill) -> bool:
    """True if a backfill's asset selection touches any Meridian factor asset.

    Backfills with no asset_selection (op/job backfills, other tenants' work)
    are not ours to verify.
    """
    selection = getattr(backfill, "asset_selection", None)
    if not selection:
        return False
    meridian_keys = set(ALL_FACTOR_ASSET_KEYS)
    for key in selection:
        try:
            name = key.to_user_string()
        except AttributeError:
            # Defensive: a shape we don't recognise is not evidence of a
            # Meridian backfill, and must not crash the tick.
            continue
        if name in meridian_keys:
            return True
    return False


def _backfill_partition_names(context, backfill) -> list[str] | None:
    """Partition names of a backfill, or None if they can't be determined.

    PartitionBackfill exposes partition names two different ways depending
    on the backfill's kind, and neither is a plain attribute read for the
    asset case:

      - non-asset backfills carry them on the `partition_names` field;
      - ASSET backfills leave that field None and keep the information in
        serialized_asset_backfill_data, which must be deserialised against
        an asset graph (PartitionBackfill.get_partition_names() takes a
        BaseWorkspaceRequestContext, which a sensor tick does not have -
        but get_asset_backfill_data() takes just an asset graph, which
        context.repository_def does expose).

    Any failure here (a definition changed since the backfill ran, a
    missing repository_def in a bare test context, an unexpected shape)
    returns None so the caller can skip and log. It NEVER raises: one
    undeserialisable historical backfill must not wedge every future tick.
    """
    names = getattr(backfill, "partition_names", None)
    if names:
        return list(names)

    try:
        repo_def = context.repository_def
        if repo_def is None:
            return None
        asset_backfill_data = backfill.get_asset_backfill_data(repo_def.asset_graph)
        names = asset_backfill_data.get_partition_names()
    except Exception:  # noqa: BLE001 - deliberately total, see docstring
        return None
    return list(names) if names else None


def _partition_range(partition_names) -> tuple[str, str] | None:
    """(min, max) ISO date over a backfill's partition names, or None.

    Partition names that are not ISO dates are ignored (Meridian's daily
    assets are all DailyPartitionsDefinition-backed, but a multi-dimensional
    or static partition key must not crash the tick). Returns None when
    nothing parseable remains.
    """
    parsed: list[date] = []
    for name in partition_names or []:
        try:
            parsed.append(date.fromisoformat(str(name)))
        except (TypeError, ValueError):
            continue
    if not parsed:
        return None
    return min(parsed).isoformat(), max(parsed).isoformat()


@sensor(
    target=backfill_verification_job,
    minimum_interval_seconds=300,
    default_status=DefaultSensorStatus.STOPPED,
    description=(
        "Tier 3 of Meridian #285: fires backfill_verification_job ONCE per "
        "finished Meridian backfill, so the set-level sweep (partition "
        "completeness, non-trading orphans, thin days, universe drift, "
        "cross-date distribution) runs one time over the whole range in a "
        "single pod instead of a pod per partition.\n\n"
        "Dagster 1.13.4 has no native backfill-completion hook. "
        "run_status_sensor fires per RUN, so a 296-run backfill would fire "
        "296 times; this is therefore a regular @sensor that polls "
        "instance.get_backfills() for TERMINAL backfills each tick. Both "
        "COMPLETED_SUCCESS and COMPLETED_FAILED count as terminal: a "
        "partially failed backfill is precisely when a completeness sweep "
        "matters most, so it must not be skipped.\n\n"
        "Regular @sensor, NOT multi_asset_sensor - the latter caps "
        "unconsumed events at 25 and dead-locks on partial-completion "
        "patterns (docs/06-CRITICAL_RULES rule 18).\n\n"
        "Cursor format (JSON): {processed_ids: [backfill_id, ...], "
        "high_water_ts: <float|null>}. processed_ids is capped at "
        "_MAX_PROCESSED_BACKFILL_IDS (200) and high_water_ts is a monotone "
        "backfill_timestamp watermark, so an id evicted from the list is "
        "still never re-fired. Bounded state is deliberate: unbounded / "
        "all-or-nothing cursor state is what wedged core_complete_sensor in "
        "Meridian #99 and silently stopped phase-2 firing for every "
        "partition.\n\n"
        "First tick SEEDS the watermark to the newest terminal backfill "
        "without firing, so enabling the sensor on an instance with a long "
        "backfill history does not stampede one verification run per "
        "historical backfill (the same bootstrap trap as #99). "
        "run_key=verify-<backfill_id> makes a re-fire for the same backfill "
        "impossible even if the cursor is reset. Findings are ADVISORY: the "
        "job logs them at WARN and always succeeds."
    ),
)
def backfill_completion_sensor(context: SensorEvaluationContext):
    """Fire one verification run per completed Meridian backfill.

    Per tick:
      1. Pull terminal backfills (COMPLETED_SUCCESS + COMPLETED_FAILED).
      2. On the very first tick, seed high_water_ts to the newest of them
         and fire nothing.
      3. Otherwise skip any backfill already in processed_ids, older than
         high_water_ts, or not touching a Meridian factor asset.
      4. Derive [start, end] from min/max of the backfill's partition names,
         skipping (with a log, never a raise) any backfill whose partition
         names are absent or unparseable.
      5. Yield RunRequest(run_key=f"verify-{backfill_id}") carrying the range
         in run config, and record the id + advance the watermark.
    """
    cursor = json.loads(context.cursor) if context.cursor else {}
    processed_ids: list[str] = list(cursor.get("processed_ids", []))
    high_water_ts = cursor.get("high_water_ts")
    bootstrapped = bool(cursor)

    try:
        backfills = context.instance.get_backfills(
            filters=BulkActionsFilter(
                statuses=[
                    BulkActionStatus.COMPLETED_SUCCESS,
                    BulkActionStatus.COMPLETED_FAILED,
                ]
            ),
            limit=_MAX_BACKFILLS_PER_TICK,
        )
    except TypeError:
        # Older/newer instance signatures that don't accept `filters` - fall
        # back to the status kwarg and filter in Python. Never let an API
        # shape difference turn into a crashed sensor.
        backfills = context.instance.get_backfills(limit=_MAX_BACKFILLS_PER_TICK)
        backfills = [
            b
            for b in backfills
            if getattr(b, "status", None)
            in (BulkActionStatus.COMPLETED_SUCCESS, BulkActionStatus.COMPLETED_FAILED)
        ]

    timestamps = [
        ts for ts in (getattr(b, "backfill_timestamp", None) for b in backfills) if ts is not None
    ]

    if not bootstrapped:
        # First tick ever: adopt the current newest terminal backfill as the
        # watermark WITHOUT firing. Without this, enabling the sensor emits
        # one verification run per historical backfill on the instance.
        #
        # The ids seen at seed time are recorded too, not just the timestamp:
        # the watermark comparison below is deliberately STRICT (`ts <`) so
        # that backfills sharing a timestamp with the newest processed one are
        # not silently dropped, which means the newest backfill at seed time
        # would otherwise sail through the watermark on the very next tick and
        # fire for work that finished before the sensor was ever enabled. The
        # seed list is inherently bounded by _MAX_BACKFILLS_PER_TICK and is
        # truncated to the same hard cap as the steady-state list.
        seed_ts = max(timestamps) if timestamps else None
        seed_ids = [
            bid for bid in (getattr(b, "backfill_id", None) for b in backfills) if bid is not None
        ][-_MAX_PROCESSED_BACKFILL_IDS:]
        context.update_cursor(json.dumps({"processed_ids": seed_ids, "high_water_ts": seed_ts}))
        context.log.info(
            f"backfill_completion_sensor: seeded high_water_ts to {seed_ts} and "
            f"{len(seed_ids)} id(s) from {len(backfills)} terminal backfill(s) "
            f"without firing (first tick)"
        )
        return

    processed_set = set(processed_ids)
    fired = 0
    skipped_unparseable = 0

    # Oldest-first so the watermark advances monotonically and a mid-tick
    # interruption can only ever lose work forward, never skip a backfill.
    for backfill in sorted(backfills, key=lambda b: getattr(b, "backfill_timestamp", 0) or 0):
        backfill_id = getattr(backfill, "backfill_id", None)
        if not backfill_id or backfill_id in processed_set:
            continue
        ts = getattr(backfill, "backfill_timestamp", None)
        if high_water_ts is not None and ts is not None and ts < high_water_ts:
            # Older than everything we have processed AND aged out of the
            # bounded id list. Skipping is the correct read of that state.
            continue
        if not _meridian_backfill(backfill):
            continue

        names = _backfill_partition_names(context, backfill)
        rng = _partition_range(names)
        if rng is None:
            # Zero, missing or unparseable partition names. Record the id so
            # we don't re-inspect it every tick, but fire nothing.
            skipped_unparseable += 1
            context.log.info(
                f"backfill_completion_sensor: backfill {backfill_id} has no parseable "
                f"partition names - skipping verification (nothing to bound a range with)"
            )
            processed_ids.append(backfill_id)
            processed_set.add(backfill_id)
            if ts is not None:
                high_water_ts = ts if high_water_ts is None else max(high_water_ts, ts)
            continue

        start, end = rng
        context.log.info(
            f"backfill_completion_sensor: backfill {backfill_id} "
            f"({getattr(backfill, 'status', None)}) finished over {start} .. {end} - "
            f"firing backfill_verification_job"
        )
        yield RunRequest(
            run_key=f"verify-{backfill_id}",
            run_config={
                "ops": {"verify_range_op": {"config": {"start_date": start, "end_date": end}}}
            },
            tags={"dagster/priority": "5", "priority_class": "low"},
        )
        fired += 1
        processed_ids.append(backfill_id)
        processed_set.add(backfill_id)
        if ts is not None:
            high_water_ts = ts if high_water_ts is None else max(high_water_ts, ts)

    # HARD BOUND on cursor growth (see _MAX_PROCESSED_BACKFILL_IDS). The list
    # is append-ordered oldest-first, so truncating from the front retains
    # the most recent ids - exactly the ones whose timestamps sit at or above
    # the watermark and therefore still need explicit de-duplication.
    if len(processed_ids) > _MAX_PROCESSED_BACKFILL_IDS:
        processed_ids = processed_ids[-_MAX_PROCESSED_BACKFILL_IDS:]

    context.update_cursor(
        json.dumps({"processed_ids": processed_ids, "high_water_ts": high_water_ts})
    )

    if fired or skipped_unparseable:
        context.log.info(
            f"backfill_completion_sensor: fired {fired} verification run(s), "
            f"skipped {skipped_unparseable} unparseable backfill(s), "
            f"{len(processed_ids)} id(s) tracked"
        )
