"""
Tests for db.assert_cross_section_single_dated / cross_section_completeness_ok
(read-during-write hardening: LDP's nightly ohlcv.prices ingest window).

Background: LDP's nightly OHLCV ingest window rewrites/repairs `ohlcv.prices`
for a period each night. A cross-section read taken mid-ingest sees a
partially-rewritten table - a mid-ingest latest-per-ticker cross-section can
return only a fraction of tickers as-of the new date, the rest one trading
day stale, indistinguishable from "did not trade". The nightly
partitioned scoring path is NOT exposed (every lookup is as-of bounded and
scores T-1), but nothing previously asserted that a scored cross-section was
actually single-dated - this guard adds that assertion.

DB-free for the pure decision function (`cross_section_completeness_ok`);
`assert_cross_section_single_dated` is exercised against a tiny fake engine
(no PostgreSQL touched) so the SQL wiring + raise-message plumbing are
covered too.
"""

import contextlib
import os
import sys
from datetime import date
from pathlib import Path

import pytest

_DAGSTER_DIR = Path(__file__).resolve().parents[1] / "dagster"
if str(_DAGSTER_DIR) not in sys.path:
    sys.path.insert(0, str(_DAGSTER_DIR))

os.environ.setdefault("PGPASS", "test-pg-pass")
os.environ.setdefault("PARTITION_START_DATE", "2010-01-02")

from meridian_pipeline import db  # noqa: E402

# ---------------------------------------------------------------------------
# cross_section_completeness_ok - pure decision function
# ---------------------------------------------------------------------------


def test_mid_ingest_incident_fires():
    """A mid-ingest reading (a collapsed fraction vs the trailing norm)
    must trip the guard - this is the failure mode the guard exists to catch."""
    trailing = [0.998, 0.997, 0.999, 0.995, 0.998, 0.996, 0.999, 0.997, 0.998, 0.996]
    ok, msg = db.cross_section_completeness_ok(0.45, trailing)
    assert ok is False
    assert "below the trailing" in msg


def test_ordinary_universe_breathing_does_not_fire():
    """Ordinary weekday-to-weekday coverage churn (new listings not yet
    backfilled, thin small-caps) is not a defect. A ~3% relative dip must NOT
    trip a check meant to catch a ~50% collapse - a check that fires on normal
    behaviour gets muted, and a muted check is how a defect goes unseen for
    months."""
    trailing = [0.9995, 0.999, 0.998, 0.997, 0.998, 0.996, 0.995, 0.997, 0.996, 0.9995]
    ok, msg = db.cross_section_completeness_ok(0.97, trailing)
    assert ok is True
    assert msg == ""


def test_no_trailing_history_does_not_fail():
    """Fewer than the minimum trailing sessions (near the partition floor, or
    just after a fresh PARTITION_START_DATE) must pass silently - absence of
    history is not evidence of a defect."""
    ok, msg = db.cross_section_completeness_ok(0.10, [0.99, 0.98])
    assert ok is True
    assert "insufficient" in msg


def test_empty_trailing_history_does_not_fail():
    ok, msg = db.cross_section_completeness_ok(0.10, [])
    assert ok is True


def test_degenerate_zero_median_does_not_fail():
    """A trailing median of exactly zero has no meaningful baseline to divide
    against - must not raise a ZeroDivisionError or manufacture a false
    alarm."""
    trailing = [0.0] * db._SINGLE_DATED_MIN_TRAILING_DAYS
    ok, msg = db.cross_section_completeness_ok(0.5, trailing)
    assert ok is True
    assert "zero" in msg


def test_current_at_or_above_median_passes():
    trailing = [0.95] * db._SINGLE_DATED_MIN_TRAILING_DAYS
    ok, msg = db.cross_section_completeness_ok(0.99, trailing)
    assert ok is True


def test_threshold_boundary_just_inside_passes():
    """Exactly at the relative-drop threshold must still pass (the guard
    fires only when the drop EXCEEDS the threshold)."""
    median = 1.0
    trailing = [median] * db._SINGLE_DATED_MIN_TRAILING_DAYS
    current = median * (1 - db._SINGLE_DATED_MAX_RELATIVE_DROP_PCT / 100.0)
    ok, _ = db.cross_section_completeness_ok(current, trailing)
    assert ok is True


def test_threshold_boundary_just_outside_fails():
    median = 1.0
    trailing = [median] * db._SINGLE_DATED_MIN_TRAILING_DAYS
    current = median * (1 - (db._SINGLE_DATED_MAX_RELATIVE_DROP_PCT + 0.5) / 100.0)
    ok, _ = db.cross_section_completeness_ok(current, trailing)
    assert ok is False


# ---------------------------------------------------------------------------
# assert_cross_section_single_dated - fake-engine integration
# ---------------------------------------------------------------------------


class _FakeResult:
    def __init__(self, fetchall_result=None, fetchone_result=None):
        self._fetchall = fetchall_result if fetchall_result is not None else []
        self._fetchone = fetchone_result

    def fetchall(self):
        return self._fetchall

    def fetchone(self):
        return self._fetchone


class _FakeConn:
    """Routes the three queries assert_cross_section_single_dated can issue:
    the watermark point lookup ('ingest_watermark' in its text, always tried
    first), the trailing GROUP BY scan ('GROUP BY 1' in its text), and the
    single-day present-count scan (neither). `watermark_row` defaults to
    None (no row for this date - the watermark is "unknown" and every test
    that doesn't care about it falls straight through to the pre-watermark
    statistical behaviour, unchanged)."""

    def __init__(self, trailing_rows, present_count, watermark_row=None):
        self.trailing_rows = trailing_rows
        self.present_count = present_count
        self.watermark_row = watermark_row
        self.n_calls = 0

    def execute(self, stmt, params=None):
        self.n_calls += 1
        sql = str(stmt)
        if "ingest_watermark" in sql:
            return _FakeResult(fetchone_result=self.watermark_row)
        if "GROUP BY 1" in sql:
            return _FakeResult(fetchall_result=self.trailing_rows)
        return _FakeResult(fetchone_result=(self.present_count,))


class _FakeEngine:
    def __init__(self, trailing_rows, present_count, watermark_row=None):
        self.conn = _FakeConn(trailing_rows, present_count, watermark_row=watermark_row)

    def connect(self):
        return contextlib.nullcontext(self.conn)


class _ExplodingEngine:
    """An engine that raises if touched at all - proves a guard path is a
    true no-op (no query issued), as opposed to a query whose result is ignored."""

    def connect(self):
        raise AssertionError("engine.connect() should not have been called")


class _NoWatermarkExplodingConn:
    """Answers the watermark lookup with 'no row for this date' (so
    _watermark_status_for_date returns None, falling through to the
    statistical heuristic below it), then raises if ANY other query is
    attempted - proves the small-universe short-circuit still skips every
    statistical query, same guarantee _ExplodingEngine gave pre-watermark."""

    def execute(self, stmt, params=None):
        sql = str(stmt)
        if "ingest_watermark" in sql:
            return _FakeResult(fetchone_result=None)
        raise AssertionError(f"unexpected statistical query issued: {sql}")


class _NoWatermarkExplodingEngine:
    def connect(self):
        return contextlib.nullcontext(_NoWatermarkExplodingConn())


def _trailing_healthy(score_date: date, n_days: int = 10, n_universe: int = 4000):
    """n_days consecutive prior trading sessions all near-full coverage."""
    rows = []
    d = score_date
    for i in range(n_days):
        # walk back one calendar day per row; exact spacing doesn't matter,
        # only order (most-recent-first) and count.
        from datetime import timedelta

        d = d - timedelta(days=1)
        rows.append((d, n_universe - 5 - i))
    return rows


def test_small_universe_is_a_silent_noop():
    """Below _SINGLE_DATED_MIN_UNIVERSE the ratio is too noisy to mean
    anything - the statistical fallback must never be reached. The watermark
    lookup itself still runs (it is cheap and universe-size-independent);
    here it has no row for this date, so it answers "unknown" and the
    function returns without issuing any statistical query."""
    tickers = [f"T{i}" for i in range(db._SINGLE_DATED_MIN_UNIVERSE - 1)]
    db.assert_cross_section_single_dated(_NoWatermarkExplodingEngine(), date(2026, 8, 14), tickers)


def test_insufficient_trailing_history_does_not_raise():
    tickers = [f"T{i}" for i in range(200)]
    engine = _FakeEngine(trailing_rows=[(date(2026, 8, 13), 190)], present_count=10)
    db.assert_cross_section_single_dated(engine, date(2026, 8, 14), tickers)
    # present-count query must not even have been reached when trailing
    # history is too thin to judge. n_calls == 2: the watermark lookup
    # (unknown - no row) plus the trailing scan.
    assert engine.conn.n_calls == 2


def test_healthy_cross_section_does_not_raise():
    n = 4000
    tickers = [f"T{i}" for i in range(n)]
    score_date = date(2026, 8, 10)
    engine = _FakeEngine(
        trailing_rows=_trailing_healthy(score_date, n_universe=n), present_count=n - 6
    )
    db.assert_cross_section_single_dated(engine, score_date, tickers)


def test_mixed_date_cross_section_raises_naming_both_dates_and_counts():
    """The core regression: a mid-ingest partial read must raise loudly,
    naming both competing as-of dates and their counts."""
    n = 4000
    tickers = [f"T{i}" for i in range(n)]
    score_date = date(2026, 8, 14)
    prior_date = date(2026, 8, 13)
    trailing_rows = [(prior_date, 3950)] + _trailing_healthy(prior_date, n_days=9, n_universe=n)
    engine = _FakeEngine(trailing_rows=trailing_rows, present_count=2000)  # ~50%

    with pytest.raises(ValueError) as exc_info:
        db.assert_cross_section_single_dated(engine, score_date, tickers)

    msg = str(exc_info.value)
    assert str(score_date) in msg
    assert str(prior_date) in msg
    assert "2000" in msg
    assert "3950" in msg


# ---------------------------------------------------------------------------
# _watermark_status_for_date / watermark-first cutover (2026-08-15)
# ---------------------------------------------------------------------------


def test_watermark_status_true_for_complete_row():
    engine = _FakeEngine(trailing_rows=[], present_count=0, watermark_row=(True,))
    assert db._watermark_status_for_date(engine, date(2026, 8, 14)) is True


def test_watermark_status_false_for_incomplete_row():
    engine = _FakeEngine(trailing_rows=[], present_count=0, watermark_row=(False,))
    assert db._watermark_status_for_date(engine, date(2026, 8, 14)) is False


def test_watermark_status_none_when_no_row():
    engine = _FakeEngine(trailing_rows=[], present_count=0, watermark_row=None)
    assert db._watermark_status_for_date(engine, date(2026, 8, 14)) is None


def test_watermark_status_none_when_engine_unreachable():
    assert db._watermark_status_for_date(_ExplodingEngine(), date(2026, 8, 14)) is None


def test_complete_watermark_short_circuits_before_any_statistical_query():
    """A complete watermark must return immediately - LDP's own gate is
    stronger than our inference, so the statistical queries never run, even
    for a universe that would otherwise trip the coverage-ratio check."""

    class _CompleteOnlyConn:
        def execute(self, stmt, params=None):
            sql = str(stmt)
            if "ingest_watermark" in sql:
                return _FakeResult(fetchone_result=(True,))
            raise AssertionError(f"unexpected statistical query issued: {sql}")

    class _CompleteOnlyEngine:
        def connect(self):
            return contextlib.nullcontext(_CompleteOnlyConn())

    n = 4000
    tickers = [f"T{i}" for i in range(n)]
    db.assert_cross_section_single_dated(_CompleteOnlyEngine(), date(2026, 8, 14), tickers)


def test_incomplete_watermark_raises_before_any_statistical_query():
    """An incomplete watermark must raise immediately, before touching the
    (universe-size-gated, potentially no-op) statistical fallback - even for
    a tiny universe that the size gate alone would have silently passed."""

    class _IncompleteOnlyConn:
        def execute(self, stmt, params=None):
            sql = str(stmt)
            if "ingest_watermark" in sql:
                return _FakeResult(fetchone_result=(False,))
            raise AssertionError(f"unexpected statistical query issued: {sql}")

    class _IncompleteOnlyEngine:
        def connect(self):
            return contextlib.nullcontext(_IncompleteOnlyConn())

    tickers = [f"T{i}" for i in range(5)]  # far below _SINGLE_DATED_MIN_UNIVERSE
    with pytest.raises(ValueError) as exc_info:
        db.assert_cross_section_single_dated(_IncompleteOnlyEngine(), date(2026, 8, 14), tickers)
    assert "ingest watermark" in str(exc_info.value)
    assert "NOT complete" in str(exc_info.value)


def test_unknown_watermark_falls_back_to_healthy_statistical_pass():
    """No watermark row for this date (unknown, not incomplete) must fall
    through to the pre-watermark statistical heuristic unchanged."""
    n = 4000
    tickers = [f"T{i}" for i in range(n)]
    score_date = date(2026, 8, 10)
    engine = _FakeEngine(
        trailing_rows=_trailing_healthy(score_date, n_universe=n),
        present_count=n - 6,
        watermark_row=None,
    )
    db.assert_cross_section_single_dated(engine, score_date, tickers)


# ---------------------------------------------------------------------------
# latest_complete_trade_date / get_new_complete_trade_dates
# (Meridian #286: watermark-driven scoring trigger, sensors.watermark_complete_sensor)
# ---------------------------------------------------------------------------


class _LatestDateConn:
    def __init__(self, result):
        self._result = result

    def execute(self, stmt, params=None):
        sql = str(stmt)
        assert "max(trade_date)" in sql
        assert "is_complete" in sql
        return _FakeResult(fetchone_result=self._result)


class _LatestDateEngine:
    def __init__(self, result):
        self._result = result

    def connect(self):
        return contextlib.nullcontext(_LatestDateConn(self._result))


def test_latest_complete_trade_date_returns_max():
    engine = _LatestDateEngine((date(2026, 8, 17),))
    assert db.latest_complete_trade_date(engine) == date(2026, 8, 17)


def test_latest_complete_trade_date_none_when_no_complete_rows():
    """Empty watermark (no rows at all, or none marked complete yet) must
    answer None, not raise - a table that only started populating this week
    is a legitimate, expected state, not a failure."""
    engine = _LatestDateEngine((None,))
    assert db.latest_complete_trade_date(engine) is None


def test_latest_complete_trade_date_propagates_connection_errors():
    """Unlike the empty-table case, a genuinely unreachable engine must
    raise so the caller (the sensor) can tell 'checked, nothing complete'
    apart from 'could not check' and log/handle them differently."""
    with pytest.raises(AssertionError):
        db.latest_complete_trade_date(_ExplodingEngine())


class _NewDatesConn:
    def __init__(self, rows):
        self._rows = rows
        self.last_params = None

    def execute(self, stmt, params=None):
        sql = str(stmt)
        assert "is_complete" in sql
        assert "trade_date > :since" in sql
        self.last_params = params
        return _FakeResult(fetchall_result=self._rows)


class _NewDatesEngine:
    def __init__(self, rows):
        self.conn = _NewDatesConn(rows)

    def connect(self):
        return contextlib.nullcontext(self.conn)


def test_get_new_complete_trade_dates_returns_rows_ascending():
    rows = [(date(2026, 8, 15),), (date(2026, 8, 16),), (date(2026, 8, 17),)]
    engine = _NewDatesEngine(rows)
    got = db.get_new_complete_trade_dates(engine, since=date(2026, 8, 14), limit=30)
    assert got == [date(2026, 8, 15), date(2026, 8, 16), date(2026, 8, 17)]
    assert engine.conn.last_params == {"since": date(2026, 8, 14), "limit": 30}


def test_get_new_complete_trade_dates_empty_is_not_an_error():
    engine = _NewDatesEngine([])
    assert db.get_new_complete_trade_dates(engine, since=date(2026, 8, 14)) == []


def test_get_new_complete_trade_dates_propagates_connection_errors():
    with pytest.raises(AssertionError):
        db.get_new_complete_trade_dates(_ExplodingEngine(), since=date(2026, 8, 14))


# ---------------------------------------------------------------------------
# trigger_cross_section_ready - sensor-side readiness predicate (Meridian
# #395: stale_factor_sensor / ldp_ohlcv_upstream_sensor firing into the guard
# above because they treated "no watermark row yet" as "proceed" during LDP's
# nightly ingest window, when a row for that date simply doesn't exist until
# ingest finishes.)
# ---------------------------------------------------------------------------


def test_trigger_ready_true_for_healthy_cross_section():
    """A trailing-healthy cross-section with near-full current coverage must
    read as ready - the acceptance criterion's "True for a complete one" half."""
    score_date = date(2026, 9, 11)
    engine = _FakeEngine(
        trailing_rows=_trailing_healthy(score_date, n_universe=2000),
        present_count=1980,
        watermark_row=None,
    )
    ready, reason = db.trigger_cross_section_ready(engine, score_date)
    assert ready is True
    assert reason == ""


def test_trigger_ready_false_for_mid_ingest_cross_section_shape():
    """The Meridian#395 acceptance criterion: a cross-section shaped like the
    mid-ingest partial write - 1000/2000 tickers (50.0%) on date D vs
    1980/2000 (99.0%) on D-1 - must read as NOT ready. The numbers are
    illustrative; the shape (a sharp drop in coverage on the newest date)
    is what matters. It is exercised
    against the predicate the SENSORS consult before launching a run, not
    just the guard the asset hits after one has already started."""
    score_date = date(2026, 9, 11)
    prior_date = date(2026, 9, 10)
    trailing_rows = [(prior_date, 1980)] + _trailing_healthy(prior_date, n_days=9, n_universe=2000)
    engine = _FakeEngine(trailing_rows=trailing_rows, present_count=1000, watermark_row=None)

    ready, reason = db.trigger_cross_section_ready(engine, score_date)

    assert ready is False
    assert "1000" in reason
    assert "1980" in reason
    assert str(score_date) in reason


def test_trigger_ready_short_circuits_on_complete_watermark():
    """A complete watermark must return ready immediately, without touching
    the statistical fallback - same short-circuit guarantee cross_section_ready
    gives the write-side guard."""

    class _CompleteOnlyConn:
        def execute(self, stmt, params=None):
            sql = str(stmt)
            if "ingest_watermark" in sql:
                return _FakeResult(fetchone_result=(True,))
            raise AssertionError(f"unexpected statistical query issued: {sql}")

    class _CompleteOnlyEngine:
        def connect(self):
            return contextlib.nullcontext(_CompleteOnlyConn())

    ready, reason = db.trigger_cross_section_ready(_CompleteOnlyEngine(), date(2026, 9, 11))
    assert ready is True
    assert reason == ""


def test_trigger_ready_defers_on_incomplete_watermark_without_statistical_query():
    """An explicit NOT-complete watermark answer must defer immediately,
    before touching the statistical fallback."""

    class _IncompleteOnlyConn:
        def execute(self, stmt, params=None):
            sql = str(stmt)
            if "ingest_watermark" in sql:
                return _FakeResult(fetchone_result=(False,))
            raise AssertionError(f"unexpected statistical query issued: {sql}")

    class _IncompleteOnlyEngine:
        def connect(self):
            return contextlib.nullcontext(_IncompleteOnlyConn())

    ready, reason = db.trigger_cross_section_ready(_IncompleteOnlyEngine(), date(2026, 9, 11))
    assert ready is False
    assert "NOT complete" in reason


def test_trigger_ready_true_when_insufficient_trailing_history():
    """Near the partition floor (or a fresh deploy) there may be too little
    trailing history to judge - must pass silently, same posture
    cross_section_ready takes."""
    engine = _FakeEngine(
        trailing_rows=[(date(2026, 9, 10), 1980)], present_count=10, watermark_row=None
    )
    ready, reason = db.trigger_cross_section_ready(engine, date(2026, 9, 11))
    assert ready is True


def test_unknown_watermark_falls_back_to_statistical_raise():
    """No watermark row for this date must still fall through to (and
    surface) the statistical heuristic's own raise - the same regression
    test_mixed_date_cross_section_raises_naming_both_dates_and_counts covers,
    now explicit about the watermark being unknown rather than absent by
    construction."""
    n = 4000
    tickers = [f"T{i}" for i in range(n)]
    score_date = date(2026, 8, 14)
    prior_date = date(2026, 8, 13)
    trailing_rows = [(prior_date, 3950)] + _trailing_healthy(prior_date, n_days=9, n_universe=n)
    engine = _FakeEngine(
        trailing_rows=trailing_rows, present_count=2000, watermark_row=None
    )  # ~50%

    with pytest.raises(ValueError) as exc_info:
        db.assert_cross_section_single_dated(engine, score_date, tickers)
    assert str(score_date) in str(exc_info.value)


# ---------------------------------------------------------------------------
# Meridian#413 - trigger-side fallback must require ret_1d as well as bars
# ---------------------------------------------------------------------------
#
# Background: #395's fallback (_cross_section_present_count_unscoped /
# _cross_section_trailing_coverage_unscoped) counted any bar dated the score
# date. A bar lands the moment LDP's price ingest writes the row; ret_1d is
# computed by a separate, later pass (ohlcv_returns) that can still be
# running once every bar for the date has landed. A run can then
# fire when bars are fully present (single-dated) but ret_1d is still NULL
# for a tail of tickers - the scoring assets
# then failed with StaleWindowError. The fallback
# must count non-NULL ret_1d, not bar presence.


class _SqlCapturingConn:
    """Records every SQL string passed to execute() (plus params) and answers
    with a fixed result - used to assert directly on the QUERY TEXT the
    unscoped/scoped helper functions build, independent of the higher-level
    predicates that call them."""

    def __init__(self, fetchone_result=None, fetchall_result=None):
        self.queries = []
        self._fetchone = fetchone_result
        self._fetchall = fetchall_result if fetchall_result is not None else []

    def execute(self, stmt, params=None):
        self.queries.append(str(stmt))
        return _FakeResult(fetchall_result=self._fetchall, fetchone_result=self._fetchone)


class _SqlCapturingEngine:
    def __init__(self, conn):
        self._conn = conn

    def connect(self):
        return contextlib.nullcontext(self._conn)


def test_present_count_unscoped_requires_non_null_ret1d():
    """Meridian#413 regression: fails on unmodified main, which built this
    query with no ret_1d condition at all."""
    conn = _SqlCapturingConn(fetchone_result=(10,))
    engine = _SqlCapturingEngine(conn)
    db._cross_section_present_count_unscoped(engine, date(2026, 9, 18))
    assert len(conn.queries) == 1
    assert "ret_1d IS NOT NULL" in conn.queries[0]


def test_trailing_coverage_unscoped_requires_non_null_ret1d():
    """The trailing-baseline query must count the SAME population as the
    current-date query above - a mismatch would silently skew the ratio
    cross_section_completeness_ok compares."""
    conn = _SqlCapturingConn(fetchall_result=[(date(2026, 9, 17), 10)])
    engine = _SqlCapturingEngine(conn)
    db._cross_section_trailing_coverage_unscoped(engine, date(2026, 9, 18), 30, 10)
    assert len(conn.queries) == 1
    assert "ret_1d IS NOT NULL" in conn.queries[0]


def test_write_side_scoped_queries_unchanged_by_413():
    """The write-side guard's population is NOT changed by
    #413 - see trigger_cross_section_ready's and cross_section_ready's
    docstrings for why the two sides diverge. This pins that boundary: a
    regression here would mean #413 accidentally tightened the write-side
    guard too, which would wrongly block assets that never touch ret_1d."""
    conn = _SqlCapturingConn(fetchone_result=(10,))
    engine = _SqlCapturingEngine(conn)
    db._cross_section_present_count(engine, ["AAA"], date(2026, 9, 18))
    assert "ret_1d" not in conn.queries[0]

    conn2 = _SqlCapturingConn(fetchall_result=[(date(2026, 9, 17), 10)])
    engine2 = _SqlCapturingEngine(conn2)
    db._cross_section_trailing_coverage(engine2, ["AAA"], date(2026, 9, 18), 30, 10)
    assert "ret_1d" not in conn2.queries[0]


class _Ret1dAwareConn:
    """Simulates ohlcv.prices as a source that answers each query with a
    DIFFERENT count depending on whether the query text actually requires
    `ret_1d IS NOT NULL` - proving the FIX's SQL is what changes trigger
    behaviour, rather than the fake's own plumbing. `bar_*` models the
    ret_1d-unaware (pre-#413) query; `ret1d_*` models the ret_1d-filtered
    (post-#413) query. Routes exactly like _FakeConn otherwise (watermark
    lookup by 'ingest_watermark' in the text, trailing scan by 'GROUP BY 1',
    else the present-count query)."""

    def __init__(
        self, bar_trailing, bar_present, ret1d_trailing, ret1d_present, watermark_row=None
    ):
        self.bar_trailing = bar_trailing
        self.bar_present = bar_present
        self.ret1d_trailing = ret1d_trailing
        self.ret1d_present = ret1d_present
        self.watermark_row = watermark_row

    def execute(self, stmt, params=None):
        sql = str(stmt)
        wants_ret1d = "ret_1d IS NOT NULL" in sql
        if "ingest_watermark" in sql:
            return _FakeResult(fetchone_result=self.watermark_row)
        if "GROUP BY 1" in sql:
            return _FakeResult(
                fetchall_result=self.ret1d_trailing if wants_ret1d else self.bar_trailing
            )
        return _FakeResult(
            fetchone_result=(self.ret1d_present if wants_ret1d else self.bar_present,)
        )


class _Ret1dAwareEngine:
    def __init__(self, **kwargs):
        self.conn = _Ret1dAwareConn(**kwargs)

    def connect(self):
        return contextlib.nullcontext(self.conn)


def test_trigger_ready_false_when_bars_full_but_ret1d_lags_main_fraction():
    """The mid-rewrite shape: bars present for the FULL cross-section
    (100%, single-dated - the old fallback would call this ready) but
    ret_1d is NULL for a material fraction (a large relative drop in
    ret_1d coverage against the trailing norm, well past the threshold).
    Must read as NOT ready.

    Fails on unmodified main: _cross_section_present_count_unscoped /
    _cross_section_trailing_coverage_unscoped counted bars only, so this
    fake's all-bars population (healthy, 100%) is what main's queries
    would resolve to (they never ask for ret_1d IS NOT NULL, so
    _Ret1dAwareConn routes them to bar_trailing/bar_present) and main
    reads this as ready=True - reproducing the failure.
    """
    score_date = date(2026, 9, 18)
    n = 2000
    engine = _Ret1dAwareEngine(
        bar_trailing=_trailing_healthy(score_date, n_universe=n),
        bar_present=n,  # every ticker has a bar dated today
        ret1d_trailing=_trailing_healthy(score_date, n_universe=n),
        ret1d_present=round(n * 0.70),  # only 70% have ret_1d yet
        watermark_row=None,
    )

    ready, reason = db.trigger_cross_section_ready(engine, score_date)

    assert ready is False
    assert "ret_1d" in reason


def test_trigger_ready_true_when_ret1d_keeps_pace_with_bars():
    """The ordinary-night shape: both bars and ret_1d are present for
    (near-)the full cross-section. Must read as ready - the fix must not
    overtighten a healthy evening into a false defer."""
    score_date = date(2026, 9, 17)
    n = 2000
    engine = _Ret1dAwareEngine(
        bar_trailing=_trailing_healthy(score_date, n_universe=n),
        bar_present=n,
        ret1d_trailing=_trailing_healthy(score_date, n_universe=n),
        ret1d_present=n - 5,  # a few bars still NULL, well within tolerance
        watermark_row=None,
    )

    ready, reason = db.trigger_cross_section_ready(engine, score_date)

    assert ready is True
    assert reason == ""
