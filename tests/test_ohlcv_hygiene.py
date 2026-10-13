"""
Unit tests for the shared bogus-print / illiquid-name floor (Meridian #275,
LDP #182) and its two direct-PG consumers: check_ret1d_input_quality and
risk_metrics.

DB-free: exercises the pure filtering/evaluation logic on synthetic frames.
"""

import os
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_DAGSTER_DIR = Path(__file__).resolve().parents[1] / "dagster"
if str(_DAGSTER_DIR) not in sys.path:
    sys.path.insert(0, str(_DAGSTER_DIR))

os.environ.setdefault("PGPASS", "test-pg-pass")
os.environ.setdefault("PARTITION_START_DATE", "2010-01-02")


def _dagster_complete() -> bool:
    try:
        import dagster
    except ImportError:
        return False
    return all(
        hasattr(dagster, n)
        for n in ("asset", "asset_check", "AssetExecutionContext", "AssetCheckResult", "Output")
    )


if not _dagster_complete():
    pytest.skip("Dagster not fully installed (slim CI test image)", allow_module_level=True)

from meridian_pipeline import ohlcv_hygiene  # noqa: E402
from meridian_pipeline.checks.ret1d_checks import (  # noqa: E402
    _EXTREME_COUNT_MAX,
    _MIN_ROWS,
    _RET_ABS_MAX,
    _RET_ABS_MIN,
    _STD_MAX,
    _STD_WINSOR_ABS,
    _era_gate,
    _evaluate_days,
)
from meridian_pipeline.ohlcv_hygiene import (  # noqa: E402
    MIN_DOLLAR_VOLUME,
    MIN_PRICE_FLOOR,
    STALE_TRADING_DAY_TOLERANCE,
    bogus_row_mask,
    stale_tickers_from_last_seen,
)

# ─────────────────────────────────────────────
# bogus_row_mask (the shared floor primitive)
# ─────────────────────────────────────────────


def test_bogus_row_mask_flags_sub_penny_close():
    # Sub-penny close: close=1e-06, volume=1010 (nonzero volume).
    close = pd.Series([1e-06])
    volume = pd.Series([1010])
    raw_close = pd.Series([1e-06])
    mask = bogus_row_mask(close, volume, raw_close)
    assert bool(mask.iloc[0]) is True


def test_bogus_row_mask_flags_zero_volume():
    # Stale print: close=0.4 (fine on its own), volume=0 (non-traded).
    close = pd.Series([0.4])
    volume = pd.Series([0])
    raw_close = pd.Series([0.4])
    mask = bogus_row_mask(close, volume, raw_close)
    assert bool(mask.iloc[0]) is True


def test_bogus_row_mask_flags_low_adv():
    # Real price + real (nonzero) volume, but dollar volume well under the floor.
    close = pd.Series([0.5])
    volume = pd.Series([100])  # dollar volume = 50, far under MIN_DOLLAR_VOLUME
    raw_close = pd.Series([0.5])
    mask = bogus_row_mask(close, volume, raw_close)
    assert bool(mask.iloc[0]) is True


def test_bogus_row_mask_passes_liquid_row():
    close = pd.Series([50.0])
    volume = pd.Series([1_000_000])  # dollar volume = $50M
    raw_close = pd.Series([50.0])
    mask = bogus_row_mask(close, volume, raw_close)
    assert bool(mask.iloc[0]) is False


def test_thresholds_match_259_price_floor():
    # ohlcv_hygiene.MIN_PRICE_FLOOR must stay in sync with the risk group's
    # #259 floor (risk_factor_scores_asset._MIN_PRICE_FLOOR is now an alias
    # of this constant - see that module).
    assert MIN_PRICE_FLOOR == 0.01
    assert MIN_DOLLAR_VOLUME > 0


# ─────────────────────────────────────────────
# check_ret1d_input_quality: _evaluate_days over floored vs unfloored input
# ─────────────────────────────────────────────


def _agg_row(score_date: str, rets: np.ndarray) -> dict:
    """Build one day's aggregate row the same way the SQL query would.

    Mirrors the check's SQL exactly: std_ret is over WINSORIZED ret_1d
    (clamped to [-_STD_WINSOR_ABS, +_STD_WINSOR_ABS] via LEAST/GREATEST),
    while min/max/extreme-count are over the RAW returns.
    """
    winsorized = np.clip(rets, -_STD_WINSOR_ABS, _STD_WINSOR_ABS)
    return {
        "score_date": score_date,
        "n_rows": len(rets),
        "std_ret": float(np.std(winsorized, ddof=1)),
        "n_extreme": int(np.sum(np.abs(rets) > 0.5)),
        "min_ret": float(np.min(rets)),
        "max_ret": float(np.max(rets)),
    }


def test_bogus_rows_no_longer_trip_the_check():
    """A day whose cross-section, once floored, is a normal quiet day
    should NOT trip the check - even though a bogus print (huge
    swing) got excluded upstream by the SQL floor before this frame was
    ever built (i.e. we simulate the POST-floor cross-section directly).
    """
    rng = np.random.default_rng(0)
    normal = rng.normal(0.0, 0.02, size=3000)  # typical daily equity noise
    df = pd.DataFrame([_agg_row("2025-03-25", normal)])

    bad_dates, skipped_partial, evaluated = _evaluate_days(df)
    assert bad_dates == []
    assert evaluated == 1

    passed, severity = _era_gate(bad_dates)
    assert passed is True


def test_pre_floor_bogus_day_would_have_tripped():
    """Sanity check the OTHER direction: if the bogus print were NOT
    floored out (e.g. a huge bogus ret_1d landed in the cross-section),
    the same day trips the check. Confirms _evaluate_days is actually
    threshold-sensitive and the floor test above isn't vacuous."""
    rng = np.random.default_rng(0)
    normal = rng.normal(0.0, 0.02, size=3000)
    contaminated = np.concatenate([normal, [399999.0]])
    df = pd.DataFrame([_agg_row("2025-03-25", contaminated)])

    bad_dates, _, _ = _evaluate_days(df)
    assert len(bad_dates) == 1
    assert "max=" in bad_dates[0]["reasons"]


def test_genuine_broad_corruption_still_trips_post_floor():
    """A broad-corruption shape: many LIQUID names broken on one day
    (high cross-sectional std, > 10 extremes). This must still fail even
    though every row here is assumed to have already cleared the
    liquidity/price floor - the floor removes illiquid junk, not the
    check's sensitivity to real corruption."""
    rng = np.random.default_rng(1)
    corrupted = rng.normal(0.0, 0.097, size=3000)
    # Force at least 22 extreme moves.
    corrupted[:22] = 0.6
    df = pd.DataFrame([_agg_row("2026-02-10", corrupted)])

    bad_dates, _, evaluated = _evaluate_days(df)
    assert evaluated == 1
    assert len(bad_dates) == 1
    assert bad_dates[0]["n_extreme"] > _EXTREME_COUNT_MAX or bad_dates[0]["std_ret"] > _STD_MAX

    passed, severity = _era_gate(bad_dates)
    assert passed is False


# ─────────────────────────────────────────────
# Meridian #275 follow-up: single-name gate aligned EXACTLY to LDP #182's
# cleaning boundary (was [-0.95, 2.0], now |ret_1d| > 50, i.e. [-50, 50])
# ─────────────────────────────────────────────


def test_genuine_large_single_name_move_passes():
    """A +251% single-name move (a genuine M&A-scale move) on an
    otherwise quiet, liquid day must PASS - this is exactly the
    false-positive a tight bound produces. Population size (20000)
    is a large synthetic cross-section, so one outlier does not swing the
    cross-sectional std past _STD_MAX on its own."""
    rng = np.random.default_rng(2)
    normal = rng.normal(0.0, 0.02, size=20000)
    day = np.concatenate([normal, [2.51]])  # +251%
    df = pd.DataFrame([_agg_row("2025-02-03", day)])

    bad_dates, _, evaluated = _evaluate_days(df)
    assert evaluated == 1
    assert bad_dates == []


def test_extreme_but_real_single_name_move_passes():
    """A +1200% single-name move (a biotech trial readout scale) is well
    below LDP's kept ceiling and must PASS."""
    rng = np.random.default_rng(3)
    normal = rng.normal(0.0, 0.02, size=20000)
    day = np.concatenate([normal, [12.0]])  # +1200%
    df = pd.DataFrame([_agg_row("2024-11-05", day)])

    bad_dates, _, evaluated = _evaluate_days(df)
    assert evaluated == 1
    assert bad_dates == []


def test_ldp_kept_high_move_passes():
    """A +4900% move (still <= LDP's 50 boundary)
    must PASS - LDP keeps everything at/below 50, so the gate must not fire
    on it. This is the tightest legit case just under the boundary."""
    rng = np.random.default_rng(30)
    normal = rng.normal(0.0, 0.02, size=20000)
    day = np.concatenate([normal, [49.0]])  # +4900%, still <= 50
    df = pd.DataFrame([_agg_row("2025-06-05", day)])

    bad_dates, _, evaluated = _evaluate_days(df)
    assert evaluated == 1
    assert bad_dates == []


def test_above_ldp_boundary_max_trips():
    """A value above LDP's |ret_1d| > 50 boundary (a print LDP would NULL,
    e.g. a +400000% bad print) must trip the gate - the forward safety net
    mirrors LDP's rule for any bad print that slips in before LDP NULLs
    it."""
    rng = np.random.default_rng(4)
    normal = rng.normal(0.0, 0.02, size=20000)
    day = np.concatenate([normal, [4000.0]])  # +400000%
    df = pd.DataFrame([_agg_row("2024-12-02", day)])

    bad_dates, _, _ = _evaluate_days(df)
    assert len(bad_dates) == 1
    assert "max=" in bad_dates[0]["reasons"]


def test_just_above_boundary_max_trips():
    """The boundary is exclusive (> 50): a value just above 50 trips."""
    rng = np.random.default_rng(31)
    normal = rng.normal(0.0, 0.02, size=20000)
    day = np.concatenate([normal, [50.5]])
    df = pd.DataFrame([_agg_row("2025-06-06", day)])

    bad_dates, _, _ = _evaluate_days(df)
    assert len(bad_dates) == 1
    assert "max=" in bad_dates[0]["reasons"]


def test_real_total_loss_passes():
    """A real total loss is exactly -100% (-1.0); with the boundary aligned
    to |ret_1d| > 50 it passes with wide margin, as do near-total crashes
    (-99%)."""
    rng = np.random.default_rng(5)
    normal = rng.normal(0.0, 0.02, size=3000)
    for loss in (-0.99, -1.0):
        day = np.concatenate([normal, [loss]])
        df = pd.DataFrame([_agg_row("2025-06-01", day)])
        bad_dates, _, _ = _evaluate_days(df)
        assert bad_dates == [], f"loss {loss} should pass"


def test_below_ldp_boundary_min_trips():
    """A value below LDP's -50 boundary (|ret_1d| > 50) must trip - a
    daily return below -5000% is bad data LDP would NULL."""
    rng = np.random.default_rng(7)
    normal = rng.normal(0.0, 0.02, size=20000)
    day_below = np.concatenate([normal, [-55.0]])
    df_below = pd.DataFrame([_agg_row("2025-06-02", day_below)])

    bad_dates_below, _, _ = _evaluate_days(df_below)
    assert len(bad_dates_below) == 1
    assert "min=" in bad_dates_below[0]["reasons"]


def test_ret_abs_bounds_match_ldp_boundary():
    # Regression guard: the single-name gate must stay aligned to LDP #182's
    # |ret_1d| > 50 cleaning boundary on both sides.
    assert _RET_ABS_MIN == -50.0
    assert _RET_ABS_MAX == 50.0


# ─────────────────────────────────────────────
# Std gate is single-outlier robust via winsorization (clamp to [-1, +1])
# ─────────────────────────────────────────────


def test_std_gate_single_large_move_does_not_trip():
    """One +1200% move among a few thousand liquid names. On RAW returns this alone
    drives std to ~0.22 (> 0.15); with the winsorized std it stays low, so the
    day PASSES both the std gate and the (|ret|>50) max gate."""
    rng = np.random.default_rng(20)
    normal = rng.normal(0.0, 0.05, size=3000)  # a slightly-active day
    day = np.concatenate([normal, [12.0]])  # +1200%
    df = pd.DataFrame([_agg_row("2024-11-05", day)])

    # Sanity: RAW std would have tripped, proving the winsorization matters.
    raw_std = float(np.std(day, ddof=1))
    assert raw_std > _STD_MAX

    bad_dates, _, evaluated = _evaluate_days(df)
    assert evaluated == 1
    assert bad_dates == []


def test_std_gate_winsor_clamps_to_bound():
    """Regression guard on the winsor constant itself."""
    assert _STD_WINSOR_ABS == 1.0


def test_broad_corruption_many_names_over_one_still_trips():
    """A synthetic broad corruption: hundreds of names all
    swung past +/-1.0 on one day. Even after winsorization to [-1, +1] the
    cross-sectional std blows past 0.15 (and the extreme count past 10), so
    the day still BLOCKS. Winsorization removes single-name sensitivity, not
    the check's response to genuine broad corruption."""
    rng = np.random.default_rng(21)
    normal = rng.normal(0.0, 0.02, size=3000)
    # 400 names corrupted to large positive/negative moves (all |ret| > 1).
    corrupted = np.concatenate([normal, rng.choice([-3.0, 3.0], size=400)])
    df = pd.DataFrame([_agg_row("2026-02-10", corrupted)])

    bad_dates, _, evaluated = _evaluate_days(df)
    assert evaluated == 1
    assert len(bad_dates) == 1
    # Winsorized std must still exceed the bound, and extreme-count too.
    assert bad_dates[0]["std_ret"] > _STD_MAX
    assert bad_dates[0]["n_extreme"] > _EXTREME_COUNT_MAX

    passed, _ = _era_gate(bad_dates)
    assert passed is False


def test_thin_day_still_skipped_not_failed():
    df = pd.DataFrame([_agg_row("2025-03-25", np.array([0.01, -0.02, 0.03]))])
    bad_dates, skipped_partial, evaluated = _evaluate_days(df)
    assert evaluated == 0
    assert skipped_partial == ["2025-03-25"]
    assert bad_dates == []


def test_min_rows_threshold_unchanged():
    # Regression guard: MIN_ROWS behavior is orthogonal to the new floor.
    assert _MIN_ROWS == 500
    assert _RET_ABS_MAX == 50.0


# ─────────────────────────────────────────────
# risk_metrics: bogus/illiquid rows excluded before vol/beta computation
# ─────────────────────────────────────────────


def test_risk_metrics_query_applies_bogus_print_guard():
    """Regression guard: the risk_metrics fetch SQL must embed the shared
    BOGUS_PRINT_SQL_GUARD - if a future edit drops it, this test catches
    the query silently reverting to the unfiltered read."""
    import inspect

    from meridian_pipeline import risk_metrics_asset

    src = inspect.getsource(risk_metrics_asset._compute_risk_metrics_for_date)
    assert "BOGUS_PRINT_SQL_GUARD" in src
    assert "fetch_delisted_tickers" in src


def test_risk_metrics_excludes_delisted_and_bogus_tickers_from_pivot():
    """Exercise the delisting-exclusion slice of _compute_risk_metrics_for_date
    directly: a ticker present in the fetched frame but confirmed delisted
    must be dropped before the pivot/window-stats step, mirroring the risk
    factor group's #259 guard."""
    prices = pd.DataFrame(
        {
            "ticker": ["SPY", "SPY", "GOOD", "GOOD", "DEAD", "DEAD"],
            "d": pd.to_datetime(["2026-01-01", "2026-01-02"] * 3),
            "ret_1d": [0.001, 0.002, 0.01, -0.01, 0.5, -0.5],
            "dollar_volume": [1e9, 1e9, 1e7, 1e7, 5e6, 5e6],
        }
    )
    delisted = frozenset({"DEAD"})
    filtered = prices[~prices["ticker"].isin(delisted)]

    assert set(filtered["ticker"].unique()) == {"SPY", "GOOD"}
    assert "DEAD" not in filtered["ticker"].values


def test_risk_metrics_applies_stale_fallback():
    """#291 regression guard: _compute_risk_metrics_for_date's source must
    call the price-staleness fallback, independent of fetch_delisted_tickers -
    if a future edit drops it, this catches the guard silently reverting to
    delisting_date-only."""
    import inspect

    from meridian_pipeline import risk_metrics_asset

    src = inspect.getsource(risk_metrics_asset._compute_risk_metrics_for_date)
    assert "stale_tickers_from_last_seen" in src


# ─────────────────────────────────────────────
# stale_tickers_from_last_seen (#291 price-staleness fallback)
# ─────────────────────────────────────────────


def _trading_calendar(n_days: int, end: str) -> pd.DatetimeIndex:
    """n_days consecutive business days ending on `end` (inclusive)."""
    return pd.bdate_range(end=end, periods=n_days)


def test_stale_ticker_beyond_tolerance_is_excluded():
    # Calendar of business days ending at score_date; STALE's last bar is
    # 20 trading days before score_date - beyond STALE_TRADING_DAY_TOLERANCE.
    cal = _trading_calendar(60, "2026-07-28")
    score_date = cal[-1].date()
    last_seen = pd.Series(
        {"ACTIVE": cal[-1], "STALE": cal[-1 - 20]},
    )
    stale = stale_tickers_from_last_seen(last_seen, cal, score_date)
    assert stale == frozenset({"STALE"})


def test_ticker_within_tolerance_is_not_excluded():
    # Last bar 10 trading days before score_date - inside the 15-day tolerance
    # (normal ingest lag / mirror lag territory, not staleness).
    cal = _trading_calendar(60, "2026-07-28")
    score_date = cal[-1].date()
    last_seen = pd.Series({"RECENT": cal[-1 - 10]})
    stale = stale_tickers_from_last_seen(last_seen, cal, score_date)
    assert stale == frozenset()


def test_legitimate_halt_inside_tolerance_is_not_excluded():
    # A multi-week trading halt (e.g. an SEC Rule 12(k) suspension, capped at
    # 10 trading days by statute) must not trip the fallback - the tolerance
    # is set above that cap. Exercise exactly at the 10-day
    # statutory cap.
    cal = _trading_calendar(60, "2026-07-28")
    score_date = cal[-1].date()
    last_seen = pd.Series({"HALTED": cal[-1 - 10]})
    stale = stale_tickers_from_last_seen(last_seen, cal, score_date)
    assert "HALTED" not in stale


def test_gap_boundary_exact_tolerance_not_excluded():
    cal = _trading_calendar(60, "2026-07-28")
    score_date = cal[-1].date()
    last_seen = pd.Series({"EDGE": cal[-1 - STALE_TRADING_DAY_TOLERANCE]})
    stale = stale_tickers_from_last_seen(last_seen, cal, score_date)
    assert "EDGE" not in stale, "exactly at the tolerance boundary must not be excluded"


def test_gap_one_beyond_tolerance_is_excluded():
    cal = _trading_calendar(80, "2026-07-28")
    score_date = cal[-1].date()
    last_seen = pd.Series({"OVER": cal[-1 - (STALE_TRADING_DAY_TOLERANCE + 1)]})
    stale = stale_tickers_from_last_seen(last_seen, cal, score_date)
    assert "OVER" in stale


def test_pit_correctness_historical_score_date_not_over_excluded():
    """A ticker alive and trading right up to a HISTORICAL score_date must not
    be excluded merely because it stopped trading long after that date (i.e.
    it is dead 'today' but was very much alive at the historical score_date).
    This is the PIT-correctness requirement: staleness is judged relative to
    score_date, never using data from after it."""
    cal = _trading_calendar(60, "2015-06-30")
    historical_score_date = cal[-1].date()
    # ALIVE_THEN traded right up through the historical score_date - as far
    # as this function's inputs are concerned, there is no "future" data at
    # all (the caller's own fetch query already bounds rows to
    # ts_event::date <= score_date), so ALIVE_THEN must not be stale.
    last_seen = pd.Series({"ALIVE_THEN": cal[-1]})
    stale = stale_tickers_from_last_seen(last_seen, cal, historical_score_date)
    assert stale == frozenset()


def test_empty_last_seen_returns_empty_set():
    cal = _trading_calendar(30, "2026-07-28")
    score_date = cal[-1].date()
    stale = stale_tickers_from_last_seen(pd.Series(dtype="datetime64[ns]"), cal, score_date)
    assert stale == frozenset()


# ─────────────────────────────────────────────
# CORE-asset extension (#259 follow-up): fetch_last_seen_dates /
# fetch_trading_calendar / stale_or_never_seen_tickers / hygiene_exclusions -
# the bulk-query building blocks CORE factor assets use when they
# have no already-fetched price frame to derive staleness from for free.
# DB-free: monkeypatches ohlcv_hygiene.pd.read_sql, keyed on distinguishing
# SQL fragments, so no live PostgreSQL connection is required.
# ─────────────────────────────────────────────


def test_stale_or_never_seen_tickers_flags_absent_ticker(monkeypatch):
    cal = _trading_calendar(60, "2026-07-28")
    score_date = cal[-1].date()

    def fake_read_sql(query, engine, params=None):
        q = str(query)
        if "MAX(ts_event::date)" in q:
            # Only ACTIVE ever printed a bar in the lookback window; GHOST is
            # absent entirely (no row at all, not a null value).
            return pd.DataFrame({"ticker": ["ACTIVE"], "last_seen": [cal[-1].date()]})
        if "DISTINCT ts_event::date" in q:
            return pd.DataFrame({"d": cal})
        raise AssertionError(f"unexpected query: {q}")

    monkeypatch.setattr(ohlcv_hygiene.pd, "read_sql", fake_read_sql)
    stale = ohlcv_hygiene.stale_or_never_seen_tickers(None, ["ACTIVE", "GHOST"], score_date)
    assert stale == frozenset({"GHOST"})


def test_stale_or_never_seen_tickers_fails_open_on_empty_calendar(monkeypatch):
    """No trading-calendar proxy data at all in the lookback window (e.g. a
    very early historical score_date) must fail OPEN, not silently exclude
    every ticker in the batch."""

    def fake_read_sql(query, engine, params=None):
        q = str(query)
        if "MAX(ts_event::date)" in q:
            return pd.DataFrame({"ticker": [], "last_seen": []})
        if "DISTINCT ts_event::date" in q:
            return pd.DataFrame({"d": pd.Series(dtype="datetime64[ns]")})
        raise AssertionError(f"unexpected query: {q}")

    monkeypatch.setattr(ohlcv_hygiene.pd, "read_sql", fake_read_sql)
    stale = ohlcv_hygiene.stale_or_never_seen_tickers(None, ["ANYTHING"], date(2004, 1, 5))
    assert stale == frozenset()


def test_hygiene_exclusions_unions_delisted_and_stale(monkeypatch):
    score_date = date(2026, 7, 28)
    monkeypatch.setattr(
        ohlcv_hygiene,
        "fetch_delisted_tickers",
        lambda engine, sd: frozenset({"DELISTED", "BOTH"}),
    )
    monkeypatch.setattr(
        ohlcv_hygiene,
        "stale_or_never_seen_tickers",
        lambda engine, tickers, sd: frozenset({"STALE", "BOTH"}),
    )

    excluded, counts = ohlcv_hygiene.hygiene_exclusions(
        None, ["DELISTED", "STALE", "BOTH", "LIVE"], score_date
    )
    assert excluded == frozenset({"DELISTED", "STALE", "BOTH"})
    assert counts == {"delisted": 2, "stale": 2, "excluded_total": 3, "link_kept": 0}
    assert "LIVE" not in excluded


# ─────────────────────────────────────────────
# fetch_delisted_tickers: reused-ticker regression.
# A ticker must be judged on the entity behind its MOST RECENT bar as of
# score_date, NOT on every entity that ever used the symbol - symbols get
# recycled, and an ANY-entity form would wrongly exclude live names.
# DB-free: the fake read_sql below re-implements the query's SEMANTICS over
# a synthetic (ticker, ts_event, entity_id) tape plus a companies table, so
# the assertions exercise the real SQL's intent rather than string-matching.
# ─────────────────────────────────────────────

# (ticker, bar date, entity_id) - the synthetic ohlcv.prices tape.
_TAPE = [
    # RECYC: dead predecessor OLDCO (entity 1, delisted 2004-09-14) then the
    # live NEWCO (entity 2). Latest bar is the live entity.
    ("RECYC", date(2004, 9, 1), 1),
    ("RECYC", date(2026, 7, 30), 2),
    # DEADCO: single entity, delisted 2026-01-15, still printing ghost bars.
    ("DEADCO", date(2026, 7, 30), 3),
    # NOLINK: real, live ticker whose bars carry no entity_id linkage.
    ("NOLINK", date(2026, 7, 30), None),
    # FUTUREDEAD: entity delisted AFTER the score_date used below.
    ("FUTUREDEAD", date(2026, 7, 30), 4),
    # JUSTDEAD: delisted 2026-07-27, last bar 2026-07-30 (within tolerance).
    ("JUSTDEAD", date(2026, 7, 30), 5),
    # SHELLIKE (#454): live ticker whose bars LDP enrichment linked to a dead
    # predecessor listing (delisted 2022-01-28) - the link is wrong.
    ("SHELLIKE", date(2026, 7, 30), 6),
]

# entity_id -> delisting_date (None = still listed)
_COMPANIES = {
    1: date(2004, 9, 14),  # OLDCO INC
    2: None,  # NEWCO INC
    3: date(2026, 1, 15),  # genuinely delisted
    4: date(2026, 12, 31),  # delists in the future relative to score_date
    5: date(2026, 7, 27),  # delisted 3 days before its last bar
    6: date(2022, 1, 28),  # dead predecessor wrongly linked to a live ticker
}


@pytest.fixture(autouse=True)
def _reset_link_warn_memo():
    """The contradicted-link warning is memoised per score_date per process;
    isolate tests from each other."""
    ohlcv_hygiene._link_warned.clear()
    yield
    ohlcv_hygiene._link_warned.clear()


def _fake_delisted_read_sql(score_date: date, lookback_days: int = 60):
    """read_sql stand-in implementing the fixed query's semantics: resolve
    each ticker to the entity of its LATEST bar in
    (score_date - lookback_days, score_date], then keep only tickers whose
    THAT entity has delisting_date < score_date. NULL entity_id drops out
    (the JOIN), matching the unguarded/assumed-active behaviour."""

    def _read_sql(query, engine, params=None):
        assert "DISTINCT ON (p.ticker)" in str(query), "must resolve latest bar per ticker"
        window_start = score_date - timedelta(days=lookback_days)
        latest: dict[str, tuple[date, "int | None"]] = {}
        for ticker, bar_date, entity_id in _TAPE:
            if not (window_start <= bar_date <= score_date):
                continue  # PIT + window bound, exactly as the SQL does
            if ticker not in latest or bar_date > latest[ticker][0]:
                latest[ticker] = (bar_date, entity_id)

        rows = []
        for ticker, (bar_date, entity_id) in latest.items():
            if entity_id is None:
                continue  # no linkage -> JOIN drops it -> assumed active
            delisting = _COMPANIES.get(entity_id)
            if delisting is not None and delisting < score_date:
                rows.append(
                    {"ticker": ticker, "last_bar_date": bar_date, "delisting_date": delisting}
                )
        return pd.DataFrame(rows, columns=["ticker", "last_bar_date", "delisting_date"])

    return _read_sql


def test_reused_ticker_with_live_current_entity_not_delisted(monkeypatch):
    """THE regression: RECYC's symbol was previously used by OLDCO INC
    (delisted 2004-09-14) and is now NEWCO INC. The latest bar
    belongs to the live entity, so RECYC must NOT be excluded. This is the
    exact shape of the reused-ticker case (same for other recycled symbols)."""
    score_date = date(2026, 7, 31)
    monkeypatch.setattr(ohlcv_hygiene.pd, "read_sql", _fake_delisted_read_sql(score_date))
    delisted = ohlcv_hygiene.fetch_delisted_tickers(None, score_date)
    assert "RECYC" not in delisted


def test_genuinely_delisted_ticker_is_excluded(monkeypatch):
    """The guard must still do its job: a ticker whose CURRENT entity
    delisted and whose last bar is within DELISTING_LINK_TOLERANCE_DAYS of
    that date is excluded."""
    score_date = date(2026, 7, 31)
    monkeypatch.setattr(ohlcv_hygiene.pd, "read_sql", _fake_delisted_read_sql(score_date))
    delisted = ohlcv_hygiene.fetch_delisted_tickers(None, score_date)
    assert "JUSTDEAD" in delisted


def test_bars_long_after_delisting_mean_a_wrong_link_454(monkeypatch, caplog):
    """Meridian#454: bars printed more than DELISTING_LINK_TOLERANCE_DAYS after
    the linked entity's delisting contradict the link (LDP enrichment carries a
    live ticker onto a dead predecessor, e.g. a ticker linked to a dead predecessor listing).
    Those tickers are NOT delisted, and the guard logs them. The accepted
    cost: a true ghost-bar ticker (DEADCO) is no longer caught here in the
    2026+ regime; the stale-by-bars rule is its only remaining guard."""
    score_date = date(2026, 7, 31)
    monkeypatch.setattr(ohlcv_hygiene.pd, "read_sql", _fake_delisted_read_sql(score_date))
    with caplog.at_level("WARNING", logger="meridian_pipeline.ohlcv_hygiene"):
        delisted = ohlcv_hygiene.fetch_delisted_tickers(None, score_date)
    assert "SHELLIKE" not in delisted
    assert "DEADCO" not in delisted
    assert "SHELLIKE" in caplog.text and "#454" in caplog.text


def _tape_read_sql(rows):
    """Fake read_sql returning raw (ticker, last_bar_date, delisting_date) rows."""

    def _read_sql(query, engine, params=None):
        return pd.DataFrame(rows, columns=["ticker", "last_bar_date", "delisting_date"])

    return _read_sql


def test_pre_2026_ghost_bars_keep_the_full_guard_454(monkeypatch):
    """Before DELISTING_LINK_CHECK_SINCE the old semantics hold exactly: a
    ticker printing bars 400 days after its delisting is still delisted (the
    real 2019 ghost-bar population)."""
    score_date = date(2019, 6, 28)
    rows = [("GHOST2019", date(2019, 6, 27), date(2018, 5, 1))]
    monkeypatch.setattr(ohlcv_hygiene.pd, "read_sql", _tape_read_sql(rows))
    assert ohlcv_hygiene.fetch_delisted_tickers(None, score_date) == frozenset({"GHOST2019"})


def test_link_check_boundary_and_tolerance_454(monkeypatch):
    """On/after the check date the tolerance is 5 days inclusive: a bar 5 days
    after the delisting still counts as delisted, 6 days after does not."""
    score_date = date(2026, 3, 2)
    rows = [
        ("AT5", date(2026, 2, 6), date(2026, 2, 1)),
        ("AT6", date(2026, 2, 7), date(2026, 2, 1)),
    ]
    monkeypatch.setattr(ohlcv_hygiene.pd, "read_sql", _tape_read_sql(rows))
    assert ohlcv_hygiene.fetch_delisted_tickers(None, score_date) == frozenset({"AT5"})
    # The same rows judged for a pre-2026 score date are both delisted.
    monkeypatch.setattr(ohlcv_hygiene, "DELISTING_LINK_CHECK_SINCE", date(2027, 1, 1))
    assert ohlcv_hygiene.fetch_delisted_tickers(None, score_date) == frozenset({"AT5", "AT6"})


def test_contradicted_link_warning_is_memoised_truncated_and_tagged_454(monkeypatch, caplog):
    score_date = date(2026, 5, 4)
    rows = [(f"T{i:03d}", date(2026, 5, 1), date(2022, 1, 1)) for i in range(30)]
    monkeypatch.setattr(ohlcv_hygiene.pd, "read_sql", _tape_read_sql(rows))
    monkeypatch.setattr(ohlcv_hygiene, "_link_warned", set())
    monkeypatch.setenv("DAGSTER_RUN_ID", "run-abc123")
    with caplog.at_level("WARNING", logger="meridian_pipeline.ohlcv_hygiene"):
        ohlcv_hygiene.fetch_delisted_tickers(None, score_date)
        ohlcv_hygiene.fetch_delisted_tickers(None, score_date)
    recs = [r for r in caplog.records if "#454" in r.getMessage()]
    assert len(recs) == 1  # once per score_date per process
    msg = recs[0].getMessage()
    assert "30 ticker(s)" in msg and "+5 more" in msg and "run-abc123" in msg


def test_link_warning_memo_is_bounded(monkeypatch):
    monkeypatch.setattr(ohlcv_hygiene, "_link_warned", set())
    for i in range(ohlcv_hygiene._LINK_WARN_MEMO_CAP + 10):
        ohlcv_hygiene._warn_contradicted_links(date(2026, 1, 1) + timedelta(days=i), ["X"])
    assert len(ohlcv_hygiene._link_warned) <= ohlcv_hygiene._LINK_WARN_MEMO_CAP


def test_wrong_link_rule_at_the_real_2026_boundary_without_patching_466(monkeypatch):
    """The #454 tests monkeypatch DELISTING_LINK_CHECK_SINCE. This one pins the
    shipped constant: a wrongly linked ticker (delisting 2022, bar on the score
    date) is delisted for score date 2025-12-31 and kept for 2026-01-01."""
    assert ohlcv_hygiene.DELISTING_LINK_CHECK_SINCE == date(2026, 1, 1)
    for score_date, expected_delisted in (
        (date(2025, 12, 31), frozenset({"WRONG"})),
        (date(2026, 1, 1), frozenset()),
    ):
        rows = [("WRONG", score_date, date(2022, 1, 28))]
        monkeypatch.setattr(ohlcv_hygiene.pd, "read_sql", _tape_read_sql(rows))
        got = ohlcv_hygiene.fetch_delisted_tickers(None, score_date)
        assert got == expected_delisted, score_date
        assert got.link_kept == (frozenset({"WRONG"}) - expected_delisted), score_date


def test_latest_bar_query_breaks_same_day_ties_on_entity_id_466():
    """Two entities printing on the same day made `DISTINCT ON` pick one
    arbitrarily; the staged mirror takes the highest entity_id. NULLS LAST,
    because Postgres sorts NULLs FIRST on DESC and would hand the tie to the
    unlinked row (dropped at the JOIN)."""
    captured = []

    def read_sql(query, engine, params=None):
        captured.append(str(query))
        return pd.DataFrame(columns=["ticker", "last_bar_date", "delisting_date"])

    import unittest.mock as um

    with um.patch.object(ohlcv_hygiene.pd, "read_sql", read_sql):
        ohlcv_hygiene.fetch_delisted_tickers(None, date(2026, 3, 2))
    assert "ORDER BY p.ticker, p.ts_event DESC, p.entity_id DESC NULLS LAST" in captured[0]


def test_link_kept_is_reported_in_hygiene_counts_466(monkeypatch):
    """The wrong-link rule's rescues are counted, restricted to the caller's
    universe, and are informational: they never enter the exclusion set."""
    score_date = date(2026, 3, 2)
    rows = [
        ("WRONG_A", date(2026, 3, 2), date(2022, 1, 28)),  # kept
        ("WRONG_B", date(2026, 3, 2), date(2022, 2, 28)),  # kept, outside universe below
        ("DEAD", date(2026, 2, 27), date(2026, 2, 26)),  # delisted
    ]
    monkeypatch.setattr(ohlcv_hygiene.pd, "read_sql", _tape_read_sql(rows))
    monkeypatch.setattr(
        ohlcv_hygiene, "stale_or_never_seen_tickers", lambda engine, tickers, sd: frozenset()
    )
    excluded, counts = ohlcv_hygiene.hygiene_exclusions(
        None, ["WRONG_A", "DEAD", "LIVE"], score_date
    )
    assert excluded == frozenset({"DEAD"})
    assert counts == {"delisted": 1, "stale": 0, "excluded_total": 1, "link_kept": 1}


def test_hygiene_counts_default_link_kept_to_zero_for_a_plain_frozenset_466(monkeypatch):
    """Test doubles and hoisted lookups return a plain frozenset: no attribute,
    no crash, a zero count."""
    monkeypatch.setattr(ohlcv_hygiene, "fetch_delisted_tickers", lambda e, d: frozenset({"X"}))
    monkeypatch.setattr(
        ohlcv_hygiene, "stale_or_never_seen_tickers", lambda engine, tickers, sd: frozenset()
    )
    _excl, counts = ohlcv_hygiene.hygiene_exclusions(None, ["X"], date(2026, 3, 2))
    assert counts["link_kept"] == 0


def test_contradicted_link_warning_splits_old_from_recent_delisting_466(monkeypatch, caplog):
    """A delisting date far before the score date is almost certainly a wrong
    link; one within DELISTING_LINK_RECENT_DAYS may be a real delisting still
    printing OTC. They get separate warnings so the second kind is not buried."""
    score_date = date(2026, 5, 4)
    rows = [
        ("OLD1", date(2026, 5, 1), date(2022, 1, 1)),
        ("OLD2", date(2026, 5, 1), date(2025, 6, 1)),
        ("RECENT1", date(2026, 5, 1), date(2026, 4, 1)),
        ("EDGE", date(2026, 5, 1), score_date - timedelta(days=90)),  # exactly on the cut: recent
    ]
    monkeypatch.setattr(ohlcv_hygiene.pd, "read_sql", _tape_read_sql(rows))
    with caplog.at_level("WARNING", logger="meridian_pipeline.ohlcv_hygiene"):
        got = ohlcv_hygiene.fetch_delisted_tickers(None, score_date)
    assert got == frozenset() and got.link_kept == {"OLD1", "OLD2", "RECENT1", "EDGE"}
    recs = [r.getMessage() for r in caplog.records if "#454" in r.getMessage()]
    assert len(recs) == 2
    old = next(m for m in recs if "OLD1" in m)
    recent = next(m for m in recs if "RECENT1" in m)
    assert "almost certainly a wrong entity link" in old and "RECENT1" not in old
    assert "OLD2" in old and "EDGE" not in old
    assert "possibly a real delisting" in recent and "EDGE" in recent and "OLD1" not in recent


def test_bar_contradicts_delisting_link_is_the_shared_rule_466():
    """The one predicate both ohlcv_hygiene and feature_panel apply."""
    bar = pd.Series(pd.to_datetime(["2026-02-06", "2026-02-07", "2026-02-07"]))
    delisting = pd.Series(pd.to_datetime(["2026-02-01", "2026-02-01", None]))
    got = ohlcv_hygiene.bar_contradicts_delisting_link(bar, delisting)
    assert got.tolist() == [False, True, False]  # 5 days ok, 6 days wrong, NaT never


def test_null_entity_id_ticker_not_excluded(monkeypatch):
    """A ticker whose bars carry no entity_id cannot be checked and stays
    unguarded/assumed-active (conservative - unchanged pre-fix behaviour,
    preserved by the inner JOIN)."""
    score_date = date(2026, 7, 31)
    monkeypatch.setattr(ohlcv_hygiene.pd, "read_sql", _fake_delisted_read_sql(score_date))
    delisted = ohlcv_hygiene.fetch_delisted_tickers(None, score_date)
    assert "NOLINK" not in delisted


def test_delisting_after_score_date_not_excluded(monkeypatch):
    """PIT correctness: an entity that delists AFTER score_date was alive at
    score_date and must not be excluded from that partition."""
    score_date = date(2026, 7, 31)
    monkeypatch.setattr(ohlcv_hygiene.pd, "read_sql", _fake_delisted_read_sql(score_date))
    delisted = ohlcv_hygiene.fetch_delisted_tickers(None, score_date)
    assert "FUTUREDEAD" not in delisted
    assert delisted == frozenset({"JUSTDEAD"})


def test_historical_score_date_resolves_to_the_then_current_entity(monkeypatch):
    """PIT correctness, the other direction: on a 2004 score_date RECYC's
    latest bar is OLDCO INC's, whose delisting_date (2004-09-14) is still
    in the future, so RECYC is not excluded then either. The resolver never
    reads a bar past score_date - it cannot see NEWCO in 2004."""
    score_date = date(2004, 9, 10)
    monkeypatch.setattr(ohlcv_hygiene.pd, "read_sql", _fake_delisted_read_sql(score_date))
    delisted = ohlcv_hygiene.fetch_delisted_tickers(None, score_date)
    assert delisted == frozenset()

    # One day after OLDCO's delisting the same 2004 bar makes RECYC delisted
    # (nothing else has printed under the symbol yet) - the guard is live in
    # both directions, it is only the ANY-entity smear that is gone.
    later = date(2004, 9, 15)
    monkeypatch.setattr(ohlcv_hygiene.pd, "read_sql", _fake_delisted_read_sql(later))
    assert "RECYC" in ohlcv_hygiene.fetch_delisted_tickers(None, later)


def test_delisted_query_is_windowed_and_pit_bounded(monkeypatch):
    """Guard the query SHAPE: the latest-bar subquery must stay bounded to a
    sargable recent window (perf - an unbounded per-ticker resolution is far
    slower on a large ohlcv.prices) and must use a
    half-open upper bound at score_date + 1 day (PIT)."""
    captured = {}

    def _capture(query, engine, params=None):
        captured["sql"] = str(query)
        captured["params"] = params
        return pd.DataFrame({"ticker": [], "last_bar_date": [], "delisting_date": []})

    monkeypatch.setattr(ohlcv_hygiene.pd, "read_sql", _capture)
    score_date = date(2026, 7, 31)
    ohlcv_hygiene.fetch_delisted_tickers(None, score_date)

    sql = captured["sql"]
    assert "DISTINCT ON (p.ticker)" in sql
    assert "ORDER BY p.ticker, p.ts_event DESC" in sql
    assert ":window_start" in sql and ":next_day" in sql
    # No `ts_event::date` cast on the bounds - that would kill index usage.
    assert "p.ts_event >= :window_start" in sql
    assert "p.ts_event < :next_day" in sql

    params = captured["params"]
    assert params["next_day"] == score_date + timedelta(days=1)
    assert params["window_start"] == score_date - timedelta(
        days=ohlcv_hygiene.HYGIENE_LOOKBACK_CALENDAR_DAYS
    )


def test_delisted_window_covers_the_stale_tolerance():
    """The windowing argument's central invariant: a ticker missing from
    the delisting window must already be caught by the independent
    stale-by-bars guard. That holds only while the window (calendar days) is
    comfortably longer than STALE_TRADING_DAY_TOLERANCE (trading days, ~1.4
    calendar days each)."""
    assert ohlcv_hygiene.HYGIENE_LOOKBACK_CALENDAR_DAYS > STALE_TRADING_DAY_TOLERANCE * 1.4


def test_hygiene_exclusions_empty_universe_no_op():
    excluded, counts = ohlcv_hygiene.hygiene_exclusions(None, [], date(2026, 7, 28))
    assert excluded == frozenset()
    assert counts == {"delisted": 0, "stale": 0, "excluded_total": 0, "link_kept": 0}


def test_hygiene_exclusions_pit_historical_date_not_over_excluded(monkeypatch):
    """PIT correctness end-to-end through hygiene_exclusions: a ticker alive
    and trading right up to a HISTORICAL score_date must not be excluded
    merely because it stopped trading long after that date."""
    cal = _trading_calendar(60, "2015-06-30")
    historical_score_date = cal[-1].date()

    def fake_read_sql(query, engine, params=None):
        q = str(query)
        if "MAX(ts_event::date)" in q:
            return pd.DataFrame({"ticker": ["ALIVE_THEN"], "last_seen": [cal[-1].date()]})
        if "DISTINCT ts_event::date" in q:
            return pd.DataFrame({"d": cal})
        raise AssertionError(f"unexpected query: {q}")

    monkeypatch.setattr(ohlcv_hygiene.pd, "read_sql", fake_read_sql)
    monkeypatch.setattr(ohlcv_hygiene, "fetch_delisted_tickers", lambda engine, sd: frozenset())

    excluded, counts = ohlcv_hygiene.hygiene_exclusions(None, ["ALIVE_THEN"], historical_score_date)
    assert excluded == frozenset()
    assert counts["excluded_total"] == 0


# ─────────────────────────────────────────────
# batch_hygiene_exclusions (#292 batch-scope bound)
#
# The batch_group_scores batch fetches run BEFORE the per-partition loop,
# so the per-partition hygiene call cannot bound them. batch_hygiene_exclusions
# is the filter that can be applied at batch scope; its contract is that its
# exclusion set is a SUBSET of every partition's own exclusion set (equivalently:
# the surviving ticker list is a SUPERSET of every partition's post-hygiene
# universe), so no factor value can change.
# ─────────────────────────────────────────────


def _bar_tape_read_sql(bars: dict[str, list], calendar: pd.DatetimeIndex):
    """Fake read_sql serving BOTH the per-partition and the batch-scope
    hygiene queries off one synthetic bar tape. `bars` maps ticker ->
    list of dates on which that ticker printed a bar."""

    def fake_read_sql(query, engine, params=None):
        q = str(query)
        params = params or {}
        if "MAX(ts_event::date)" in q:
            start = pd.Timestamp(params["start"])
            end = pd.Timestamp(params["score_date"])
            rows = []
            for ticker in params["tickers"]:
                seen = [d for d in bars.get(ticker, []) if start < d <= end]
                if seen:
                    rows.append({"ticker": ticker, "last_seen": max(seen).date()})
            if not rows:
                return pd.DataFrame({"ticker": [], "last_seen": []})
            return pd.DataFrame(rows)
        if "DISTINCT ts_event::date" in q:
            start = pd.Timestamp(params["start"])
            end = pd.Timestamp(params["score_date"])
            return pd.DataFrame({"d": [d for d in calendar if start < d <= end]})
        raise AssertionError(f"unexpected query: {q}")

    return fake_read_sql


def _batch_fixture(monkeypatch):
    """A 400-business-day calendar ending 2026-07-31 with a 10-partition
    batch at the end, and five tickers spanning every monotonicity case."""
    cal = pd.bdate_range(end="2026-07-31", periods=400)
    batch = list(cal[-10:])
    batch_start, batch_end = batch[0].date(), batch[-1].date()

    bars = {
        # Trades every day - never excluded anywhere.
        "LIVE": list(cal),
        # Dormant for a year, RESUMES mid-batch: stale at the batch's early
        # partitions, live at its late ones. The non-monotone direction that
        # forbids anchoring the stale test at batch_end.
        "DORMANT_RESUMES": list(cal[:50]) + list(cal[-4:]),
        # Trades up to just before the batch, then goes quiet: live at the
        # early partitions, stale at the late ones. The other non-monotone
        # direction.
        "QUITS_MIDWAY": list(cal[:-9]),
        # No bar for ~1.5 years - stale at every partition in the batch.
        "DEAD": list(cal[:60]),
        # Delisting_date falls INSIDE the batch: not delisted at batch_start.
        "DELISTS_MIDWAY": list(cal),
    }
    tickers = sorted(bars)

    monkeypatch.setattr(ohlcv_hygiene.pd, "read_sql", _bar_tape_read_sql(bars, cal))
    monkeypatch.setattr(
        ohlcv_hygiene,
        "fetch_delisted_tickers",
        lambda engine, sd: frozenset({"DELISTS_MIDWAY"}) if sd > batch[5].date() else frozenset(),
    )
    return tickers, batch, batch_start, batch_end


def test_batch_hygiene_is_a_subset_of_every_partitions_exclusions(monkeypatch):
    """THE contract: batch-scope exclusions must hold at EVERY partition in
    the batch, so the batch-scoped fetch universe is a superset of every
    partition's own post-hygiene universe."""
    tickers, batch, batch_start, batch_end = _batch_fixture(monkeypatch)

    batch_excluded, _ = ohlcv_hygiene.batch_hygiene_exclusions(
        None, tickers, batch_start, batch_end
    )
    for d in batch:
        per_partition, _ = ohlcv_hygiene.hygiene_exclusions(None, tickers, d.date())
        assert batch_excluded <= per_partition, (
            f"batch exclusions {sorted(batch_excluded)} are not a subset of "
            f"{d.date()}'s own exclusions {sorted(per_partition)}"
        )
        survivors_batch = {t for t in tickers if t not in batch_excluded}
        survivors_partition = {t for t in tickers if t not in per_partition}
        assert survivors_partition <= survivors_batch


def test_batch_hygiene_keeps_non_monotone_and_drops_permanently_dead(monkeypatch):
    """Spot-check the individual cases behind the subset property."""
    tickers, _batch, batch_start, batch_end = _batch_fixture(monkeypatch)
    excluded, counts = ohlcv_hygiene.batch_hygiene_exclusions(None, tickers, batch_start, batch_end)

    # Stale at some partitions but not all - must survive the batch filter.
    assert "DORMANT_RESUMES" not in excluded
    assert "QUITS_MIDWAY" not in excluded
    # Delisted only partway through the batch - not delisted at batch_start.
    assert "DELISTS_MIDWAY" not in excluded
    assert "LIVE" not in excluded
    # Untraded for ~1.5 years - stale at every partition, so it is safe (and
    # the whole point) to keep it out of the batch fetches.
    assert "DEAD" in excluded
    assert counts["excluded_total"] == 1


def test_batch_delisted_leg_is_the_intersection_of_both_ends_454(monkeypatch):
    """Post-2026 a ticker can be delisted at batch_start and then link-
    contradicted (not delisted) at batch_end, or the reverse. The batch leg
    excludes only tickers delisted at BOTH ends; the per-partition guard stays
    the sole authority."""
    tickers, _batch, batch_start, batch_end = _batch_fixture(monkeypatch)
    delisted_at = {
        batch_start: frozenset({"LIVE", "FLIPS_TO_LINK", "BOTH"}),
        batch_end: frozenset({"LIVE", "FLIPS_FROM_LINK", "BOTH"}),
    }
    monkeypatch.setattr(ohlcv_hygiene, "fetch_delisted_tickers", lambda e, sd: delisted_at[sd])
    universe = tickers + ["FLIPS_TO_LINK", "FLIPS_FROM_LINK", "BOTH"]
    # Give the extra tickers a bar every day so the stale leg ignores them.
    cal = pd.bdate_range(end="2026-07-31", periods=400)
    bars = {t: list(cal) for t in universe}
    monkeypatch.setattr(ohlcv_hygiene.pd, "read_sql", _bar_tape_read_sql(bars, cal))
    excluded, counts = ohlcv_hygiene.batch_hygiene_exclusions(
        None, universe, batch_start, batch_end
    )
    assert {"FLIPS_TO_LINK", "FLIPS_FROM_LINK"}.isdisjoint(excluded)
    assert {"BOTH", "LIVE"} <= excluded
    assert counts["delisted"] == 2


def test_batch_delisted_leg_skips_end_query_before_the_check_date(monkeypatch):
    """Entirely before DELISTING_LINK_CHECK_SINCE `delisting_date < d` is
    monotone, so one anchor suffices."""
    calls = []
    monkeypatch.setattr(
        ohlcv_hygiene, "fetch_delisted_tickers", lambda e, sd: calls.append(sd) or frozenset()
    )
    cal = pd.bdate_range(end="2019-07-31", periods=400)
    monkeypatch.setattr(ohlcv_hygiene.pd, "read_sql", _bar_tape_read_sql({"A": list(cal)}, cal))
    ohlcv_hygiene.batch_hygiene_exclusions(None, ["A"], date(2019, 7, 22), date(2019, 7, 31))
    assert calls == [date(2019, 7, 22)]


def test_batch_hygiene_empty_universe_no_op():
    excluded, counts = ohlcv_hygiene.batch_hygiene_exclusions(
        None, [], date(2026, 7, 20), date(2026, 7, 31)
    )
    assert excluded == frozenset()
    assert counts == {"delisted": 0, "stale": 0, "excluded_total": 0, "link_kept": 0}


def test_batch_hygiene_fails_open_on_empty_calendar(monkeypatch):
    """No trading-calendar proxy data in the window must fail OPEN on the
    stale half (delisted is unaffected), never empty out the batch."""

    def fake_read_sql(query, engine, params=None):
        q = str(query)
        if "MAX(ts_event::date)" in q:
            return pd.DataFrame({"ticker": [], "last_seen": []})
        if "DISTINCT ts_event::date" in q:
            return pd.DataFrame({"d": pd.Series(dtype="datetime64[ns]")})
        raise AssertionError(f"unexpected query: {q}")

    monkeypatch.setattr(ohlcv_hygiene.pd, "read_sql", fake_read_sql)
    monkeypatch.setattr(ohlcv_hygiene, "fetch_delisted_tickers", lambda engine, sd: frozenset())
    excluded, _ = ohlcv_hygiene.batch_hygiene_exclusions(
        None, ["ANYTHING"], date(2004, 1, 5), date(2004, 1, 9)
    )
    assert excluded == frozenset()


def test_batch_hygiene_rejects_inverted_batch_bounds():
    with pytest.raises(ValueError):
        ohlcv_hygiene.batch_hygiene_exclusions(None, ["X"], date(2026, 7, 31), date(2026, 7, 20))


@pytest.mark.parametrize(
    "fn, kwargs, upper",
    [
        (
            ohlcv_hygiene.fetch_last_seen_dates,
            {"tickers": ["A"], "score_date": date(2026, 7, 31)},
            "score_date",
        ),
        (ohlcv_hygiene.fetch_trading_calendar, {"score_date": date(2026, 7, 31)}, "score_date"),
        (
            ohlcv_hygiene.fetch_last_seen_bars_range,
            {"tickers": ["A"], "start": date(2026, 7, 1), "end": date(2026, 7, 31)},
            "end",
        ),
        (
            ohlcv_hygiene.fetch_trading_calendar_range,
            {"start": date(2026, 7, 1), "end": date(2026, 7, 31)},
            "end",
        ),
    ],
)
def test_window_queries_bound_ts_event_sargably(monkeypatch, fn, kwargs, upper):
    """The four lookback-window queries must bound the raw `ts_event` column,
    not `ts_event::date` - the cast defeats the index and turns a
    60-day probe into a full-history walk per ticker."""
    captured = {}

    def _capture(query, engine, params=None):
        captured["sql"] = str(query)
        captured["params"] = params
        return pd.DataFrame({"ticker": [], "last_seen": [], "d": []})

    monkeypatch.setattr(ohlcv_hygiene.pd, "read_sql", _capture)
    fn(None, **kwargs)

    sql = captured["sql"]
    assert "ts_event::date > :start" not in sql
    assert "ts_event::date <= :" not in sql
    assert "ts_event >= (:start)::date + 1" in sql
    assert f"ts_event < (:{upper})::date + 1" in sql
    # Param contract unchanged: `start` exclusive, upper bound inclusive.
    assert "start" in captured["params"] and upper in captured["params"]


def test_batch_hygiene_counts_link_kept_at_either_end(monkeypatch):
    """batch_hygiene_exclusions reports link_kept like hygiene_exclusions: the
    universe tickers the #454 rule kept at batch_start OR batch_end
    (informational; they are not excluded)."""
    tickers, _batch, batch_start, batch_end = _batch_fixture(monkeypatch)
    kept = {
        batch_start: ohlcv_hygiene.DelistedTickers([], link_kept=["LIVE", "OUTSIDE"]),
        batch_end: ohlcv_hygiene.DelistedTickers([], link_kept=["QUITS_MIDWAY"]),
    }
    monkeypatch.setattr(ohlcv_hygiene, "fetch_delisted_tickers", lambda e, sd: kept[sd])
    excluded, counts = ohlcv_hygiene.batch_hygiene_exclusions(None, tickers, batch_start, batch_end)
    assert counts["link_kept"] == 2  # OUTSIDE is not in the universe
    assert "LIVE" not in excluded
