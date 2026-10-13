"""
Data quality asset checks (meridian-factors code location, Meridian #144).

Seven checks attached to composite_scores:

1. check_dq_row_coverage      - latest-partition row count in daily_scores is within
                                 an expected band (a minimum number of tickers
                                 per group per day).
2. check_dq_zscore_drift      - per factor-group mean ~0 and std within sane bounds;
                                 catches broken sector neutralisation.
3. check_dq_split_bias        - vol_surge lies within sane physical bounds
                                 (the split-bias class). Its
                                 drawdown_all / drawdown_52w legs were
                                 DELETED 2026-08-03 (#295 D5) - they bounded a
                                 quantity that is algebraically confined to
                                 [-1, 0] and so could never fire; (7) owns that
                                 class now. See _SPLIT_BIAS_BOUNDS.
4. check_dq_universe_drift    - scored universe size for the latest partition vs the
                                 prior partition does not jump beyond a threshold (the
                                 universe-alignment class of drift).
5. check_dq_composite_freshness - composite_scores has been materialised for the
                                 most recent expected trading weekday.
6. check_dq_liquid_unscored   - no meaningful set of tickers printed a LIQUID bar on
                                 the score date while missing from the scored
                                 cross-section (the reused-ticker hygiene
                                 class; the contradiction-shaped counterpart to
                                 check_dq_universe_drift's count delta).
7. check_dq_pinned_at_floor   - the share of the cross-section with
                                 drawdown_all / drawdown_52w pinned just
                                 INSIDE the bottom of their physical range
                                 stays below a baseline. A range check
                                 cannot catch a value pinned just inside the
                                 range (see _PINNED_FLOOR_MAX_PCT).

Tier 1 inlining (Meridian #285): these checks used to be standalone
`@asset_check` step pods. The pass/fail DECISIONS still live in pure,
DB-free helper functions (unchanged, still unit-tested directly):

  coverage_ok(n)                  -> bool
  zscore_drift_ok(mean, std)      -> tuple[bool, str]
  split_bias_ok(factor, value)    -> tuple[bool, str]
  pinned_at_floor_ok(factor, pinned, total) -> tuple[bool, str]
  universe_drift_ok(curr, prev)   -> tuple[bool, str]
  liquid_unscored_ok(n)           -> tuple[bool, str]
  composite_freshness_ok(latest, expected, max_lag_days) -> tuple[bool, str]

but the per-date evaluation and DB access now live in `evaluate_*`
functions that `composite_scores` (assets.py) calls once per partition
inside its existing loop, reusing DataFrames it already has in memory
(`daily`) rather than re-querying. `evaluate_dq_composite_freshness` is the
one exception: it's a global (non-partition-scoped) probe, so it runs once
after the loop with a single query.
"""

from __future__ import annotations

import functools
from datetime import date, datetime, timedelta
from datetime import timezone as _tz

import pandas as pd
from sqlalchemy import text

from dagster import (
    AssetCheckResult,
    AssetCheckSeverity,
    MetadataValue,
)

# Thresholds (documented here so they appear in diffs when tuned)
# Row coverage: we require a conservative minimum number of distinct tickers
# per factor group per day (see the constant below) to catch major upstream gaps
# while tolerating normal universe churn (new listings, delistings, weekends
# with stale data). The absolute minimum is intentionally conservative so that
# a large drop (e.g. an upstream LDP partition gap) triggers a warning before
# the composite is distorted.
_COVERAGE_MIN_TICKERS = 2_000  # illustrative value

# Z-score drift: after cross-sectional sector neutralisation the per-group
# distribution should be roughly N(0,1). We allow |mean| <= 0.3 and std in
# [0.5, 2.0] matching factor_checks.check_factor_distribution (shared bands).
_ZSCORE_MEAN_MAX_ABS = 0.3
_ZSCORE_STD_LOW = 0.5
_ZSCORE_STD_HIGH = 2.0

# Split-bias: physical-sanity bounds for the price-level factors that are
# exposed to split-adjustment errors.
#
# DELETED 2026-08-03 (Meridian #295 D5): the `drawdown_all` and
# `drawdown_52w` legs. They asserted an ALGEBRAIC IDENTITY, not a
# property of the data:
#     drawdown_all = (close - ath) / ath with close <= ath by construction,
#     so the quantity is confined to [-1, 0] and a [-1, 1] bound cannot be
#     violated by any corruption of any magnitude - the SAME corrupted series
#     produces both numerator and denominator.
# The check therefore had a ZERO probability of firing. That is strictly
# worse than no check: it converts "unmonitored" into "monitored and clean".
# `check_dq_pinned_at_floor` owns this defect class properly, by testing the
# SHAPE of the distribution - the share of the cross-section within 1% of the
# floor - which is a statistic the corruption actually moves.
#
# What REMAINS here is the one leg that is not an identity:
#   vol_surge      - a short-window / long-window volume ratio; unbounded
#                       above, so a bound is a real assertion. The cap is
#                       loose: it catches pathological split-bias
#                       outliers without false-positives on genuine volume
#                       spikes. NOTE the bound is un-tuned and illustrative -
#                       retightening or converting it to a distribution-shape
#                       test is tracked as design-doc D5 remainder, phase 3, and
#                       NOT done here because it needs its own
#                       calibration against volume history.
_SPLIT_BIAS_BOUNDS: dict[str, tuple[float, float]] = {
    "vol_surge": (0.0, 75.0),
}

# Pinned-at-the-floor: _SPLIT_BIAS_BOUNDS above is a RANGE check, and a range
# check cannot see a value pinned just INSIDE the range. A corrupted split
# factor drives the price ratio toward 0, so drawdown_all would collapse onto
# its floor, which sits comfortably inside a [-1, 1] bound. This check tests
# the SHAPE of that failure instead: the share of the cross-section within 1%
# of the floor.
#
# A ticker legitimately 99%+ below its high exists (post-blowup microcaps),
# so the signal is the SHARE, not any single value. The threshold must clear
# the healthy share and a crash-era share (these checks also run on
# historical backfills, so crash eras are the false-positive risk), and sit
# below the share a corrupted day produces. Tune it against fresh
# measurements of factors.daily_scores if that balance changes.
_PINNED_FLOOR_VALUE = -0.99
_PINNED_FLOOR_MAX_PCT: dict[str, float] = {
    "drawdown_all": 2.0,
    "drawdown_52w": 2.0,
}

# Universe drift: a sudden >20% change in the scored ticker count between
# consecutive trading days indicates a universe alignment problem. Normal
# day-over-day churn is far smaller.
_UNIVERSE_DRIFT_MAX_PCT = 20.0

# Liquid-but-unscored (the reused-ticker hygiene class).
#
# Universe drift alone is an ambiguous signal: a large change in the scored
# universe can have several unrelated causes at once, and part of it can be
# CORRECT (for example newly-visible dead tickers being properly excluded by
# the hygiene guard). The bug class is the residual: reused ticker symbols
# whose live issuer inherited a dead predecessor's delisting_date. A pure
# count-delta alarm therefore cries wolf on every legitimate upstream universe
# change, which is precisely how a real defect hides inside an "expected"
# drift.
#
# This check watches the CONTRADICTION instead, which no legitimate universe
# change produces: a ticker that printed a genuinely LIQUID bar on the score
# date but is missing from the day's scored cross-section. Dead names fail
# this test by construction - their post-delisting ghost prints are thin or
# zero-volume - so the ingest of delisted tickers would not trip it,
# while wrongly-excluded live names would.
_LIQUID_UNSCORED_MIN_DOLLAR_VOLUME = 1_000_000.0

# Tolerance before the contradiction is reported. Never expected to be zero:
# a handful of liquid names legitimately sit outside the scored universe on
# any given day (fresh IPOs before their first fundamentals land, ADRs and
# other instruments the universe filter excludes by design). A genuine
# exclusion bug puts far more names than this in the bucket.
_LIQUID_UNSCORED_MAX_TICKERS = 50


# Exception-safety decorator: all @asset_check functions are WARN-only
# monitoring checks and must never raise into the run.


def _warn_on_error(fn):
    """Wrap an asset_check function so any uncaught exception returns WARN.

    A monitoring check must never fail the run.  This decorator catches any
    exception (including transient DB errors or schema bugs) and returns an
    AssetCheckResult with passed=False, severity=WARN so the run continues.

    The result carries `outcome="errored"` (Meridian #295 D3.2), and
    `notify_errored_check` is the notification side of it; checks/alerting.py
    wires both for the standalone checks.

    Apply directly UNDER @asset_check so that @asset_check sees the preserved
    function name (functools.wraps copies __name__ / __wrapped__ etc.).
    """

    @functools.wraps(fn)
    def wrapper(context):
        try:
            result = fn(context)
        except Exception as e:  # noqa: BLE001 - monitoring check must never fail the run
            try:
                from meridian_pipeline.notify import notify_errored_check

                notify_errored_check(
                    "composite_scores", fn.__name__, e, log=getattr(context, "log", None)
                )
            except Exception:  # noqa: BLE001 - alerting must never fail the run
                pass
            return AssetCheckResult(
                passed=False,
                severity=AssetCheckSeverity.WARN,
                metadata={
                    "outcome": MetadataValue.text("errored"),
                    "error": MetadataValue.text(f"{type(e).__name__}: {e}"[:500]),
                    "note": MetadataValue.text(
                        "check RAISED - this is missing coverage, not a data finding"
                    ),
                },
            )
        return result

    return wrapper


# Pure pass/fail helpers (DB-free, unit-testable)


def coverage_ok(n: int) -> bool:
    """Return True if the partition row count meets the minimum threshold.

    Args:
        n: number of distinct tickers scored for the partition.

    Returns:
        True when n >= _COVERAGE_MIN_TICKERS.
    """
    return n >= _COVERAGE_MIN_TICKERS


def zscore_drift_ok(mean: float, std: float) -> tuple[bool, str]:
    """Return (passed, message) for a single group's z-score distribution.

    Args:
        mean: average z-score across tickers for the group.
        std:  standard deviation of z-scores for the group.

    Returns:
        (True, "") if within bounds, else (False, human-readable reason).
    """
    if abs(mean) > _ZSCORE_MEAN_MAX_ABS:
        return False, f"mean={mean:.3f} outside [-{_ZSCORE_MEAN_MAX_ABS}, {_ZSCORE_MEAN_MAX_ABS}]"
    if std < _ZSCORE_STD_LOW or std > _ZSCORE_STD_HIGH:
        return False, f"std={std:.3f} outside [{_ZSCORE_STD_LOW}, {_ZSCORE_STD_HIGH}]"
    return True, ""


def split_bias_ok(factor: str, value: float) -> tuple[bool, str]:
    """Return (passed, message) for a single risk price-level factor value.

    Only checks factors listed in _SPLIT_BIAS_BOUNDS; unknown factors pass.

    Args:
        factor: factor_name string (e.g. "drawdown_all").
        value:  numeric value to validate.

    Returns:
        (True, "") if within bounds or unknown factor, else (False, reason).
    """
    if factor not in _SPLIT_BIAS_BOUNDS:
        return True, ""
    lo, hi = _SPLIT_BIAS_BOUNDS[factor]
    if value < lo or value > hi:
        return (
            False,
            f"{factor}={value:.4f} outside [{lo}, {hi}]",
        )
    return True, ""


def universe_drift_ok(curr: int, prev: int) -> tuple[bool, str]:
    """Return (passed, message) for ticker-count drift between two partitions.

    Args:
        curr: ticker count for the current (latest) partition.
        prev: ticker count for the prior partition.

    Returns:
        (True, "") if drift is within threshold, else (False, reason).
    """
    if prev == 0:
        return True, ""  # no prior data - nothing to compare
    pct = abs(curr - prev) / prev * 100.0
    if pct > _UNIVERSE_DRIFT_MAX_PCT:
        return (
            False,
            f"universe drifted {pct:.1f}% (curr={curr}, prev={prev}); threshold={_UNIVERSE_DRIFT_MAX_PCT}%",
        )
    return True, ""


# Meridian#298. A factor's row count may fall to this fraction of its
# own trailing median before we call it a collapse. Generous: real
# cross-sections breathe with the universe, and a check that cries wolf gets
# muted, which is how a real gap goes unseen.
_CONTINUITY_MIN_FRACTION = 0.5
# Below this trailing median a factor is too small for a ratio to mean anything;
# only a fall to exactly zero is reported.
_CONTINUITY_MIN_MEDIAN = 5


def factor_continuity_ok(factor: str, curr: int, trailing_median: float) -> tuple[bool, str]:
    """Return (passed, message) for one factor's row count against its own history.

    Why this exists, and why it is per-factor rather than global (Meridian#298):
    a cut to the scoreable universe costs broad factors a small share of their
    cross-section, which reads as noise, but can cost a narrow factor 100%, in
    which case it is silently omitted from the output.

    Every other data-quality check in this module asks "do the values look
    right". None of them can see this, because there are no values - range,
    null and distribution checks all pass vacuously on an empty set. And a
    single global row-count threshold cannot catch it either: one calibrated to
    notice a narrow factor vanishing would fire constantly on the broad
    ones. Judging each factor against ITS OWN trailing median is what makes the
    narrow factors visible, and they are the ones that die first.

    Args:
        factor: factor name, for the message.
        curr: row count for this factor on the current partition.
        trailing_median: median row count over the recent trailing window.

    Returns:
        (True, "") if the count is plausible, else (False, reason).
    """
    if trailing_median <= 0:
        return True, ""  # never seen before - nothing to compare against
    if curr == 0:
        return (
            False,
            f"{factor}: produced ZERO rows (trailing median {trailing_median:.0f}). "
            "A factor that emits nothing leaves no values for any other check to "
            "inspect - look upstream at the universe before the factor's own code.",
        )
    if trailing_median < _CONTINUITY_MIN_MEDIAN:
        return True, ""  # too small for a ratio to be meaningful
    frac = curr / trailing_median
    if frac < _CONTINUITY_MIN_FRACTION:
        return (
            False,
            f"{factor}: {curr} rows is {frac:.0%} of its trailing median "
            f"({trailing_median:.0f}); threshold {_CONTINUITY_MIN_FRACTION:.0%}",
        )
    return True, ""


def pinned_at_floor_ok(factor: str, pinned: int, total: int) -> tuple[bool, str]:
    """Return (passed, message) for one factor's pinned-at-the-floor share.

    Complements split_bias_ok: that one asks "is any value outside the
    physical range", this one asks "are implausibly many values sitting just
    INSIDE the bottom of that range" - the shape upstream adj_close
    corruption produces (see _PINNED_FLOOR_MAX_PCT).

    Only checks factors listed in _PINNED_FLOOR_MAX_PCT; unknown factors and
    empty cross-sections pass.

    Args:
        factor: factor_name string (e.g. "drawdown_all").
        pinned: count of non-null values below _PINNED_FLOOR_VALUE.
        total:  count of non-null values for the factor on the date.

    Returns:
        (True, "") if the share is within threshold, else (False, reason).
    """
    if factor not in _PINNED_FLOOR_MAX_PCT:
        return True, ""
    if total <= 0:
        return True, ""
    max_pct = _PINNED_FLOOR_MAX_PCT[factor]
    pct = pinned / total * 100.0
    if pct > max_pct:
        return (
            False,
            f"{factor}: {pinned}/{total} ({pct:.2f}%) below {_PINNED_FLOOR_VALUE}; "
            f"threshold={max_pct}% - suspect upstream adj_close corruption",
        )
    return True, ""


# Helper shared by all checks below


def _date_range_label(kr) -> str:
    if kr.start == kr.end:
        return kr.start
    return f"{kr.start}..{kr.end}"


# Tier 1 inlined checks (#285): pure, DB-free evaluators built on the
# decision helpers above. Called by composite_scores (assets.py) once per
# partition against `daily` (the DataFrame it already read for the date) -
# no new DB round-trips for row_coverage/zscore_drift/split_bias.
# check_dq_universe_drift needs the prior scored trading day's ticker count,
# which composite_scores fetches with one small reused-engine query per date
# (same cost as the original check; still folds into the existing pod).


def evaluate_dq_row_coverage(daily: "pd.DataFrame") -> dict:
    """Distinct ticker count for the date must be >= _COVERAGE_MIN_TICKERS."""
    n_tickers = int(daily["ticker"].nunique()) if not daily.empty else 0
    return {
        "passed": coverage_ok(n_tickers),
        "n_tickers": n_tickers,
        "coverage_threshold": _COVERAGE_MIN_TICKERS,
    }


def evaluate_dq_zscore_drift(daily: "pd.DataFrame") -> dict:
    """Per-group z-score mean/std must stay within the shared sane bands."""
    z = daily[daily["z_score"].notna()]
    if z.empty:
        return {"passed": True, "note": "no z-scores for this date"}

    violations: list[str] = []
    stats = z.groupby("group_name")["z_score"].agg(["mean", "std"])
    for group_name, row in stats.iterrows():
        std_z = row["std"]
        ok, msg = zscore_drift_ok(float(row["mean"]), float(std_z) if not pd.isna(std_z) else 1.0)
        if not ok:
            violations.append(f"{group_name}: {msg}")

    return {"passed": len(violations) == 0, "violations": violations}


def evaluate_dq_split_bias(daily: "pd.DataFrame") -> dict:
    """The split-bias-prone risk price-level factors must stay within their
    physical-sanity bounds. WARNs if >0.5% of non-null rows violate for any
    one factor.

    Scoped to _SPLIT_BIAS_BOUNDS, which is `vol_surge` only since #295 D5
    deleted the two legs that asserted an algebraic identity - see that
    constant's comment. The ATH-distance factors are now covered by
    evaluate_dq_pinned_at_floor, which tests a statistic the defect moves.
    """
    factor_results: dict[str, dict] = {}
    any_violation = False

    for factor in _SPLIT_BIAS_BOUNDS:
        sub = daily[(daily["factor_name"] == factor) & daily["raw_value"].notna()]
        n = int(len(sub))
        bad = 0
        if n:
            bad = int(sum(1 for v in sub["raw_value"] if not split_bias_ok(factor, v)[0]))
        pct = (bad / n * 100.0) if n > 0 else 0.0
        passed_factor = pct <= 0.5
        if not passed_factor:
            any_violation = True
        factor_results[factor] = {
            "total": n,
            "out_of_range": bad,
            "pct": round(pct, 3),
            "passed": passed_factor,
        }

    return {"passed": not any_violation, "factors": factor_results}


def evaluate_dq_pinned_at_floor(daily: "pd.DataFrame") -> dict:
    """The ATH-distance factors must not have an implausible share of the
    cross-section pinned just inside the bottom of their physical range.

    Reuses the `daily` DataFrame composite_scores already holds for the date
    (no new DB round-trip), exactly like evaluate_dq_split_bias. WARN-only.
    """
    factor_results: dict[str, dict] = {}
    violations: list[str] = []

    for factor in _PINNED_FLOOR_MAX_PCT:
        sub = daily[(daily["factor_name"] == factor) & daily["raw_value"].notna()]
        n = int(len(sub))
        pinned = int((sub["raw_value"] < _PINNED_FLOOR_VALUE).sum()) if n else 0
        ok, msg = pinned_at_floor_ok(factor, pinned, n)
        if not ok:
            violations.append(msg)
        factor_results[factor] = {
            "total": n,
            "pinned": pinned,
            "pct": round((pinned / n * 100.0) if n > 0 else 0.0, 3),
            "threshold_pct": _PINNED_FLOOR_MAX_PCT[factor],
            "passed": ok,
        }

    return {
        "passed": len(violations) == 0,
        "factors": factor_results,
        "violations": violations,
        "floor": _PINNED_FLOOR_VALUE,
    }


def evaluate_dq_universe_drift(curr_tickers: int, prev_tickers: "int | None") -> dict:
    """Distinct ticker count vs the prior scored trading day must not jump
    beyond _UNIVERSE_DRIFT_MAX_PCT.
    """
    if prev_tickers is None:
        return {"passed": True, "note": "no prior scored trading day available"}

    ok, msg = universe_drift_ok(curr_tickers, prev_tickers)
    return {
        "passed": ok,
        "curr": curr_tickers,
        "prev": prev_tickers,
        "detail": msg,
    }


def liquid_unscored_ok(n_liquid_unscored: int) -> tuple[bool, str]:
    """Decision helper (pure): at most _LIQUID_UNSCORED_MAX_TICKERS names may
    print a liquid bar on the score date without being scored."""
    if n_liquid_unscored > _LIQUID_UNSCORED_MAX_TICKERS:
        return (
            False,
            f"{n_liquid_unscored} ticker(s) printed a bar with dollar volume >= "
            f"${_LIQUID_UNSCORED_MIN_DOLLAR_VOLUME:,.0f} on the score date but are "
            f"missing from the scored cross-section; threshold="
            f"{_LIQUID_UNSCORED_MAX_TICKERS}",
        )
    return True, ""


def evaluate_dq_liquid_unscored(engine, score_date: date, scored_tickers: set) -> dict:
    """Tickers that traded LIQUIDLY on score_date but were not scored.

    The targeted counterpart to check_dq_universe_drift: drift measures a
    count delta (ambiguous - a legitimate upstream universe change moves it
    just as hard as a defect does), while this measures a contradiction that
    only a defect produces. See the _LIQUID_UNSCORED_* constants above for
    the incident that motivated it.

    ONE indexed single-day query on ohlcv.prices (the (ts_event, resolution)
    index), evaluated per partition inside composite_scores' existing loop.
    """
    next_day = score_date + timedelta(days=1)
    df = pd.read_sql(
        text("""
            SELECT p.ticker, (p.raw_close * p.volume) AS dollar_volume
            FROM ohlcv.prices p
            WHERE p.resolution = 'daily'
              AND p.security_type IN ('equity', 'etf')
              AND p.ts_event >= :score_date
              AND p.ts_event < :next_day
              AND p.volume > 0
              AND p.raw_close IS NOT NULL
              AND (p.raw_close * p.volume) >= :min_dollar_volume
        """),
        engine,
        params={
            "score_date": score_date,
            "next_day": next_day,
            "min_dollar_volume": _LIQUID_UNSCORED_MIN_DOLLAR_VOLUME,
        },
    )
    if df.empty:
        return {"passed": True, "note": "no liquid bars for this date"}

    missing = df[~df["ticker"].isin(scored_tickers)].sort_values("dollar_volume", ascending=False)
    n_missing = int(len(missing))
    ok, msg = liquid_unscored_ok(n_missing)
    return {
        "passed": ok,
        "n_liquid": int(len(df)),
        "n_liquid_unscored": n_missing,
        "examples": missing["ticker"].head(20).tolist(),
        "detail": msg,
    }


def evaluate_dq_factor_continuity(engine, daily: "pd.DataFrame", score_date) -> dict:
    """Per-factor row-count continuity against each factor's own recent history.

    Complements evaluate_dq_row_coverage, which asks whether the PARTITION has
    enough rows overall. This asks whether each FACTOR still exists - the
    failure mode row-coverage cannot see, because one narrow factor vanishing
    barely moves a partition-level total.

    See factor_continuity_ok for the failure mode (Meridian#298): a universe
    cut that costs broad factors a small share can cost a narrow factor 100%,
    and that factor is then silently omitted.

    The trailing window is the 20 most recent PRIOR score dates that carry any
    rows, so the baseline follows trading days rather than calendar days and a
    long weekend does not depress the median.
    """
    if daily is None or daily.empty:
        return {"passed": True, "note": "no rows in this partition"}

    curr = daily.groupby("factor_name").size().to_dict()

    hist = pd.read_sql(
        text("""
            WITH recent AS (
                SELECT DISTINCT score_date
                FROM factors.daily_scores
                WHERE score_date < :score_date
                ORDER BY score_date DESC
                LIMIT 20
            )
            SELECT factor_name, score_date, count(*) AS n
            FROM factors.daily_scores
            WHERE score_date IN (SELECT score_date FROM recent)
            GROUP BY factor_name, score_date
        """),
        engine,
        params={"score_date": score_date},
    )
    if hist.empty:
        return {"passed": True, "note": "no trailing history to compare against"}

    medians = hist.groupby("factor_name")["n"].median().to_dict()

    # Union of the history and today's factors: a factor that vanished
    # ENTIRELY is absent from `curr`, and that is precisely the case worth
    # catching. Iterating only over what is present would make the check blind
    # to its own motivating defect.
    failures: list[str] = []
    for factor in sorted(set(medians) | set(curr)):
        ok, msg = factor_continuity_ok(
            factor, int(curr.get(factor, 0)), float(medians.get(factor, 0))
        )
        if not ok:
            failures.append(msg)

    if failures:
        return {
            "passed": False,
            "n_factors_flagged": len(failures),
            "detail": "; ".join(failures[:6]),
        }
    return {"passed": True, "n_factors_checked": len(medians)}


def evaluate_dq_composite_freshness(engine) -> dict:
    """Global (non-partition-scoped) freshness probe: factors.composite_scores
    must have a row for the most recent expected trading weekday. Runs once
    per composite_scores materialisation (after the per-date loop), not
    once per partition - a single cheap MAX(score_date) query.
    """
    row = pd.read_sql(
        text("SELECT MAX(score_date) AS latest FROM factors.composite_scores"),
        engine,
    ).iloc[0]

    if row["latest"] is None:
        return {"passed": False, "note": "no rows in factors.composite_scores"}

    today_utc = datetime.now(_tz.utc).date()
    latest = row["latest"] if isinstance(row["latest"], date) else row["latest"].date()
    expected = _latest_expected_trading_day(today_utc)
    passed, msg = composite_freshness_ok(latest, expected)

    return {
        "passed": passed,
        "latest_score_date": str(latest),
        "expected_trading_day": str(expected),
        "lag_days": (expected - latest).days,
        "detail": msg if msg else "fresh",
    }


# Freshness helpers - composite_scores must not be stale by > 1 calendar day
# behind the most recent expected trading weekday (weekends are skipped; a
# Friday close is expected to persist until Monday morning without alarming).
# evaluate_dq_composite_freshness() above is NOT partition-scoped - it runs
# once per composite_scores materialisation as a global freshness probe.


def _latest_expected_trading_day(as_of: date) -> date:
    """Return the most recent weekday on or before `as_of`.

    Very conservative approximation (ignores US market holidays). Used only
    to determine whether the composite is stale - false negatives (holiday
    treated as trading day) result in at most one extra WARN per holiday.
    """
    d = as_of
    # Roll back to Friday if Saturday/Sunday.
    while d.weekday() >= 5:  # 5=Saturday, 6=Sunday
        d -= timedelta(days=1)
    return d


def composite_freshness_ok(
    latest_date: date, expected_date: date, max_lag_days: int = 1
) -> tuple[bool, str]:
    """Return (passed, message) for composite_scores freshness.

    Args:
        latest_date:  most recent score_date in factors.composite_scores.
        expected_date: the latest expected trading day (weekday <= today).
        max_lag_days: number of business days of lag tolerated before warning.

    Returns:
        (True, "") if within tolerance, else (False, human-readable reason).
    """
    lag = (expected_date - latest_date).days
    if lag <= max_lag_days:
        return True, ""
    return (
        False,
        f"composite_scores latest={latest_date} is {lag} calendar days behind expected={expected_date}",
    )
