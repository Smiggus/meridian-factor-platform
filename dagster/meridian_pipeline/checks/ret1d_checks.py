"""
ret_1d input-quality asset check (meridian-factors code location).

Attached to momentum_scores (daily-partitioned). momentum_scores chains
ret_1d straight out of ohlcv.prices, so a corrupted ret_1d column upstream
silently poisons momentum (and, downstream, the composite and the factor
risk model). This check is a tripwire: it inspects
the raw ret_1d cross-section for the partition date(s) BEFORE the corruption
propagates and FAILs the partition (blocking=True) if the day looks broken.

Failure shape: a bad ret_1d recompute produces cross-sections with
many returns beyond |0.5| and a grossly inflated cross-sectional std - values
that are physically impossible for a daily equity cross-section.

Calibration reference (shapes, not measurements):
    normal day       : small std, no extremes, |ret_1d| <= 50          -> PASS
    broad corruption : inflated std or many extremes                   -> FAIL

The single-name min/max gate is deliberately wide: a tight bound would treat
genuine biotech/M&A/SPAC moves as corruption and halt the composite on live,
liquid, correctly-scored names. The gate is aligned to the upstream platform's
cleaning boundary: |ret_1d| > 50 is NULLed upstream and everything at/below 50
is kept, so this gate blocks only |ret_1d| > 50 - no value the upstream
deliberately kept can halt the composite. On cleaned data the gate effectively
never trips (those rows are already NULL and excluded); it is a forward safety
net that mirrors the upstream rule. See the _RET_ABS_MIN / _RET_ABS_MAX
threshold comment below.

Broad-corruption detection stays: the std gate (std > 0.15) is now computed
over WINSORIZED ret_1d (clamped to [-1, +1]) so a single genuine large
single-name move does not inflate the cross-sectional std past the bound on
its own (one very large genuine move would otherwise drive std past the
bound); the extreme-count gate (> 10 names with |ret_1d| > 0.5, on RAW
values) remains the primary broad-corruption signal. Both still fire on the
broad-corruption shape (many liquid names extreme together on one day).

Note: checks use context.partition_key_range (not context.partition_key) so
they work in both single-partition (daily schedule) and multi-partition
batched (meridian_backfill_job) runs. Mirrors factor_checks / risk_metrics_checks.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
from sqlalchemy import text

from dagster import (
    AssetCheckResult,
    AssetCheckSeverity,
    MetadataValue,
    asset_check,
)
from meridian_pipeline.k8s_config import LIGHT_OP_TAGS
from meridian_pipeline.ohlcv_hygiene import (
    BOGUS_PRINT_SQL_GUARD,
    BOGUS_PRINT_SQL_PARAMS,
    MIN_DOLLAR_VOLUME,
    MIN_PRICE_FLOOR,
)
from meridian_pipeline.risk_metrics_asset import _get_engine

# ── Thresholds ──────────────────────────────────────────────────────────
# A daily equity cross-section is tight: std ~0.03-0.05, essentially no
# returns beyond +/-50%. These bounds sit comfortably above normal-day noise
# (incl. earnings-day single-name moves) while catching the corruption.
#
# The std is computed over WINSORIZED ret_1d (clamped to [-1.0, +1.0];
# _STD_WINSOR_ABS below) so a single genuine large single-name move does
# NOT trip it. Without winsorizing, one very large genuine biotech/M&A/SPAC
# move can alone drive the cross-sectional std past the bound - the
# same false-positive class as the old min/max gate, just via std.
# Clamping the outlier to +100% for the std
# calculation removes that single-name sensitivity; BROAD corruption (many
# names extreme) still pushes the winsorized std past 0.15
# and still blocks. The extreme-count gate (below) remains the primary
# broad-corruption signal and is unchanged.
_STD_MAX = 0.15  # cross-sectional stddev of WINSORIZED ret_1d
_STD_WINSOR_ABS = 1.0  # clamp ret_1d to [-1, +1] before the std (single-outlier robust)
_EXTREME_ABS = 0.5  # |ret_1d| beyond this counts as an extreme
_EXTREME_COUNT_MAX = 10  # tolerate a handful of legit single-name moves
# Below this row count a date is a partial / mid-ingestion day; coverage is
# owned by check_factor_coverage, so we skip (do NOT fail) thin days here.
_MIN_ROWS = 500

# Single-name min/max gate (Meridian #275 follow-up). A tight bound such as
# +200% is NOT non-physical for small/mid-cap names (trial readouts, M&A,
# de-SPACs), so after the universe/liquidity floor the bounds are set wide;
# treating legitimate large moves as errors would halt the composite on
# real market moves.
#
# The single-name gate is now ALIGNED EXACTLY to LDP's cleaning boundary:
# LDP #182 NULLs ret_1d rows with |ret_1d| > 50 in ohlcv.prices and KEEPS
# everything at/below 50 (genuine moves of several hundred percent and
# beyond are all preserved). This
# gate MUST match that boundary so there is NO gap where a value LDP
# deliberately kept would still halt the composite here. Both sides gate
# on |ret_1d| > 50:
#   - max: block max_ret > 50.0 (LDP keeps everything <= 50).
#   - min: block min_ret < -50.0 (LDP keeps everything >= -50; a real total
#     loss is exactly -1.0 and passes with wide margin).
# Because LDP already NULLed |ret_1d| > 50, those rows are NULL and excluded
# from the aggregate (the query's `ret_1d IS NOT NULL` filter), so on
# cleaned data this gate effectively never trips - it is a forward safety
# net that mirrors LDP's cleaning rule for any future bad print that slips
# in before LDP NULLs it.
# Broad-corruption detection is unaffected: _STD_MAX and _EXTREME_COUNT_MAX
# (unchanged above) still catch the broad-corruption shape (many liquid
# names moving together on one day), which is what this check exists to
# stop - a single extreme-but-real name should never block the composite.
_RET_ABS_MIN = -50.0  # aligned to LDP #182: |ret_1d| > 50 is NULLed upstream
_RET_ABS_MAX = 50.0  # aligned to LDP #182: LDP keeps everything <= 50

# Universe/liquidity floor (Meridian #275, LDP #182): a separate class of
# bad close prints in PG ohlcv.prices (sub-penny / zero-volume / illiquid
# shell prints, e.g. a sub-penny close -> ret_1d=-100%, then
# a huge positive ret_1d the next day on volume=0) trips this check on names nobody scores.
# The cross-sectional stats below are now computed over the SAME
# bogus/illiquid-print floor as the risk factor group (#259, extended in
# ohlcv_hygiene.py): close >= MIN_PRICE_FLOOR, volume > 0, dollar volume >=
# MIN_DOLLAR_VOLUME. This does NOT weaken the check's ability to catch
# genuine broad corruption - which hits the whole liquid
# cross-section, and the floor only removes rows that were never part of
# that liquid cross-section to begin with. See ohlcv_hygiene.py's docstring
# for the rationale of the floor.

# Historical-era gate: the thresholds above are tuned for the modern data era
# and the check exists to stop LIVE corruption propagating - blocking every
# historical backfill month on older vendor data quirks would yield no data
# at all for those windows.
# Violations on dates BEFORE this floor therefore downgrade to a passing WARN
# with full metadata (still visible in the UI/logs); any violation ON/AFTER
# the floor still fails the check (blocking) exactly as before.
_BLOCKING_FLOOR = "2015-01-01"


def _era_gate(bad_dates: list) -> tuple:
    """(passed, severity) for the era gate - see _BLOCKING_FLOOR's comment.

    Violations confined to pre-floor dates pass with WARN (metadata still
    carries them); any modern-era violation fails with ERROR (blocking).
    """
    from dagster import AssetCheckSeverity

    modern_bad = [b for b in bad_dates if str(b["score_date"]) >= _BLOCKING_FLOOR]
    historical_only = bool(len(bad_dates) > 0 and len(modern_bad) == 0)
    passed = bool(len(modern_bad) == 0)
    return passed, (AssetCheckSeverity.WARN if historical_only else AssetCheckSeverity.ERROR)


def _date_range_label(kr) -> str:
    if kr.start == kr.end:
        return kr.start
    return f"{kr.start}..{kr.end}"


def _evaluate_days(df: pd.DataFrame) -> tuple[list[dict], list[str], int]:
    """Pure evaluation of the per-date aggregate rows (score_date, n_rows,
    std_ret, n_extreme, min_ret, max_ret) against the thresholds.

    Extracted from check_ret1d_input_quality so it can be unit-tested
    without a database connection - `df` is expected to already reflect the
    universe/liquidity floor (BOGUS_PRINT_SQL_GUARD) applied upstream by the
    SQL query; this function only judges the resulting cross-section.

    Returns (bad_dates, skipped_partial, evaluated_count).
    """
    bad_dates: list[dict] = []
    skipped_partial: list[str] = []
    evaluated = 0

    for _, row in df.iterrows():
        d = str(row["score_date"])
        n_rows = int(row["n_rows"])
        if n_rows < _MIN_ROWS:
            skipped_partial.append(d)
            continue

        evaluated += 1
        std_ret = float(row["std_ret"]) if row["std_ret"] is not None else 0.0
        n_extreme = int(row["n_extreme"])
        min_ret = float(row["min_ret"])
        max_ret = float(row["max_ret"])

        reasons = []
        if std_ret > _STD_MAX:
            reasons.append(f"std={std_ret:.4f}>{_STD_MAX}")
        if n_extreme > _EXTREME_COUNT_MAX:
            reasons.append(f"extremes={n_extreme}>{_EXTREME_COUNT_MAX}")
        if min_ret < _RET_ABS_MIN:
            reasons.append(f"min={min_ret:.4f}<{_RET_ABS_MIN}")
        if max_ret > _RET_ABS_MAX:
            reasons.append(f"max={max_ret:.4f}>{_RET_ABS_MAX}")

        if reasons:
            bad_dates.append(
                {
                    "score_date": d,
                    "n_rows": n_rows,
                    "std_ret": round(std_ret, 4),
                    "n_extreme": n_extreme,
                    "min_ret": round(min_ret, 4),
                    "max_ret": round(max_ret, 4),
                    "reasons": ", ".join(reasons),
                }
            )

    return bad_dates, skipped_partial, evaluated


@asset_check(
    asset="momentum_scores",
    description=(
        "Tripwire for upstream ret_1d corruption. Inspects the raw "
        "ret_1d cross-section in ohlcv.prices - restricted to equity/etf rows "
        "that clear the same bogus/illiquid-print floor as the risk factor "
        "group (close >= $0.01, volume > 0, dollar volume >= $10k; #259/#275) "
        "- for the partition date(s) and FAILs (blocking) if any full trading "
        "day has cross-sectional std (over ret_1d winsorized to [-1, +1], so "
        "a single legit large move does not trip it) > 0.15, more than 10 "
        "returns with |ret_1d| > 0.5, or a single-name min/max with "
        "|ret_1d| > 50 "
        "(#275 recalibration: the single-name gate is aligned exactly to "
        "LDP #182's cleaning boundary - LDP NULLs |ret_1d| > 50 and keeps "
        "everything <= 50, so this gate never fires on a value LDP kept and "
        "acts as a forward safety net mirroring LDP's rule). Dates with "
        "fewer than 500 rows (post-floor) are treated as partial and "
        "skipped "
        "(coverage is owned by check_factor_coverage). Blocking so a corrupt "
        "day fails the partition before the composite consumes it."
    ),
    blocking=True,
    op_tags=LIGHT_OP_TAGS,  # right-sized resources (Meridian#82) - cheap PG read
)
def check_ret1d_input_quality(context) -> AssetCheckResult:
    kr = context.partition_key_range
    start_date = date.fromisoformat(kr.start)
    end_date = date.fromisoformat(kr.end)
    label = _date_range_label(kr)
    engine = _get_engine()
    try:
        # One aggregate row per date: cross-sectional std, extreme count,
        # min/max and the row count (for the partial-day guard). Restricted
        # to security_type IN ('equity','etf') (indices are not scored) and
        # floored via BOGUS_PRINT_SQL_GUARD (#275) so bogus/illiquid prints
        # (sub-penny closes, zero-volume stale prints, sub-$10k-ADV shells)
        # do not trip the check on names nobody scores.
        #
        # std_ret is computed over WINSORIZED ret_1d (clamped to
        # [-_STD_WINSOR_ABS, +_STD_WINSOR_ABS] via LEAST/GREATEST) so a
        # single genuine large single-name move does not
        # inflate the cross-sectional std past _STD_MAX on its own. min_ret /
        # max_ret stay on the RAW ret_1d - the single-name gate needs the
        # true extreme to catch a bad print. n_extreme also counts RAW
        # |ret_1d| > _EXTREME_ABS, so broad corruption (many names extreme)
        # still trips via both the winsorized std and the extreme count.
        df = pd.read_sql(
            text(f"""
                SELECT
                    ts_event::date AS score_date,
                    COUNT(*)       AS n_rows,
                    STDDEV(LEAST(GREATEST(ret_1d, -:winsor_abs), :winsor_abs)) AS std_ret,
                    COUNT(*) FILTER (WHERE ABS(ret_1d) > :extreme_abs) AS n_extreme,
                    MIN(ret_1d)    AS min_ret,
                    MAX(ret_1d)    AS max_ret
                FROM ohlcv.prices
                WHERE resolution = 'daily'
                  AND security_type IN ('equity', 'etf')
                  AND ts_event::date BETWEEN :s AND :e
                  AND ret_1d IS NOT NULL
                  AND {BOGUS_PRINT_SQL_GUARD}
                GROUP BY ts_event::date
                ORDER BY ts_event::date
            """),
            engine,
            params={
                "s": start_date,
                "e": end_date,
                "extreme_abs": _EXTREME_ABS,
                "winsor_abs": _STD_WINSOR_ABS,
                **BOGUS_PRINT_SQL_PARAMS,
            },
        )

        if df.empty:
            # No ret_1d rows in range (e.g. holidays only). Nothing to judge;
            # coverage owns the "where did the data go" question.
            return AssetCheckResult(
                passed=True,
                severity=AssetCheckSeverity.WARN,
                metadata={
                    "score_date_range": MetadataValue.text(label),
                    "note": MetadataValue.text("no ret_1d rows in range"),
                },
            )

        bad_dates, skipped_partial, evaluated = _evaluate_days(df)

        # bool() cast: numpy scalar comparisons yield numpy.bool_ which
        # Dagster's AssetCheckResult rejects with ParameterCheckError.
        passed, severity = _era_gate(bad_dates)
        return AssetCheckResult(
            passed=passed,
            severity=severity,
            metadata={
                "score_date_range": MetadataValue.text(label),
                "dates_evaluated": MetadataValue.int(evaluated),
                "dates_skipped_partial": MetadataValue.json(skipped_partial),
                "bad_date_count": MetadataValue.int(len(bad_dates)),
                "bad_dates": MetadataValue.json(bad_dates),
                "thresholds": MetadataValue.json(
                    {
                        "std_max": _STD_MAX,
                        "std_winsor_abs": _STD_WINSOR_ABS,
                        "extreme_abs": _EXTREME_ABS,
                        "extreme_count_max": _EXTREME_COUNT_MAX,
                        "ret_abs_min": _RET_ABS_MIN,
                        "ret_abs_max": _RET_ABS_MAX,
                        "min_rows": _MIN_ROWS,
                        "min_price_floor": MIN_PRICE_FLOOR,
                        "min_dollar_volume": MIN_DOLLAR_VOLUME,
                    }
                ),
            },
        )
    finally:
        engine.dispose()
