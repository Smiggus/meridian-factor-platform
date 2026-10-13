"""
LDP OHLCV read-boundary contract check (Meridian #295, design doc D1).

This is the ONE BLOCKING detective control in the repo. Every
other check here is advisory by invariant ("a data-quality WARN must never
halt daily scoring and starve every downstream consumer"). The read boundary
is the documented exception, for one reason: a violation here means the
INPUTS are wrong, so proceeding writes durable corruption into
factors.daily_scores, and the cost of that is exactly the multi-year
recompute this program exists to eliminate. Refusing to write is strictly
cheaper than writing and recomputing.

The contract itself (thresholds, calibration evidence, the source-aware
adj_close invariant) lives in `parquet_io.evaluate_ohlcv_contract` - the read
chokepoint - so it is a pure, DB-free function that this check merely feeds
and reports. Statistics are gathered from the PARQUET/ICEBERG mirror when the
assets are reading it, and from PostgreSQL otherwise, so the contract is
evaluated on whichever path actually served the day's factors.

WHAT BLOCKS AND WHAT ALERTS (see parquet_io for the numbers)
  BLOCKING - required columns missing (this replaces a SILENT degrade in
             _ohlcv_columns()), zero bars, ret_1d non-null share below 90%,
             and adj_close above 1.1x close on the sources that carry no
             dividend adjustment (fmp/wrds/databento), where the correct
             share is zero.
  ALERTING - an adj_close excess on the total-return-adjusted sources
             (already mitigated row-by-row by adj_close_guard) and a
             bar-count step against the prior date (a legitimate end of an
             upstream tape can produce one; blocking it would wedge the
             recompute campaign on correct data).

Attached to momentum_scores alongside check_ret1d_input_quality - the same
asset, the same blocking semantics, the same reason (momentum is the first
consumer of the raw price/return tape).
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
from meridian_pipeline import parquet_io
from meridian_pipeline.k8s_config import LIGHT_OP_TAGS
from meridian_pipeline.notify import notify_failed_checks
from meridian_pipeline.risk_metrics_asset import _get_engine

# Historical-era gate, same rationale as ret1d_checks._BLOCKING_FLOOR: the
# thresholds are tuned for the era Meridian actually scores, and older data
# must not halt a historical backfill. Violations before the
# floor downgrade to a passing WARN with full metadata.
_BLOCKING_FLOOR = "2004-01-01"

_PG_STATS_SQL = text("""
    SELECT source,
           COUNT(*)                                   AS rows,
           COUNT(*) FILTER (WHERE close > 0
                              AND adj_close > :ratio * close) AS adj_above,
           MAX(adj_close / NULLIF(close, 0))          AS max_ratio,
           COUNT(ret_1d)                              AS ret1d_nonnull
    FROM ohlcv.prices
    WHERE resolution = 'daily'
      AND security_type IN ('equity', 'etf')
      AND ts_event::date = :d
    GROUP BY source
""")

_PG_PRIOR_SQL = text("""
    SELECT ts_event::date AS d, COUNT(*) AS rows
    FROM ohlcv.prices
    WHERE resolution = 'daily'
      AND security_type IN ('equity', 'etf')
      AND ts_event::date < :d
      AND ts_event::date >= :floor
    GROUP BY ts_event::date
    ORDER BY ts_event::date DESC
    LIMIT 1
""")

_PG_COLUMNS_SQL = text("""
    SELECT column_name FROM information_schema.columns
    WHERE table_schema = 'ohlcv' AND table_name = 'prices'
""")


def pg_contract_stats(engine, score_date: date) -> dict:
    """Contract statistics from PostgreSQL - the fallback read path."""
    from datetime import timedelta

    cols = {r[0] for r in pd.read_sql(_PG_COLUMNS_SQL, engine).itertuples(index=False, name=None)}
    df = pd.read_sql(
        _PG_STATS_SQL,
        engine,
        params={"d": score_date, "ratio": parquet_io.ADJ_CLOSE_MAX_RATIO},
    )
    prior = pd.read_sql(
        _PG_PRIOR_SQL,
        engine,
        params={"d": score_date, "floor": score_date - timedelta(days=14)},
    )
    by_source = [
        {
            "source": r["source"],
            "rows": int(r["rows"]),
            "adj_above": int(r["adj_above"] or 0),
            "max_ratio": float(r["max_ratio"]) if pd.notna(r["max_ratio"]) else None,
        }
        for _, r in df.iterrows()
    ]
    return {
        "score_date": score_date,
        "columns": cols,
        "rows": int(df["rows"].sum()) if not df.empty else 0,
        "ret1d_nonnull": int(df["ret1d_nonnull"].sum()) if not df.empty else 0,
        "prior_date": prior.iloc[0]["d"] if not prior.empty else None,
        "prior_rows": int(prior.iloc[0]["rows"]) if not prior.empty else None,
        "by_source": by_source,
        "source_path": "postgres",
    }


def contract_stats_for(engine, score_date: date) -> dict:
    """Prefer the mirror the assets read; fall back to PostgreSQL."""
    prior_date = None
    stats = None
    try:
        prior = pd.read_sql(
            _PG_PRIOR_SQL,
            engine,
            params={"d": score_date, "floor": score_date - pd.Timedelta(days=14)},
        )
        prior_date = prior.iloc[0]["d"] if not prior.empty else None
    except Exception:  # noqa: BLE001 - prior date is an alerting leg only
        prior_date = None
    try:
        stats = parquet_io.ohlcv_contract_stats(score_date, prior_date)
    except Exception:  # noqa: BLE001 - fall back to PG, never fail on the probe
        stats = None
    return stats if stats is not None else pg_contract_stats(engine, score_date)


def evaluate_contract_dates(stats_by_date: list[dict]) -> dict:
    """Pure aggregation of per-date contract results into one check outcome.

    Split out from the DB read so the era gate and the pass/fail decision are
    unit-testable without a database. Returns
    `{"passed", "severity", "metadata", "blocking", "alerting"}`.

    `passed` reflects the BLOCKING legs only. Alerting legs (the total-return
    adj_close residual, a bar-count step) are carried in `alerting` and are
    notified by the caller: this check is blocking=True, so making an
    alerting-only finding `passed=False` would risk halting the run on a leg
    that is non-halting. Visibility for those comes from ntfy,
    not from the run's exit status - the repo's standing rule.
    """
    blocking: list[dict] = []
    alerting: list[dict] = []
    for stats in stats_by_date:
        result = parquet_io.evaluate_ohlcv_contract(stats)
        d = str(stats.get("score_date"))
        for v in result["violations"]:
            blocking.append({"score_date": d, **v})
        for w in result["warnings"]:
            alerting.append({"score_date": d, **w})

    modern_blocking = [b for b in blocking if b["score_date"] >= _BLOCKING_FLOOR]
    passed = not modern_blocking
    severity = AssetCheckSeverity.ERROR if modern_blocking else AssetCheckSeverity.WARN
    metadata = {
        "dates_evaluated": MetadataValue.int(len(stats_by_date)),
        "blocking_violations": MetadataValue.json(blocking[:20]),
        "alerting_violations": MetadataValue.json(alerting[:20]),
        "n_blocking": MetadataValue.int(len(modern_blocking)),
        "n_alerting": MetadataValue.int(len(alerting)),
        "read_paths": MetadataValue.json(
            sorted({str(s.get("source_path", "unknown")) for s in stats_by_date})
        ),
        "thresholds": MetadataValue.json(
            {
                "adj_close_max_ratio": parquet_io.ADJ_CLOSE_MAX_RATIO,
                "non_crsp_adj_above_max_pct": parquet_io.NON_CRSP_ADJ_ABOVE_MAX_PCT,
                "crsp_adj_above_max_pct": parquet_io.CRSP_ADJ_ABOVE_MAX_PCT,
                "ret1d_nonnull_min_pct": parquet_io.RET1D_NONNULL_MIN_PCT,
                "bar_count_min_ratio": parquet_io.BAR_COUNT_MIN_RATIO,
                "required_columns": sorted(parquet_io.REQUIRED_OHLCV_COLUMNS),
                "blocking_floor": _BLOCKING_FLOOR,
            }
        ),
    }
    # A blocking violation confined to the pre-2004 era passes with WARN, the
    # metadata still carrying it (mirrors ret1d_checks._era_gate).
    if blocking and not modern_blocking:
        metadata["note"] = MetadataValue.text(
            f"contract violations confined to dates before {_BLOCKING_FLOOR} - reported, not blocked"
        )
    return {
        "passed": passed,
        "severity": severity,
        "metadata": metadata,
        "blocking": blocking,
        "alerting": alerting,
    }


@asset_check(
    asset="momentum_scores",
    description=(
        "BLOCKING read-boundary contract on the LDP OHLCV tape (Meridian "
        "#295 D1). Per partition date: required columns present (replacing a "
        "silent degrade to plain `close`), bars present, ret_1d non-null "
        "share >= 90%, and adj_close never above 1.1x close on the sources "
        "that carry no dividend adjustment (fmp/wrds/databento). A "
        "total-return "
        "chain legitimately sits BELOW close, so any residual "
        "above-close excess alerts rather than blocks, as "
        "does a bar-count step against the prior date. Evaluated on the "
        "Parquet/Iceberg mirror when the assets read it, PostgreSQL "
        "otherwise."
    ),
    blocking=True,
    op_tags=LIGHT_OP_TAGS,
)
def check_ohlcv_read_contract(context) -> AssetCheckResult:
    kr = context.partition_key_range
    label = kr.start if kr.start == kr.end else f"{kr.start}..{kr.end}"
    dates = pd.date_range(kr.start, kr.end, freq="D").date
    engine = _get_engine()
    try:
        stats_by_date = []
        for d in dates:
            stats = contract_stats_for(engine, d)
            # Non-trading days have no bars at all; they are not contract
            # violations, they are days the market was shut.
            if stats["rows"] == 0:
                continue
            stats_by_date.append(stats)

        if not stats_by_date:
            return AssetCheckResult(
                passed=True,
                severity=AssetCheckSeverity.WARN,
                metadata={
                    "score_date_range": MetadataValue.text(label),
                    "note": MetadataValue.text("no OHLCV bars in range (non-trading days)"),
                },
            )

        outcome = evaluate_contract_dates(stats_by_date)
        metadata = outcome["metadata"]
        metadata["score_date_range"] = MetadataValue.text(label)
        if not outcome["passed"]:
            context.log.error(
                f"[check_ohlcv_read_contract] {label}: contract violated - "
                f"see blocking_violations metadata"
            )
        # Alerting legs never fail the check (see evaluate_contract_dates), so
        # ntfy is their ONLY route to a human. Fail-open, as everywhere else.
        if outcome["alerting"]:
            for a in outcome["alerting"][:20]:
                context.log.warning(
                    f"[check_ohlcv_read_contract] {a['score_date']} {a['leg']}: {a['detail']}"
                )
            try:
                notify_failed_checks(
                    "momentum_scores",
                    [
                        {
                            "name": "check_ohlcv_read_contract (alerting legs)",
                            "failed_partitions": sorted(
                                {a["score_date"] for a in outcome["alerting"]}
                            ),
                            "detail": outcome["alerting"][0]["detail"],
                        }
                    ],
                    log=context.log,
                )
            except Exception as exc:  # noqa: BLE001 - alerting must never fail the run
                context.log.warning(f"[check_ohlcv_read_contract] ntfy alert failed: {exc}")
        return AssetCheckResult(
            passed=outcome["passed"], severity=outcome["severity"], metadata=metadata
        )
    finally:
        engine.dispose()
