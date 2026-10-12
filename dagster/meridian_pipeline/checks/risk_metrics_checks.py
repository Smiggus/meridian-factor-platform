"""
Risk metrics asset checks (meridian-factors code location).

Tier 1 inlining (Meridian #285): these checks used to be four separate
`@asset_check` step pods attached to `risk_metrics`. The #285 audit found
that per-check step pods spent most of their lifetime on k8s startup rather
than on the check itself.

Now the pass/fail DECISIONS live here as pure, DB-free functions that
operate on the `rows` list already built in-memory by
`_compute_risk_metrics_for_date` for each partition (score_date) - the
exact rows about to be upserted into factors.risk_metrics. Evaluating
against `rows` directly is behaviourally identical to the old checks
(which queried factors.risk_metrics AFTER the upsert) since `rows` IS
what gets upserted, and is strictly cheaper (no second DB round-trip).

`risk_metrics_asset.py` calls these once per partition inside its existing
loop, accumulates failures per check name, and yields exactly one
AssetCheckResult per declared AssetCheckSpec after the loop (Dagster
requires exactly one result per spec per run; see the zero-partitions-
evaluated handling there).

All four checks keep their exact pre-existing name and WARN severity so
UI history / any downstream queries stay continuous.
"""

from __future__ import annotations


def evaluate_spy_self_consistency(rows: list[dict]) -> dict:
    """SPY's beta_252d should be ~1.0 (SPY regressed against itself).

    Mirrors the old check_risk_metrics_spy_self_consistency query.
    """
    spy_rows = [r for r in rows if r["ticker"] == "SPY"]
    if not spy_rows:
        return {
            "passed": False,
            "note": "SPY not found in risk_metrics for this date",
        }

    beta_252d = spy_rows[0]["beta_252d"]
    beta_60d = spy_rows[0]["beta_60d"]
    passed = beta_252d is not None and 0.95 <= beta_252d <= 1.05
    return {
        "passed": passed,
        "beta_252d": beta_252d,
        "beta_60d": beta_60d,
    }


def evaluate_beta_range(rows: list[dict]) -> dict:
    """No non-null beta should have abs > 10 (verifies _f_beta() clipping).

    Mirrors the old check_risk_metrics_beta_range query.
    """
    bad = [
        r
        for r in rows
        if (r["beta_252d"] is not None and abs(r["beta_252d"]) > 10)
        or (r["beta_60d"] is not None and abs(r["beta_60d"]) > 10)
    ]
    return {
        "passed": len(bad) == 0,
        "total_rows": len(rows),
        "violations": len(bad),
        "sample": [
            {"ticker": r["ticker"], "beta_252d": r["beta_252d"], "beta_60d": r["beta_60d"]}
            for r in bad[:10]
        ],
    }


def evaluate_vol_range(rows: list[dict]) -> dict:
    """vol_60d/vol_20d must fall within [0, 5] (daily-scale std of log returns).

    WARNs if more than 0.5% of relevant rows violate. Mirrors the old
    check_risk_metrics_vol_range query.
    """
    relevant = [r for r in rows if r["vol_60d"] is not None or r["vol_20d"] is not None]
    total = len(relevant)

    def _bad60(r):
        return r["vol_60d"] is not None and (r["vol_60d"] < 0 or r["vol_60d"] > 5)

    def _bad20(r):
        return r["vol_20d"] is not None and (r["vol_20d"] < 0 or r["vol_20d"] > 5)

    v60_bad = sum(1 for r in relevant if _bad60(r))
    v20_bad = sum(1 for r in relevant if _bad20(r))
    worst = max(v60_bad, v20_bad)
    pct = (worst / total * 100) if total > 0 else 0.0

    offenders = sorted(
        (r for r in relevant if _bad60(r) or _bad20(r)),
        key=lambda r: max(r["vol_60d"] or 0, r["vol_20d"] or 0),
        reverse=True,
    )[:10]

    return {
        "passed": pct <= 0.5,
        "total_rows": total,
        "vol_60d_out_of_range": v60_bad,
        "vol_20d_out_of_range": v20_bad,
        "offenders": [
            {"ticker": r["ticker"], "vol_60d": r["vol_60d"], "vol_20d": r["vol_20d"]}
            for r in offenders
        ],
    }


def evaluate_coverage(rows: list[dict]) -> dict:
    """n_obs distribution across tickers. WARNs if >10% have n_obs < 60.

    Mirrors the old check_risk_metrics_coverage query. Percentiles use
    nearest-rank rather than PostgreSQL's PERCENTILE_CONT interpolation -
    an acceptable difference for a WARN-severity diagnostic.
    """
    relevant = [r for r in rows if r["n_obs"] is not None]
    total = len(relevant)
    low = sum(1 for r in relevant if r["n_obs"] < 60)
    pct = (low / total * 100) if total > 0 else 0.0

    n_obs_sorted = sorted(r["n_obs"] for r in relevant)

    def _pctl(p: float):
        if not n_obs_sorted:
            return None
        idx = min(len(n_obs_sorted) - 1, int(p * len(n_obs_sorted)))
        return n_obs_sorted[idx]

    return {
        "passed": pct <= 10.0,
        "total_tickers": total,
        "low_obs_count": low,
        "low_obs_pct": pct,
        "n_obs_p10": _pctl(0.10),
        "n_obs_p50": _pctl(0.50),
        "n_obs_p90": _pctl(0.90),
    }
