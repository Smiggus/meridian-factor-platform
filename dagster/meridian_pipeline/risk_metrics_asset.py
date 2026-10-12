"""
risk_metrics_asset.py
---------------------
Daily PIT risk metrics for all tickers in ohlcv.prices.

Computes vol (60d, 20d), ADV (20d, 60d), beta vs SPY (252d, 60d),
and idiosyncratic vol (60d residual after market regression).

Universe: all tickers with at least 60 daily bars before score_date, after
the bogus/illiquid-print floor (close >= $0.01, volume > 0, dollar volume
>= $10k), the delisting guard shared with the risk factor group (#259,
extended in ohlcv_hygiene.py for #275), and the price-staleness fallback
(#291: no bar within STALE_TRADING_DAY_TOLERANCE trading days of score_date,
independent of the nullable delisting_date column) exclude rows/tickers
that would otherwise poison a ticker's own vol/beta window.
Uses ret_1d for return computation (dividend-corrected, split-invariant,
PIT-safe). Dollar volume uses raw_close * volume for consistent notional.
"""

from __future__ import annotations

import os

# Must be set before any CUDA context is created. Forces managed-memory
# onto device memory.
# Step pod env-var injection is unreliable (dagster_k8s parser
# bug with "dict[str, str]" annotations), so we set it here instead.
os.environ.setdefault("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1")

from datetime import date, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import text

from dagster import (
    AssetCheckResult,
    AssetCheckSeverity,
    AssetCheckSpec,
    AssetMaterialization,
    MetadataValue,
    Output,
    asset,
)
from meridian_pipeline import db
from meridian_pipeline.gpu_vram import log_vram_peak
from meridian_pipeline.k8s_config import GPU_OP_TAGS
from meridian_pipeline.ohlcv_hygiene import (
    BOGUS_PRINT_SQL_GUARD,
    BOGUS_PRINT_SQL_PARAMS,
    STALE_TRADING_DAY_TOLERANCE,
    fetch_delisted_tickers,
    stale_tickers_from_last_seen,
)
from meridian_pipeline.partitions import DAILY_BACKFILL_POLICY, DAILY_PARTITIONS

# Tier 1 inlining (#285): these four check names are unchanged from the
# formerly-standalone @asset_check steps in checks/risk_metrics_checks.py -
# preserved exactly so Dagster UI check history stays continuous.
_RISK_METRICS_CHECK_NAMES = (
    "check_risk_metrics_spy_self_consistency",
    "check_risk_metrics_beta_range",
    "check_risk_metrics_vol_range",
    "check_risk_metrics_coverage",
)


def _get_engine():
    """Delegate to db._engine so this module shares the bounded QueuePool."""
    return db._engine(
        pguser=os.environ.get("PGUSER", "dbuser"),
        pgpass=os.environ["PGPASS"],
        pghost=os.environ.get("PGHOST", "db.internal"),
        pgport=int(os.environ.get("PGPORT", 5432)),
        pgdb=os.environ.get("PGDB", "marketdata"),
    )


def _compute_risk_metrics_for_date(engine, score_date: date) -> tuple[list[dict], dict[str, int]]:
    """
    Compute vol, ADV, beta, idio_vol for all tickers as of score_date.
    Returns (rows, exclusion_counts). rows is a list of dicts ready for
    upsert to factors.risk_metrics. NaN values are converted to None at row
    construction so they land as PostgreSQL NULL not IEEE NaN.
    exclusion_counts (#291 observability) is {"delisted": n, "stale": n} -
    counts of tickers dropped by the two hygiene guards before the vol/beta
    tensors are built; can overlap (a ticker can be both).

    GPU-accelerated: uses PyTorch tensor ops for vectorized batch computation
    of vol/beta across all tickers simultaneously. Falls back to CPU if no CUDA.
    """
    # 1. Pull last 380 trading days of ret_1d + raw_close*volume for all tickers
    #    (380 days covers 252 trading days for beta_252d with weekend buffer)
    #
    # Return source: ret_1d is the PIT-safe workhorse return, stored as
    #   (close[t] + cash_div_on_ex[t]) / close[t-1] - 1
    # Split-invariant, dividend-corrected, frozen at write time. Using it
    # directly avoids the lossy log(adj_close) approximation of total return.
    #
    # Dollar volume uses raw_close * volume for a consistent notional
    # (raw price * raw volume = actual traded dollar value, no adjustment mix).
    #
    # Bogus/illiquid-print floor (Meridian #275, LDP #182): a separate class
    # of bad close prints in PG ohlcv.prices (sub-penny / zero-volume /
    # illiquid-shell rows, e.g. a sub-penny close -> ret_1d=-100%,
    # then a volume=0 stale print -> a huge positive ret_1d) would otherwise feed straight
    # into this row's own vol/beta window (per-ticker, so it only corrupts
    # that ticker's own metrics, but corrupts them badly). Filtered here on
    # the same floor as the risk factor group (#259) via ohlcv_hygiene.py so
    # both risk consumers agree on what counts as a real print.
    # TICKER-SPLICE GUARD (LDP #189, entity_id now in ohlcv.prices at near-complete
    # coverage): current_entity resolves each ticker's entity_id as of its
    # own most recent row on/before score_date; the main SELECT keeps only
    # rows matching that entity_id (or all rows, when the current entity_id
    # itself is unmapped/NULL - e.g. SPY, most ETFs). A reused ticker
    # symbol's old, unrelated company never reaches the vol/beta tensors
    # below. NULL-current-entity_id tickers fall through to the date-gap
    # guard applied further down (numpy_window_has_gap / splice_guard.py).
    lookback_start = score_date - timedelta(days=380)
    prices = pd.read_sql(
        text(f"""
            WITH current_entity AS (
                SELECT DISTINCT ON (ticker) ticker, entity_id AS current_eid
                FROM ohlcv.prices
                WHERE resolution = 'daily'
                  AND security_type IN ('equity', 'etf')
                  AND ts_event::date <= :score_date
                ORDER BY ticker, ts_event DESC
            )
            SELECT p.ticker, p.ts_event::date AS d,
                   p.ret_1d,
                   raw_close * volume AS dollar_volume
            FROM ohlcv.prices p
            JOIN current_entity c ON c.ticker = p.ticker
            WHERE p.resolution = 'daily'
              AND p.security_type IN ('equity', 'etf')
              AND p.ts_event::date <= :score_date
              AND p.ts_event::date > :lookback_start
              AND {BOGUS_PRINT_SQL_GUARD}
              AND (c.current_eid IS NULL OR p.entity_id = c.current_eid)
            ORDER BY p.ticker, p.ts_event
        """),
        engine,
        params={
            "score_date": score_date,
            "lookback_start": lookback_start,
            **BOGUS_PRINT_SQL_PARAMS,
        },
    )

    empty_counts = {"delisted": 0, "stale": 0}
    if prices.empty:
        return [], empty_counts

    # Delisting guard (#259/#275): drop tickers confirmed delisted before
    # score_date so their stale post-delisting bars (which still clear the
    # bogus-print floor above) don't ghost-score vol/beta for months.
    delisted = fetch_delisted_tickers(engine, score_date)
    n_delisted = 0
    if delisted:
        pre = set(prices["ticker"])
        prices = prices[~prices["ticker"].isin(delisted)]
        n_delisted = len(pre & set(delisted))
        if prices.empty:
            return [], {"delisted": n_delisted, "stale": 0}

    # Price-staleness fallback (#259/#291): same guard as
    # risk_factor_scores_asset.py, independent of delisting_date (nullable
    # upstream column - see ohlcv_hygiene.py PRICE-STALENESS FALLBACK note).
    # Derived from `prices` (already fetched, PIT-bounded to <= score_date) -
    # no new query.
    last_seen = prices.groupby("ticker")["d"].max()
    stale = stale_tickers_from_last_seen(last_seen, prices["d"], score_date)
    n_stale = len(stale)
    if stale:
        prices = prices[~prices["ticker"].isin(stale)]
        if prices.empty:
            return [], {"delisted": n_delisted, "stale": n_stale}

    exclusion_counts = {"delisted": n_delisted, "stale": n_stale}

    # Pivot to wide format (date x ticker)
    rets_wide = prices.pivot_table(index="d", columns="ticker", values="ret_1d")
    dv = prices.pivot_table(index="d", columns="ticker", values="dollar_volume")

    if "SPY" not in rets_wide.columns:
        return [], exclusion_counts  # need SPY for beta

    # Align columns: some tickers may have ret_1d but no dollar_volume (or vice
    # versa) when raw_close is NULL for early dates. Build both tensors from the
    # common column set to prevent shape mismatches in torch indexing.
    common_cols = rets_wide.columns.intersection(dv.columns)
    rets_wide = rets_wide[common_cols]
    dv = dv[common_cols]

    # Align DATE axes too (#292). `pivot_table` silently drops an index value
    # whose aggregate is all-NaN, so the two pivots built from the SAME frame
    # can end up with DIFFERENT date indexes: a date where prices published but
    # ret_1d has not (routine LDP publication lag - a day can have all its
    # equity bars and ZERO non-null ret_1d) exists in `dv` and is absent from
    # `rets_wide`. Everything downstream - the window slices, `n_dates`, and the
    # splice guard's `date_index` - is derived from `rets_wide.index`, so an
    # unequal `dv` index produced (a) an IndexError in numpy_window_has_gap
    # ("size of axis is 60 but size of corresponding boolean axis is 61") and
    # (b) a SILENT off-by-N in adv_20d/adv_60d, which sliced the last N rows of
    # a differently-indexed dv.
    #
    # DIRECTION: `rets_wide` is the authority. The returns panel defines this
    # asset's trading-day axis - every window size, every min_obs gate, and the
    # splice guard's calendar are expressed in rows of it - so a date carrying
    # no return information must not consume a window slot. Reindexing dv onto
    # it also repairs the opposite case (a date in rets_wide but not in dv)
    # by inserting an all-NaN dv row, which the ADV NaN-mask already ignores
    # correctly. The alternative (union index) would inject an all-NaN row into
    # the RETURNS panel, diluting every window's observation count against its
    # min_obs threshold for no gain; the only thing it would buy is one extra
    # day in a 20/60-day NaN-masked ADV mean, an effect of order 1/N on a
    # quantity used for liquidity sizing.
    dv = dv.reindex(index=rets_wide.index)

    if "SPY" not in rets_wide.columns:
        return [], exclusion_counts  # SPY dropped from common set - can't compute beta

    # GPU-accelerated batch computation
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tickers = list(rets_wide.columns)
    spy_idx = tickers.index("SPY")

    # Convert to torch tensors: (n_dates, n_tickers)
    # Use float32 - sufficient precision for financial returns
    rets_simple = torch.tensor(rets_wide.values, dtype=torch.float32, device=device)
    dv_t = torch.tensor(dv.values, dtype=torch.float32, device=device)

    # Convert simple (dividend-corrected) returns to log returns for
    # additive-stability in vol/beta/idio computations.
    # ret_1d values rarely hit the -1 boundary for equities, so log1p is safe;
    # any NaN stays NaN and is masked below.
    rets = torch.log1p(rets_simple)  # (n_dates, n_tickers)
    n_dates = rets.shape[0]

    # Per-ticker n_obs (count non-NaN returns)
    valid = ~torch.isnan(rets)
    n_obs_per_ticker = valid.sum(dim=0)  # (n_tickers,)

    spy_r = rets[:, spy_idx]  # (n_dates,)

    # Helper: compute vol/beta for a window (last N rows of rets).
    # min_obs is the minimum number of valid (non-NaN) observations required
    # in this window for the metric to be considered defined. Below the
    # threshold the metric is set to NaN (later converted to NULL at insert).
    # Scalar NaN tensor used for masking below-threshold windows. Created once
    # here rather than inside _window_stats to avoid repeated small allocations
    # that increase HMM page churn under kernel 6.17 (k3s#85 / Meridian#88).
    _nan = torch.tensor(float("nan"), device=device)

    def _window_stats(window_rets, window_spy, min_obs: int):
        """Returns (vol, beta, idio_vol, n_per_ticker) for the window."""
        # window_rets: (window_n, n_keep), window_spy: (window_n,)
        # Mask NaNs - replace with 0 then track valid count
        mask = ~torch.isnan(window_rets) & ~torch.isnan(window_spy.unsqueeze(1))
        clean_rets = torch.where(mask, window_rets, torch.zeros_like(window_rets))
        clean_spy = window_spy.unsqueeze(1).expand_as(window_rets)
        clean_spy = torch.where(mask, clean_spy, torch.zeros_like(clean_spy))

        n = mask.sum(dim=0).float()  # (n_keep,) per-ticker valid count IN THIS WINDOW

        # Vol: stddev of stock log returns.
        # BASIS: this is DAILY std (no sqrt(252)) by design. `factors.risk_metrics`
        # vol_20d/vol_60d/idio_vol_60d are all DAILY-scale (typical 0.01-0.05).
        # This is INTENTIONALLY different from `factors.risk_factor_scores`, whose
        # vol factors are annualised (x sqrt(252)) and then z-scored - the z-score
        # makes annualisation scale-invariant there, and downstream sizing consumers of
        # risk_metrics use inverse-vol sizing (also scale-invariant). Do NOT
        # "fix" this by adding sqrt(252); daily is the canonical basis here and
        # the docs / API contract document it as daily.
        sum_r = clean_rets.sum(dim=0)
        mean_r = sum_r / n.clamp(min=1)
        var_r = ((clean_rets - mean_r) ** 2 * mask).sum(dim=0) / (n - 1).clamp(min=1)
        vol = torch.sqrt(var_r)

        # Beta = cov(r, m) / var(m), where each is masked
        sum_m = clean_spy.sum(dim=0)
        mean_m = sum_m / n.clamp(min=1)
        cov = ((clean_rets - mean_r) * (clean_spy - mean_m) * mask).sum(dim=0) / (n - 1).clamp(
            min=1
        )
        var_m = ((clean_spy - mean_m) ** 2 * mask).sum(dim=0) / (n - 1).clamp(min=1)
        beta = cov / var_m.clamp(min=1e-12)

        # Residuals: e[t] = r[t] - beta * m[t]; idio vol = std(e)
        residuals = (clean_rets - beta * clean_spy) * mask
        sum_e = residuals.sum(dim=0)
        mean_e = sum_e / n.clamp(min=1)
        var_e = ((residuals - mean_e) ** 2 * mask).sum(dim=0) / (n - 1).clamp(min=1)
        # idio_vol is DAILY std of residuals (no sqrt(252)) - same basis as `vol`
        # above. Do NOT annualise; see the BASIS note at the vol computation.
        idio_vol = torch.sqrt(var_e)

        # Per-metric minimum observation threshold
        too_few = n < min_obs
        vol = torch.where(too_few, _nan, vol)
        beta = torch.where(too_few, _nan, beta)
        idio_vol = torch.where(too_few, _nan, idio_vol)

        return vol, beta, idio_vol, n

    # Per-metric minimum observation thresholds. The mask requires BOTH the
    # ticker AND SPY (the market) to have non-NaN returns on each day, so 1-2
    # missing SPY days (holidays, halts, data gaps) drop the per-window count
    # for ALL tickers. Threshold is set to ~90% of the window size to allow
    # for normal data hiccups while still rejecting genuinely sparse data.
    #
    # .clone() gives each window its own contiguous GPU allocation so kernel
    # 6.17's HMM page reclaim cannot migrate pages that are still referenced
    # by other live tensors (Meridian#88 / k3s#85).
    win60 = rets[max(0, n_dates - 60) :].clone()
    spy_60 = spy_r[max(0, n_dates - 60) :].clone()
    vol_60d, beta_60d, idio_60d, n60 = _window_stats(win60, spy_60, min_obs=55)

    win20 = rets[max(0, n_dates - 20) :].clone()
    spy_20 = spy_r[max(0, n_dates - 20) :].clone()
    vol_20d, _, _, _ = _window_stats(win20, spy_20, min_obs=18)

    # 252-day window: require >=240 obs (~95%). Garbage betas came from
    # running with only a small number of observations.
    win252 = rets[max(0, n_dates - 252) :].clone()
    spy_252 = spy_r[max(0, n_dates - 252) :].clone()
    _, beta_252d, _, _ = _window_stats(win252, spy_252, min_obs=240)

    # ADV (dollar volume) - simple mean over windows, ignoring NaN.
    # Sliced on `n_dates` (the returns panel's row count), NOT on
    # dv_t.shape[0]: after the reindex above the two are identical by
    # construction, and using n_dates makes the shared date axis explicit -
    # slicing on dv_t's own length is what silently mis-windowed ADV whenever
    # the two pivots disagreed (see the date-axis alignment note above).
    dv_60 = dv_t[max(0, n_dates - 60) :]
    dv_20 = dv_t[max(0, n_dates - 20) :]
    dv_60_mask = ~torch.isnan(dv_60)
    dv_20_mask = ~torch.isnan(dv_20)
    adv_60d = torch.where(dv_60_mask, dv_60, torch.zeros_like(dv_60)).sum(dim=0) / dv_60_mask.sum(
        dim=0
    ).float().clamp(min=1)
    adv_20d = torch.where(dv_20_mask, dv_20, torch.zeros_like(dv_20)).sum(dim=0) / dv_20_mask.sum(
        dim=0
    ).float().clamp(min=1)

    # ── TICKER-SPLICE GUARD (see splice_guard.py) ───────────────────────────
    # `valid`/`mask` above only count non-NaN observations in a window; they
    # do not check whether those observations are temporally contiguous. A
    # reused ticker symbol whose old and new companies each individually
    # clear a window's min_obs threshold would otherwise blend both
    # companies' returns into one vol/beta/idio_vol number. Compute, per
    # window, whether each ticker's valid observations straddle an internal
    # gap larger than the splice threshold, using the SAME dense date index
    # (rets_wide.index) the tensors above were built from.
    from meridian_pipeline.splice_guard import numpy_window_has_gap

    date_index = rets_wide.index.to_numpy()
    valid_np = valid.cpu().numpy()  # (n_dates, n_tickers), aligned to `tickers`
    gap60 = numpy_window_has_gap(
        date_index[max(0, n_dates - 60) :], valid_np[max(0, n_dates - 60) :]
    )
    gap20 = numpy_window_has_gap(
        date_index[max(0, n_dates - 20) :], valid_np[max(0, n_dates - 20) :]
    )
    gap252 = numpy_window_has_gap(
        date_index[max(0, n_dates - 252) :], valid_np[max(0, n_dates - 252) :]
    )
    dv_valid_np = (~torch.isnan(dv_t)).cpu().numpy()
    dv_gap60 = numpy_window_has_gap(
        date_index[max(0, n_dates - 60) :], dv_valid_np[max(0, n_dates - 60) :]
    )
    dv_gap20 = numpy_window_has_gap(
        date_index[max(0, n_dates - 20) :], dv_valid_np[max(0, n_dates - 20) :]
    )

    gap60_t = torch.from_numpy(gap60).to(device)
    gap20_t = torch.from_numpy(gap20).to(device)
    gap252_t = torch.from_numpy(gap252).to(device)
    dv_gap60_t = torch.from_numpy(dv_gap60).to(device)
    dv_gap20_t = torch.from_numpy(dv_gap20).to(device)

    vol_60d = torch.where(gap60_t, _nan, vol_60d)
    beta_60d = torch.where(gap60_t, _nan, beta_60d)
    idio_60d = torch.where(gap60_t, _nan, idio_60d)
    vol_20d = torch.where(gap20_t, _nan, vol_20d)
    beta_252d = torch.where(gap252_t, _nan, beta_252d)
    adv_60d = torch.where(dv_gap60_t, _nan, adv_60d)
    adv_20d = torch.where(dv_gap20_t, _nan, adv_20d)

    # Flush all pending GPU kernels before transferring to CPU. Without this,
    # device pages can be reclaimed while async CUDA work is still
    # referencing them.
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Move back to CPU and assemble DataFrame
    vol_60d_arr = vol_60d.cpu().numpy()
    vol_20d_arr = vol_20d.cpu().numpy()
    beta_60d_arr = beta_60d.cpu().numpy()
    beta_252d_arr = beta_252d.cpu().numpy()
    idio_60d_arr = idio_60d.cpu().numpy()
    adv_60d_arr = adv_60d.cpu().numpy()
    adv_20d_arr = adv_20d.cpu().numpy()
    n_obs_arr = n_obs_per_ticker.cpu().numpy()

    def _f(x):
        """Convert NaN/inf to None so they land as PostgreSQL NULL.
        Without this, NumPy NaN passes through psycopg2 as IEEE NaN
        (not NULL), and PostgreSQL sorts NaN as MAX in ORDER BY queries -
        poisoning every consumer that doesn't explicitly filter it."""
        if x is None:
            return None
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return None
        return v

    def _f_beta(x):
        """Same as _f, but clips extreme betas to None. Real betas are
        bounded by ~±5; anything outside ±10 is a single-event price spike
        leaking into the regression and should be treated as no-signal."""
        v = _f(x)
        if v is None or abs(v) > 10:
            return None
        return v

    rows = []
    for i, ticker in enumerate(tickers):
        n = int(n_obs_arr[i])
        # Filter on the global n_obs only at a very loose threshold - per-metric
        # thresholds are now applied inside _window_stats so each metric gets
        # its own NaN if its window is short. Keep at least 20 daily observations
        # so we don't write meaningless rows.
        if n < 20:
            continue
        rows.append(
            {
                "score_date": score_date,
                "ticker": ticker,
                "vol_60d": _f(vol_60d_arr[i]),
                "vol_20d": _f(vol_20d_arr[i]),
                "adv_20d": _f(adv_20d_arr[i]),
                "adv_60d": _f(adv_60d_arr[i]),
                "beta_252d": _f_beta(beta_252d_arr[i]),
                "beta_60d": _f_beta(beta_60d_arr[i]),
                "idio_vol_60d": _f(idio_60d_arr[i]),
                "n_obs": n,
            }
        )

    # Return list of dicts directly. Going through pd.DataFrame coerces None
    # to NaN in float columns, which then leaks through psycopg2 as IEEE NaN.
    return rows, exclusion_counts


def _upsert_risk_metrics(engine, rows: list[dict]) -> int:
    """Upsert pre-built rows (NaN already converted to None) into factors.risk_metrics.

    Supports multi-date batches (historical backfill path): the previous
    implementation only DELETEd rows[0]["score_date"], so passing rows for
    multiple score_dates with force=True would violate the PK on the second
    and later dates. Now uses ON CONFLICT DO UPDATE on the full PK
    (score_date, ticker), so any combination of dates in a single batch
    upserts cleanly.
    """
    if not rows:
        return 0
    with engine.begin() as conn:
        # Fix #17 (#161): delete-replace per score_date so orphan tickers from
        # a shrinking universe do not accumulate across reruns.
        dates = list({r["score_date"] for r in rows})
        conn.execute(
            text("DELETE FROM factors.risk_metrics WHERE score_date = ANY(:dates)"),
            {"dates": dates},
        )
        # Insert in chunks to avoid parameter overflow
        for i in range(0, len(rows), 1000):
            chunk = rows[i : i + 1000]
            conn.execute(
                text("""
                    INSERT INTO factors.risk_metrics
                        (score_date, ticker, vol_60d, vol_20d, adv_20d, adv_60d,
                         beta_252d, beta_60d, idio_vol_60d, n_obs)
                    VALUES (:score_date, :ticker, :vol_60d, :vol_20d, :adv_20d,
                            :adv_60d, :beta_252d, :beta_60d, :idio_vol_60d, :n_obs)
                    ON CONFLICT (score_date, ticker) DO UPDATE SET
                        vol_60d      = EXCLUDED.vol_60d,
                        vol_20d      = EXCLUDED.vol_20d,
                        adv_20d      = EXCLUDED.adv_20d,
                        adv_60d      = EXCLUDED.adv_60d,
                        beta_252d    = EXCLUDED.beta_252d,
                        beta_60d     = EXCLUDED.beta_60d,
                        idio_vol_60d = EXCLUDED.idio_vol_60d,
                        n_obs        = EXCLUDED.n_obs,
                        computed_at  = NOW()
                """),
                chunk,
            )
    return len(rows)


@asset(
    partitions_def=DAILY_PARTITIONS,
    backfill_policy=DAILY_BACKFILL_POLICY,
    # gpu pool caps cluster-wide concurrency at the op level. Stays
    # alongside `dagster/gpu: "true"` in op_tags until platform-side
    # `dagster-platform#58` Phase 3 decommissions the run-tag cap.
    pool="gpu",
    group_name="meridian_factors",
    description=(
        "Daily PIT risk metrics: vol (60d, 20d), ADV (20d, 60d), beta vs SPY "
        "(252d, 60d), idiosyncratic vol (60d residual). All tickers with >=60 "
        "daily bars before score_date. Uses ret_1d for return calculations."
    ),
    # op_tags so the K8s JSON config bypasses asset-tag strict validation
    # (see gpu_asset comment).
    op_tags=GPU_OP_TAGS,
    # Tier 1 check inlining (#285): these were four separate check-step pods.
    # Evaluated per partition inside the loop below (essentially free - the
    # rows are already in memory) and aggregated into one AssetCheckResult
    # per spec per run. No partitions_def here by design: AssetCheckResult
    # has no partition field in dagster 1.13.4, so attribution to specific
    # dates lives in each result's metadata instead.
    check_specs=[
        AssetCheckSpec(
            name=name,
            asset="risk_metrics",
            blocking=False,
            description="Tier 1 inlined check (Meridian #285) - see checks/risk_metrics_checks.py",
        )
        for name in _RISK_METRICS_CHECK_NAMES
    ],
)
def risk_metrics(context):
    def _skip_non_trading(context, engine, score_date) -> bool:
        """True (and log) when score_date is not a trading day (#280)."""
        if db.is_trading_day(engine, score_date):
            return False
        context.log.info(f"{score_date} is not a trading day - skipping (no scores written)")
        return True

    # Lazy import: `meridian_pipeline.checks.__init__` imports factor_checks,
    # which imports _get_engine back from THIS module, so a module-level
    # import here is circular. Same pattern as _skip_non_trading above.
    from meridian_pipeline.checks.risk_metrics_checks import (
        evaluate_beta_range,
        evaluate_coverage,
        evaluate_spy_self_consistency,
        evaluate_vol_range,
    )

    engine = _get_engine()

    # Per-check accumulator: partitions_evaluated + failed partition list +
    # a representative detail payload from the worst/most-recent failure.
    check_state = {
        name: {"n_eval": 0, "failed_partitions": [], "last_detail": None}
        for name in _RISK_METRICS_CHECK_NAMES
    }

    def _record(name: str, partition_key: str, result: dict) -> None:
        state = check_state[name]
        state["n_eval"] += 1
        if not result.get("passed", True):
            state["failed_partitions"].append(partition_key)
            state["last_detail"] = {k: v for k, v in result.items() if k != "passed"}

    try:
        for partition_key in context.partition_keys:
            score_date = date.fromisoformat(partition_key)
            db.assert_score_date_not_future(score_date)
            # Trading-day gate (#280). risk_metrics computes off a TRAILING
            # lookback window, so on a weekend/holiday it still produces a
            # full cross-section derived from the prior session and writes it
            # to factors.risk_metrics. That is the mechanism behind the orphan
            # dates the #280 audit found: this was the one
            # scored asset with no gate at all.
            if _skip_non_trading(context, engine, score_date):
                continue

            rows, exclusion_counts = _compute_risk_metrics_for_date(engine, score_date)
            if exclusion_counts["delisted"] or exclusion_counts["stale"]:
                context.log.info(
                    f"Hygiene exclusions for {score_date} (#259/#291): "
                    f"delisted={exclusion_counts['delisted']} "
                    f"stale={exclusion_counts['stale']} "
                    f"(tolerance={STALE_TRADING_DAY_TOLERANCE}td)"
                )
            if not rows:
                context.log.warning(f"No data for {score_date}")
                yield AssetMaterialization(
                    asset_key=context.asset_key,
                    partition=partition_key,
                    metadata={"rows": MetadataValue.int(0)},
                )
                continue

            n = _upsert_risk_metrics(engine, rows)
            vol_60d_nn = sum(1 for r in rows if r["vol_60d"] is not None)
            beta_252d_nn = sum(1 for r in rows if r["beta_252d"] is not None)
            context.log.info(
                f"[risk_metrics] {score_date}: {n} tickers "
                f"(vol_60d non-null: {vol_60d_nn}, "
                f"beta_252d non-null: {beta_252d_nn})"
            )
            yield AssetMaterialization(
                asset_key=context.asset_key,
                partition=partition_key,
                metadata={
                    "rows": MetadataValue.int(n),
                    "tickers": MetadataValue.int(len({r["ticker"] for r in rows})),
                    "as_of": MetadataValue.text(str(score_date)),
                    "excluded_delisted": MetadataValue.int(exclusion_counts["delisted"]),
                    "excluded_stale": MetadataValue.int(exclusion_counts["stale"]),
                },
            )

            # Tier 1 inlined checks (#285): evaluate against `rows`, the same
            # data just upserted - no second DB round-trip.
            try:
                _record(
                    "check_risk_metrics_spy_self_consistency",
                    partition_key,
                    evaluate_spy_self_consistency(rows),
                )
                _record(
                    "check_risk_metrics_beta_range",
                    partition_key,
                    evaluate_beta_range(rows),
                )
                _record(
                    "check_risk_metrics_vol_range",
                    partition_key,
                    evaluate_vol_range(rows),
                )
                _record(
                    "check_risk_metrics_coverage",
                    partition_key,
                    evaluate_coverage(rows),
                )
            except Exception as exc:  # noqa: BLE001 - checks must never fail the run
                context.log.warning(
                    f"[risk_metrics checks] evaluation failed for {score_date}: {exc}"
                )

        # Flush CUDA cache before yielding so the gpu pool slot is not
        # released while CUDA context teardown is still in-flight. (Meridian#96)
        try:
            import gc

            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                # Peak BEFORE empty_cache: see gpu_vram / Meridian#296.
                log_vram_peak(context.log, "risk_metrics")
                torch.cuda.empty_cache()
                gc.collect()
        except Exception:
            pass

        # Dagster requires exactly one AssetCheckResult per declared spec per
        # run - even when zero partitions were evaluated (non-trading date,
        # or every partition skipped). passed=True in that case: there is
        # nothing to flag.
        for name in _RISK_METRICS_CHECK_NAMES:
            state = check_state[name]
            yield AssetCheckResult(
                check_name=name,
                passed=len(state["failed_partitions"]) == 0,
                severity=AssetCheckSeverity.WARN,
                metadata={
                    "partitions_evaluated": MetadataValue.int(state["n_eval"]),
                    "failed_count": MetadataValue.int(len(state["failed_partitions"])),
                    "failed_partitions": MetadataValue.json(state["failed_partitions"][:20]),
                    "last_failure_detail": MetadataValue.json(state["last_detail"])
                    if state["last_detail"] is not None
                    else MetadataValue.text("none"),
                },
            )

        yield Output(value=None)
    finally:
        engine.dispose()
