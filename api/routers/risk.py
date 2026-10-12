"""
Risk metrics API endpoints.

Serves volatility, ADV, beta, and idio vol from PostgreSQL (factors.risk_metrics
populated by the lean-data-platform Dagster pipeline). Falls back to on-the-fly
SQL computation if no pre-computed data exists for the requested date.

Used for inverse-vol position sizing, ADV-based capacity limits and beta
targeting by downstream consumers.
"""

import sys
from datetime import date as date_cls
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from api.auth import verify_token
from api.dates import parse_iso_date
from risk_metrics import get_risk_metrics

router = APIRouter()

# dagster package path in the API container (audit batch D #6). An unguarded
# sys.path.insert(0, ...) per request grows sys.path by one duplicate entry per
# request for the process lifetime. Do it once at import time, guarded,
# matching the pattern in api/factor_catalogue.py.
if "/app" not in sys.path:
    sys.path.insert(0, "/app")


@router.get("/metrics", dependencies=[Depends(verify_token)])
def get_metrics(
    tickers: str = Query(..., description="Comma-separated tickers, e.g. AAPL,MSFT,NVDA"),
    as_of: Optional[str] = Query(None, description="As-of date YYYY-MM-DD. Omit for today."),
) -> dict:
    """
    Returns risk metrics for the requested tickers as-of a date.

    Response:
    ```
    {
        "as_of": "2025-06-15",
        "count": 487,
        "metrics": {
            "AAPL": {
                "vol_60d": 0.0152, "vol_20d": 0.0148,
                "adv_20d": 12400000000, "adv_60d": 11800000000,
                "beta_252d": 1.18, "beta_60d": 1.22,
                "idio_vol_60d": 0.0089,
                "score_date": "2025-06-13", "source": "canonical"
            },
            ...
        },
        "fallback_tickers": []
    }
    ```

    Units: vol_60d, vol_20d and idio_vol_60d are DAILY std of log returns
    (not annualised; typical ~0.01-0.05). adv_* are USD notional. beta_* are
    dimensionless vs SPY.

    Tickers with insufficient history (e.g. <60 daily bars) may be missing
    from the response.

    `score_date` and `source` (Meridian#408, ADDITIVE) are on every ticker's
    entry. `score_date` is the date the numbers actually reflect - it can lag
    `as_of` (a "latest score_date <= as_of" lookup, same pattern as
    `/risk/model`'s `exposures_score_date`), so it is NOT always equal to the
    requested `as_of`. `source` is `"canonical"` (read from
    `factors.risk_metrics`, written by the Dagster asset under its own
    trading-day windows and minimum-observation floors) or `"fallback"`
    (computed on-the-fly for a ticker the asset has never scored - windows
    and floors are aligned to the asset's since #408, but the fallback does
    not apply the asset's universe/hygiene filters, so treat it as lower
    confidence). `fallback_tickers` is a convenience top-level list of every
    ticker served via `source: "fallback"`, mirroring `/risk/model`'s
    `stale_exposure_tickers`.
    """
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    if not ticker_list:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one ticker required",
        )

    if as_of:
        as_of_date = parse_iso_date(as_of, "as_of")
    else:
        as_of_date = date_cls.today()

    metrics = get_risk_metrics(ticker_list, as_of_date)
    fallback_tickers = sorted(tk for tk, m in metrics.items() if m.get("source") == "fallback")

    return {
        "as_of": as_of_date.isoformat(),
        "count": len(metrics),
        "metrics": metrics,
        "fallback_tickers": fallback_tickers,
    }
