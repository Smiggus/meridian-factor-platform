"""GET /scores: read-only lookup of one ticker's factor scores.

This router reads `factors.daily_scores`, which the Dagster code server writes. There is no live
FMP, FRED or NLP model fallback; that path was retired (issue #77). If the database has no row
for the requested (symbol, date) the API returns 404, and the caller waits for the pipeline to
backfill that partition.
"""

from __future__ import annotations

import json

import numpy as np
from fastapi import APIRouter, HTTPException, Query, status

from api.dates import parse_iso_date
from api.factor_catalogue import FACTOR_GROUPS

router = APIRouter()


# Canonical factor -> group lookup, derived from FACTOR_GROUPS so it stays
# in lockstep with the catalogue (Meridian #59). Stamping `group` on each
# factor entry in the /scores response lets downstream consumers
# render new groups without a per-factor code change.
_FACTOR_TO_GROUP: dict[str, str] = {
    factor: group for group, factors in FACTOR_GROUPS.items() for factor in factors
}


@router.get("")
def get_scores(
    symbol: str = Query(..., description="Ticker symbol, e.g. AAPL"),
    date: str = Query(
        ...,
        description=(
            "As-of date YYYY-MM-DD. Required - the API only serves "
            "precomputed scores from factors.daily_scores."
        ),
    ),
) -> dict:
    """Factor scores for one symbol on one date, or 404 if the row is not there."""
    from factor_store import get_stock_factors

    symbol = symbol.upper().strip()

    as_of = parse_iso_date(date, "date")

    rec = get_stock_factors(symbol, as_of)
    if rec is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No factor scores in database for {symbol} on {as_of.isoformat()}",
        )

    return _build_response_from_record(symbol, as_of.isoformat(), rec)


def _build_response_from_record(symbol: str, date: str, rec: dict) -> dict:
    """Shape a factor_store.get_stock_factors() record into the response."""
    factors = {
        name: {
            "raw": _safe_float(detail.get("raw")),
            "z_score": _safe_float(detail.get("z_score")),
            "direction": int(detail.get("direction", 1)),
            # Catalogue wins; fall back to the DB-stored group_name so
            # factors not (yet) in FACTOR_GROUPS never serve group:null (#130).
            "group": _FACTOR_TO_GROUP.get(name) or detail.get("group"),
        }
        for name, detail in rec.get("factors", {}).items()
    }

    group_scores = {g: _safe_float(s) for g, s in rec.get("group_scores", {}).items()}
    provenance = rec.get("blend_provenance")
    if isinstance(provenance, str):
        # Same tolerance as factor_store._n_weighted_groups: a driver may hand
        # the jsonb back as text.
        try:
            provenance = json.loads(provenance)
        except ValueError:
            provenance = None
    if not isinstance(provenance, dict):
        provenance = {}

    return {
        "symbol": symbol,
        "date": date,
        "factors": factors,
        "group_scores": group_scores,
        "composite_score": _safe_float(rec.get("composite_score")),
        "quintile": rec.get("quintile"),
        # #329: count of non-null groups in the composite input pivot, INCLUDING
        # unweighted diagnostic groups (see n_weighted_groups below). Additive;
        # a downstream consumer's floor is calibrated on this value, so its meaning is fixed.
        "n_groups": rec.get("n_groups"),
        # RANK-003: n_groups counts every non-null group in the composite input
        # (including unweighted diagnostic groups); n_weighted_groups counts only
        # the positive-weight groups that actually entered the blend. Taken from the
        # record (factor_store computes it once); null on rows written before the
        # field existed (not backfilled).
        "n_weighted_groups": rec.get("n_weighted_groups"),
        # Keep captured precision: rounding weights compounds score rounding.
        # NULL means an older score has no captured explanation.
        "weights_used": provenance.get("weights"),
        "composite_group_scores": provenance.get("group_scores"),
        # Part of the frozen response contract. Always empty in this export.
        "regime_factors": {},
    }


def _safe_float(val) -> float | None:
    if val is None:
        return None
    try:
        f = float(val)
        return None if not np.isfinite(f) else round(f, 4)
    except (TypeError, ValueError):
        return None
