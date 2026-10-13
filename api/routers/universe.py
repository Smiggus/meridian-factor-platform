"""
api/routers/universe.py

GET /universe - list of scoreable tickers for a date.
GET /universe/scores - full cross-section of composite + group scores for a date.

Both endpoints read factors.daily_scores directly (written by the Dagster
code-server). No FMP / FRED / NLP model fallback - that path was retired
(issue #77). When no date is supplied, the latest score_date in the DB is
used.
"""

from __future__ import annotations

import logging
import math
from datetime import date as date_type
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from api.cache import ScoreCache
from api.dates import parse_iso_date
from api.dependencies import get_cache
from api.factor_catalogue import FACTOR_GROUPS

logger = logging.getLogger(__name__)

router = APIRouter()


def _json_safe(v):
    """Response guard for JSON serialisation: map non-finite floats (NaN, inf,
    -inf) to None so the downstream C# `decimal` consumer never sees them.

    NaN-only guards silently leak inf/-inf, which crash C# decimal parsing.
    Non-float values pass through unchanged. Mirrors the math.isfinite pattern
    used by api/routers/dcf.py and api/routers/scores.py.
    """
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def _resolve_score_date(date: Optional[str]) -> date_type:
    """Parse the user-supplied date, or fall back to the latest score_date
    available in factors.daily_scores."""
    if date is not None:
        return parse_iso_date(date, "date")

    from factor_store import get_available_dates

    dates = get_available_dates(limit=1)
    if not dates:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="factors.daily_scores is empty - no universe to serve",
        )
    return dates[0]


# Threshold (calendar days) beyond which the resolved score date is considered
# stale and a data_age_days field is appended to the response. 6 calendar days
# covers a long weekend (Fri->Fri) without false-alarming on a normal Monday
# refresh. Intentionally conservative - raise it if weekly batches are fine.
_STALENESS_THRESHOLD_DAYS = 6


def _data_age_days(score_date: date_type) -> Optional[int]:
    """Calendar days since score_date, but only when that exceeds the staleness
    threshold; None otherwise.

    Purely additive signal for consumers (audit batch B #161, Fix #15): the
    endpoints never 503 or change shape on staleness, they just surface the age
    so a caller can tell "latest available" apart from "current".
    """
    delta = (date_type.today() - score_date).days
    return delta if delta > _STALENESS_THRESHOLD_DAYS else None


@router.get("")
def list_universe(
    date: Optional[str] = Query(
        None, description="As-of date YYYY-MM-DD. Omit for the latest available date."
    ),
    include_spx_fallback: bool = Query(
        True,
        description=(
            "When true (default - preserves historical behaviour), a date with "
            "zero rows in factors.daily_scores falls back to SPX index membership "
            "so this endpoint still returns a non-empty list. Pass false to get a "
            "404 instead when there are genuinely no scores for the date, rather "
            "than silently serving SPX membership as if it were the scored "
            "universe (audit batch D #3)."
        ),
    ),
    cache: ScoreCache = Depends(get_cache),
) -> dict:
    """
    Returns the list of tickers with scores on the given date.

    Reads distinct tickers from factors.daily_scores. Cached by score_date
    (and by the include_spx_fallback flag, so the two modes never share a
    cache entry).
    """
    score_date = _resolve_score_date(date)

    cache_key = f"universe|{score_date.isoformat()}|{include_spx_fallback}"
    cached = cache.get(cache_key)
    if cached is not None and isinstance(cached, list):
        tickers = cached
    else:
        from factor_store import get_scoreable_universe

        tickers = get_scoreable_universe(as_of=score_date, allow_spx_fallback=include_spx_fallback)
        if not tickers:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No tickers in factors.daily_scores for {score_date}",
            )
        cache.set(cache_key, tickers)

    resp: dict = {
        "date": score_date.isoformat(),
        "count": len(tickers),
        "tickers": tickers,
    }
    age = _data_age_days(score_date)
    if age is not None:
        resp["data_age_days"] = age
    return resp


@router.get("/scores")
def universe_scores(
    date: Optional[str] = Query(
        None, description="As-of date YYYY-MM-DD. Omit for the latest available date."
    ),
    cache: ScoreCache = Depends(get_cache),
) -> dict:
    """
    Returns composite scores, group scores, and quintiles for the full
    universe on the given date.

    Reads from PostgreSQL (factors.daily_scores) via factor_store.py. Cached
    per score_date.
    """
    from factor_store import get_universe_scores

    score_date = _resolve_score_date(date)
    all_groups = list(FACTOR_GROUPS.keys())

    cache_key = f"universe_scores|{score_date.isoformat()}"
    cached = cache.get(cache_key)
    if cached is not None and isinstance(cached, list):
        clean_records = cached
    else:
        records = get_universe_scores(score_date)
        if records is None or len(records) == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No factor scores in database for {score_date}",
            )

        # Enrich with sector classification (DB-backed)
        try:
            from sector_classifier import get_sector_map

            tickers = [r["symbol"] for r in records]
            sector_map = get_sector_map(tickers)
            for r in records:
                r["sector"] = sector_map.get(r["symbol"])
        except Exception:
            for r in records:
                r["sector"] = None

        # Clean NaN / inf for JSON serialisation (inf/-inf crash C# decimal)
        clean_records = [{k: _json_safe(v) for k, v in r.items()} for r in records]

        cache.set(cache_key, clean_records)

    resp: dict = {
        "date": score_date.isoformat(),
        "count": len(clean_records),
        "groups": all_groups,
        "scores": clean_records,
    }
    age = _data_age_days(score_date)
    if age is not None:
        resp["data_age_days"] = age
    return resp
