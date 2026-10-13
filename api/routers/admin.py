"""Operational routes: /health, and the two cache routes that need admin rights (API-004).

The private service serves more routes from this router. They are not in this export.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from fastapi import APIRouter, Depends, Query

from api.auth import require_admin
from api.cache import ScoreCache
from api.dates import parse_iso_date
from api.dependencies import get_cache

logger = logging.getLogger(__name__)

router = APIRouter()

_start_time = time.monotonic()


@router.get("/health")
def health(cache: ScoreCache = Depends(get_cache)) -> dict:
    """Uptime and cache statistics. No authentication."""
    return {
        "status": "ok",
        "uptime_seconds": round(time.monotonic() - _start_time, 1),
        "cache": cache.stats(),
    }


@router.post("/cache/invalidate", dependencies=[Depends(require_admin)])
def invalidate_cache(cache: ScoreCache = Depends(get_cache)) -> dict:
    """Drop every cached universe and score entry; the next request refetches."""
    cleared = cache.invalidate()
    return {
        "cleared": cleared,
        "message": f"Cleared {cleared} cache entries. Next request will re-score.",
    }


@router.post("/admin/cache/warm", dependencies=[Depends(require_admin)])
def warm_cache(
    date: Optional[str] = Query(
        None,
        description=(
            "Score date YYYY-MM-DD whose cache entries should be invalidated. "
            "Omit to flush all entries (equivalent to /cache/invalidate)."
        ),
    ),
    cache: ScoreCache = Depends(get_cache),
) -> dict:
    """Clear cached entries for a score date, or everything when no date is given.

    composite_cache_warm_sensor calls this once composite_scores has materialised a
    partition, so readers do not get the old scores until the TTL runs out. Despite the name
    it does not fill the cache. The first real request after the clear pays the database
    round trip and repopulates it.

    With a date it removes every key that contains the date string, which covers
    full-universe and per-ticker calls alike, and returns how many it removed.
    """
    if date is None:
        cleared = cache.invalidate()
        return {
            "date": None,
            "cleared": cleared,
            "message": f"Cleared {cleared} cache entries (full flush).",
        }

    date_str = parse_iso_date(date, "date").isoformat()

    # ScoreCache has no per-date invalidation, so this takes its lock and edits the store.
    # Keys are "<tickers>|<date>|<period>" (make_score_key), so a substring match works.
    with cache._lock:
        to_evict = [k for k in list(cache._store) if date_str in k]
        for k in to_evict:
            del cache._store[k]
    cleared = len(to_evict)

    logger.info("cache warm (invalidate) for %s: cleared %d entries", date_str, cleared)
    return {
        "date": date_str,
        "cleared": cleared,
        "message": f"Cleared {cleared} cache entries for {date_str}.",
    }
