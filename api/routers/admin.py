"""
api/routers/admin.py

Operational endpoints:
  GET  /health                  -- uptime + cache stats (no auth)
  POST /cache/invalidate        -- flush in-process score cache (admin only, API-004)
  POST /admin/cache/warm        -- invalidate cache entries for a date so the next request re-hydrates (admin only, API-004)

The private service serves further routes from this router; they are not part
of this export.
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


# ─────────────────────────────────────────────
# GET /health  (no auth)
# ─────────────────────────────────────────────


@router.get("/health")
def health(cache: ScoreCache = Depends(get_cache)) -> dict:
    """
    Returns API uptime and cache statistics.
    Does not require authentication.
    """
    return {
        "status": "ok",
        "uptime_seconds": round(time.monotonic() - _start_time, 1),
        "cache": cache.stats(),
    }


# ─────────────────────────────────────────────
# POST /cache/invalidate  (admin required, API-004)
# ─────────────────────────────────────────────


@router.post("/cache/invalidate", dependencies=[Depends(require_admin)])
def invalidate_cache(cache: ScoreCache = Depends(get_cache)) -> dict:
    """
    Flushes all cached universe and score data.
    The next request to /scores or /universe/scores will trigger a fresh fetch.
    """
    cleared = cache.invalidate()
    return {
        "cleared": cleared,
        "message": f"Cleared {cleared} cache entries. Next request will re-score.",
    }


# ─────────────────────────────────────────────
# POST /admin/cache/warm  (admin required, API-004)
# ─────────────────────────────────────────────


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
    """
    Invalidate in-process cache entries for *date* so the next consumer
    request re-hydrates from PostgreSQL with the freshly written scores.

    Called by composite_cache_warm_sensor immediately after composite_scores
    materialises for a partition, closing the write->serve gap where the
    stale cache would otherwise serve old data until TTL expiry.

    The endpoint is intentionally simple: it does NOT pre-populate the cache
    (that would couple it to the scoring params the eventual caller will use).
    Instead it clears the relevant slot so the first real request pays one DB
    round-trip and then populates the cache for subsequent callers.

    With ?date=YYYY-MM-DD: clears all cache keys that contain the date string
    (covers full-universe and per-ticker calls for that date). Returns the
    count of cleared entries.

    Without ?date=: flushes all entries (same as POST /cache/invalidate).
    """
    if date is None:
        cleared = cache.invalidate()
        return {
            "date": None,
            "cleared": cleared,
            "message": f"Cleared {cleared} cache entries (full flush).",
        }

    date_str = parse_iso_date(date, "date").isoformat()

    # Scan and evict any cache key that embeds the date string.
    # Cache keys are built as "<tickers>|<date>|<period>" by make_score_key(),
    # so a substring match on the date is correct and cheap for a small cache.
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
