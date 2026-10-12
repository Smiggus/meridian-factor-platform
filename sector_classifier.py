"""
sector_classifier.py

Fetches and caches sector classifications for a ticker universe.
Used by the API for sector enrichment of universe responses.

DB-only: reads fundamentals.company_profile in PostgreSQL (populated by LDP).
The FMP /stable/profile fallback was retired with the live-fallback sunset
(#77) - the API is read-only against the DB.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# In-memory cache TTL (populated once per process lifetime, refreshed on restart)
_sector_cache: dict[str, str] = {}
_cache_loaded = False


# -----------------------------------------------
# DB-backed sector lookup (primary)
# -----------------------------------------------


def _load_sectors_from_db() -> dict[str, str]:
    """Load all sector mappings from fundamentals.company_profile."""
    global _sector_cache, _cache_loaded
    try:
        from factor_store import _engine

        engine = _engine()
        from sqlalchemy import text

        df = pd.read_sql(
            text(
                "SELECT ticker, sector FROM fundamentals.company_profile WHERE sector IS NOT NULL"
            ),
            engine,
        )
        result = dict(zip(df["ticker"], df["sector"]))
        _sector_cache = result
        _cache_loaded = True
        logger.info("Loaded %d sector mappings from DB", len(result))
        return result
    except Exception as e:
        logger.warning("Failed to load sectors from DB: %s", e)
        return {}


def _ensure_cache():
    """Load sector cache from DB if not already loaded."""
    global _cache_loaded
    if not _cache_loaded:
        _load_sectors_from_db()


# -----------------------------------------------
# Public API
# -----------------------------------------------


def get_sector(ticker: str) -> str | None:
    """Returns the GICS sector for a ticker from the DB cache, or None."""
    _ensure_cache()
    return _sector_cache.get(ticker)


def get_sector_map(tickers: list[str]) -> dict[str, str]:
    """
    Returns a dict of ticker -> sector for the given universe.
    Bulk lookup from the DB-backed in-process cache. Tickers without a
    fundamentals.company_profile row are simply omitted from the result.
    """
    _ensure_cache()
    return {t: _sector_cache[t] for t in tickers if t in _sector_cache}


def invalidate_cache():
    """Force reload from DB on next call."""
    global _cache_loaded
    _cache_loaded = False
    _sector_cache.clear()


# -----------------------------------------------
# Sector dummy matrix
# -----------------------------------------------


def build_sector_dummies(
    tickers: list[str],
    sector_map: dict[str, str],
) -> pd.DataFrame:
    """
    Builds a DataFrame of 0/1 sector dummy variables, indexed by ticker.
    Used by orthogonalise() in scorer.py for OLS residualisation.
    """
    sectors = pd.Series(sector_map).reindex(tickers).fillna("Unknown")
    dummies = pd.get_dummies(sectors, prefix="sector", dtype=float)
    if len(dummies.columns) > 1:
        dummies = dummies.iloc[:, 1:]
    return dummies


# -----------------------------------------------
# Diagnostics
# -----------------------------------------------


def sector_summary(sector_map: dict[str, str]) -> pd.DataFrame:
    """Returns a count breakdown of sectors in the universe."""
    s = pd.Series(sector_map)
    counts = s.value_counts().reset_index()
    counts.columns = ["sector", "count"]
    counts["pct"] = (counts["count"] / counts["count"].sum() * 100).round(1)
    return counts


if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "GOOGL", "NVDA", "JPM", "JNJ", "WMT"]
    print("[sector_classifier] Fetching sectors (DB-backed)...")
    sector_map = get_sector_map(tickers)
    for t, s in sector_map.items():
        print(f"  {t:<8} {s}")
    print(f"\n{len(sector_map)} / {len(tickers)} tickers classified")
