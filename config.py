"""config.py - API-side configuration.

Trimmed down with the live-fallback sunset (#77). The API is read-only
against PostgreSQL, so vendor-API config is no longer needed
here - the Dagster code-server keeps its own config under
dagster/meridian_pipeline/.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _require_env(key: str) -> str:
    val = os.environ.get(key)
    if not val:
        raise ValueError(f"{key} environment variable is required")
    return val


# Domain-grouped config dataclasses


@dataclass(frozen=True)
class _APIConfig:
    jwt_secret: str = field(default_factory=lambda: _require_env("JWT_SECRET"))
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = field(
        default_factory=lambda: int(os.environ.get("JWT_EXPIRE_MINUTES", "1440"))
    )
    score_cache_ttl: int = field(
        default_factory=lambda: int(os.environ.get("API_SCORE_CACHE_TTL", "3600"))
    )
    # Upper bound on in-process ScoreCache entries (0 disables the bound).
    score_cache_max_entries: int = field(
        default_factory=lambda: int(os.environ.get("API_SCORE_CACHE_MAX_ENTRIES", "256"))
    )
    port: int = 8100


@dataclass(frozen=True)
class _DataConfig:
    # Sectors to exclude from the scoreable universe served by /universe.
    # Empty by default. To exclude sectors, set
    # MERIDIAN_EXCLUDED_SECTORS="SectorA,SectorB" in the env.
    excluded_sectors: list = field(
        default_factory=lambda: [
            s.strip()
            for s in os.environ.get("MERIDIAN_EXCLUDED_SECTORS", "").split(",")
            if s.strip()
        ]
    )


# Instantiate domain configs
api = _APIConfig()
data = _DataConfig()


# Backward-compatible module-level aliases
# API Auth
JWT_SECRET: str = api.jwt_secret
JWT_ALGORITHM: str = api.jwt_algorithm
JWT_EXPIRE_MINUTES: int = api.jwt_expire_minutes

# Score cache
API_SCORE_CACHE_TTL: int = api.score_cache_ttl
API_SCORE_CACHE_MAX_ENTRIES: int = api.score_cache_max_entries

# Universe / sector filter (read by factor_store.get_scoreable_universe)
EXCLUDED_SECTORS: list = data.excluded_sectors
