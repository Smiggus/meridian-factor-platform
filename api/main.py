from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI

import config
from api.auth import verify_token
from api.cache import ScoreCache
from api.routers import admin, risk, scores, universe


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.cache = ScoreCache(
        ttl_seconds=config.API_SCORE_CACHE_TTL,
        max_entries=config.API_SCORE_CACHE_MAX_ENTRIES,
    )
    yield


app = FastAPI(
    title="Meridian Factor API",
    description=(
        "Standalone factor scoring engine. "
        "Ingests FMP data, computes cross-sectional factor scores, "
        "and exposes them via authenticated REST endpoints."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

_auth = [Depends(verify_token)]

# /health is unauthed; the cache-mutating admin routes carry their own
# Depends(require_admin), which wraps verify_token
app.include_router(admin.router, prefix="", tags=["admin"])
app.include_router(scores.router, prefix="/scores", tags=["scores"], dependencies=_auth)
app.include_router(universe.router, prefix="/universe", tags=["universe"], dependencies=_auth)
app.include_router(risk.router, prefix="/risk", tags=["risk"], dependencies=_auth)
