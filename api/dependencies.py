from fastapi import Request

from api.cache import ScoreCache


def get_cache(request: Request) -> ScoreCache:
    """Injects the ScoreCache instance from app state."""
    return request.app.state.cache
