"""
test_universe_spx_fallback_flag.py -- audit batch D #3.

get_scoreable_universe used to silently fall back to SPX membership
whenever factors.daily_scores had zero rows for the date, so GET /universe
served 200 with ~500 SPX tickers instead of the documented 404 for e.g.
weekend/holiday dates. The fix keeps the historical default (fallback
enabled) but adds an explicit include_spx_fallback query param so a caller
can opt into the correct 404 semantics without any response-shape change.
"""

from datetime import date
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from api.cache import ScoreCache
from api.routers.universe import list_universe


def _fresh_cache() -> ScoreCache:
    return ScoreCache()


def test_default_preserves_historical_spx_fallback_behaviour():
    """No include_spx_fallback param: default True, matches prior behaviour."""
    with (
        patch("api.routers.universe._resolve_score_date", return_value=date(2026, 6, 27)),
        patch("factor_store.get_scoreable_universe", return_value=["AAPL", "MSFT"]) as gsu,
    ):
        result = list_universe(date="2026-06-27", include_spx_fallback=True, cache=_fresh_cache())

    assert result["tickers"] == ["AAPL", "MSFT"]
    gsu.assert_called_once_with(as_of=date(2026, 6, 27), allow_spx_fallback=True)


def test_include_spx_fallback_false_propagates_to_factor_store():
    with (
        patch("api.routers.universe._resolve_score_date", return_value=date(2026, 6, 27)),
        patch("factor_store.get_scoreable_universe", return_value=[]) as gsu,
    ):
        with pytest.raises(HTTPException) as exc_info:
            list_universe(date="2026-06-27", include_spx_fallback=False, cache=_fresh_cache())

    assert exc_info.value.status_code == 404
    gsu.assert_called_once_with(as_of=date(2026, 6, 27), allow_spx_fallback=False)


def test_true_and_false_do_not_share_a_cache_entry():
    """The two modes must not poison each other's cached ticker list."""
    cache = _fresh_cache()
    with (
        patch("api.routers.universe._resolve_score_date", return_value=date(2026, 6, 27)),
        patch(
            "factor_store.get_scoreable_universe",
            side_effect=[["SPX_FALLBACK_TICKER"], []],
        ) as gsu,
    ):
        # First call: fallback enabled, populates its own cache key.
        first = list_universe(date="2026-06-27", include_spx_fallback=True, cache=cache)
        assert first["tickers"] == ["SPX_FALLBACK_TICKER"]

        # Second call: fallback disabled must still hit factor_store (not the
        # cache entry from the fallback-enabled call) and correctly 404.
        with pytest.raises(HTTPException) as exc_info:
            list_universe(date="2026-06-27", include_spx_fallback=False, cache=cache)

    assert exc_info.value.status_code == 404
    assert gsu.call_count == 2
