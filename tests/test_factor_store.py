"""
test_factor_store.py -- light unit tests for factor_store.py

All database calls are mocked; no PostgreSQL connection required.
Tests verify the data-shaping logic and the None-guard on missing rows.
"""

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd

from factor_store import get_stock_factors

# ----------------------------------------------------------------
# Shared test data helpers
# ----------------------------------------------------------------


def _make_daily_df():
    return pd.DataFrame(
        {
            "factor_name": ["earnings_yield", "roe"],
            "group_name": ["value", "quality"],
            "raw_value": [0.05, 0.12],
            "z_score": [1.2, -0.3],
            "direction": [1, 1],
        }
    )


def _make_groups_df():
    return pd.DataFrame(
        {
            "group_name": ["value", "quality"],
            "group_score": [0.8, -0.1],
        }
    )


def _make_composite_df():
    return pd.DataFrame({"composite_score": [0.45], "quintile": [4]})


def _sequential_read_sql(*frames):
    """Return a side_effect that yields each frame in order."""
    it = iter(frames)

    def _inner(*args, **kwargs):
        return next(it)

    return _inner


# ----------------------------------------------------------------
# Tests
# ----------------------------------------------------------------


class TestGetStockFactors:
    """factor_store.get_stock_factors() shapes DB rows into the response dict.

    _engine() is patched directly (not create_engine) - audit batch D #1
    made _engine() an lru_cache singleton, so patching create_engine would
    leak a cached engine across test functions within the same process.
    Patching the function itself is the same convention already used by
    api/routers/dcf.py's tests.
    """

    def test_returns_none_when_no_data(self):
        with (
            patch("factor_store._engine", return_value=MagicMock()),
            patch("factor_store.pd.read_sql", return_value=pd.DataFrame()),
        ):
            result = get_stock_factors("FAKE", date(2024, 6, 1))

        assert result is None

    def test_returns_dict_structure_with_data(self):
        with (
            patch("factor_store._engine", return_value=MagicMock()),
            patch(
                "factor_store.pd.read_sql",
                side_effect=_sequential_read_sql(
                    _make_daily_df(), _make_groups_df(), _make_composite_df()
                ),
            ),
        ):
            result = get_stock_factors("AAPL", date(2024, 6, 1))

        assert result is not None
        assert "factors" in result
        assert "earnings_yield" in result["factors"]
        assert "group_scores" in result
        assert "composite_score" in result
        assert "quintile" in result

    def test_factor_dict_has_expected_keys(self):
        with (
            patch("factor_store._engine", return_value=MagicMock()),
            patch(
                "factor_store.pd.read_sql",
                side_effect=_sequential_read_sql(
                    _make_daily_df(), _make_groups_df(), _make_composite_df()
                ),
            ),
        ):
            result = get_stock_factors("AAPL", date(2024, 6, 1))

        factor = result["factors"]["earnings_yield"]
        assert set(factor.keys()) >= {"raw", "z_score", "direction"}

    def test_safe_float_handles_nan(self):
        """_safe() in factor_store must not crash on NaN raw_value."""
        nan_daily = pd.DataFrame(
            {
                "factor_name": ["roe"],
                "group_name": ["quality"],
                "raw_value": [float("nan")],
                "z_score": [0.0],
                "direction": [1],
            }
        )
        groups = pd.DataFrame({"group_name": ["quality"], "group_score": [0.0]})
        composite = pd.DataFrame({"composite_score": [0.0], "quintile": [3]})

        with (
            patch("factor_store._engine", return_value=MagicMock()),
            patch(
                "factor_store.pd.read_sql",
                side_effect=_sequential_read_sql(nan_daily, groups, composite),
            ),
        ):
            result = get_stock_factors("AAPL", date(2024, 6, 1))

        assert result["factors"]["roe"]["raw"] is None


class TestEngineUrl:
    """_engine_url() must read DB connection vars from the environment."""

    def test_engine_url_uses_env_vars(self, monkeypatch):
        monkeypatch.setenv("FACTOR_DB_HOST", "test-host")
        monkeypatch.setenv("FACTOR_DB_USER", "test-user")
        monkeypatch.setenv("FACTOR_DB_PASSWORD", "test-pass")
        monkeypatch.setenv("FACTOR_DB_PORT", "5432")
        monkeypatch.setenv("FACTOR_DB_NAME", "TestDB")

        import factor_store

        url = factor_store._engine_url()

        assert "test-host" in url
        assert "test-user" in url
        assert "TestDB" in url


class TestScoreableUniverseFallback:
    """get_scoreable_universe's SPX fallback (audit batch D #3) must be
    explicitly gate-able via allow_spx_fallback, defaulting to the
    historical always-fallback behaviour."""

    def _empty_then_spx(self, spx_df):
        """read_sql side_effect: first call (daily_scores) empty, second
        call (SPX fallback) returns spx_df."""
        calls = iter([pd.DataFrame({"ticker": []}), spx_df])

        def _inner(*args, **kwargs):
            return next(calls)

        return _inner

    def test_default_falls_back_to_spx(self):
        import factor_store

        spx_df = pd.DataFrame({"ticker": ["AAPL", "MSFT"]})
        with (
            patch("factor_store._engine", return_value=MagicMock()),
            patch(
                "factor_store.pd.read_sql",
                side_effect=self._empty_then_spx(spx_df),
            ),
        ):
            tickers = factor_store.get_scoreable_universe(
                as_of=date(2026, 6, 27), excluded_sectors=[]
            )

        assert tickers == ["AAPL", "MSFT"]

    def test_allow_spx_fallback_false_returns_empty(self):
        import factor_store

        with (
            patch("factor_store._engine", return_value=MagicMock()),
            patch("factor_store.pd.read_sql", return_value=pd.DataFrame({"ticker": []})),
        ):
            tickers = factor_store.get_scoreable_universe(
                as_of=date(2026, 6, 27),
                excluded_sectors=[],
                allow_spx_fallback=False,
            )

        assert tickers == []
