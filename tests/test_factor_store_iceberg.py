"""
test_factor_store_iceberg.py -- M14.4: Iceberg-first / Postgres-fallback
choke point in factor_store.py.

All DB calls (Postgres and Iceberg) are mocked; no PostgreSQL connection or
live lakehouse catalog is required. Covers:
  (a) flag OFF -> pure PG path (byte-identical to pre-M14.4 behaviour)
  (b) flag ON + Iceberg reachable -> Iceberg path used, PG not touched
  (c) Iceberg read raises -> exception is swallowed, PG fallback runs
  (d) response shape is identical between the two paths for the same fixture
  (e) the API path is gated by MERIDIAN_ICEBERG_API_READ /
      lakehouse.iceberg_read_primary_api(), a flag STRUCTURALLY SEPARATE from
      MERIDIAN_ICEBERG_READ_PRIMARY / lakehouse.iceberg_read_primary() (the
      research-substrate switch parquet_io.py reads) - setting the research
      flag alone must never route the API to Iceberg (docs/18-ICEBERG_MIGRATION.md
      section 6, "Superseded decisions")
"""

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import factor_store
from factor_store import _iceberg_first, get_stock_factors


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
    it = iter(frames)

    def _inner(*args, **kwargs):
        return next(it)

    return _inner


# ----------------------------------------------------------------
# _iceberg_first helper, in isolation
# ----------------------------------------------------------------


class TestIcebergFirstHelper:
    def test_flag_off_uses_pg_and_never_touches_iceberg(self, monkeypatch):
        monkeypatch.setattr(factor_store.lakehouse, "iceberg_read_primary_api", lambda: False)
        available = MagicMock(side_effect=AssertionError("should not be called"))
        monkeypatch.setattr(factor_store.lakehouse, "daily_scores_available", available)

        pg_read = MagicMock(return_value="pg-result")
        iceberg_read = MagicMock(side_effect=AssertionError("should not be called"))

        result = _iceberg_first(pg_read, iceberg_read)

        assert result == "pg-result"
        pg_read.assert_called_once()
        iceberg_read.assert_not_called()
        available.assert_not_called()

    def test_flag_on_and_available_uses_iceberg_not_pg(self, monkeypatch):
        monkeypatch.setattr(factor_store.lakehouse, "iceberg_read_primary_api", lambda: True)
        monkeypatch.setattr(factor_store.lakehouse, "daily_scores_available", lambda: True)

        pg_read = MagicMock(side_effect=AssertionError("should not be called"))
        iceberg_read = MagicMock(return_value="iceberg-result")

        result = _iceberg_first(pg_read, iceberg_read)

        assert result == "iceberg-result"
        iceberg_read.assert_called_once()
        pg_read.assert_not_called()

    def test_flag_on_but_unavailable_falls_back_to_pg(self, monkeypatch):
        monkeypatch.setattr(factor_store.lakehouse, "iceberg_read_primary_api", lambda: True)
        monkeypatch.setattr(factor_store.lakehouse, "daily_scores_available", lambda: False)

        pg_read = MagicMock(return_value="pg-result")
        iceberg_read = MagicMock(side_effect=AssertionError("should not be called"))

        result = _iceberg_first(pg_read, iceberg_read)

        assert result == "pg-result"
        iceberg_read.assert_not_called()
        pg_read.assert_called_once()

    def test_iceberg_error_is_swallowed_and_falls_back_to_pg(self, monkeypatch, caplog):
        monkeypatch.setattr(factor_store.lakehouse, "iceberg_read_primary_api", lambda: True)
        monkeypatch.setattr(factor_store.lakehouse, "daily_scores_available", lambda: True)

        pg_read = MagicMock(return_value="pg-result")
        iceberg_read = MagicMock(side_effect=RuntimeError("catalog unreachable"))

        result = _iceberg_first(pg_read, iceberg_read)

        assert result == "pg-result"
        iceberg_read.assert_called_once()
        pg_read.assert_called_once()


# ----------------------------------------------------------------
# get_stock_factors wired through _iceberg_first
# ----------------------------------------------------------------


class TestGetStockFactorsIcebergRouting:
    def test_flag_off_reads_pg_only(self, monkeypatch):
        monkeypatch.setattr(factor_store.lakehouse, "iceberg_read_primary_api", lambda: False)
        read_daily_scores = MagicMock(side_effect=AssertionError("should not be called"))
        monkeypatch.setattr(factor_store.lakehouse, "read_daily_scores", read_daily_scores)

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

        read_daily_scores.assert_not_called()
        assert result is not None
        assert "earnings_yield" in result["factors"]

    def test_flag_on_and_available_reads_iceberg_for_daily_scores(self, monkeypatch):
        monkeypatch.setattr(factor_store.lakehouse, "iceberg_read_primary_api", lambda: True)
        monkeypatch.setattr(factor_store.lakehouse, "daily_scores_available", lambda: True)
        read_daily_scores = MagicMock(return_value=_make_daily_df())
        monkeypatch.setattr(factor_store.lakehouse, "read_daily_scores", read_daily_scores)

        # pd.read_sql should only be hit for group_scores + composite_scores
        # (2 calls), never for the daily_scores portion.
        with (
            patch("factor_store._engine", return_value=MagicMock()),
            patch(
                "factor_store.pd.read_sql",
                side_effect=_sequential_read_sql(_make_groups_df(), _make_composite_df()),
            ) as mock_read_sql,
        ):
            result = get_stock_factors("AAPL", date(2024, 6, 1))

        read_daily_scores.assert_called_once()
        _, kwargs = read_daily_scores.call_args
        assert kwargs["tickers"] == ["AAPL"]
        assert kwargs["score_date"] == date(2024, 6, 1)
        assert kwargs["as_pandas"] is True
        assert mock_read_sql.call_count == 2
        assert result is not None
        assert "earnings_yield" in result["factors"]

    def test_iceberg_exception_falls_back_to_pg_without_raising(self, monkeypatch):
        monkeypatch.setattr(factor_store.lakehouse, "iceberg_read_primary_api", lambda: True)
        monkeypatch.setattr(factor_store.lakehouse, "daily_scores_available", lambda: True)
        read_daily_scores = MagicMock(side_effect=RuntimeError("boom"))
        monkeypatch.setattr(factor_store.lakehouse, "read_daily_scores", read_daily_scores)

        with (
            patch("factor_store._engine", return_value=MagicMock()),
            patch(
                "factor_store.pd.read_sql",
                side_effect=_sequential_read_sql(
                    _make_daily_df(), _make_groups_df(), _make_composite_df()
                ),
            ),
        ):
            # Must not raise even though the iceberg read blew up.
            result = get_stock_factors("AAPL", date(2024, 6, 1))

        read_daily_scores.assert_called_once()
        assert result is not None
        assert "earnings_yield" in result["factors"]

    def test_pg_and_iceberg_paths_produce_identical_shape(self, monkeypatch):
        """Same underlying data via either path -> identical response dict."""

        def _via_pg():
            monkeypatch.setattr(factor_store.lakehouse, "iceberg_read_primary_api", lambda: False)
            with (
                patch("factor_store._engine", return_value=MagicMock()),
                patch(
                    "factor_store.pd.read_sql",
                    side_effect=_sequential_read_sql(
                        _make_daily_df(), _make_groups_df(), _make_composite_df()
                    ),
                ),
            ):
                return get_stock_factors("AAPL", date(2024, 6, 1))

        def _via_iceberg():
            monkeypatch.setattr(factor_store.lakehouse, "iceberg_read_primary_api", lambda: True)
            monkeypatch.setattr(factor_store.lakehouse, "daily_scores_available", lambda: True)
            monkeypatch.setattr(
                factor_store.lakehouse, "read_daily_scores", lambda **kw: _make_daily_df()
            )
            with (
                patch("factor_store._engine", return_value=MagicMock()),
                patch(
                    "factor_store.pd.read_sql",
                    side_effect=_sequential_read_sql(_make_groups_df(), _make_composite_df()),
                ),
            ):
                return get_stock_factors("AAPL", date(2024, 6, 1))

        pg_result = _via_pg()
        iceberg_result = _via_iceberg()

        assert pg_result == iceberg_result
        assert set(pg_result.keys()) == {
            "blend_provenance",
            "factors",
            "group_scores",
            "composite_score",
            "quintile",
            "n_groups",
            "n_weighted_groups",
        }


# ----------------------------------------------------------------
# get_scoreable_universe: explicit as_of branch routes through Iceberg
# ----------------------------------------------------------------


class TestGetScoreableUniverseIcebergRouting:
    def test_flag_on_uses_iceberg_tickers_plus_pg_sector_filter(self, monkeypatch):
        monkeypatch.setattr(factor_store.lakehouse, "iceberg_read_primary_api", lambda: True)
        monkeypatch.setattr(factor_store.lakehouse, "daily_scores_available", lambda: True)
        monkeypatch.setattr(
            factor_store.lakehouse,
            "read_daily_scores",
            lambda **kw: pd.DataFrame({"ticker": ["AAPL", "XYZ", "MSFT"]}),
        )

        # Sector-exclusion follow-up query excludes XYZ.
        with (
            patch("factor_store._engine", return_value=MagicMock()),
            patch(
                "factor_store.pd.read_sql",
                return_value=pd.DataFrame({"ticker": ["XYZ"]}),
            ),
        ):
            tickers = factor_store.get_scoreable_universe(
                as_of=date(2026, 6, 27), excluded_sectors=["Utilities"]
            )

        assert tickers == ["AAPL", "MSFT"]

    def test_flag_off_uses_pg_only(self, monkeypatch):
        monkeypatch.setattr(factor_store.lakehouse, "iceberg_read_primary_api", lambda: False)
        read_daily_scores = MagicMock(side_effect=AssertionError("should not be called"))
        monkeypatch.setattr(factor_store.lakehouse, "read_daily_scores", read_daily_scores)

        with (
            patch("factor_store._engine", return_value=MagicMock()),
            patch(
                "factor_store.pd.read_sql",
                return_value=pd.DataFrame({"ticker": ["AAPL", "MSFT"]}),
            ),
        ):
            tickers = factor_store.get_scoreable_universe(
                as_of=date(2026, 6, 27), excluded_sectors=[]
            )

        read_daily_scores.assert_not_called()
        assert tickers == ["AAPL", "MSFT"]


# ----------------------------------------------------------------
# The API read path must be structurally incapable of Iceberg-first via the
# research-substrate flag (M14.4 audit finding, HIGH: setting
# MERIDIAN_ICEBERG_READ_PRIMARY on the API deployment must never route the
# API to Iceberg).
# ----------------------------------------------------------------


class TestApiFlagIsSeparateFromResearchFlag:
    def test_iceberg_first_ignores_research_flag_when_api_flag_unset(self, monkeypatch):
        """Setting MERIDIAN_ICEBERG_READ_PRIMARY (research substrate) alone,
        with MERIDIAN_ICEBERG_API_READ left unset/default, must not route
        _iceberg_first through Iceberg."""
        monkeypatch.setenv("MERIDIAN_ICEBERG_READ_PRIMARY", "true")
        monkeypatch.delenv("MERIDIAN_ICEBERG_API_READ", raising=False)
        available = MagicMock(side_effect=AssertionError("should not be called"))
        monkeypatch.setattr(factor_store.lakehouse, "daily_scores_available", available)

        pg_read = MagicMock(return_value="pg-result")
        iceberg_read = MagicMock(side_effect=AssertionError("should not be called"))

        result = _iceberg_first(pg_read, iceberg_read)

        # Sanity: the research flag really is on, proving this isn't a
        # false-negative from a broken env read.
        assert factor_store.lakehouse.iceberg_read_primary() is True
        assert factor_store.lakehouse.iceberg_read_primary_api() is False
        assert result == "pg-result"
        pg_read.assert_called_once()
        iceberg_read.assert_not_called()
        available.assert_not_called()

    def test_iceberg_first_uses_iceberg_only_when_api_flag_explicitly_set(self, monkeypatch):
        monkeypatch.delenv("MERIDIAN_ICEBERG_READ_PRIMARY", raising=False)
        monkeypatch.setenv("MERIDIAN_ICEBERG_API_READ", "true")
        monkeypatch.setattr(factor_store.lakehouse, "daily_scores_available", lambda: True)

        pg_read = MagicMock(side_effect=AssertionError("should not be called"))
        iceberg_read = MagicMock(return_value="iceberg-result")

        result = _iceberg_first(pg_read, iceberg_read)

        assert factor_store.lakehouse.iceberg_read_primary() is False
        assert result == "iceberg-result"
        iceberg_read.assert_called_once()
        pg_read.assert_not_called()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
