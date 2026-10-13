"""
conftest.py -- shared fixtures for the Meridian test suite.

All fixtures are pure in-memory; no DB, no FMP, no FRED calls.
"""

import os

import numpy as np
import pandas as pd
import pytest

# ---- env stubs so config.py loads without real secrets ----


def pytest_configure(config):
    """Inject sentinel env vars before any module is imported."""
    os.environ.setdefault("JWT_SECRET", "test-jwt-secret-for-pytest")
    # Point ntfy at a black hole for the whole session. notify.ntfy_url()
    # resolves at call time and defaults to the LIVE homelab topic, so any
    # test touching a notifying path pages the operator's real phone and
    # tracks CI activity rather than production health.
    #
    # Second guard: notify.send_ntfy also refuses outright under pytest.
    # This override is the one that keeps the URL wrong even if that guard is
    # ever removed; the guard is the one that still holds for a test run
    # outside this conftest. Neither alone is sufficient.
    os.environ["MERIDIAN_NTFY_URL"] = "http://127.0.0.1:9/meridian-test-blackhole"


@pytest.fixture
def sample_universe_df():
    """10-ticker universe with realistic factor values including some NaN."""
    np.random.seed(42)
    n = 10
    tickers = [f"TICK{i:02d}" for i in range(n)]
    df = pd.DataFrame(
        {
            "earnings_yield": np.random.normal(0.05, 0.02, n),
            "book_yield": np.random.normal(0.15, 0.05, n),
            "roe": np.random.normal(0.12, 0.04, n),
            "momentum_12_1": np.random.normal(0.08, 0.15, n),
            "sector": (["Technology"] * 4 + ["Financials"] * 3 + ["Healthcare"] * 3),
        },
        index=tickers,
    )
    # Introduce some NaN to test robustness
    df.loc["TICK03", "earnings_yield"] = np.nan
    df.loc["TICK07", "book_yield"] = np.nan
    return df


@pytest.fixture
def small_group_series():
    """Series with only 3 non-NaN values -- tests small-group handling."""
    return pd.Series([0.1, 0.2, np.nan, 0.3, np.nan], index=["A", "B", "C", "D", "E"])


@pytest.fixture
def numeric_df_no_nan():
    """Clean 10-row DataFrame with two numeric columns, zero NaN."""
    np.random.seed(0)
    return pd.DataFrame(
        {
            "earnings_yield": np.random.normal(0.05, 0.02, 10),
            "book_yield": np.random.normal(0.15, 0.05, 10),
        },
        index=[f"T{i}" for i in range(10)],
    )
