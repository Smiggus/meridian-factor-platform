"""
tests/test_risk_metrics_fallback_parity.py

Regression tests for Meridian#408: `risk_metrics.py::_compute_on_the_fly`
(the fallback estimator behind /risk/metrics and /dcf) used CALENDAR-day
windows and minimum-observation thresholds up to an order of magnitude
laxer than the canonical Dagster asset, and fired silently whenever
factors.risk_metrics had no row for the EXACT requested date - the default
path on any weekend, holiday, or before phase 2 completes for today.

Two independent things had to change:
  1. The pre-computed read must be a "latest score_date <= as_of" lookup
     (like the #215/#375 pattern already used for factor_covariance /
     group_scores), so a weekend/holiday request is served the prior
     trading day's CANONICAL row instead of falling to the fallback at all.
  2. Whatever still reaches the fallback must use the same trading-day
     windows and minimum-observation floors as the asset, and the response
     must say which estimator produced each value.

DB-free: monkeypatches risk_metrics.pd.read_sql (dispatching on the SQL
text, same idiom as tests/test_api_risk_asof.py) and a stub engine's
.connect() for the MAX(score_date) lookup. No live PostgreSQL connection
required.
"""

import os
from datetime import date, timedelta

os.environ.setdefault("PGPASS", "test-pg-pass")

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import risk_metrics  # noqa: E402

FRIDAY = date(2026, 9, 11)
SATURDAY = date(2026, 9, 12)


class _StubRow:
    def __init__(self, value):
        self._value = value

    def fetchone(self):
        return (self._value,) if self._value is not None else None


class _StubConn:
    """Answers the MAX(score_date) lookup get_risk_metrics does via
    engine.connect() before touching pd.read_sql."""

    def __init__(self, max_date):
        self._max_date = max_date

    def execute(self, stmt, params=None):
        assert "MAX(score_date)" in str(stmt)
        assert "factors.risk_metrics" in str(stmt)
        return _StubRow(self._max_date)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _StubEngine:
    def __init__(self, max_date):
        self._max_date = max_date

    def connect(self):
        return _StubConn(self._max_date)


# ─────────────────────────────────────────────
# The reported defect, end to end
# ─────────────────────────────────────────────


def test_weekend_request_served_from_fridays_canonical_row_not_the_fallback(monkeypatch):
    """FAILS ON MAIN. Main's pre-computed read is an EXACT match on the
    requested date (`WHERE score_date = :d`), so a Saturday request always
    misses even when Friday's canonical row exists, and falls straight
    through to the far-laxer on-the-fly fallback with nothing in the
    response saying so. After #408, get_risk_metrics() must resolve
    "latest score_date <= as_of" first, serve Friday's row labelled
    `source: "canonical"`, and never invoke the fallback at all."""
    monkeypatch.setattr(risk_metrics, "_engine", lambda: _StubEngine(FRIDAY))

    def _fake_read_sql(stmt, engine, params=None):
        sql = str(stmt)
        assert "factors.risk_metrics" in sql
        # The defect under test: main queries with params["d"] == the
        # REQUESTED date (Saturday); the fix must query the RESOLVED date
        # (Friday) returned by the MAX(score_date) lookup above.
        assert params["d"] == FRIDAY, (
            f"expected the resolved score_date {FRIDAY}, got {params['d']!r} "
            "- this is the #408 exact-date-match bug if it fired"
        )
        return pd.DataFrame(
            [
                {
                    "ticker": "AAPL",
                    "vol_60d": 0.02,
                    "vol_20d": 0.018,
                    "adv_20d": 1.0e9,
                    "adv_60d": 1.0e9,
                    "beta_252d": 1.10,
                    "beta_60d": 1.05,
                    "idio_vol_60d": 0.01,
                }
            ]
        )

    monkeypatch.setattr(risk_metrics.pd, "read_sql", _fake_read_sql)

    # A fallback that looks nothing like main's real one (no score_date/
    # source stamping) - it must never be reached once a canonical row is
    # found for the resolved date.
    def _stub_fallback(engine, tickers, as_of_date):
        return {
            "AAPL": {
                "vol_60d": 0.5,
                "vol_20d": 0.5,
                "adv_20d": 1.0,
                "adv_60d": 1.0,
                "beta_252d": 999.0,
                "beta_60d": 999.0,
                "idio_vol_60d": None,
            }
        }

    monkeypatch.setattr(risk_metrics, "_compute_on_the_fly", _stub_fallback)

    result = risk_metrics.get_risk_metrics(["AAPL"], SATURDAY)

    assert "AAPL" in result
    assert result["AAPL"]["source"] == "canonical"
    assert result["AAPL"]["score_date"] == FRIDAY.isoformat()
    assert result["AAPL"]["beta_252d"] == pytest.approx(1.10)


def test_true_fallback_is_labelled_and_dated(monkeypatch):
    """FAILS ON MAIN. When there is genuinely no precomputed row on or
    before as_of, the value served must still say it is an estimate. Main's
    fallback dict carries only the seven numeric keys - indistinguishable
    from a canonical row."""
    monkeypatch.setattr(risk_metrics, "_engine", lambda: _StubEngine(None))

    def _stub_fallback(engine, tickers, as_of_date):
        return risk_metrics._df_to_dict(
            pd.DataFrame(
                [
                    {
                        "ticker": "AAPL",
                        "vol_60d": 0.02,
                        "vol_20d": 0.018,
                        "adv_20d": 1.0e9,
                        "adv_60d": 1.0e9,
                        "beta_252d": 1.10,
                        "beta_60d": 1.05,
                        "idio_vol_60d": 0.01,
                    }
                ]
            ),
            score_date=as_of_date,
            source="fallback",
        )

    monkeypatch.setattr(risk_metrics, "_compute_on_the_fly", _stub_fallback)

    result = risk_metrics.get_risk_metrics(["AAPL"], SATURDAY)

    assert result["AAPL"]["source"] == "fallback"
    assert result["AAPL"]["source"] != "canonical"
    assert result["AAPL"]["score_date"] == SATURDAY.isoformat()


# ─────────────────────────────────────────────
# Source guard: the laxer calendar-day windows/thresholds must be gone
# ─────────────────────────────────────────────


def test_fallback_source_matches_the_canonical_windows_and_floors():
    """FAILS ON MAIN. Structural guard so a future edit cannot quietly
    reintroduce calendar-day windows or the old order-of-magnitude-laxer
    thresholds (min_obs 30 for a 252-observation requirement, an effective
    non-floor of 20 for vol_20d/beta_60d)."""
    import inspect

    src = inspect.getsource(risk_metrics._compute_on_the_fly)

    # New: trading-day calendar anchored on the market (SPY), not a naive
    # calendar-day INTERVAL, and the named min-obs constants mirroring the
    # canonical asset (55/60d, 18/20d, 240/252d).
    assert "_trading_day_calendar" in src
    assert "_MIN_OBS_60D" in src
    assert "_MIN_OBS_20D" in src
    assert "_MIN_OBS_252D" in src

    # Old: the laxer thresholds/windows this issue reported must be gone.
    assert "INTERVAL '380 days'" not in src
    assert "INTERVAL '90 days'" not in src
    assert "HAVING COUNT(*) >= 30" not in src
    assert "HAVING COUNT(*) >= 5" not in src

    assert risk_metrics._MIN_OBS_60D == 55
    assert risk_metrics._MIN_OBS_20D == 18
    assert risk_metrics._MIN_OBS_252D == 240


def test_idio_vol_inputs_use_sample_not_population_variance():
    """Structural guard for a bug found in review: `var_r60`/`var_m60` (the
    inputs to the idio_vol_60d residual-variance identity) must come from
    VAR_SAMP (n-1 denominator), matching vol_60d's own STDDEV (Postgres
    STDDEV = STDDEV_SAMP) and the canonical asset's (n-1)-normalised
    residual variance. VAR_POP (n denominator) here would silently
    under-state idio_vol_60d by sqrt((n-1)/n) (~1% at min_obs=55) - beta_60d/
    beta_252d themselves are unaffected (the n cancels in a cov/var ratio),
    so this only shows up in idio_vol_60d and is easy to miss on inspection.
    """
    import inspect

    src = inspect.getsource(risk_metrics._compute_on_the_fly)

    assert "VAR_SAMP(r) FILTER (WHERE d >= :d60) AS var_r60" in src
    assert "VAR_SAMP(m_r) FILTER (WHERE d >= :d60) AS var_m60" in src
    assert "VAR_POP(r) FILTER (WHERE d >= :d60) AS var_r60" not in src
    assert "VAR_POP(m_r) FILTER (WHERE d >= :d60) AS var_m60" not in src


# ─────────────────────────────────────────────
# Runtime check of the threshold masking itself
# ─────────────────────────────────────────────


def test_beta_252d_below_canonical_min_obs_is_masked_to_none(monkeypatch):
    """Direct runtime check of _compute_on_the_fly: a ticker with 30 matched
    (stock, SPY) observations in the 252-trading-day window - enough to pass
    the OLD `HAVING COUNT(*) >= 30` - must be masked to None under the
    canonical asset's min_obs=240. This is the exact gap the issue's evidence
    named: garbage betas came from running with only a
    small number of observations."""
    calendar = [FRIDAY - timedelta(days=i) for i in range(252)]

    def _fake_read_sql(stmt, engine, params=None):
        sql = str(stmt)
        if "DISTINCT ts_event::date" in sql:
            return pd.DataFrame({"d": calendar})
        if "raw_close" in sql:
            return pd.DataFrame(
                {
                    "ticker": ["THIN"],
                    "adv_20d": [1.0e6],
                    "n20": [20],
                    "adv_60d": [1.0e6],
                    "n60": [60],
                }
            )
        if "beta_252d" in sql:
            # 30 matched observations - the OLD floor let this through.
            return pd.DataFrame(
                {
                    "ticker": ["THIN"],
                    "beta_252d": [1.4],
                    "n252": [30],
                    "beta_60d": [1.2],
                    "var_r60": [0.0004],
                    "var_m60": [0.0002],
                    "n60": [55],
                }
            )
        if "vol_60d" in sql:
            return pd.DataFrame(
                {"ticker": ["THIN"], "vol_60d": [0.02], "n60": [60], "vol_20d": [0.02], "n20": [20]}
            )
        raise AssertionError(f"unexpected query: {sql}")

    monkeypatch.setattr(risk_metrics.pd, "read_sql", _fake_read_sql)

    result = risk_metrics._compute_on_the_fly(object(), ["THIN"], FRIDAY)

    assert result["THIN"]["beta_252d"] is None
    # beta_60d had enough observations (55, the min_obs floor) and must
    # still be served.
    assert result["THIN"]["beta_60d"] == pytest.approx(1.2)
    assert result["THIN"]["source"] == "fallback"


def test_idio_vol_60d_matches_direct_sample_residual_variance(monkeypatch):
    """Regression test for a bug found in review: the residual-variance
    identity Var(r - beta*m) = Var(r) - beta^2 * Var(m) only reduces to the
    canonical asset's own (n-1)-normalised residual variance when var_r60/
    var_m60 are themselves SAMPLE variance (VAR_SAMP), matching vol_60d's
    own STDDEV (Postgres STDDEV = STDDEV_SAMP). Feeding it POPULATION
    variance (VAR_POP, the n-not-(n-1) aggregate) instead under-states
    idio_vol_60d by a systematic sqrt((n-1)/n) factor (~1% at min_obs=55) -
    silent because it is a small, direction-consistent bias, not a crash.

    This test does not care which SQL aggregate produced the numbers; it
    independently recomputes the true sample residual variance from a raw
    (ticker, market) return series and checks _compute_on_the_fly's output
    matches it, so it would have failed against a VAR_POP-based var_r60/
    var_m60 regardless of how the SQL text itself is phrased.
    """
    import numpy as np

    rng = np.random.default_rng(1234)
    n = 60
    m = rng.normal(0, 0.011, n)
    r = 1.35 * m + rng.normal(0, 0.007, n)

    rbar, mbar = r.mean(), m.mean()
    cov_s = ((r - rbar) * (m - mbar)).sum() / (n - 1)
    var_m_s = ((m - mbar) ** 2).sum() / (n - 1)
    var_r_s = ((r - rbar) ** 2).sum() / (n - 1)
    beta = cov_s / var_m_s

    resid = r - beta * m
    expected_idio_vol = np.sqrt(((resid - resid.mean()) ** 2).sum() / (n - 1))

    calendar = [FRIDAY - timedelta(days=i) for i in range(252)]

    def _fake_read_sql(stmt, engine, params=None):
        sql = str(stmt)
        if "DISTINCT ts_event::date" in sql:
            return pd.DataFrame({"d": calendar})
        if "raw_close" in sql:
            return pd.DataFrame(
                {
                    "ticker": ["ACME"],
                    "adv_20d": [1.0e6],
                    "n20": [20],
                    "adv_60d": [1.0e6],
                    "n60": [60],
                }
            )
        if "beta_252d" in sql:
            return pd.DataFrame(
                {
                    "ticker": ["ACME"],
                    "beta_252d": [beta],
                    "n252": [n],
                    "beta_60d": [beta],
                    "var_r60": [var_r_s],
                    "var_m60": [var_m_s],
                    "n60": [n],
                }
            )
        if "vol_60d" in sql:
            return pd.DataFrame(
                {"ticker": ["ACME"], "vol_60d": [0.02], "n60": [n], "vol_20d": [0.02], "n20": [20]}
            )
        raise AssertionError(f"unexpected query: {sql}")

    monkeypatch.setattr(risk_metrics.pd, "read_sql", _fake_read_sql)

    result = risk_metrics._compute_on_the_fly(object(), ["ACME"], FRIDAY)

    assert result["ACME"]["idio_vol_60d"] == pytest.approx(expected_idio_vol, rel=1e-9)


# ─────────────────────────────────────────────
# Edge cases called out in review
# ─────────────────────────────────────────────


def test_empty_calendar_returns_empty_dict(monkeypatch):
    """as_of_date predates any OHLCV history (empty SPY calendar) - must
    fail closed to an empty dict rather than computing window boundaries
    against an empty list."""

    def _fake_read_sql(stmt, engine, params=None):
        assert "DISTINCT ts_event::date" in str(stmt)
        return pd.DataFrame({"d": []})

    monkeypatch.setattr(risk_metrics.pd, "read_sql", _fake_read_sql)

    result = risk_metrics._compute_on_the_fly(object(), ["AAPL"], date(2001, 1, 1))

    assert result == {}


def test_canonical_row_with_all_null_metrics_is_not_retried_via_fallback(monkeypatch):
    """A ticker the canonical asset scored (a row exists on the resolved
    score_date) but with every metric NULL - e.g. too new to clear the
    asset's own min_obs floors as of that date - must be served as-is with
    `source: "canonical"`. It must NOT be silently re-routed to the fallback,
    which would compute against a different reference date (as_of, not the
    resolved score_date) and blur which estimator actually produced the
    ticker's numbers."""
    monkeypatch.setattr(risk_metrics, "_engine", lambda: _StubEngine(FRIDAY))

    def _fake_read_sql(stmt, engine, params=None):
        assert "factors.risk_metrics" in str(stmt)
        assert params["d"] == FRIDAY
        return pd.DataFrame(
            [
                {
                    "ticker": "NEWCO",
                    "vol_60d": None,
                    "vol_20d": None,
                    "adv_20d": None,
                    "adv_60d": None,
                    "beta_252d": None,
                    "beta_60d": None,
                    "idio_vol_60d": None,
                }
            ]
        )

    monkeypatch.setattr(risk_metrics.pd, "read_sql", _fake_read_sql)

    def _fail_if_called(engine, tickers, as_of_date):
        raise AssertionError(
            "on-the-fly fallback must not run for a ticker with an existing "
            "(all-NULL) canonical row"
        )

    monkeypatch.setattr(risk_metrics, "_compute_on_the_fly", _fail_if_called)

    result = risk_metrics.get_risk_metrics(["NEWCO"], SATURDAY)

    assert result["NEWCO"]["source"] == "canonical"
    assert result["NEWCO"]["score_date"] == FRIDAY.isoformat()
    assert result["NEWCO"]["beta_252d"] is None
    assert result["NEWCO"]["vol_60d"] is None
