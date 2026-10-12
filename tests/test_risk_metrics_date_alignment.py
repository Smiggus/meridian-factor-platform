"""
Regression tests for the risk_metrics date-axis alignment bug (#292).

`_compute_risk_metrics_for_date` builds TWO wide pivots from ONE long frame:

    rets_wide = prices.pivot_table(index="d", columns="ticker", values="ret_1d")
    dv        = prices.pivot_table(index="d", columns="ticker", values="dollar_volume")

`pivot_table` silently drops an index value whose aggregate is all-NaN, so
the two pivots can end up with DIFFERENT date indexes. Everything downstream
(window slices, `n_dates`, the splice guard's `date_index`) is derived from
`rets_wide.index`, so an unequal `dv` index produced:

  - a hard IndexError in splice_guard.numpy_window_has_gap ("size of axis is
    60 but size of corresponding boolean axis is 61"), and
  - a SILENT off-by-N in adv_20d/adv_60d, which sliced the last N rows of a
    differently-indexed dv.

Trigger shape: a day with equity bars and ZERO non-null ret_1d (prices
published before returns). The risk-metrics write must handle that day
rather than stop.

DB-free: monkeypatches risk_metrics_asset.pd.read_sql and
fetch_delisted_tickers, so no live PostgreSQL connection is required.
"""

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

_DAGSTER_DIR = Path(__file__).resolve().parents[1] / "dagster"
if str(_DAGSTER_DIR) not in sys.path:
    sys.path.insert(0, str(_DAGSTER_DIR))

os.environ.setdefault("PGPASS", "test-pg-pass")
os.environ.setdefault("PARTITION_START_DATE", "2010-01-02")


def _dagster_complete() -> bool:
    try:
        import dagster
    except ImportError:
        return False
    return all(
        hasattr(dagster, n)
        for n in ("asset", "asset_check", "AssetExecutionContext", "AssetCheckResult", "Output")
    )


if not _dagster_complete():
    pytest.skip("Dagster not fully installed (slim CI test image)", allow_module_level=True)

pytest.importorskip("torch", reason="risk_metrics computes on torch tensors")

from meridian_pipeline import risk_metrics_asset  # noqa: E402

_N_DAYS = 70
_BASE_DV = 1_000_000.0
_SPIKE_DV = 5_000_000.0


def _prices_frame(*, ret_nan_on_last: bool, dv_nan_on_last: bool) -> pd.DataFrame:
    """Long (ticker, d, ret_1d, dollar_volume) frame for SPY + ACME over
    _N_DAYS business days. The LAST date carries a deliberately distinctive
    dollar volume (_SPIKE_DV) so a mis-sliced ADV window is detectable, and
    optionally has ret_1d / dollar_volume all-NaN so the corresponding pivot
    drops that date."""
    days = pd.bdate_range(end="2026-07-31", periods=_N_DAYS)
    rows = []
    for i, d in enumerate(days):
        last = i == _N_DAYS - 1
        for ticker, ret in (("SPY", 0.001), ("ACME", 0.002)):
            rows.append(
                {
                    "ticker": ticker,
                    "d": d.date(),
                    "ret_1d": None if (last and ret_nan_on_last) else ret,
                    "dollar_volume": (
                        None if (last and dv_nan_on_last) else (_SPIKE_DV if last else _BASE_DV)
                    ),
                }
            )
    return pd.DataFrame(rows)


def _run(monkeypatch, frame: pd.DataFrame):
    monkeypatch.setattr(risk_metrics_asset.pd, "read_sql", lambda *a, **k: frame.copy())
    monkeypatch.setattr(risk_metrics_asset, "fetch_delisted_tickers", lambda e, d: frozenset())
    return risk_metrics_asset._compute_risk_metrics_for_date(
        None, pd.Timestamp("2026-07-31").date()
    )


def test_date_present_in_dv_absent_from_rets_does_not_crash(monkeypatch):
    """The exact production shape: the final date has dollar_volume but no
    ret_1d anywhere, so `dv` has 70 rows and `rets_wide` has 69. Before the
    fix this raised IndexError inside numpy_window_has_gap."""
    rows, counts = _run(monkeypatch, _prices_frame(ret_nan_on_last=True, dv_nan_on_last=False))
    assert rows, "expected risk metric rows despite the return-less final date"
    assert {r["ticker"] for r in rows} == {"SPY", "ACME"}


def test_return_less_date_is_excluded_from_adv_windows(monkeypatch):
    """Alignment DIRECTION check: rets_wide is the authority, so the
    return-less final date must not occupy a slot in the 20/60-day ADV
    windows. Every remaining day has dollar_volume == _BASE_DV, so a correct
    ADV is exactly _BASE_DV; the pre-fix slice (last 20 rows of a
    differently-indexed dv) would have pulled the _SPIKE_DV row in."""
    rows, _ = _run(monkeypatch, _prices_frame(ret_nan_on_last=True, dv_nan_on_last=False))
    by_ticker = {r["ticker"]: r for r in rows}
    for ticker in ("SPY", "ACME"):
        assert by_ticker[ticker]["adv_20d"] == pytest.approx(_BASE_DV)
        assert by_ticker[ticker]["adv_60d"] == pytest.approx(_BASE_DV)


def test_date_present_in_rets_absent_from_dv_does_not_crash(monkeypatch):
    """The mirror case (a date with returns but no dollar volume anywhere):
    reindexing dv onto rets_wide inserts an all-NaN dv row, which the ADV
    NaN-mask already ignores - no crash, no window shift."""
    rows, _ = _run(monkeypatch, _prices_frame(ret_nan_on_last=False, dv_nan_on_last=True))
    by_ticker = {r["ticker"]: r for r in rows}
    assert set(by_ticker) == {"SPY", "ACME"}
    for ticker in ("SPY", "ACME"):
        assert by_ticker[ticker]["adv_20d"] == pytest.approx(_BASE_DV)


def test_dv_is_reindexed_onto_rets_wide_index():
    """Source guard: if a future edit drops the reindex, the two pivots can
    silently diverge again."""
    import inspect

    src = inspect.getsource(risk_metrics_asset._compute_risk_metrics_for_date)
    assert "dv.reindex(index=rets_wide.index)" in src
    # ADV must slice on the shared returns-panel row count, never on dv's own.
    assert "dv_t[max(0, dv_t.shape[0]" not in src
