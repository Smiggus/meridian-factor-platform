"""
Tests for the Iceberg-first branch of fetch_factor_scores_chunks() (M14.3).

fetch_factor_scores_chunks() must:
  - take the Iceberg branch (lakehouse.read_daily_scores()) only when BOTH
    lakehouse.iceberg_read_primary() and lakehouse.daily_scores_available()
    return True;
  - otherwise fall back to the existing flat-parquet/DuckDB path unchanged;
  - yield pandas DataFrame chunks with exactly the columns
    [score_date, ticker, factor_name, z_score], chunked to approximately
    chunk_rows per yield, regardless of which path served the data.

No real DuckDB/S3/Iceberg network calls are made - the flat path is exercised
via meridian_parquet_available() returning False (short-circuits before any
DuckDB connection is built), and the Iceberg path is exercised via a
monkeypatched lakehouse module.
"""

import sys
from datetime import date
from pathlib import Path

import polars as pl
import pytest

_DAGSTER_DIR = Path(__file__).resolve().parents[1] / "dagster"
if str(_DAGSTER_DIR) not in sys.path:
    sys.path.insert(0, str(_DAGSTER_DIR))

import meridian_pipeline.lakehouse as lakehouse  # noqa: E402
import meridian_pipeline.parquet_io as pio  # noqa: E402


def _fake_frame(n_rows: int) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "score_date": [date(2024, 1, 2)] * n_rows,
            "ticker": [f"T{i:04d}" for i in range(n_rows)],
            "factor_name": ["pe_ratio"] * n_rows,
            "z_score": [0.1 * i for i in range(n_rows)],
        }
    )


def test_iceberg_branch_used_when_both_flags_true(monkeypatch):
    frame = _fake_frame(5)
    captured = {}

    def fake_read_daily_scores(**kwargs):
        captured.update(kwargs)
        return frame

    monkeypatch.setattr(lakehouse, "iceberg_read_primary", lambda: True)
    monkeypatch.setattr(lakehouse, "daily_scores_available", lambda: True)
    monkeypatch.setattr(lakehouse, "read_daily_scores", fake_read_daily_scores)

    # Guard: if the Iceberg branch is NOT taken, this would attempt a real
    # DuckDB/S3 connection and blow up loudly instead of silently passing.
    def _boom():
        raise AssertionError("flat-parquet path should not run when Iceberg is primary")

    monkeypatch.setattr(pio, "_meridian_con", _boom)

    chunks = list(
        pio.fetch_factor_scores_chunks(
            ["pe_ratio"],
            date(2024, 1, 1),
            date(2024, 1, 31),
            chunk_rows=1_000_000,
            tickers=None,
        )
    )

    assert len(chunks) == 1
    chunk = chunks[0]
    assert list(chunk.columns) == ["score_date", "ticker", "factor_name", "z_score"]
    assert len(chunk) == 5

    # Filters were forwarded to read_daily_scores with the same semantics.
    assert captured["start"] == date(2024, 1, 1)
    assert captured["end"] == date(2024, 1, 31)
    assert captured["factor_names"] == ["pe_ratio"]
    assert captured["tickers"] is None
    assert captured["columns"] == ["score_date", "ticker", "factor_name", "z_score"]


def test_iceberg_branch_chunks_respect_chunk_rows(monkeypatch):
    frame = _fake_frame(10)

    monkeypatch.setattr(lakehouse, "iceberg_read_primary", lambda: True)
    monkeypatch.setattr(lakehouse, "daily_scores_available", lambda: True)
    monkeypatch.setattr(lakehouse, "read_daily_scores", lambda **kwargs: frame)

    chunks = list(
        pio.fetch_factor_scores_chunks(
            ["pe_ratio"],
            date(2024, 1, 1),
            date(2024, 1, 31),
            chunk_rows=4,
        )
    )

    assert [len(c) for c in chunks] == [4, 4, 2]
    for c in chunks:
        assert list(c.columns) == ["score_date", "ticker", "factor_name", "z_score"]


def test_iceberg_branch_filters_null_z_score(monkeypatch):
    frame = pl.DataFrame(
        {
            "score_date": [date(2024, 1, 2)] * 3,
            "ticker": ["AAA", "BBB", "CCC"],
            "factor_name": ["pe_ratio"] * 3,
            "z_score": [0.1, None, 0.3],
        }
    )

    monkeypatch.setattr(lakehouse, "iceberg_read_primary", lambda: True)
    monkeypatch.setattr(lakehouse, "daily_scores_available", lambda: True)
    monkeypatch.setattr(lakehouse, "read_daily_scores", lambda **kwargs: frame)

    chunks = list(
        pio.fetch_factor_scores_chunks(
            ["pe_ratio"], date(2024, 1, 1), date(2024, 1, 31), chunk_rows=1_000_000
        )
    )

    assert len(chunks) == 1
    assert len(chunks[0]) == 2
    assert set(chunks[0]["ticker"]) == {"AAA", "CCC"}


def test_iceberg_branch_empty_result_yields_nothing(monkeypatch):
    empty = pl.DataFrame(
        schema={
            "score_date": pl.Date,
            "ticker": pl.Utf8,
            "factor_name": pl.Utf8,
            "z_score": pl.Float64,
        }
    )

    monkeypatch.setattr(lakehouse, "iceberg_read_primary", lambda: True)
    monkeypatch.setattr(lakehouse, "daily_scores_available", lambda: True)
    monkeypatch.setattr(lakehouse, "read_daily_scores", lambda **kwargs: empty)

    chunks = list(
        pio.fetch_factor_scores_chunks(
            ["pe_ratio"], date(2024, 1, 1), date(2024, 1, 31), chunk_rows=1_000_000
        )
    )
    assert chunks == []


def test_iceberg_branch_pages_reads_per_calendar_year(monkeypatch):
    """A multi-year range must page lakehouse.read_daily_scores() one
    calendar year at a time instead of materialising the whole range in a
    single call - bounds peak memory to one year's slice (audit finding:
    the un-paged implementation reproduced the #98/#188 whole-population OOM
    bug class the moment the Iceberg flag flips for a wide research read)."""
    calls: list[tuple[date, date]] = []

    def fake_read_daily_scores(**kwargs):
        calls.append((kwargs["start"], kwargs["end"]))
        n = 3
        return pl.DataFrame(
            {
                "score_date": [kwargs["start"]] * n,
                "ticker": [f"T{i}" for i in range(n)],
                "factor_name": ["pe_ratio"] * n,
                "z_score": [1.0] * n,
            }
        )

    monkeypatch.setattr(lakehouse, "iceberg_read_primary", lambda: True)
    monkeypatch.setattr(lakehouse, "daily_scores_available", lambda: True)
    monkeypatch.setattr(lakehouse, "read_daily_scores", fake_read_daily_scores)

    chunks = list(
        pio.fetch_factor_scores_chunks(
            ["pe_ratio"], date(2022, 6, 1), date(2024, 3, 15), chunk_rows=1_000_000
        )
    )

    # One call per calendar year touched by the range, each clipped to the
    # requested [start, end] - never a single call spanning all three years.
    assert calls == [
        (date(2022, 6, 1), date(2022, 12, 31)),
        (date(2023, 1, 1), date(2023, 12, 31)),
        (date(2024, 1, 1), date(2024, 3, 15)),
    ]
    assert sum(len(c) for c in chunks) == 9
    for c in chunks:
        assert list(c.columns) == ["score_date", "ticker", "factor_name", "z_score"]


def test_iceberg_branch_single_year_range_unpaged_call_unchanged(monkeypatch):
    """A range within one calendar year must still issue exactly one
    read_daily_scores() call spanning the original [start, end] - paging
    must not change behaviour for the common (single-year) case."""
    captured = {}

    def fake_read_daily_scores(**kwargs):
        captured.update(kwargs)
        return _fake_frame(5)

    monkeypatch.setattr(lakehouse, "iceberg_read_primary", lambda: True)
    monkeypatch.setattr(lakehouse, "daily_scores_available", lambda: True)
    monkeypatch.setattr(lakehouse, "read_daily_scores", fake_read_daily_scores)

    chunks = list(
        pio.fetch_factor_scores_chunks(
            ["pe_ratio"], date(2024, 1, 1), date(2024, 12, 31), chunk_rows=1_000_000
        )
    )

    assert captured["start"] == date(2024, 1, 1)
    assert captured["end"] == date(2024, 12, 31)
    assert len(chunks) == 1
    assert len(chunks[0]) == 5


def test_iceberg_branch_chunk_boundary_carries_across_page(monkeypatch):
    """Rows that straddle a year-page boundary must still be assembled into
    chunk_rows-sized chunks identical to what an unpaged single call would
    have produced - the leftover-carry logic must not leak undersized chunks
    mid-range."""
    pages = {
        2023: pl.DataFrame(
            {
                "score_date": [date(2023, 12, 31)] * 6,
                "ticker": [f"A{i}" for i in range(6)],
                "factor_name": ["pe_ratio"] * 6,
                "z_score": [1.0] * 6,
            }
        ),
        2024: pl.DataFrame(
            {
                "score_date": [date(2024, 1, 2)] * 4,
                "ticker": [f"B{i}" for i in range(4)],
                "factor_name": ["pe_ratio"] * 4,
                "z_score": [1.0] * 4,
            }
        ),
    }

    def fake_read_daily_scores(**kwargs):
        return pages[kwargs["start"].year]

    monkeypatch.setattr(lakehouse, "iceberg_read_primary", lambda: True)
    monkeypatch.setattr(lakehouse, "daily_scores_available", lambda: True)
    monkeypatch.setattr(lakehouse, "read_daily_scores", fake_read_daily_scores)

    chunks = list(
        pio.fetch_factor_scores_chunks(
            ["pe_ratio"], date(2023, 6, 1), date(2024, 6, 1), chunk_rows=4
        )
    )

    # 6 rows (2023) + 4 rows (2024) = 10 rows total, chunked into [4, 4, 2] -
    # exactly matching the pre-paging single-call behaviour on 10 rows.
    assert [len(c) for c in chunks] == [4, 4, 2]
    assert sum(len(c) for c in chunks) == 10


@pytest.mark.parametrize(
    "iceberg_primary,ds_available",
    [(False, True), (True, False), (False, False)],
)
def test_flat_path_used_unless_both_flags_true(monkeypatch, iceberg_primary, ds_available):
    monkeypatch.setattr(lakehouse, "iceberg_read_primary", lambda: iceberg_primary)
    monkeypatch.setattr(lakehouse, "daily_scores_available", lambda: ds_available)

    def _boom(**kwargs):
        raise AssertionError("Iceberg path should not run when either flag is False")

    monkeypatch.setattr(lakehouse, "read_daily_scores", _boom)

    # Force the flat path to fail loudly and distinctly (instead of attempting
    # a real DuckDB/S3 connection) so we can assert it - not the Iceberg
    # path - is the one that ran.
    class _FlatPathSentinel(RuntimeError):
        pass

    def _fake_meridian_con():
        raise _FlatPathSentinel("flat-parquet path reached, as expected")

    monkeypatch.setattr(pio, "_meridian_con", _fake_meridian_con)

    with pytest.raises(_FlatPathSentinel):
        list(
            pio.fetch_factor_scores_chunks(
                ["pe_ratio"], date(2024, 1, 1), date(2024, 1, 31), chunk_rows=1_000_000
            )
        )
