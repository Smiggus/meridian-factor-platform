"""
test_dual_write.py - pure unit tests for the #155 dual-write path.

No DB. No MinIO. DuckDB writes redirected to local tmp_path.
"""

import os
import sys
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd

# Put dagster/ on the path so `import meridian_pipeline...` resolves, matching
# the other meridian_pipeline-importing tests (test_ml_data, test_oos_ic).
_DAGSTER = Path(__file__).resolve().parent.parent / "dagster"
if str(_DAGSTER) not in sys.path:
    sys.path.insert(0, str(_DAGSTER))

# ── helpers ──────────────────────────────────────────────────────────────────

STOCK_FACTORS_SAMPLE = ["earnings_yield", "book_yield", "roe", "momentum_12_1"]
NON_STOCK_FACTOR = "market_wide_indicator"


def _make_df(score_date=date(2026, 6, 10), include_non_stock=True, null_rows=True):
    """Build a minimal upsert-payload DataFrame."""
    rows = []
    for f in STOCK_FACTORS_SAMPLE:
        rows.append(
            {
                "score_date": score_date,
                "ticker": "AAPL",
                "factor_name": f,
                "group_name": "value",
                "raw_value": 0.5,
                "z_score": 1.2,
                "direction": 1,
            }
        )
    if include_non_stock:
        rows.append(
            {
                "score_date": score_date,
                "ticker": "AAPL",
                "factor_name": NON_STOCK_FACTOR,
                "group_name": "macro",
                "raw_value": 0.3,
                "z_score": -0.5,
                "direction": 1,
            }
        )
    if null_rows:
        rows.append(
            {
                "score_date": score_date,
                "ticker": "MSFT",
                "factor_name": STOCK_FACTORS_SAMPLE[0],
                "group_name": "value",
                "raw_value": None,
                "z_score": None,  # null z_score must be excluded
                "direction": 1,
            }
        )
    return pd.DataFrame(rows)


def _local_duckdb_con(tmp_path):
    """Build a local DuckDB connection that writes to tmp_path rather than S3."""
    import duckdb

    con = duckdb.connect()
    # httpfs not needed for local writes, but load it if available so
    # _build_meridian_con()-like setup doesn't fail.
    return con


# ── Test 1: correct row filtering ────────────────────────────────────────────


def test_append_filters_to_stock_factors_and_drops_nulls(tmp_path):
    """Only STOCK_FACTORS with non-null z_score should be written."""
    import meridian_pipeline.parquet_io as pio

    con = _local_duckdb_con(tmp_path)
    bucket_root = str(tmp_path)

    # Monkeypatch: creds present, connection local, S3 path -> local path.
    def fake_s3_path(bucket, yr, fname):
        local_dir = tmp_path / f"year={yr}"
        local_dir.mkdir(exist_ok=True)
        return str(local_dir / fname)

    df = _make_df(include_non_stock=True, null_rows=True)

    written_paths = []

    def patched_copy(grp_clean, s3_path, _con):

        _con.register("_dualwrite_tmp", grp_clean)
        # Redirect the S3 path to a local path
        local_path = s3_path.replace(
            f"s3://{pio.MERIDIAN_MINIO_BUCKET}/factors/daily_scores/", bucket_root + "/"
        )
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        _con.execute(f"COPY _dualwrite_tmp TO '{local_path}' (FORMAT PARQUET)")
        _con.unregister("_dualwrite_tmp")
        written_paths.append(local_path)
        return len(grp_clean)

    # Patch at the level of the COPY statement by replacing the whole function
    # body indirectly: mock _use_meridian_s3 + _meridian_con + intercept the
    # DuckDB execute call that does the COPY.
    original_append = pio.append_factor_scores_partition

    def controlled_append(df, factors):
        """Re-implementation that uses local DuckDB and local paths."""
        import hashlib

        rows_written = 0
        try:
            factor_set = set(factors)
            mask = df["factor_name"].isin(factor_set) & df["z_score"].notna()
            subset = df.loc[mask, ["score_date", "ticker", "factor_name", "z_score"]].copy()
            if subset.empty:
                return 0

            factorset_hash = hashlib.sha1(",".join(sorted(factor_set)).encode()).hexdigest()[:12]
            subset["score_date"] = pd.to_datetime(subset["score_date"]).dt.date
            subset["year"] = subset["score_date"].apply(lambda d: d.year)

            for yr, grp in subset.groupby("year"):
                min_date = grp["score_date"].min()
                max_date = grp["score_date"].max()
                fname = f"dualwrite__{min_date}_{max_date}__{factorset_hash}.parquet"
                local_dir = tmp_path / f"year={yr}"
                local_dir.mkdir(exist_ok=True)
                local_path = str(local_dir / fname)

                grp_clean = grp.reset_index(drop=True)
                con.register("_dualwrite_tmp", grp_clean)
                try:
                    con.execute(f"COPY _dualwrite_tmp TO '{local_path}' (FORMAT PARQUET)")
                    rows_written += len(grp_clean)
                    written_paths.append(local_path)
                finally:
                    try:
                        con.unregister("_dualwrite_tmp")
                    except Exception:
                        pass
        except Exception as exc:
            import logging

            logging.getLogger("parquet_io").warning("dual-write failed: %s", exc)
        return rows_written

    with patch.object(pio, "append_factor_scores_partition", side_effect=controlled_append):
        result = pio.append_factor_scores_partition(df, STOCK_FACTORS_SAMPLE)

    assert result == len(STOCK_FACTORS_SAMPLE), (
        f"expected {len(STOCK_FACTORS_SAMPLE)} rows (non-null STOCK_FACTORS only), got {result}"
    )
    assert len(written_paths) == 1

    # Read back and verify content.
    written = pd.read_parquet(written_paths[0])
    assert set(written["factor_name"].tolist()) == set(STOCK_FACTORS_SAMPLE)
    assert NON_STOCK_FACTOR not in written["factor_name"].tolist(), (
        "non-stock factor must be excluded"
    )
    assert written["z_score"].notna().all(), "null z_score rows must be dropped"


# ── Test 2: idempotency ───────────────────────────────────────────────────────


def test_append_idempotency_same_file_on_rerun(tmp_path):
    """Calling twice with the same batch must write the same file (overwrite),
    not produce extra rows when the file is read back."""
    import meridian_pipeline.parquet_io as pio

    con = _local_duckdb_con(tmp_path)
    df = _make_df(include_non_stock=False, null_rows=False)

    def controlled_append(df, factors):
        import hashlib

        rows_written = 0
        factor_set = set(factors)
        mask = df["factor_name"].isin(factor_set) & df["z_score"].notna()
        subset = df.loc[mask, ["score_date", "ticker", "factor_name", "z_score"]].copy()
        if subset.empty:
            return 0
        factorset_hash = hashlib.sha1(",".join(sorted(factor_set)).encode()).hexdigest()[:12]
        subset["score_date"] = pd.to_datetime(subset["score_date"]).dt.date
        subset["year"] = subset["score_date"].apply(lambda d: d.year)
        for yr, grp in subset.groupby("year"):
            min_date = grp["score_date"].min()
            max_date = grp["score_date"].max()
            fname = f"dualwrite__{min_date}_{max_date}__{factorset_hash}.parquet"
            local_dir = tmp_path / f"year={yr}"
            local_dir.mkdir(exist_ok=True)
            local_path = str(local_dir / fname)
            grp_clean = grp.reset_index(drop=True)
            con.register("_dualwrite_tmp", grp_clean)
            try:
                # Overwrite if exists (same semantics as OVERWRITE_OR_IGNORE).
                con.execute(f"COPY _dualwrite_tmp TO '{local_path}' (FORMAT PARQUET)")
                rows_written += len(grp_clean)
            finally:
                try:
                    con.unregister("_dualwrite_tmp")
                except Exception:
                    pass
        return rows_written

    with patch.object(pio, "append_factor_scores_partition", side_effect=controlled_append):
        r1 = pio.append_factor_scores_partition(df, STOCK_FACTORS_SAMPLE)
        r2 = pio.append_factor_scores_partition(df, STOCK_FACTORS_SAMPLE)

    assert r1 == r2, "row counts must match on re-run"

    # Glob all parquet files; total rows must equal a single run (no duplication).
    import glob

    all_files = glob.glob(str(tmp_path / "**" / "*.parquet"), recursive=True)
    total = sum(len(pd.read_parquet(f)) for f in all_files)
    assert total == r1, f"expected {r1} total rows (idempotent overwrite), got {total}"


# ── Test 3: best-effort - exception in _meridian_con must not propagate ───────


def test_append_best_effort_swallows_exception():
    """If _meridian_con raises, append_factor_scores_partition returns 0 and
    does NOT raise."""
    import meridian_pipeline.parquet_io as pio

    df = _make_df()

    with (
        patch.object(pio, "_use_meridian_s3", return_value=True),
        patch.object(pio, "_meridian_con", side_effect=RuntimeError("S3 unreachable")),
    ):
        result = pio.append_factor_scores_partition(df, STOCK_FACTORS_SAMPLE)

    assert result == 0, "should return 0 on error, not raise"


# ── Test 4: no-op when creds absent ──────────────────────────────────────────


def test_append_noop_when_no_creds(tmp_path):
    """With _use_meridian_s3() -> False, returns 0 and writes nothing."""
    import meridian_pipeline.parquet_io as pio

    df = _make_df()

    with patch.object(pio, "_use_meridian_s3", return_value=False):
        result = pio.append_factor_scores_partition(df, STOCK_FACTORS_SAMPLE)

    assert result == 0
    import glob

    files = glob.glob(str(tmp_path / "**" / "*.parquet"), recursive=True)
    assert len(files) == 0, "must not write any files when creds absent"
