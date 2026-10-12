"""test_dual_write_generations.py - Meridian #275 mirror generation hygiene.

Covers the two fixes that stop dual-write generations layering in the
`meridian-parquet` mirror:

  (a) every mirrored row carries a `computed_at` stamp, so a duplicate key is
      resolvable by recency (and a layered mirror is diagnosable at all);
  (b) a write deletes the fragments it fully supersedes instead of landing
      alongside them.

No DB, no MinIO: the DuckDB COPY is redirected to tmp_path and the prune step
is exercised through its pure key-selection helper.
"""

import sys
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

_DAGSTER = Path(__file__).resolve().parent.parent / "dagster"
if str(_DAGSTER) not in sys.path:
    sys.path.insert(0, str(_DAGSTER))

import meridian_pipeline.parquet_io as pio  # noqa: E402

# ── (a) computed_at stamp ────────────────────────────────────────────────────


class _LocalCon:
    """DuckDB connection wrapper that rewrites s3:// COPY targets to tmp_path."""

    def __init__(self, con, root: Path, bucket: str):
        self._con = con
        self._root = root
        self._bucket = bucket
        self.copied: list[str] = []

    def register(self, name, df):
        return self._con.register(name, df)

    def unregister(self, name):
        return self._con.unregister(name)

    def execute(self, sql, *args):
        prefix = f"s3://{self._bucket}/"
        if prefix in sql:
            head, _, rest = sql.partition(prefix)
            key = rest.split("'", 1)[0]
            local = self._root / key
            local.parent.mkdir(parents=True, exist_ok=True)
            sql = head + str(local) + "'" + rest.split("'", 1)[1]
            self.copied.append(str(local))
        return self._con.execute(sql, *args)


def _payload(score_date=date(2026, 7, 24)):
    rows = [
        {
            "score_date": score_date,
            "ticker": t,
            "factor_name": f,
            "group_name": "value",
            "raw_value": 1.0,
            "z_score": 0.5,
            "direction": 1,
        }
        for t in ("AAPL", "MSFT")
        for f in ("earnings_yield", "book_yield")
    ]
    return pd.DataFrame(rows)


def test_dualwrite_rows_carry_computed_at(tmp_path):
    duckdb = pytest.importorskip("duckdb")

    con = _LocalCon(duckdb.connect(), tmp_path, pio.MERIDIAN_MINIO_BUCKET)
    with (
        patch.object(pio, "_use_meridian_s3", return_value=True),
        patch.object(pio, "_meridian_con", return_value=con),
        patch.object(pio, "_prune_superseded_dualwrite", return_value=0),
    ):
        written = pio.append_factor_scores_partition(_payload(), [])

    assert written == 4
    assert len(con.copied) == 1
    out = pd.read_parquet(con.copied[0])
    assert "computed_at" in out.columns, "mirror rows must carry a generation stamp"
    assert out["computed_at"].notna().all()
    assert out["computed_at"].nunique() == 1, "one batch, one generation stamp"
    # `year` stays in the path only, matching the materialisation layout.
    assert "year" not in out.columns
    assert set(out.columns) == {"score_date", "ticker", "factor_name", "z_score", "computed_at"}


def test_dualwrite_prunes_after_write(tmp_path):
    """The writer must call the prune step with the batch's own span/factors."""
    duckdb = pytest.importorskip("duckdb")

    con = _LocalCon(duckdb.connect(), tmp_path, pio.MERIDIAN_MINIO_BUCKET)
    seen = {}

    def _fake_prune(c, year, min_date, max_date, batch_factors, keep_key):
        seen.update(
            year=year,
            min_date=min_date,
            max_date=max_date,
            batch_factors=batch_factors,
            keep_key=keep_key,
        )
        return 0

    with (
        patch.object(pio, "_use_meridian_s3", return_value=True),
        patch.object(pio, "_meridian_con", return_value=con),
        patch.object(pio, "_prune_superseded_dualwrite", side_effect=_fake_prune),
    ):
        pio.append_factor_scores_partition(_payload(), [])

    assert seen["year"] == 2026
    assert seen["min_date"] == date(2026, 7, 24)
    assert seen["max_date"] == date(2026, 7, 24)
    assert seen["batch_factors"] == {"earnings_yield", "book_yield"}
    assert seen["keep_key"].startswith("factors/daily_scores/year=2026/dualwrite__")


def test_shared_dual_write_helper_stamps_and_prunes(tmp_path):
    """Both PG write paths stamp and prune, because both go through the one
    `db._dual_write_daily_scores` helper !407 extracted.

    Pinning it here means a future refactor that reintroduces a PG write path
    bypassing the helper (the Meridian#266 silent-bypass class) also loses the
    #275 generation hygiene, and this test catches it.
    """
    pytest.importorskip("sqlalchemy")
    duckdb = pytest.importorskip("duckdb")
    from meridian_pipeline import db

    con = _LocalCon(duckdb.connect(), tmp_path, pio.MERIDIAN_MINIO_BUCKET)
    payload = _payload()

    with (
        patch.object(pio, "_use_meridian_s3", return_value=True),
        patch.object(pio, "_meridian_con", return_value=con),
        patch.object(pio, "_prune_superseded_dualwrite", return_value=0) as prune,
        patch("meridian_pipeline.lakehouse.iceberg_dual_write_enabled", return_value=False),
    ):
        db._dual_write_daily_scores(payload, payload, replace=True, owned_factors=None)

    assert len(con.copied) == 1, "the shared helper did not reach the flat-parquet mirror"
    out = pd.read_parquet(con.copied[0])
    assert "computed_at" in out.columns
    assert out["computed_at"].notna().all()
    prune.assert_called_once()


# ── (b) supersession selection ───────────────────────────────────────────────

_P = "factors/daily_scores/year=2024/"
_KEEP = _P + "dualwrite__2024-03-01_2024-03-01__0000ffff0000.parquet"


def _sel(keys, factors_by_key, batch_factors, lo=date(2024, 3, 1), hi=date(2024, 3, 1)):
    return pio.superseded_dualwrite_keys(
        keys,
        lo,
        hi,
        set(batch_factors),
        lambda k: factors_by_key[k],
        _KEEP,
    )


def test_supersedes_prior_generation_same_dates_same_factors():
    old = _P + "dualwrite__2024-03-01_2024-03-01__0111ddd0aaaa.parquet"
    got = _sel([_KEEP, old], {old: {"roe", "book_yield"}}, {"roe", "book_yield"})
    assert got == [old]


def test_supersedes_strict_subset_of_batch_factors():
    old = _P + "dualwrite__2024-03-01_2024-03-01__0111ddd0aaaa.parquet"
    got = _sel([old], {old: {"roe"}}, {"roe", "book_yield"})
    assert got == [old]


def test_keeps_sibling_asset_fragment_with_disjoint_factors():
    """value_scores must never delete quality_scores' fragment for the same day."""
    sibling = _P + "dualwrite__2024-03-01_2024-03-01__5ba0000000ab.parquet"
    got = _sel([sibling], {sibling: {"roe", "gross_profitability"}}, {"book_yield"})
    assert got == []


def test_keeps_fragment_with_partially_overlapping_factors():
    other = _P + "dualwrite__2024-03-01_2024-03-01__0e0000000abc.parquet"
    got = _sel([other], {other: {"roe", "book_yield"}}, {"book_yield"})
    assert got == []


def test_keeps_fragment_covering_dates_outside_the_batch():
    wider = _P + "dualwrite__2024-02-01_2024-04-01__0d0000000dad.parquet"
    got = _sel([wider], {wider: {"roe"}}, {"roe"})
    assert got == []


def test_never_touches_materialised_or_unknown_files():
    mat = _P + "data_0.parquet"
    junk = _P + "_SUCCESS"
    got = _sel([mat, junk], {}, {"roe"})
    assert got == []


def test_never_deletes_the_key_just_written():
    got = _sel([_KEEP], {_KEEP: {"roe"}}, {"roe"})
    assert got == []


def test_unreadable_fragment_is_never_deleted():
    bad = _P + "dualwrite__2024-03-01_2024-03-01__bad0000000ad.parquet"

    def _boom(_key):
        raise OSError("s3 read failed")

    got = pio.superseded_dualwrite_keys(
        [bad], date(2024, 3, 1), date(2024, 3, 1), {"roe"}, _boom, _KEEP
    )
    assert got == []


def test_multi_day_batch_supersedes_inner_single_day_fragments():
    d1 = _P + "dualwrite__2024-03-01_2024-03-01__aaaaaaaaaaaa.parquet"
    d2 = _P + "dualwrite__2024-03-05_2024-03-05__bbbbbbbbbbbb.parquet"
    outside = _P + "dualwrite__2024-04-05_2024-04-05__cccccccccccc.parquet"
    got = pio.superseded_dualwrite_keys(
        [d1, d2, outside],
        date(2024, 3, 1),
        date(2024, 3, 31),
        {"roe"},
        lambda k: {"roe"},
        _KEEP,
    )
    assert got == [d1, d2]


def test_prune_uses_per_key_delete_not_batch(monkeypatch):
    """MinIO rejects botocore>=1.36 DeleteObjects (MissingContentMD5).

    Observed live 2026-09-02 during the #332 promote: every year-chunk logged
    "dual-write prune failed" and superseded fragments were left behind. The
    prune must issue per-key delete_object calls, never the batch API.
    """
    from datetime import date

    class _Client:
        def __init__(self):
            self.deleted = []

        def list_objects_v2(self, **kw):
            return {
                "Contents": [
                    {"Key": "factors/daily_scores/year=2009/old_2009-01-02_2009-12-31.parquet"},
                    {"Key": "factors/daily_scores/year=2009/new_2009-01-02_2009-12-31.parquet"},
                ],
                "IsTruncated": False,
            }

        def delete_object(self, Bucket, Key):
            self.deleted.append(Key)

        def delete_objects(self, **kw):  # pragma: no cover - must never be called
            raise AssertionError("batch DeleteObjects is not MinIO-safe")

    client = _Client()
    monkeypatch.setattr(pio, "_meridian_s3_client", lambda: client)
    monkeypatch.setattr(
        pio,
        "superseded_dualwrite_keys",
        lambda keys, *a, **k: [k_ for k_ in keys if "old_" in k_],
    )
    n = pio._prune_superseded_dualwrite(
        con=None,
        year=2009,
        min_date=date(2009, 1, 2),
        max_date=date(2009, 12, 31),
        batch_factors={"roe"},
        keep_key="factors/daily_scores/year=2009/new_2009-01-02_2009-12-31.parquet",
    )
    assert n == 1
    assert client.deleted == ["factors/daily_scores/year=2009/old_2009-01-02_2009-12-31.parquet"]
