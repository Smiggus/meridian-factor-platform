"""
Tests for the Iceberg gold -> PostgreSQL promotion path (Meridian#265):
  * lakehouse.write_gold_scores / read_gold_snapshot - scope stamping + scoped
    read-back (a non-owned factor present in the table is excluded by the
    snapshot's own owned_factors scope).
  * db.upsert_daily_scores owned_factors delete-replace - only the owned
    factors are ever deleted (a non-owned factor row is untouched).
  * gold_scores_pg_sync asset - threads the snapshot's owned_factors scope into
    upsert_daily_scores(replace=True, owned_factors=...), and skips clean when
    the gold table is unavailable.

Dagster is only partially installed on the slim CI image, so the whole module
is skipped there via a hasattr check (NOT importorskip, which would pass on the
partial install and then blow up on a missing attribute - see
a repo convention for slim CI). pyiceberg/polars are genuinely-optional deps
and use importorskip inside the tests that need them (matching test_lakehouse).

There is intentionally NO live Iceberg<->PG round-trip here: that needs real
catalog creds + a real Postgres and never runs in CI. It is exercised manually
against the live warehouse (see the MR notes); the seam tests below prove the
scope contract on both sides without fabricating a live pass.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

_DAGSTER_DIR = Path(__file__).resolve().parents[1] / "dagster"
if str(_DAGSTER_DIR) not in sys.path:
    sys.path.insert(0, str(_DAGSTER_DIR))


def _dagster_complete() -> bool:
    try:
        import dagster
    except ImportError:
        return False
    return all(
        hasattr(dagster, n) for n in ("Config", "asset", "materialize", "build_asset_context")
    )


pytestmark = pytest.mark.skipif(
    not _dagster_complete(),
    reason="Dagster not fully installed (slim CI test image)",
)

from meridian_pipeline import db, lakehouse  # noqa: E402

from tests._stage_copy import decode_copy_text  # noqa: E402

# ── write_gold_scores: scope stamping ─────────────────────────────────────────


class _FakeGoldTable:
    """Minimal pyiceberg Table stand-in observing how write_gold_scores calls
    overwrite()/append() and what snapshot_properties it stamps."""

    def __init__(self):
        self.overwrite_calls = []  # (arrow, overwrite_filter, snapshot_properties)
        self.append_calls = []  # (arrow, snapshot_properties)
        self._snapshot_id = 100

    def overwrite(self, arrow, overwrite_filter=None, snapshot_properties=None):
        self.overwrite_calls.append((arrow, overwrite_filter, snapshot_properties))
        self._snapshot_id += 1

    def append(self, arrow, snapshot_properties=None):
        self.append_calls.append((arrow, snapshot_properties))
        self._snapshot_id += 1

    def refresh(self):
        pass

    def current_snapshot(self):
        class _Snap:
            def __init__(self, sid):
                self.snapshot_id = sid

        return _Snap(self._snapshot_id)


def _gold_df():
    import polars as pl

    return pl.DataFrame(
        {
            "score_date": [dt.date(2020, 3, 2), dt.date(2020, 3, 3)],
            "ticker": ["AAPL", "AAPL"],
            "factor_name": ["cand_x", "cand_x"],
            "group_name": ["research", "research"],
            "raw_value": [1.0, 2.0],
            "z_score": [0.5, -0.5],
            "direction": [1, 1],
        }
    )


def test_write_gold_scores_stamps_scope_and_scopes_delete(monkeypatch):
    """A gold batch write must (a) stamp owned_factors CSV + score_date min/max
    on the snapshot summary and (b) delete-replace scoped to (date-range,
    owned_factors) so a re-run is idempotent and only owned factors move."""
    pytest.importorskip("polars")
    pytest.importorskip("pyiceberg")
    from pyiceberg.expressions import And, GreaterThanOrEqual, In, LessThanOrEqual

    fake = _FakeGoldTable()
    monkeypatch.setattr(lakehouse, "ensure_gold_promotion_table", lambda: fake)

    snap_id = lakehouse.write_gold_scores(_gold_df(), owned_factors=["cand_x"], source="unit-test")

    assert fake.append_calls == []
    assert len(fake.overwrite_calls) == 1
    arrow, del_filter, props = fake.overwrite_calls[0]
    assert arrow.num_rows == 2
    # Scope properties stamped on the snapshot.
    assert props[lakehouse.GOLD_SCOPE_OWNED_FACTORS] == "cand_x"
    assert props[lakehouse.GOLD_SCOPE_DATE_MIN] == "2020-03-02"
    assert props[lakehouse.GOLD_SCOPE_DATE_MAX] == "2020-03-03"
    assert props[lakehouse.GOLD_SCOPE_SOURCE] == "unit-test"
    # Delete-replace scoped to (date range, owned_factors) - never a bare factor
    # match, so a factor outside owned_factors is never deleted.
    assert del_filter == And(
        And(
            GreaterThanOrEqual("score_date", dt.date(2020, 3, 2)),
            LessThanOrEqual("score_date", dt.date(2020, 3, 3)),
        ),
        In("factor_name", ["cand_x"]),
    )
    assert snap_id == fake._snapshot_id


def test_write_gold_scores_requires_owned_factors(monkeypatch):
    pytest.importorskip("polars")
    pytest.importorskip("pyiceberg")
    monkeypatch.setattr(lakehouse, "ensure_gold_promotion_table", lambda: _FakeGoldTable())
    with pytest.raises(ValueError, match="owned_factors must be a non-empty list"):
        lakehouse.write_gold_scores(_gold_df(), owned_factors=[])


# ── read_gold_snapshot: scoped read-back ──────────────────────────────────────


class _FakeSnap:
    def __init__(self, sid, props):
        self.snapshot_id = sid
        self.summary = _FakeSummary(props)


class _FakeSummary:
    def __init__(self, props):
        self._props = props
        self.additional_properties = props

    def get(self, key):
        return self._props.get(key)


class _FakeReadScan:
    def __init__(self, row_filter, snapshot_id, arrow):
        self.row_filter = row_filter
        self.snapshot_id = snapshot_id
        self._arrow = arrow

    def to_arrow(self):
        return self._arrow


class _FakeReadTable:
    def __init__(self, snap, arrow):
        self._snap = snap
        self._arrow = arrow
        self.scan_calls = []

    def current_snapshot(self):
        return self._snap

    def snapshot_by_id(self, sid):
        return self._snap if sid == self._snap.snapshot_id else None

    def scan(self, row_filter=None, snapshot_id=None, selected_fields=None):
        self.scan_calls.append((row_filter, snapshot_id))
        return _FakeReadScan(row_filter, snapshot_id, self._arrow)


def test_read_gold_snapshot_scopes_to_owned_factors_and_dates(monkeypatch):
    """read_gold_snapshot builds its row filter from the SNAPSHOT's own scope
    (owned_factors + date range), so factors outside owned_factors that also
    live in the gold table are excluded by construction - a non-owned factor is
    never read back for promotion."""
    pytest.importorskip("polars")
    pytest.importorskip("pyiceberg")
    import pyarrow as pa
    from pyiceberg.expressions import And, GreaterThanOrEqual, In, LessThanOrEqual

    props = {
        lakehouse.GOLD_SCOPE_OWNED_FACTORS: "cand_x",
        lakehouse.GOLD_SCOPE_DATE_MIN: "2020-03-02",
        lakehouse.GOLD_SCOPE_DATE_MAX: "2020-03-03",
        lakehouse.GOLD_SCOPE_SOURCE: "unit-test",
    }
    snap = _FakeSnap(4242, props)
    # The scan is stubbed to return only cand_x rows (as a real scoped scan
    # would); the assertion below proves the SCOPE predicate is what drives it.
    arrow = pa.table(
        {
            "score_date": pa.array([dt.date(2020, 3, 2)], pa.date32()),
            "ticker": pa.array(["AAPL"], pa.string()),
            "factor_name": pa.array(["cand_x"], pa.string()),
            "group_name": pa.array(["research"], pa.string()),
            "raw_value": pa.array([1.0], pa.float64()),
            "z_score": pa.array([0.5], pa.float64()),
            "direction": pa.array([1], pa.int32()),
        }
    )
    fake_tbl = _FakeReadTable(snap, arrow)

    class _FakeCatalog:
        def load_table(self, name):
            return fake_tbl

    monkeypatch.setattr(lakehouse, "meridian_catalog", lambda: _FakeCatalog())

    df, scope = lakehouse.read_gold_snapshot(snapshot_id=None)

    assert scope["owned_factors"] == ["cand_x"]
    assert scope["snapshot_id"] == 4242
    assert scope["score_date_min"] == "2020-03-02"
    assert list(df["factor_name"].unique()) == ["cand_x"]
    # Scan pinned the snapshot and filtered by the snapshot's own scope.
    assert len(fake_tbl.scan_calls) == 1
    row_filter, snapshot_id = fake_tbl.scan_calls[0]
    assert snapshot_id == 4242
    assert row_filter == And(
        And(
            In("factor_name", ["cand_x"]),
            GreaterThanOrEqual("score_date", "2020-03-02"),
        ),
        LessThanOrEqual("score_date", "2020-03-03"),
    )


def test_read_gold_snapshot_empty_scope_returns_empty(monkeypatch):
    """A snapshot carrying no owned_factors scope promotes nothing (empty df),
    never a full-table scan."""
    pytest.importorskip("polars")
    pytest.importorskip("pyiceberg")

    snap = _FakeSnap(7, {})  # no scope props
    fake_tbl = _FakeReadTable(snap, None)

    class _FakeCatalog:
        def load_table(self, name):
            return fake_tbl

    monkeypatch.setattr(lakehouse, "meridian_catalog", lambda: _FakeCatalog())

    df, scope = lakehouse.read_gold_snapshot()
    assert df.empty
    assert scope["owned_factors"] == []
    assert fake_tbl.scan_calls == []  # no scan issued with no scope


# ── db.upsert_daily_scores owned_factors scoping (only owned touched) ──────────


class _FakeCursor:
    """Records the stage COPY (#432) as (sql, decoded row dicts)."""

    def __init__(self, log):
        self.log = log

    def copy_expert(self, sql, buf):
        self.log.append((sql, decode_copy_text(sql, buf)))

    def close(self):
        pass


class _FakeConn:
    def __init__(self, log):
        self.log = log
        self.connection = SimpleNamespace(cursor=lambda: _FakeCursor(log))

    def execute(self, statement, params=None):
        self.log.append((str(statement), params))
        return None


class _FakeBegin:
    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        return self.conn

    def __exit__(self, *exc):
        return False


class _FakeEngine:
    def __init__(self):
        self.log = []

    def begin(self):
        return _FakeBegin(_FakeConn(self.log))


def test_upsert_daily_scores_owned_factors_only_deletes_owned(monkeypatch):
    """The promotion write path is db.upsert_daily_scores(replace=True,
    owned_factors=<scope>). Its DELETE must be scoped to score_date +
    owned_factors ONLY - so a factor outside owned_factors is never deleted
    (untouched), and the promoted rows are inserted verbatim (PG rows match)."""
    # Keep the best-effort dual-writes inert + offline for this unit test.
    monkeypatch.setattr(lakehouse, "iceberg_dual_write_enabled", lambda: False)
    # No database behind _FakeEngine: the lineage probe must not (and, since a
    # probe error now re-raises, cannot) be left to fail silently.
    monkeypatch.setattr(db, "_campaign_column_available", lambda engine, table: False)
    from meridian_pipeline import parquet_io

    monkeypatch.setattr(parquet_io, "append_factor_scores_partition", lambda *a, **k: None)

    df = pd.DataFrame(
        [
            {
                "score_date": dt.date(2020, 3, 2),
                "ticker": "AAPL",
                "factor_name": "cand_x",
                "group_name": "research",
                "raw_value": 1.0,
                "z_score": 0.5,
                "direction": 1,
            },
            {
                "score_date": dt.date(2020, 3, 2),
                "ticker": "MSFT",
                "factor_name": "cand_x",
                "group_name": "research",
                "raw_value": 2.0,
                "z_score": -0.5,
                "direction": 1,
            },
        ]
    )
    eng = _FakeEngine()
    n = db.upsert_daily_scores(eng, df, replace=True, owned_factors=["cand_x"])
    assert n == 2

    deletes = [(s, p) for s, p in eng.log if "DELETE" in s]
    # #432: rows are staged in a temp table (one COPY), then one INSERT ... SELECT.
    inserts = [(s, p) for s, p in eng.log if s.lstrip().startswith("COPY _m432_stage")]

    # Exactly one delete for the single score_date, scoped to owned_factors.
    assert len(deletes) == 1
    _, del_params = deletes[0]
    assert del_params["factor_names"] == ["cand_x"]
    assert del_params["score_dates"] == [dt.date(2020, 3, 2)]
    # A factor the batch does not own is NEVER named in any delete.
    for _, p in deletes:
        assert "some_other_factor" not in (p.get("factor_names") or [])

    # Promoted rows inserted verbatim (PG rows match the gold batch).
    insert_rows = []
    for _, chunk in inserts:
        insert_rows.extend(chunk)
    assert {r["ticker"] for r in insert_rows} == {"AAPL", "MSFT"}
    assert all(r["factor_name"] == "cand_x" for r in insert_rows)


# ── gold_scores_pg_sync asset: threads scope owned_factors into PG write ───────


def test_gold_scores_pg_sync_promotes_scope_owned_factors(monkeypatch):
    """The asset reads the target snapshot's scope and delete-replaces exactly
    that owned_factors set in PG (replace=True) - the end-to-end scope thread."""
    from meridian_pipeline import gold_pg_sync_asset as gps

    from dagster import build_asset_context

    df = pd.DataFrame(
        [
            {
                "score_date": dt.date(2020, 3, 2),
                "ticker": "AAPL",
                "factor_name": "cand_x",
                "group_name": "research",
                "raw_value": 1.0,
                "z_score": 0.5,
                "direction": 1,
            }
        ]
    )
    scope = {
        "snapshot_id": 5150,
        "owned_factors": ["cand_x"],
        "score_date_min": "2020-03-02",
        "score_date_max": "2020-03-02",
        "source": "unit-test",
    }
    captured = {}

    monkeypatch.setattr(lakehouse, "gold_promotion_available", lambda: True)
    monkeypatch.setattr(gps.lakehouse, "read_gold_snapshot", lambda snapshot_id=None: (df, scope))
    monkeypatch.setattr(gps.db, "engine_from_env", lambda: object())

    def _fake_upsert(
        engine,
        frame,
        replace=True,
        owned_factors=None,
        allow_unwritten_owned_factors=False,
        replace_date_range=None,
        allow_interior_gaps=False,
    ):
        captured["replace"] = replace
        captured["owned_factors"] = owned_factors
        assert replace_date_range == (dt.date(2020, 3, 2), dt.date(2020, 3, 2))
        # the snapshot scope is the authority here and may name a factor this
        # batch has no rows for, so the asset opts out of the #378 check
        captured["allow_unwritten_owned_factors"] = allow_unwritten_owned_factors
        # #401: default run config does not opt into the interior-gap override
        captured["allow_interior_gaps"] = allow_interior_gaps
        captured["rows"] = len(frame)
        return len(frame)

    monkeypatch.setattr(gps.db, "upsert_daily_scores", _fake_upsert)

    out = gps.gold_scores_pg_sync(build_asset_context(), gps.GoldPgSyncConfig(snapshot_id=5150))
    assert captured["replace"] is True
    assert captured["owned_factors"] == ["cand_x"]
    assert captured["allow_interior_gaps"] is False
    assert captured["rows"] == 1
    assert out.metadata["rows_promoted"].value == 1
    assert out.metadata["allow_interior_gaps"].value is False


def test_gold_scores_pg_sync_skips_when_unavailable(monkeypatch):
    from meridian_pipeline import gold_pg_sync_asset as gps

    from dagster import build_asset_context

    monkeypatch.setattr(lakehouse, "gold_promotion_available", lambda: False)

    def _boom(*a, **k):
        raise AssertionError("must not read/write when gold table is unavailable")

    monkeypatch.setattr(gps.lakehouse, "read_gold_snapshot", _boom)
    monkeypatch.setattr(gps.db, "upsert_daily_scores", _boom)

    out = gps.gold_scores_pg_sync(build_asset_context(), gps.GoldPgSyncConfig())
    assert out.metadata["skipped"].value is True


def test_gold_scores_pg_sync_scoped_run_fails_when_unavailable(monkeypatch):
    """A skip is a SUCCESS run and the sensor acknowledges it: a run requested
    for a specific snapshot must FAIL instead, so the failure pager fires."""
    from meridian_pipeline import gold_pg_sync_asset as gps

    from dagster import Failure, build_asset_context

    monkeypatch.setattr(lakehouse, "gold_promotion_available", lambda: False)
    with pytest.raises(Failure, match="5150"):
        gps.gold_scores_pg_sync(build_asset_context(), gps.GoldPgSyncConfig(snapshot_id=5150))


@pytest.mark.parametrize("scope_owned", [[], None])
def test_gold_scores_pg_sync_scoped_run_fails_on_missing_or_empty_snapshot(
    monkeypatch, scope_owned
):
    from meridian_pipeline import gold_pg_sync_asset as gps

    from dagster import Failure, build_asset_context

    monkeypatch.setattr(lakehouse, "gold_promotion_available", lambda: True)
    scope = {"snapshot_id": 5150, "owned_factors": scope_owned}
    monkeypatch.setattr(
        gps.lakehouse, "read_gold_snapshot", lambda snapshot_id=None: (pd.DataFrame(), scope)
    )

    def _boom(*a, **k):
        raise AssertionError("must not write when there is nothing to promote")

    monkeypatch.setattr(gps.db, "upsert_daily_scores", _boom)
    with pytest.raises(Failure, match="5150"):
        gps.gold_scores_pg_sync(build_asset_context(), gps.GoldPgSyncConfig(snapshot_id=5150))


def test_gold_scores_pg_sync_unscoped_run_still_skips_on_empty_scope(monkeypatch):
    from meridian_pipeline import gold_pg_sync_asset as gps

    from dagster import build_asset_context

    monkeypatch.setattr(lakehouse, "gold_promotion_available", lambda: True)
    monkeypatch.setattr(
        gps.lakehouse,
        "read_gold_snapshot",
        lambda snapshot_id=None: (pd.DataFrame(), {"snapshot_id": 7, "owned_factors": []}),
    )
    out = gps.gold_scores_pg_sync(build_asset_context(), gps.GoldPgSyncConfig())
    assert out.metadata["skipped"].value is True
