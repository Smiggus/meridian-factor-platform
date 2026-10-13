"""Exercise the physical overwrite boundary with a broken date."""

import sys
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pyarrow.compute as pc
import pytest

pytest.importorskip("pyiceberg")
_DAGSTER_DIR = Path(__file__).resolve().parents[1] / "dagster"
if str(_DAGSTER_DIR) not in sys.path:
    sys.path.insert(0, str(_DAGSTER_DIR))
from meridian_pipeline import checked_mirror
from meridian_pipeline.checked_mirror import _completed_dates as real_completed_dates
from pyiceberg.exceptions import CommitFailedException
from pyiceberg.expressions import EqualTo

DAY = date(2026, 9, 24)
OTHER = date(2026, 9, 23)
NOW = datetime(2026, 9, 25, tzinfo=timezone.utc)


@pytest.mark.parametrize("stage", ["catalog", "commit"])
def test_failures_redact_credentials_from_evidence_and_traceback(monkeypatch, caplog, stage):
    import logging
    import traceback

    canary = "synthetic-mirror-value"
    monkeypatch.setattr(checked_mirror.lakehouse, "MERIDIAN_MINIO_SECRET_KEY", canary)
    events = []
    if stage == "catalog":

        def load(name):
            raise LookupError(f"catalog rejected {canary}")

        monkeypatch.setattr(checked_mirror, "_table", load)
    else:
        monkeypatch.setattr(checked_mirror, "_table", lambda name: object())
        monkeypatch.setattr(checked_mirror, "_pg_chunks", lambda *args: iter(()))
        monkeypatch.setattr(checked_mirror, "_lake_date", lambda *args: iter(()))
        evidence = iter([{"rows": 1}, {"rows": 0}] * 3)
        monkeypatch.setattr(checked_mirror, "_chunk_evidence", lambda *args: next(evidence))
        monkeypatch.setattr(checked_mirror.time, "sleep", lambda *args: None)

        def replace(*args):
            raise CommitFailedException(f"commit rejected {canary}")

        monkeypatch.setattr(checked_mirror, "_replace_month", replace)
        monkeypatch.setattr(
            checked_mirror, "_table", lambda name: SimpleNamespace(refresh=lambda: None)
        )
    with caplog.at_level(logging.INFO), pytest.raises(Exception) as caught:
        checked_mirror.reconcile_date(FakeEngine(), "group_scores", DAY, on_event=events.append)
    rendered = "".join(traceback.format_exception(caught.value))
    assert canary not in rendered
    assert canary not in repr(events)
    assert canary not in caplog.text
    assert "***REDACTED***" in rendered


@pytest.fixture(autouse=True)
def staging(monkeypatch):
    import pyiceberg.io.pyarrow

    monkeypatch.setattr(
        pyiceberg.io.pyarrow, "_dataframe_to_data_files", lambda **kwargs: iter([kwargs["df"]])
    )
    monkeypatch.setattr(checked_mirror, "_completed_dates", lambda *args: [OTHER])


class FakeTable:
    def __init__(self, rows, name):
        self.name = name
        self.rows = rows
        self.commits = 0
        self.conflict = None

    def refresh(self):
        pass

    def scan(self, row_filter, selected_fields=None):
        selected_fields = selected_fields or checked_mirror.CONTRACTS[self.name][0]
        arrow = checked_mirror._arrow(self.name, pd.DataFrame(self.rows, columns=selected_fields))
        if isinstance(row_filter, EqualTo):
            day = date.fromordinal(date(1970, 1, 1).toordinal() + row_filter.value.value)
            arrow = arrow.filter(pc.equal(arrow["score_date"], day))
        return SimpleNamespace(
            to_arrow_batch_reader=lambda: arrow.to_reader(),
            plan_files=lambda: [SimpleNamespace(file=None, delete_files=[])],
        )

    @property
    def metadata(self):
        return SimpleNamespace(
            specs=lambda: {0: checked_mirror.lakehouse.group_scores_partition_spec()}
        )

    def schema(self):
        return (
            checked_mirror.lakehouse._daily_scores_schema()
            if self.name == "daily_scores"
            else checked_mirror.lakehouse.group_scores_schema()
        )

    io = None

    def transaction(self):
        table = self

        class Transaction:
            table_metadata = SimpleNamespace(schema=table.schema, current_schema_id=0)

            def _apply(self, updates, requirements=()):
                assert updates == ()
                assert requirements[0].current_schema_id == 0
                return self

            def __enter__(self):
                return self

            def __exit__(self, typ, *args):
                pass

            def update_snapshot(self):
                return self

            def overwrite(self):
                class Overwrite:
                    commit_uuid = None

                    def __enter__(self):
                        self.chunks = []
                        return self

                    def append_data_file(self, chunk):
                        self.chunks.append(chunk)

                    def delete_data_file(self, file):
                        pass

                    def __exit__(self, typ, *args):
                        if typ is None:
                            import pyarrow as pa

                            table.overwrite(
                                pa.concat_tables(self.chunks)
                                if self.chunks
                                else checked_mirror._arrow(
                                    table.name,
                                    pd.DataFrame(columns=checked_mirror.CONTRACTS[table.name][0]),
                                ),
                                None,
                            )

                return Overwrite()

        return Transaction()

    def overwrite(self, payload, overwrite_filter):
        if self.conflict:
            callback, self.conflict = self.conflict, None
            callback()
            raise CommitFailedException("competing commit")
        self.commits += 1
        # Whole month snapshot replacement, not an append or per-day mutation.
        self.rows = payload.to_pylist()

    def current_snapshot(self):
        return SimpleNamespace(snapshot_id=self.commits)


class FakeConnection:
    def execute(self, *args, **kwargs):
        return None

    def rollback(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class FakeEngine:
    def connect(self):
        return FakeConnection()


@pytest.mark.parametrize("name", ["daily_scores", "group_scores"])
@pytest.mark.parametrize("damage", ["value", "missing", "orphan", "all_deleted"])
def test_broken_date_repaired_at_overwrite_boundary(monkeypatch, name, damage):
    if name == "daily_scores":

        def row(day, ticker, value):
            return dict(
                score_date=day,
                ticker=ticker,
                factor_name="quality",
                group_name="quality",
                raw_value=value,
                z_score=0.5,
                direction=1,
                computed_at=NOW,
            )
    else:

        def row(day, ticker, value):
            return dict(
                score_date=day,
                ticker=ticker,
                group_name="quality",
                group_score=value,
                computed_at=NOW,
            )

    source = [row(DAY, "A", 1.0), row(DAY, "B", 2.0), row(OTHER, "A", 3.0)]
    lake = list(source)
    if damage == "value":
        lake[0] = row(DAY, "A", 99.0)
    elif damage == "missing":
        lake.pop(0)
    elif damage == "orphan":
        lake.append(row(DAY, "X", 4.0))
    else:
        source = [source[-1]]
    table = FakeTable(lake, name)
    monkeypatch.setattr(checked_mirror, "_table", lambda _: table)
    monkeypatch.setattr(
        checked_mirror,
        "_pg",
        lambda engine, n, lo, hi: pd.DataFrame(
            [r for r in source if lo <= r["score_date"] < hi],
            columns=checked_mirror.CONTRACTS[name][0],
        ),
    )
    evidence = []
    result = checked_mirror.reconcile_date(FakeEngine(), name, DAY, on_event=evidence.append)
    assert result["verdict"] == ("deleted_orphans" if damage == "all_deleted" else "repaired")
    assert table.commits == 1

    def order(r):
        return r["score_date"], r["ticker"]

    assert sorted(table.rows, key=order) == sorted(
        checked_mirror._arrow(name, pd.DataFrame(source)).to_pylist(), key=order
    )
    assert [event["phase"] for event in evidence] == ["before", "commit", "after", "month_recheck"]
    assert evidence[-1]["pg"] == evidence[-1]["iceberg"]
    assert checked_mirror.reconcile_date(FakeEngine(), name, OTHER)["verdict"] == "matched"


def test_conflict_reloads_pg_not_stale_payload(monkeypatch):
    name = "group_scores"
    source = [dict(score_date=DAY, ticker="A", group_name="risk", group_score=1.0, computed_at=NOW)]
    table = FakeTable([], name)
    table.conflict = lambda: source[0].update(group_score=2.0)
    monkeypatch.setattr(checked_mirror, "_table", lambda _: table)
    monkeypatch.setattr(
        checked_mirror,
        "_pg",
        lambda engine, n, lo, hi: pd.DataFrame(
            [r for r in source if lo <= r["score_date"] < hi],
            columns=checked_mirror.CONTRACTS[name][0],
        ),
    )
    monkeypatch.setattr(checked_mirror, "_completed_dates", lambda *args: [])
    monkeypatch.setattr(checked_mirror.time, "sleep", lambda _: None)
    result = checked_mirror.reconcile_date(FakeEngine(), name, DAY)
    assert result["verdict"] == "repaired"
    assert table.rows[0]["group_score"] == 2.0
    assert table.commits == 1


def test_other_day_pg_write_during_month_commit_is_not_copied(monkeypatch):
    name = "group_scores"
    source = [
        dict(score_date=DAY, ticker="A", group_name="risk", group_score=1.0, computed_at=NOW),
        dict(score_date=OTHER, ticker="A", group_name="risk", group_score=2.0, computed_at=NOW),
    ]
    table = FakeTable([source[0] | {"group_score": 9.0}, source[1]], name)
    original_overwrite = table.overwrite

    def competing_pg_write(payload, overwrite_filter):
        original_overwrite(payload, overwrite_filter)
        source[1]["group_score"] = 3.0

    table.overwrite = competing_pg_write
    monkeypatch.setattr(checked_mirror, "_table", lambda _: table)
    monkeypatch.setattr(
        checked_mirror,
        "_pg",
        lambda engine, n, lo, hi: pd.DataFrame([r for r in source if lo <= r["score_date"] < hi]),
    )
    assert checked_mirror.reconcile_date(FakeEngine(), name, DAY)["verdict"] == "repaired"
    assert table.rows[1]["group_score"] == 2.0
    assert table.commits == 1


def test_both_empty_is_not_success(monkeypatch):
    name = "group_scores"
    table = FakeTable([], name)
    monkeypatch.setattr(checked_mirror, "_table", lambda _: table)
    monkeypatch.setattr(
        checked_mirror,
        "_pg",
        lambda engine, n, lo, hi: pd.DataFrame(columns=checked_mirror.CONTRACTS[name][0]),
    )
    with pytest.raises(RuntimeError, match="missing date"):
        checked_mirror.reconcile_date(FakeEngine(), name, DAY)
    assert table.commits == 0


@pytest.mark.parametrize("completed", [False, True])
def test_uncompleted_pg_days_are_preserved_not_copied(monkeypatch, completed):
    name = "group_scores"

    def row(day, score):
        return dict(
            score_date=day, ticker="A", group_name="risk", group_score=score, computed_at=NOW
        )

    source = [row(DAY, 1.0), row(OTHER, 22.0)]
    table = FakeTable([row(DAY, 9.0), row(OTHER, 2.0)], name)
    monkeypatch.setattr(checked_mirror, "_table", lambda _: table)
    monkeypatch.setattr(
        checked_mirror, "_completed_dates", lambda *args: [OTHER] if completed else []
    )
    monkeypatch.setattr(
        checked_mirror,
        "_pg",
        lambda engine, n, lo, hi: iter(
            [pd.DataFrame([r for r in source if lo <= r["score_date"] < hi])]
        ),
    )
    original = table.overwrite

    def commit(payload, filt):
        original(payload, filt)
        source[1]["group_score"] = 33.0

    table.overwrite = commit
    assert checked_mirror.reconcile_date(FakeEngine(), name, DAY)["verdict"] == "repaired"
    assert row(OTHER, 2.0) in table.rows


def test_group_repair_preserves_composite_only_lake_date(monkeypatch):
    import sqlite3

    from sqlalchemy import create_engine, text

    name = "group_scores"

    def row(day, score):
        return dict(
            score_date=day, ticker="A", group_name="risk", group_score=score, computed_at=NOW
        )

    source = [row(DAY, 1.0)]
    preserved = row(OTHER, 2.0)
    table = FakeTable([row(DAY, 9.0), preserved], name)
    monkeypatch.setattr(checked_mirror, "_table", lambda _: table)
    monkeypatch.setattr(checked_mirror, "_completed_dates", real_completed_dates)
    monkeypatch.setattr(
        checked_mirror,
        "_pg",
        lambda engine, n, lo, hi: pd.DataFrame(
            [r for r in source if lo <= r["score_date"] < hi],
            columns=checked_mirror.CONTRACTS[name][0],
        ),
    )
    # Execute the actual inventory SQL locally; only PG advisory locks are stubbed.
    sql_engine = create_engine("sqlite://", connect_args={"detect_types": sqlite3.PARSE_DECLTYPES})
    try:
        with sql_engine.connect() as sql_conn:
            sql_conn.exec_driver_sql("ATTACH DATABASE ':memory:' AS factors")
            for relation, days in (
                ("composite_scores", [OTHER, DAY, DAY, date(2026, 8, 31), date(2026, 10, 1)]),
                ("group_scores", [DAY, date(2026, 9, 22), date(2026, 8, 31), date(2026, 10, 1)]),
            ):
                sql_conn.exec_driver_sql(f"CREATE TABLE factors.{relation} (score_date DATE)")
                for day in days:
                    sql_conn.execute(
                        text(f"INSERT INTO factors.{relation} VALUES (:day)"), {"day": day}
                    )

            class InventoryConnection(FakeConnection):
                def execute(self, statement, *args, **kwargs):
                    if "pg_advisory_" in str(statement):
                        return None
                    return sql_conn.execute(statement, *args, **kwargs)

            engine = SimpleNamespace(connect=InventoryConnection)
            result = checked_mirror.reconcile_date(engine, name, DAY)
            assert result["verdict"] == "repaired"
            assert table.commits == 1
            assert sorted(table.rows, key=lambda r: r["score_date"]) == [preserved, source[0]]
            assert real_completed_dates(engine, date(2026, 9, 1), date(2026, 10, 1)) == [DAY]
            recheck = result["events"][-1]
            assert recheck["phase"] == "month_recheck"
            assert set(recheck["committed"]) == {str(DAY)}
    finally:
        sql_engine.dispose()


def test_unlock_error_invalidates_connection_preserves_original(monkeypatch):
    class BrokenConnection(FakeConnection):
        invalidated = False

        def execute(self, statement, *args):
            if "unlock" in str(statement):
                raise RuntimeError("unlock broke")

        def invalidate(self):
            self.invalidated = True

    conn = BrokenConnection()
    engine = SimpleNamespace(connect=lambda: conn)
    monkeypatch.setattr(
        checked_mirror, "_table", lambda _: (_ for _ in ()).throw(LookupError("original"))
    )
    with pytest.raises(LookupError, match="original"):
        checked_mirror.reconcile_date(engine, "group_scores", DAY)
    assert conn.invalidated


def test_pg_reads_use_server_side_chunks(monkeypatch):
    conn = FakeConnection()
    options = {}
    conn.execution_options = lambda **kwargs: options.update(kwargs) or conn

    def read_sql(query, connection, *, params, chunksize):
        assert connection is conn
        assert chunksize == checked_mirror._CHUNK_ROWS
        assert options["stream_results"] is True
        yield pd.DataFrame()

    monkeypatch.setattr(pd, "read_sql", read_sql)
    assert (
        len(
            list(
                checked_mirror._pg(
                    SimpleNamespace(connect=lambda: conn), "group_scores", OTHER, DAY
                )
            )
        )
        == 1
    )


def test_duplicate_pg_payload_never_commits(monkeypatch):
    row = dict(score_date=DAY, ticker="A", group_name="risk", group_score=1.0, computed_at=NOW)
    table = FakeTable([], "group_scores")
    monkeypatch.setattr(checked_mirror, "_table", lambda _: table)
    monkeypatch.setattr(
        checked_mirror, "_pg", lambda *args: iter([pd.DataFrame([row]), pd.DataFrame([row])])
    )
    with pytest.raises(ValueError, match="duplicate"):
        checked_mirror.reconcile_date(FakeEngine(), "group_scores", DAY)
    assert table.commits == 0


def test_retry_refreshes_new_lake_orphan_when_pg_becomes_empty(monkeypatch):
    name = "group_scores"
    row = dict(score_date=DAY, ticker="A", group_name="risk", group_score=1.0, computed_at=NOW)
    source = [row]
    table = FakeTable([], name)

    def conflict():
        source.clear()
        table.rows = [row]

    table.conflict = conflict
    monkeypatch.setattr(checked_mirror, "_table", lambda _: table)
    monkeypatch.setattr(checked_mirror, "_completed_dates", lambda *args: [])
    monkeypatch.setattr(
        checked_mirror,
        "_pg",
        lambda *args: pd.DataFrame(source, columns=checked_mirror.CONTRACTS[name][0]),
    )
    monkeypatch.setattr(checked_mirror.time, "sleep", lambda _: None)
    result = checked_mirror.reconcile_date(FakeEngine(), name, DAY)
    assert result["verdict"] == "deleted_orphans"
    assert table.rows == []
    before = [event for event in result["events"] if event["phase"] == "before"]
    assert [event["iceberg"]["rows"] for event in before] == [0, 1]


def test_unrelated_completed_marker_does_not_require_pg_payload(monkeypatch):
    name = "group_scores"
    row = dict(score_date=DAY, ticker="A", group_name="risk", group_score=1.0, computed_at=NOW)
    table = FakeTable([], name)
    monkeypatch.setattr(checked_mirror, "_table", lambda _: table)
    monkeypatch.setattr(
        checked_mirror,
        "_pg",
        lambda engine, n, lo, hi: pd.DataFrame(
            [row] if lo == DAY else [], columns=checked_mirror.CONTRACTS[name][0]
        ),
    )
    assert checked_mirror.reconcile_date(FakeEngine(), name, DAY)["verdict"] == "repaired"
    assert table.commits == 1
    assert table.rows == [row]


def test_source_change_to_requested_payload_rebuilds_before_commit(monkeypatch):
    name = "group_scores"
    source = [
        dict(score_date=d, ticker="A", group_name="risk", group_score=1.0, computed_at=NOW)
        for d in (OTHER, DAY)
    ]
    table = FakeTable([], name)
    monkeypatch.setattr(checked_mirror, "_table", lambda _: table)
    monkeypatch.setattr(
        checked_mirror,
        "_pg",
        lambda engine, n, lo, hi: pd.DataFrame([r for r in source if lo <= r["score_date"] < hi]),
    )
    import pyiceberg.io.pyarrow

    writes = 0

    def stage(**kwargs):
        nonlocal writes
        writes += 1
        if writes == 1:
            source[1]["group_score"] = 2.0
        yield kwargs["df"]

    monkeypatch.setattr(pyiceberg.io.pyarrow, "_dataframe_to_data_files", stage)
    monkeypatch.setattr(checked_mirror.time, "sleep", lambda _: None)
    assert checked_mirror.reconcile_date(FakeEngine(), name, DAY)["verdict"] == "repaired"
    assert table.commits == 1
    assert any(row["score_date"] == DAY and row["group_score"] == 2.0 for row in table.rows)


def test_missing_catalog_table_fails_closed(monkeypatch):
    monkeypatch.setattr(
        checked_mirror, "_table", lambda _: (_ for _ in ()).throw(LookupError("no table"))
    )
    with pytest.raises(LookupError, match="no table"):
        checked_mirror.reconcile_date(FakeEngine(), "group_scores", DAY)


def test_matched_read_rechecks_pg_stability(monkeypatch):
    name = "group_scores"
    row = dict(score_date=DAY, ticker="A", group_name="risk", group_score=1.0, computed_at=NOW)
    table = FakeTable([row.copy()], name)
    monkeypatch.setattr(checked_mirror, "_table", lambda _: table)
    monkeypatch.setattr(checked_mirror, "_completed_dates", lambda *args: [])
    monkeypatch.setattr(checked_mirror, "_pg", lambda *args: pd.DataFrame([row]))
    original = checked_mirror._lake_date

    def concurrent_write(*args):
        chunks = list(original(*args))
        row["group_score"] = 2.0
        return iter(chunks)

    monkeypatch.setattr(checked_mirror, "_lake_date", concurrent_write)
    result = checked_mirror.reconcile_date(FakeEngine(), name, DAY)
    assert result["verdict"] == "repaired"
    assert table.rows[0]["group_score"] == 2.0
    assert any(e["phase"] == "retry" for e in result["events"])
