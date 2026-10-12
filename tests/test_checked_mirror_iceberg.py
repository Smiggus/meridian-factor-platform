"""Isolated real PyIceberg commit, not a mock of the write boundary."""

import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("pyiceberg")
_DAGSTER_DIR = Path(__file__).resolve().parents[1] / "dagster"
if str(_DAGSTER_DIR) not in sys.path:
    sys.path.insert(0, str(_DAGSTER_DIR))
from meridian_pipeline import checked_mirror, lakehouse
from pyiceberg.catalog.sql import SqlCatalog

DAY = date(2026, 9, 24)
OTHER = date(2026, 9, 23)
NOW = datetime(2026, 9, 25, tzinfo=timezone.utc)


@pytest.mark.parametrize("name", ["daily_scores", "group_scores"])
def test_evolved_schema_fails_without_replacing_snapshot(tmp_path, monkeypatch, name):
    import pyarrow as pa
    from pyiceberg.types import StringType

    catalog = SqlCatalog(
        "schema_test",
        uri=f"sqlite:///{tmp_path / 'catalog.db'}",
        warehouse=(tmp_path / "warehouse").as_uri(),
    )
    catalog.create_namespace("factors")
    schema, spec = (
        (lakehouse._daily_scores_schema(), lakehouse._daily_scores_partition_spec())
        if name == "daily_scores"
        else (lakehouse.group_scores_schema(), lakehouse.group_scores_partition_spec())
    )
    table = catalog.create_table(f"factors.{name}", schema=schema, partition_spec=spec)
    with table.update_schema() as update:
        update.add_column("extra", StringType())

    def row(day, value):
        result = dict(score_date=day, ticker="ABC", group_name="value", computed_at=NOW)
        if name == "daily_scores":
            result.update(factor_name="value", raw_value=value, z_score=0.5, direction=1)
        else:
            result.update(group_score=value)
        return result

    payload = checked_mirror._arrow(name, pd.DataFrame([row(DAY, 9.0), row(OTHER, 2.0)]))
    table.append(payload.append_column("extra", pa.array(["requested-extra", "survivor-extra"])))
    original_snapshot = table.current_snapshot().snapshot_id
    original_rows = table.scan().to_arrow().to_pylist()
    monkeypatch.setattr(lakehouse, "meridian_catalog", lambda: catalog)
    monkeypatch.setattr(checked_mirror, "_pg", lambda *args: pd.DataFrame([row(DAY, 1.0)]))
    with pytest.raises(ValueError, match="unsupported.*schema"):
        checked_mirror.reconcile_date(Engine(), name, DAY)
    table.refresh()
    assert table.current_snapshot().snapshot_id == original_snapshot
    assert table.scan().to_arrow().to_pylist() == original_rows


@pytest.mark.parametrize("name", ["daily_scores", "group_scores"])
@pytest.mark.parametrize("matched", [False, True])
def test_concurrent_schema_evolution_cannot_publish_repair(tmp_path, monkeypatch, name, matched):
    from pyiceberg.types import StringType

    catalog = SqlCatalog(
        "schema_race_test",
        uri=f"sqlite:///{tmp_path / 'catalog.db'}",
        warehouse=(tmp_path / "warehouse").as_uri(),
    )
    catalog.create_namespace("factors")
    schema, spec = (
        (lakehouse._daily_scores_schema(), lakehouse._daily_scores_partition_spec())
        if name == "daily_scores"
        else (lakehouse.group_scores_schema(), lakehouse.group_scores_partition_spec())
    )
    table = catalog.create_table(f"factors.{name}", schema=schema, partition_spec=spec)

    def row(day, value):
        result = dict(score_date=day, ticker="ABC", group_name="value", computed_at=NOW)
        if name == "daily_scores":
            result.update(factor_name="value", raw_value=value, z_score=0.5, direction=1)
        else:
            result.update(group_score=value)
        return result

    table.append(
        checked_mirror._arrow(
            name, pd.DataFrame([row(DAY, 1.0 if matched else 9.0), row(OTHER, 2.0)])
        )
    )
    original_snapshot = table.current_snapshot().snapshot_id
    reads = 0

    def pg(*args):
        nonlocal reads
        reads += 1
        if reads == (2 if matched else 3):
            # Independent catalog client changes only schema, not the data head,
            # after all survivor scans and immediately before overwrite commit.
            other_client = catalog.load_table(f"factors.{name}")
            with other_client.update_schema() as update:
                update.add_column("extra", StringType())
        return pd.DataFrame([row(DAY, 1.0)])

    monkeypatch.setattr(checked_mirror, "_table", lambda _: table)
    monkeypatch.setattr(checked_mirror, "_pg", pg)
    monkeypatch.setattr(checked_mirror.time, "sleep", lambda _: None)
    with pytest.raises(ValueError, match="unsupported.*schema"):
        checked_mirror.reconcile_date(Engine(), name, DAY)
    table.refresh()
    assert table.current_snapshot().snapshot_id == original_snapshot
    assert len(table.snapshots()) == 1


@pytest.mark.parametrize("name", ["daily_scores", "group_scores"])
@pytest.mark.parametrize(
    "change", ["schema", "compatible_schema", "snapshot", "continuous_snapshot"]
)
def test_matched_scan_requires_stable_catalog_head(tmp_path, monkeypatch, name, change):
    import pyarrow.parquet as pq
    from pyiceberg.types import StringType

    catalog = SqlCatalog(
        "matched_scan_race",
        uri=f"sqlite:///{tmp_path / 'catalog.db'}",
        warehouse=(tmp_path / "warehouse").as_uri(),
    )
    catalog.create_namespace("factors")
    schema, spec = (
        (lakehouse._daily_scores_schema(), lakehouse._daily_scores_partition_spec())
        if name == "daily_scores"
        else (lakehouse.group_scores_schema(), lakehouse.group_scores_partition_spec())
    )
    table = catalog.create_table(f"factors.{name}", schema=schema, partition_spec=spec)

    def row(day, ticker):
        result = dict(score_date=day, ticker=ticker, group_name="value", computed_at=NOW)
        if name == "daily_scores":
            result.update(factor_name="value", raw_value=1.0, z_score=0.5, direction=1)
        else:
            result.update(group_score=1.0)
        return result

    source = pd.DataFrame([row(DAY, "ABC")])
    table.append(checked_mirror._arrow(name, source))
    original_snapshot = table.current_snapshot().snapshot_id
    original_file = pq.ParquetFile
    scans = 0
    mutations = 0

    def racing_file(*args, **kwargs):
        parquet = original_file(*args, **kwargs)
        original_iter = parquet.iter_batches

        def racing_iter(**options):
            nonlocal scans, mutations
            scans += 1
            if scans % 2 == 0 and (mutations == 0 or change == "continuous_snapshot"):
                other_client = catalog.load_table(f"factors.{name}")
                mutations += 1
                if change == "compatible_schema":
                    from pyiceberg.schema import Schema
                    from pyiceberg.table.update import AddSchemaUpdate, SetCurrentSchemaUpdate

                    # A metadata client can publish identical fields under a new
                    # ID. The high-level update API deduplicates such schemas.
                    new_schema = Schema(*other_client.schema().fields, schema_id=1)
                    with other_client.transaction() as transaction:
                        transaction._apply(
                            (
                                AddSchemaUpdate(schema=new_schema),
                                SetCurrentSchemaUpdate(schema_id=-1),
                            )
                        )
                elif change == "schema":
                    with other_client.update_schema() as update:
                        update.add_column("extra", StringType())
                else:
                    # An unrelated date leaves the requested digest unchanged,
                    # but its publication still invalidates this read boundary.
                    other_client.append(
                        checked_mirror._arrow(name, pd.DataFrame([row(OTHER, str(mutations))]))
                    )
            yield from original_iter(**options)

        parquet.iter_batches = racing_iter
        return parquet

    monkeypatch.setattr(pq, "ParquetFile", racing_file)
    monkeypatch.setattr(checked_mirror, "_table", lambda _: table)
    monkeypatch.setattr(checked_mirror, "_pg", lambda *args: source)
    events = []
    if change == "schema":
        with pytest.raises(ValueError, match="unsupported.*schema"):
            checked_mirror.reconcile_date(Engine(), name, DAY, on_event=events.append)
    elif change == "continuous_snapshot":
        with pytest.raises(RuntimeError, match="source changed during parity read"):
            checked_mirror.reconcile_date(Engine(), name, DAY, on_event=events.append)
        assert mutations == checked_mirror._ATTEMPTS
        assert len([event for event in events if event["phase"] == "retry"]) == mutations
    else:
        result = checked_mirror.reconcile_date(Engine(), name, DAY, on_event=events.append)
        assert result["verdict"] == "matched"
        assert len([event for event in events if event["phase"] == "retry"]) == 1
        assert scans == 4
    monkeypatch.setattr(pq, "ParquetFile", original_file)
    table.refresh()
    if change in {"schema", "compatible_schema"}:
        assert table.current_snapshot().snapshot_id == original_snapshot
        assert len(table.snapshots()) == 1
        expected = row(DAY, "ABC")
        if change == "schema":
            expected["extra"] = None
        else:
            assert table.schema().fields == schema.fields
            assert table.schema().schema_id != schema.schema_id
        assert table.scan().to_arrow().to_pylist() == [expected]
    else:
        assert len(table.snapshots()) == mutations + 1
        assert row(DAY, "ABC") in table.scan().to_arrow().to_pylist()
    assert not any(event["phase"] == "commit" for event in events)


@pytest.mark.parametrize("name", ["daily_scores", "group_scores"])
@pytest.mark.parametrize(
    "change", ["schema", "compatible_schema", "requested_snapshot", "snapshot"]
)
def test_repair_scan_rejects_changed_catalog_head(tmp_path, monkeypatch, name, change):
    import pyarrow.parquet as pq
    from pyiceberg.expressions import EqualTo
    from pyiceberg.types import StringType

    catalog = SqlCatalog(
        "repair_scan_race",
        uri=f"sqlite:///{tmp_path / 'catalog.db'}",
        warehouse=(tmp_path / "warehouse").as_uri(),
    )
    catalog.create_namespace("factors")
    schema, spec = (
        (lakehouse._daily_scores_schema(), lakehouse._daily_scores_partition_spec())
        if name == "daily_scores"
        else (lakehouse.group_scores_schema(), lakehouse.group_scores_partition_spec())
    )
    table = catalog.create_table(f"factors.{name}", schema=schema, partition_spec=spec)

    def row(day, value):
        result = dict(score_date=day, ticker="ABC", group_name="value", computed_at=NOW)
        if name == "daily_scores":
            result.update(factor_name="value", raw_value=value, z_score=0.5, direction=1)
        else:
            result.update(group_score=value)
        return result

    table.append(checked_mirror._arrow(name, pd.DataFrame([row(DAY, 9.0)])))
    source = pd.DataFrame([row(DAY, 1.0)])
    original_verify = checked_mirror._verify_month
    original_file = pq.ParquetFile
    mutations = []

    def racing_file(*args, **kwargs):
        parquet = original_file(*args, **kwargs)
        original_iter = parquet.iter_batches

        def racing_iter(**options):
            if not mutations:
                other = catalog.load_table(f"factors.{name}")
                mutations.append(other.current_snapshot().snapshot_id)
                if change == "schema":
                    with other.update_schema() as update:
                        update.add_column("extra", StringType())
                elif change == "compatible_schema":
                    from pyiceberg.schema import Schema
                    from pyiceberg.table.update import AddSchemaUpdate, SetCurrentSchemaUpdate

                    with other.transaction() as transaction:
                        transaction._apply(
                            (
                                AddSchemaUpdate(schema=Schema(*other.schema().fields, schema_id=1)),
                                SetCurrentSchemaUpdate(schema_id=-1),
                            )
                        )
                elif change == "requested_snapshot":
                    other.overwrite(
                        checked_mirror._arrow(name, pd.DataFrame([row(DAY, 777.0)])),
                        overwrite_filter=EqualTo("score_date", DAY),
                    )
                else:
                    other.append(checked_mirror._arrow(name, pd.DataFrame([row(OTHER, 2.0)])))
            yield from original_iter(**options)

        parquet.iter_batches = racing_iter
        return parquet

    def verify(*args):
        # Only final repair verification consumption races; the repair has
        # already published, so rejection must not imply rollback.
        with monkeypatch.context() as scoped:
            scoped.setattr(pq, "ParquetFile", racing_file)
            return original_verify(*args)

    monkeypatch.setattr(checked_mirror, "_verify_month", verify)
    monkeypatch.setattr(checked_mirror, "_table", lambda _: table)
    monkeypatch.setattr(checked_mirror, "_pg", lambda *args: source)
    events = []
    error = ValueError if change == "schema" else RuntimeError
    message = "unsupported.*schema" if change == "schema" else "lake head changed during repair"
    with pytest.raises(error, match=message):
        checked_mirror.reconcile_date(Engine(), name, DAY, on_event=events.append)
    assert len(mutations) == 1
    assert len([event for event in events if event["phase"] == "commit"]) == 1
    assert events[-1]["phase"] == "month_recheck"
    assert events[-1]["lake_head_stable"] is False
    evidence = events[-1]
    assert evidence["read_snapshot_id"] == mutations[0]
    assert evidence["read_schema_id"] == schema.schema_id
    assert evidence["pg"] == evidence["iceberg"] == evidence["committed"]
    table.refresh()
    assert evidence["current_schema_id"] == table.schema().schema_id
    assert evidence["current_snapshot_id"] == table.current_snapshot().snapshot_id
    # PyIceberg overwrite publishes delete and append snapshots atomically.
    expected_snapshots = 2 if "schema" in change else (4 if change == "requested_snapshot" else 3)
    assert len(table.snapshots()) == expected_snapshots
    rows = table.scan().to_arrow().to_pylist()
    expected = row(DAY, 777.0 if change == "requested_snapshot" else 1.0)
    if change == "schema":
        expected["extra"] = None
    assert expected in rows
    if change == "snapshot":
        assert row(OTHER, 2.0) in rows
    if "schema" in change:
        assert table.current_snapshot().snapshot_id == mutations[0]
    else:
        assert table.current_snapshot().snapshot_id != mutations[0]


@pytest.mark.parametrize("name", ["daily_scores", "group_scores"])
@pytest.mark.parametrize("change", ["schema", "snapshot"])
def test_empty_repair_verification_rejects_head_changes(tmp_path, monkeypatch, name, change):
    from pyiceberg.types import StringType

    catalog = SqlCatalog(
        "empty_repair_race",
        uri=f"sqlite:///{tmp_path / 'catalog.db'}",
        warehouse=(tmp_path / "warehouse").as_uri(),
    )
    catalog.create_namespace("factors")
    schema, spec = (
        (lakehouse._daily_scores_schema(), lakehouse._daily_scores_partition_spec())
        if name == "daily_scores"
        else (lakehouse.group_scores_schema(), lakehouse.group_scores_partition_spec())
    )
    table = catalog.create_table(f"factors.{name}", schema=schema, partition_spec=spec)
    row = dict(score_date=DAY, ticker="ABC", group_name="value", computed_at=NOW)
    if name == "daily_scores":
        row.update(factor_name="value", raw_value=9.0, z_score=0.5, direction=1)
    else:
        row.update(group_score=9.0)
    table.append(checked_mirror._arrow(name, pd.DataFrame([row])))
    empty = pd.DataFrame(columns=checked_mirror.CONTRACTS[name][0])
    original_verify = checked_mirror._verify_month
    mutations = []

    def final_pg(*args):
        other = catalog.load_table(f"factors.{name}")
        mutations.append(other.current_snapshot().snapshot_id)
        if change == "schema":
            with other.update_schema() as update:
                update.add_column("extra", StringType())
        else:
            other.append(checked_mirror._arrow(name, pd.DataFrame([row])))
        return empty

    def verify(*args):
        # An empty lake scan consumes no Parquet batches. Mutate during the
        # final PG read, after verification captured its initial lake head.
        with monkeypatch.context() as scoped:
            scoped.setattr(checked_mirror, "_pg", final_pg)
            return original_verify(*args)

    monkeypatch.setattr(checked_mirror, "_verify_month", verify)
    monkeypatch.setattr(checked_mirror, "_table", lambda _: table)
    monkeypatch.setattr(checked_mirror, "_pg", lambda *args: empty)
    events = []
    error = ValueError if change == "schema" else RuntimeError
    message = "unsupported.*schema" if change == "schema" else "lake head changed during repair"
    with pytest.raises(error, match=message):
        checked_mirror.reconcile_date(Engine(), name, DAY, on_event=events.append)
    assert len(mutations) == 1
    assert len([event for event in events if event["phase"] == "commit"]) == 1
    assert events[-1]["lake_head_stable"] is False
    evidence = events[-1]
    assert evidence["phase"] == "month_recheck"
    assert evidence["read_snapshot_id"] == mutations[0]
    assert evidence["read_schema_id"] == schema.schema_id
    assert evidence["pg"] == evidence["iceberg"] == evidence["committed"]
    table.refresh()
    assert evidence["current_schema_id"] == table.schema().schema_id
    assert evidence["current_snapshot_id"] == table.current_snapshot().snapshot_id
    assert len(table.snapshots()) == (2 if change == "schema" else 3)
    assert table.scan().to_arrow().to_pylist() == ([] if change == "schema" else [row])


class Connection:
    def execute(self, *args, **kwargs):
        return None

    def rollback(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class Engine:
    def connect(self):
        return Connection()


@pytest.mark.parametrize("name", ["daily_scores", "group_scores"])
@pytest.mark.parametrize("all_deleted", [False, True])
def test_real_month_snapshot_replaces_broken_day_only_after_pg_read(
    tmp_path, monkeypatch, name, all_deleted
):
    catalog = SqlCatalog(
        "mirror_test",
        uri=f"sqlite:///{tmp_path / 'catalog.db'}",
        warehouse=(tmp_path / "warehouse").as_uri(),
    )
    catalog.create_namespace_if_not_exists("factors")
    if name == "daily_scores":
        schema, spec = lakehouse._daily_scores_schema(), lakehouse._daily_scores_partition_spec()

        def row(day, value):
            return dict(
                score_date=day,
                ticker="ABC",
                factor_name="value",
                group_name="value",
                raw_value=value,
                z_score=0.5,
                direction=1,
                computed_at=NOW,
            )
    else:
        schema, spec = lakehouse.group_scores_schema(), lakehouse.group_scores_partition_spec()

        def row(day, value):
            return dict(
                score_date=day, ticker="ABC", group_name="value", group_score=value, computed_at=NOW
            )

    table = catalog.create_table(f"factors.{name}", schema=schema, partition_spec=spec)
    source = [row(DAY, 1.0), row(OTHER, 2.0)]
    orphan_day = date(2026, 9, 22)
    preserved = row(orphan_day, 77.0)
    source.append(row(orphan_day, 42.0))  # Uncompleted PG data must not replace lake history.
    outside = row(date(2026, 8, 31), 88.0)
    table.append(
        checked_mirror._arrow(name, pd.DataFrame([row(DAY, 9.0), source[1], preserved, outside]))
    )
    monkeypatch.setattr(checked_mirror, "_completed_dates", lambda *args: [OTHER], raising=False)
    if all_deleted:
        source.pop(0)
    monkeypatch.setattr(lakehouse, "meridian_catalog", lambda: catalog)
    import pyarrow.parquet as pq
    from pyiceberg.io.pyarrow import ArrowScan

    original_file = pq.ParquetFile
    reads = []

    def bounded_file(*args, **kwargs):
        assert kwargs["pre_buffer"] is False
        reads.append(kwargs)
        parquet = original_file(*args, **kwargs)
        original_iter = parquet.iter_batches

        def bounded_iter(**options):
            assert options == {"batch_size": checked_mirror._CHUNK_ROWS, "use_threads": False}
            yield from original_iter(**options)

        parquet.iter_batches = bounded_iter
        return parquet

    monkeypatch.setattr(pq, "ParquetFile", bounded_file)

    def eager_read_forbidden(*args, **kwargs):
        raise AssertionError("native PyIceberg 0.11 reader buffers whole files")

    original_batches = ArrowScan.to_record_batches
    monkeypatch.setattr(ArrowScan, "to_record_batches", eager_read_forbidden)
    monkeypatch.setattr(
        checked_mirror,
        "_pg",
        lambda engine, n, lo, hi: pd.DataFrame(
            [r for r in source if lo <= r["score_date"] < hi],
            columns=checked_mirror.CONTRACTS[n][0],
        ),
    )
    result = checked_mirror.reconcile_date(Engine(), name, DAY)
    assert result["verdict"] == ("deleted_orphans" if all_deleted else "repaired")
    assert result["events"][-1]["phase"] == "month_recheck"
    assert reads
    monkeypatch.setattr(pq, "ParquetFile", original_file)
    table.refresh()
    monkeypatch.setattr(ArrowScan, "to_record_batches", original_batches)
    assert table.scan().to_arrow().num_rows == (3 if all_deleted else 4)
    assert preserved in table.scan().to_arrow().to_pylist()
    assert outside in table.scan().to_arrow().to_pylist()
    assert len(table.snapshots()) == 2
    assert checked_mirror.reconcile_date(Engine(), name, OTHER)["verdict"] == "matched"


def test_real_many_chunks_share_one_snapshot_and_unique_paths(tmp_path, monkeypatch):
    catalog = SqlCatalog(
        "chunk_test",
        uri=f"sqlite:///{tmp_path / 'catalog.db'}",
        warehouse=(tmp_path / "warehouse").as_uri(),
    )
    catalog.create_namespace("factors")
    table = catalog.create_table(
        "factors.group_scores",
        schema=lakehouse.group_scores_schema(),
        partition_spec=lakehouse.group_scores_partition_spec(),
    )
    source = [
        dict(
            score_date=DAY, ticker=str(i), group_name="risk", group_score=float(i), computed_at=NOW
        )
        for i in range(3)
    ]
    monkeypatch.setattr(checked_mirror, "_table", lambda _: table)
    monkeypatch.setattr(checked_mirror, "_completed_dates", lambda *args: [])
    monkeypatch.setattr(
        checked_mirror, "_pg", lambda *args: (pd.DataFrame([row]) for row in source)
    )
    assert checked_mirror.reconcile_date(Engine(), "group_scores", DAY)["verdict"] == "repaired"
    table.refresh()
    assert len(table.snapshots()) == 1
    tasks = list(table.scan().plan_files())
    assert len(tasks) == 3
    assert len({task.file.file_path for task in tasks}) == 3
    assert table.scan().to_arrow().num_rows == 3
