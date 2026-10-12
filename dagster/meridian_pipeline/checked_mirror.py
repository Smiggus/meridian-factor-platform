"""Checked, bounded-chunk PG gold to Iceberg score mirror.

Only the requested date is sourced from PG. Other lake dates survive the
single month-aligned snapshot replacement, even with stale completion markers.
"""

import itertools
import logging
import random
import sys
import time
from contextlib import contextmanager
from datetime import date, timedelta

from meridian_pipeline import lakehouse
from meridian_pipeline.mirror_digest import content_evidence

logger = logging.getLogger(__name__)
CONTRACTS = {
    "daily_scores": (lakehouse._DAILY_SCORES_COLUMNS, ("score_date", "ticker", "factor_name")),
    "group_scores": (
        ("score_date", "ticker", "group_name", "group_score", "computed_at"),
        ("score_date", "ticker", "group_name"),
    ),
}
_ATTEMPTS = 3
_CHUNK_ROWS = 4096


def _table(name):
    if name not in CONTRACTS:
        raise ValueError(f"unsupported mirror table: {name}")
    return lakehouse.meridian_catalog().load_table(f"factors.{name}")


def _pg(engine, name, lo, hi):
    import pandas as pd
    from sqlalchemy import text

    cols, _ = CONTRACTS[name]
    query = text(
        f"SELECT {', '.join(cols)} FROM factors.{name} WHERE score_date >= :lo AND score_date < :hi"
    )
    with engine.connect() as conn:
        conn = conn.execution_options(stream_results=True, max_row_buffer=_CHUNK_ROWS)
        yield from pd.read_sql(query, conn, params={"lo": lo, "hi": hi}, chunksize=_CHUNK_ROWS)


def _completed_dates(engine, lo, hi):
    from sqlalchemy import text

    with engine.connect() as conn:
        return list(
            conn.execute(
                text(
                    "SELECT DISTINCT c.score_date FROM factors.composite_scores c "
                    "WHERE c.score_date >= :lo AND c.score_date < :hi "
                    "AND EXISTS (SELECT 1 FROM factors.group_scores g "
                    "WHERE g.score_date = c.score_date) ORDER BY c.score_date"
                ),
                {"lo": lo, "hi": hi},
            ).scalars()
        )


def _arrow(name, frame):
    import pyarrow as pa
    import pyarrow.compute as pc

    if name == "daily_scores":
        return lakehouse._to_daily_scores_arrow(frame)
    schema = pa.schema(
        [
            ("score_date", pa.date32()),
            ("ticker", pa.string()),
            ("group_name", pa.string()),
            ("group_score", pa.float64()),
            ("computed_at", pa.timestamp("us", tz="UTC")),
        ]
    )
    if frame.empty:
        return pa.Table.from_pylist([], schema=schema)
    arrow = pa.Table.from_pandas(frame, preserve_index=False).select(schema.names)
    fields = []
    for field in schema:
        values = arrow[field.name].cast(field.type)
        if field.name == "group_score":
            values = pc.if_else(pc.is_finite(values), values, pa.scalar(None, pa.float64()))
        fields.append(values)
    return pa.Table.from_arrays(fields, schema=schema)


def _rows(arrow, cols):
    for batch in arrow.to_batches(max_chunksize=_CHUNK_ROWS):
        for row in batch.to_pylist():
            yield tuple(row[c] for c in cols)


def _evidence(arrow, name, day):
    import pyarrow.compute as pc

    cols, key = CONTRACTS[name]
    return content_evidence(
        _rows(arrow.filter(pc.equal(arrow["score_date"], day)), cols), cols, key
    )


def _month_evidence(arrow, name):
    cols, key = CONTRACTS[name]
    return content_evidence(_rows(arrow, cols), cols, key)


def _pg_chunks(engine, name, day):
    if hasattr(engine, "mirror_chunks"):
        yield from engine.mirror_chunks(name, day, day + timedelta(days=1))
        return
    frames = _pg(engine, name, day, day + timedelta(days=1))
    # Compatibility for callers/tests supplying a small single DataFrame.
    if hasattr(frames, "columns"):
        frames = (frames,)
    for frame in frames:
        yield _arrow(name, frame)


def _validate_schema(schema, name):
    """Reject projections that would silently erase evolved survivor fields."""
    expected = (
        lakehouse._daily_scores_schema()
        if name == "daily_scores"
        else lakehouse.group_scores_schema()
    )
    if schema.fields != expected.fields:
        raise ValueError(
            f"{name}: unsupported Iceberg schema; checked mirror requires declared fields"
        )


def _lake_chunks(table, name, filt):
    import pyarrow as pa

    _validate_schema(table.schema(), name)
    cols, _ = CONTRACTS[name]
    scan = table.scan(row_filter=filt, selected_fields=cols)
    for task in scan.plan_files():
        file_scan = table.scan(row_filter=filt, selected_fields=cols)
        # Keep a reusable tuple: PyIceberg visits tasks twice.
        file_scan.plan_files = lambda task=task: (task,)
        reader = _batch_reader(file_scan)
        try:
            for batch in reader:
                for bounded in pa.Table.from_batches([batch]).to_batches(max_chunksize=_CHUNK_ROWS):
                    yield pa.Table.from_batches([bounded])
        finally:
            reader.close()


def _batch_reader(scan):
    """Read Parquet without PyIceberg's eager batches or Arrow file prebuffering."""
    import pyarrow as pa
    from pyiceberg.io.pyarrow import schema_to_pyarrow
    from pyiceberg.table import DataScan

    if not isinstance(scan, DataScan):
        return scan.to_arrow_batch_reader()
    projection = scan.projection()
    schema = schema_to_pyarrow(projection)
    return pa.RecordBatchReader.from_batches(schema, _parquet_batches(scan)).cast(schema)


def _parquet_batches(scan):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from pyiceberg.expressions.visitors import bind
    from pyiceberg.io.pyarrow import (
        _get_column_projection_values,
        _to_requested_schema,
        expression_to_pyarrow,
        pyarrow_to_schema,
    )
    from pyiceberg.manifest import FileFormat

    projection = scan.projection()
    predicate = expression_to_pyarrow(
        bind(projection, scan.row_filter, case_sensitive=scan.case_sensitive), projection
    )
    for task in scan.plan_files():
        if task.delete_files or task.file.file_format != FileFormat.PARQUET:
            raise ValueError("checked mirror requires Parquet without delete files")
        with scan.io.new_input(task.file.file_path).open() as source:
            parquet = pq.ParquetFile(source, pre_buffer=False, buffer_size=65536)
            file_schema = pyarrow_to_schema(
                parquet.schema_arrow,
                scan.table_metadata.name_mapping(),
                downcast_ns_timestamp_to_us=True,
                format_version=scan.table_metadata.format_version,
            )
            missing = _get_column_projection_values(
                task.file,
                projection,
                scan.table_metadata.schema(),
                scan.table_metadata.specs().get(task.file.spec_id),
                file_schema.field_ids,
            )
            # No scanner readahead, file prebuffer, or row-count-sized batch list.
            for batch in parquet.iter_batches(batch_size=_CHUNK_ROWS, use_threads=False):
                projected = _to_requested_schema(
                    projection,
                    file_schema,
                    batch,
                    downcast_ns_timestamp_to_us=True,
                    projected_missing_fields=missing,
                    allow_timestamp_tz_mismatch=True,
                )
                yield from pa.Table.from_batches([projected]).filter(predicate).to_batches()


def _lake_date(table, name, day):
    from pyiceberg.expressions import EqualTo

    table.refresh()
    return _lake_chunks(table, name, EqualTo("score_date", day))


def _chunk_evidence(chunks, name, stage=None):
    cols, key = CONTRACTS[name]

    def rows():
        for chunk in chunks:
            if stage is not None and chunk.num_rows:
                stage(chunk)
            yield from _rows(chunk, cols)

    return content_evidence(rows(), cols, key)


def _month(day):
    lo = day.replace(day=1)
    return lo, (lo + timedelta(days=32)).replace(day=1)


class _SourceChanged(RuntimeError):
    pass


def _replace_month(engine, table, name, day, lo, hi, days):
    """Stage bounded chunks and publish exactly one overwrite snapshot.

    This uses PyIceberg 0.11's low-level writer. One shared counter prevents
    chunk file-name collisions. Failed staging leaves unreferenced files for
    normal orphan cleanup, never a partial visible replacement.
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    from pyiceberg.expressions import And, GreaterThanOrEqual, LessThan
    from pyiceberg.io.pyarrow import _dataframe_to_data_files
    from pyiceberg.table.update import AssertCurrentSchemaId
    from pyiceberg.transforms import MonthTransform

    field_id = table.schema().find_field("score_date").field_id
    # Whole-file deletion is safe only with the declared month partition spec.
    for spec in table.metadata.specs().values():
        if not any(
            field.source_id == field_id and isinstance(field.transform, MonthTransform)
            for field in spec.fields
        ):
            raise ValueError("checked mirror requires month(score_date) partitioning")
    filt = And(GreaterThanOrEqual("score_date", lo), LessThan("score_date", hi))
    committed = {}
    count = 0
    with table.transaction() as transaction:
        _validate_schema(transaction.table_metadata.schema(), name)
        # Schema-only commits need not move the data snapshot head. Pin the
        # validated schema as well as PyIceberg's built-in snapshot requirement.
        transaction._apply(
            (),
            (
                AssertCurrentSchemaId(
                    current_schema_id=transaction.table_metadata.current_schema_id
                ),
            ),
        )
        with transaction.update_snapshot().overwrite() as overwrite:
            counter = itertools.count()

            def stage(chunk):
                nonlocal count
                for data_file in _dataframe_to_data_files(
                    table_metadata=transaction.table_metadata,
                    df=chunk,
                    io=table.io,
                    write_uuid=overwrite.commit_uuid,
                    counter=counter,
                ):
                    overwrite.append_data_file(data_file)
                count += chunk.num_rows

            for task in table.scan(row_filter=filt).plan_files():
                if task.delete_files:
                    raise ValueError("checked mirror does not support Iceberg delete files")
                overwrite.delete_data_file(task.file)
            for payload_day in days:
                committed[payload_day] = _chunk_evidence(
                    _pg_chunks(engine, name, payload_day), name, stage
                )
                if not committed[payload_day]["rows"] and payload_day != day:
                    raise RuntimeError(f"{name} {payload_day}: completed payload date is empty")

            def survivors():
                for chunk in _lake_chunks(table, name, filt):
                    yield chunk.filter(
                        pc.invert(
                            pc.is_in(
                                chunk["score_date"], value_set=pa.array(days, type=pa.date32())
                            )
                        )
                    )

            # Also fail closed on duplicate identities in preserved lake rows.
            _chunk_evidence(survivors(), name, stage)
            # All data buffers are out of scope before independent rereads.
            for payload_day, expected in committed.items():
                if _chunk_evidence(_pg_chunks(engine, name, payload_day), name) != expected:
                    raise _SourceChanged(f"{name} {payload_day}: PG changed before commit")
    return committed, count


def reconcile_date(engine, name, day, *, on_event=None, target=None):
    """Redact storage failures before they reach Dagster's traceback logger."""
    try:
        return _reconcile_date(engine, name, day, on_event=on_event, target=target)
    except Exception as exc:
        # Do not retain a credential-bearing cause/context or exception-specific
        # attributes. Keep validation and missing-table errors distinguishable.
        error_type = RuntimeError
        if isinstance(exc, ValueError):
            error_type = ValueError
        elif isinstance(exc, LookupError):
            error_type = LookupError
        raise error_type(lakehouse._scrub_secrets(str(exc))) from None


def _reconcile_date(engine, name, day, *, on_event=None, target=None):
    """Fail closed, with a session advisory lock covering retries and verification."""
    from pyiceberg.exceptions import CommitFailedException

    if not isinstance(day, date) or name not in CONTRACTS:
        raise ValueError("expected a date and supported score table")
    if target is not None:
        target.validate_source(engine)
    lo, hi = _month(day)
    events = []

    def record(phase, **fields):
        event = {"table": name, "date": str(day), "phase": phase, **fields}
        events.append(event)
        logger.info("checked mirror: %s", event)
        if on_event is not None:
            on_event(event)

    lock = _production_lock(engine, lo) if target is None else target.lock(name, lo)
    attempts = _ATTEMPTS if target is None else target.attempts
    with lock:
        try:
            table = _table(name) if target is None else target.table(name)
            for attempt in range(attempts):
                # Refresh lake evidence on every retry, including newly created orphans.
                before_pg = _chunk_evidence(_pg_chunks(engine, name, day), name)
                before_lake = _chunk_evidence(_lake_date(table, name, day), name)
                record("before", pg=before_pg, iceberg=before_lake)
                if before_pg == before_lake:
                    if not before_pg["rows"]:
                        raise RuntimeError(
                            f"{name} {day}: both sides empty; missing date is not parity"
                        )
                    stable_pg = _chunk_evidence(_pg_chunks(engine, name, day), name)
                    stable_chunks = _lake_date(table, name, day)
                    read_schema_id = table.schema().schema_id
                    read_snapshot = table.current_snapshot()
                    read_snapshot_id = read_snapshot.snapshot_id if read_snapshot else None
                    stable_lake = _chunk_evidence(stable_chunks, name)
                    # Scans use captured metadata. Refresh after consumption so
                    # schema-only evolution or a new head cannot certify an old
                    # read. This is a point-in-time check, not cross-store atomicity.
                    table.refresh()
                    _validate_schema(table.schema(), name)
                    current_snapshot = table.current_snapshot()
                    current_snapshot_id = current_snapshot.snapshot_id if current_snapshot else None
                    if (
                        stable_pg != before_pg
                        or stable_lake != before_lake
                        or table.schema().schema_id != read_schema_id
                        or current_snapshot_id != read_snapshot_id
                    ):
                        record(
                            "retry", attempt=attempt + 1, reason="source changed during parity read"
                        )
                        if attempt == attempts - 1:
                            raise _SourceChanged(f"{name} {day}: source changed during parity read")
                        continue
                    return {"verdict": "matched", "events": events}
                # Old composite/group rows do not certify that a historical
                # rerun is finished. Never recruit unrelated PG dates.
                days = [day]
                try:
                    if target is not None:
                        target.guard(table, name)
                    committed, rows = _replace_month(engine, table, name, day, lo, hi, days)
                except (CommitFailedException, _SourceChanged) as exc:
                    record("retry", attempt=attempt + 1, reason=lakehouse._scrub_secrets(str(exc)))
                    if attempt == attempts - 1:
                        raise
                    table.refresh()
                    time.sleep(random.uniform(0.2, 0.6) * (attempt + 1))
                    continue
                snapshot = table.current_snapshot()
                record(
                    "commit",
                    attempt=attempt + 1,
                    rows=rows,
                    snapshot_id=snapshot.snapshot_id if snapshot else None,
                )
                after_pg = _chunk_evidence(_pg_chunks(engine, name, day), name)
                after_lake = _chunk_evidence(_lake_date(table, name, day), name)
                record("after", pg=after_pg, iceberg=after_lake)
                if after_pg != after_lake or after_pg != committed[day]:
                    raise RuntimeError(f"{name} {day}: post-commit digest mismatch")
                _verify_month(engine, table, name, lo, hi, committed, record)
                if not after_pg["rows"] and not before_lake["rows"]:
                    raise RuntimeError(f"{name} {day}: no source or orphan to repair")
                return {
                    "verdict": "repaired" if after_pg["rows"] else "deleted_orphans",
                    "events": events,
                }
        finally:
            pass


@contextmanager
def _production_lock(engine, lo):
    """Unchanged production session-lock behavior; isolated targets never enter."""
    from sqlalchemy import text

    month_id = lo.year * 12 + lo.month
    with engine.connect() as conn:
        conn.execute(text("SELECT pg_advisory_lock(118120, :month)"), {"month": month_id})
        try:
            yield
        finally:
            original_error = sys.exc_info()[0] is not None
            try:
                conn.execute(text("SELECT pg_advisory_unlock(118120, :month)"), {"month": month_id})
                conn.rollback()
            except Exception:
                try:
                    conn.invalidate()
                except Exception as exc:
                    logger.error(
                        "failed to invalidate advisory lock connection: %s",
                        lakehouse._scrub_secrets(str(exc)),
                    )
                if not original_error:
                    raise
                logger.error("advisory unlock failed; original error retained")


def reconcile_both(engine, day, *, on_event=None):
    return {name: reconcile_date(engine, name, day, on_event=on_event) for name in CONTRACTS}


def _verify_month(engine, table, name, lo, hi, committed, record):
    """Recheck only PG payload days, not unrelated in-flight scoring dates."""
    pg = {}
    iceberg = {}
    table.refresh()
    read_schema_id = table.schema().schema_id
    read_snapshot = table.current_snapshot()
    read_snapshot_id = read_snapshot.snapshot_id if read_snapshot else None
    from pyiceberg.expressions import EqualTo

    for day in committed:
        pg[day] = _chunk_evidence(_pg_chunks(engine, name, day), name)
        iceberg[day] = _chunk_evidence(_lake_chunks(table, name, EqualTo("score_date", day)), name)
    # The repair is already published. Reject stale verification without
    # rollback or another repair: a later run can reconcile the current head.
    table.refresh()
    current_snapshot = table.current_snapshot()
    current_snapshot_id = current_snapshot.snapshot_id if current_snapshot else None
    lake_head_stable = (
        table.schema().schema_id == read_schema_id and current_snapshot_id == read_snapshot_id
    )
    # String keys keep emitted evidence JSON-serializable.
    record(
        "month_recheck",
        pg={str(d): v for d, v in pg.items()},
        iceberg={str(d): v for d, v in iceberg.items()},
        committed={str(d): v for d, v in committed.items()},
        read_schema_id=read_schema_id,
        current_schema_id=table.schema().schema_id,
        read_snapshot_id=read_snapshot_id,
        current_snapshot_id=current_snapshot_id,
        lake_head_stable=lake_head_stable,
    )
    _validate_schema(table.schema(), name)
    if not lake_head_stable:
        raise RuntimeError(
            f"{name} {lo}: lake head changed during repair verification; repair already committed"
        )
    if pg != iceberg or iceberg != committed:
        raise RuntimeError(f"{name} {lo}: physical month changed during repair")
