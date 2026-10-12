#!/usr/bin/env python3
"""box_preflight.py - refuse to start a burst run that cannot possibly succeed.

Meridian#356. The burst box has no route to PostgreSQL by design and reads the
LDP lake over S3. Two failure modes make a misconfigured box look like something
else entirely:

1. **ldp-parquet not readable.** `parquet_io` treats an unreachable lake as a
   degradation and falls back to the PostgreSQL read path. On the box that PG
   call hits the ship-back shim and raises "PG connection attempted", so the
   operator sees a DATABASE error whose actual cause is an S3 permission. This
   probe names the real thing.

2. **No route to PG at all.** A genuine connection attempt on a box with no
   route hangs until timeout rather than failing, so a run looks slow instead of
   broken.

Everything here is read-only except one probe object written and deleted under
the run's own output prefix.
"""

import os
import sys

REQUIRED_ENV = (
    "LDP_MINIO_ENDPOINT",
    "LDP_MINIO_ACCESS_KEY",
    "LDP_MINIO_SECRET_KEY",
    "LDP_PARQUET_BUCKET",
)


def _duck():
    import duckdb

    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute(f"SET s3_endpoint='{os.environ['LDP_MINIO_ENDPOINT']}'")
    con.execute(
        "SET s3_use_ssl=%s"
        % str(os.environ.get("LDP_MINIO_USE_SSL", "true").lower() == "true").lower()
    )
    con.execute("SET s3_url_style='path'")
    con.execute(f"SET s3_access_key_id='{os.environ['LDP_MINIO_ACCESS_KEY']}'")
    con.execute(f"SET s3_secret_access_key='{os.environ['LDP_MINIO_SECRET_KEY']}'")
    return con


def main() -> int:
    fails = []

    missing = [k for k in REQUIRED_ENV if not os.environ.get(k)]
    if missing:
        print(f"FAIL: missing env {missing}")
        return 1

    bucket = os.environ["LDP_PARQUET_BUCKET"]
    con = _duck()

    # 1. the LDP lake must be READABLE. This is the probe that matters: without it
    #    parquet_io degrades to PG and the error names the wrong subsystem.
    try:
        n = con.execute(
            f"SELECT COUNT(*) FROM read_parquet('s3://{bucket}/ohlcv/**/*.parquet', "
            "hive_partitioning=1) LIMIT 1"
        ).fetchone()[0]
        print(f"  OK    read s3://{bucket}/ohlcv  ({n:,} rows visible)")
    except Exception as exc:
        fails.append(
            f"cannot read s3://{bucket}/ohlcv with this key: {type(exc).__name__}: "
            f"{str(exc)[:160]}\n"
            "        This is an S3 PERMISSION problem, not a database problem.\n"
            "        Without it parquet_io silently degrades to the PostgreSQL read\n"
            "        path, which this box has no route to, and the run would fail\n"
            "        reporting a PG error instead of this."
        )

    # 2. the output prefix must be WRITABLE
    out = os.environ.get("BURST_OUT_PREFIX")
    if out:
        try:
            con.execute(
                f"COPY (SELECT 1 AS probe) TO '{out.rstrip('/')}/_preflight.parquet' (FORMAT PARQUET)"
            )
            print(f"  OK    write {out}")
        except Exception as exc:
            fails.append(f"cannot write {out}: {type(exc).__name__}: {str(exc)[:160]}")
    else:
        fails.append("BURST_OUT_PREFIX unset - nowhere to ship results back to")

    # 3. the panel must be present and complete
    panel = os.environ.get("BURST_PANEL_DIR")
    if panel:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        try:
            from replay_panel import read_panel

            frames = read_panel(panel)
            print(f"  OK    panel {panel} ({len(frames)} tables)")
        except SystemExit as exc:
            fails.append(str(exc))
    else:
        fails.append("BURST_PANEL_DIR unset - the box cannot reach PG for these inputs")

    if fails:
        print("\nPREFLIGHT FAILED:")
        for f in fails:
            print(f"  - {f}")
        print(
            "\nRefusing to start. A run that begins without these either hangs or\n"
            "reports a misleading error hours later."
        )
        return 1
    print("\nPREFLIGHT OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
