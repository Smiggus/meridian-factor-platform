"""replay_out.py - write factor rows to object storage instead of PostgreSQL.

Meridian#356 ship-back. The burst box computes but cannot reach PG, so it writes
per-date parquet to `s3://meridian-parquet/burst/out/<run_id>/` and an in-cluster
promote step upserts from there under the explicit campaign.

## The contract

Each date becomes one parquet file whose columns are exactly what
`db.upsert_daily_scores` consumes, so the promote step does not transform
anything - it reads and upserts. A transformation on the promote side would be a
second place for the two halves to disagree.

`campaign_id` is deliberately NOT written here. The box has no business deciding
lineage; promote stamps it from the explicit campaign so every row lands under
one auditable id regardless of how many box runs produced it.

## The manifest is the completeness gate

`manifest.json` records the git SHA, asset, date range, per-date row counts and a
heartbeat. Promote refuses a run whose manifest disagrees with the files present:
without that, a box run that died two thirds through would ship a prefix that
looks complete, and the missing third would surface as a coverage hole days later
rather than as a refusal at promote time.
"""

# NB: no `from __future__ import annotations` - shares the asset import graph.

import json
import os
import subprocess
import time

import pandas as pd

MANIFEST = "manifest.json"


class ShipBackWriter:
    """Stand-in for db.upsert_daily_scores that writes parquet to `out_root`.

    Callable with the same signature the asset uses, so it drops straight into
    the panel shim without the asset knowing anything changed.
    """

    def __init__(self, out_root: str, run_id: str, asset: str, repo: str = None):
        self.out_root = out_root.rstrip("/")
        self.run_id = run_id
        self.asset = asset
        self.repo = repo or os.environ.get("MERIDIAN_REPO_ROOT", ".")
        self.dates = {}
        self.started = time.time()
        os.makedirs(self.out_root, exist_ok=True) if not self.out_root.startswith("s3://") else None

    def __call__(self, engine, df, **kwargs):
        """Signature-compatible with db.upsert_daily_scores; ignores `engine`."""
        if df is None or len(df) == 0:
            return 0
        frame = pd.DataFrame(df) if not isinstance(df, pd.DataFrame) else df.copy()
        # campaign_id is promote's to assign - see module docstring.
        frame = frame.drop(columns=[c for c in ("campaign_id",) if c in frame.columns])
        dates = sorted({str(d) for d in frame["score_date"]})
        if len(dates) != 1:
            raise RuntimeError(
                f"ship-back expects one score_date per write, got {len(dates)}. "
                "Writing a multi-date frame would break per-date completeness accounting."
            )
        d = dates[0]
        path = f"{self.out_root}/date={d}/part.parquet"
        if not path.startswith("s3://"):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        frame.to_parquet(path, index=False)
        self.dates[d] = {
            "rows": int(len(frame)),
            "factors": sorted(map(str, frame["factor_name"].unique())),
        }
        return int(len(frame))

    def _git_sha(self) -> str:
        try:
            return subprocess.run(
                ["git", "-C", self.repo, "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
        except Exception:
            return "unknown"

    def manifest(self, expected_dates=None) -> dict:
        return {
            "run_id": self.run_id,
            "asset": self.asset,
            "git_sha": self._git_sha(),
            "input_source": {
                "bucket": os.environ.get("LDP_PARQUET_BUCKET"),
                "endpoint": os.environ.get("LDP_MINIO_ENDPOINT"),
            },
            "dates_written": sorted(self.dates),
            "expected_dates": sorted(expected_dates) if expected_dates else None,
            "per_date": self.dates,
            "total_rows": int(sum(v["rows"] for v in self.dates.values())),
            "started_at": self.started,
            "heartbeat_at": time.time(),
            "complete": (
                None if expected_dates is None else sorted(self.dates) == sorted(expected_dates)
            ),
        }

    def write_manifest(self, expected_dates=None) -> dict:
        m = self.manifest(expected_dates)
        path = f"{self.out_root}/{MANIFEST}"
        if not path.startswith("s3://"):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as fh:
                json.dump(m, fh, indent=2, sort_keys=True)
        else:
            import io

            import pyarrow.fs as pafs  # noqa: F401  (s3 handled by the caller's fs)

            buf = io.BytesIO(json.dumps(m, indent=2, sort_keys=True).encode())
            _s3_put(path, buf.getvalue())
        return m


def _s3_put(uri: str, payload: bytes) -> None:
    """Minimal S3 PUT via duckdb's httpfs, so we carry no extra dependency."""
    import duckdb

    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute(f"SET s3_endpoint='{os.environ['LDP_MINIO_ENDPOINT']}'")
    con.execute("SET s3_url_style='path'")
    con.execute("SET s3_use_ssl=true")
    con.execute(f"SET s3_access_key_id='{os.environ['LDP_MINIO_ACCESS_KEY']}'")
    con.execute(f"SET s3_secret_access_key='{os.environ['LDP_MINIO_SECRET_KEY']}'")
    tmp = "/tmp/_manifest_tmp.json"
    with open(tmp, "wb") as fh:
        fh.write(payload)
    con.execute(f"COPY (SELECT content FROM read_text('{tmp}')) TO '{uri}' (FORMAT CSV, HEADER 0)")
