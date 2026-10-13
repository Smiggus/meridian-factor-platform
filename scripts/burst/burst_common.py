"""
burst_common.py - shared building blocks for the elastic burst-compute pattern.

Design write-up: docs/proposals/07-elastic-burst-compute-pipes.md in the private repo.

This module is dependency-light (psycopg2 + pandas + boto3 +
pyarrow) so it runs unchanged on any burst box (a dev machine, an EC2
spot node) as well as inside a cluster run pod. It owns four concerns that are
identical across every backend and every kernel:

  1. Config     - env-driven PG (gold) + MinIO (staging) connections.
  2. Partitions - discover the requested trading days from gold.
  3. Kernel     - per-partition compute (pluggable; PoC ships `verify`).
  4. Staging IO - per-partition parquet + sentinel to a MinIO run prefix,
                  with resume + a fingerprint that the cluster parity gate diffs.

Only the *kernel* and the *launcher* change per use case. Everything here
stays put. The gold store is PostgreSQL `factors.daily_scores` in the
marketdata database; staging is a MinIO parquet prefix. The intermittent
box NEVER writes gold - it writes staging, and the cluster promotes.
"""

import hashlib
import io
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

import boto3
import pandas as pd
import psycopg2
import pyarrow as pa
import pyarrow.parquet as pq

# Ingest-window preflight lives in its own dependency-free module (pure
# stdlib) so it can be imported without pulling in boto3/psycopg2/pyarrow -
# see ingest_window.py's module docstring. BurstConfig.from_env below only
# needs refuse_if_ingest_window_open itself; the window-parsing internals
# (_parse_ingest_window, _time_in_window, the env-var constants) stay
# private to ingest_window.py and are tested directly there
# (tests/test_ingest_window_guard.py), not re-exported through this module.
#
# Support both import styles used across scripts/burst/: flat
# (`from burst_common import ...`, scripts/burst itself on sys.path - the
# driver-script pattern) and package-qualified (`from scripts.burst.burst_common
# import ...`, repo root on sys.path - the pipes_dask_driver/Dask-worker
# pattern).
try:
    from ingest_window import refuse_if_ingest_window_open
except ImportError:
    from scripts.burst.ingest_window import refuse_if_ingest_window_open

# Ingest-completion watermark preflight (supersedes the clock-window guard)
#
# LDP's completion watermark (`ohlcv.ingest_watermark`, live 2026-08-15)
# answers "is date D's ingest actually done" directly from LDP's own
# coverage/returns-ratio thresholds, instead of guessing from wall-clock like
# `refuse_if_ingest_window_open` does. It does not go stale when LDP's window
# shifts and needs no retuning.
#
# Preferred whenever the table is reachable and has at least one row for the
# most recently evaluated trade_date. The clock window remains the fallback
# for when the watermark is unreachable (connection error, missing
# table/schema) or empty (freshly created, not yet populated by LDP) - a
# MISSING watermark must not silently mean "safe to run", so those two cases
# fall straight through to the clock-window check rather than skipping the
# guard outright. This lives in burst_common.py (not ingest_window.py)
# because it needs a real DB connection; ingest_window.py stays pure-stdlib
# on purpose (see its module docstring).
#
# Decision rule: look at the single most-recently-evaluated trade_date row
# (regardless of ITS OWN is_complete value - that value IS the decision). If
# it is marked complete, LDP considers its most recent evaluation caught up
# and it is safe to start. If not, LDP is still mid-evaluation (or coverage
# hasn't cleared its threshold) for its most recent date and this refuses,
# the same way the clock-window guard would refuse during a live rewrite.
_INGEST_WATERMARK_QUERY = """
    SELECT trade_date, is_complete, coverage_ratio, observed_tickers, expected_tickers
    FROM ohlcv.ingest_watermark
    ORDER BY trade_date DESC
    LIMIT 1
"""

# Same override switch as ingest_window.py's _LDP_INGEST_WINDOW_FORCE_ENV -
# the SAME env var (not re-imported, since ingest_window.py's
# module docstring says its internals stay private and untested-through-here;
# duplicating one string literal is cheaper than coupling the two modules).
# An operator setting this means "I have verified it is safe to launch right
# now", which applies to both the watermark and the clock-window mechanism.
_INGEST_SAFETY_FORCE_ENV = "MERIDIAN_BURST_FORCE_INGEST_WINDOW"


def _ingest_safety_force_requested(force: bool) -> bool:
    if force:
        return True
    return os.environ.get(_INGEST_SAFETY_FORCE_ENV, "").strip().lower() in ("1", "true", "yes")


def refuse_if_ingest_incomplete(conn, *, force: bool = False) -> bool:
    """Refuse to START a burst campaign if LDP's completion watermark shows
    its most recently evaluated trade_date is not yet complete.

    Returns True if the watermark was usable (reachable, at least one row)
    and therefore answered the safety question - the caller must NOT also
    run the clock-window fallback in that case. Returns False if the
    watermark is unreachable or empty; the caller must fall back to
    `refuse_if_ingest_window_open`.

    Raises:
        RuntimeError: the watermark is usable and its latest row is not
            complete (or `force` bypassed the check - handled by returning
            True before any query runs).
    """
    if _ingest_safety_force_requested(force):
        return True
    try:
        with conn.cursor() as cur:
            cur.execute(_INGEST_WATERMARK_QUERY)
            row = cur.fetchone()
    except Exception:
        return False  # unreachable (bad connection, missing table/schema, ...)
    if row is None:
        return False  # empty table - LDP hasn't populated it yet

    trade_date, is_complete, coverage_ratio, observed, expected = row
    if is_complete:
        return True
    raise RuntimeError(
        f"Refusing to start a burst campaign: LDP's ingest watermark shows "
        f"{trade_date} is NOT yet complete ({observed}/{expected} tickers, "
        f"{float(coverage_ratio):.1%} coverage). Wait for ohlcv.ingest_watermark "
        f"to mark {trade_date} complete, or set {_INGEST_SAFETY_FORCE_ENV}=1 if "
        f"you have independently verified it is safe to start now."
    )


def _refuse_if_unsafe_to_start(
    *, pg_host: str, pg_port: int, pg_user: str, pg_password: str, gold_db: str, force: bool
) -> None:
    """Campaign-START preflight: prefer the completion watermark, fall back
    to the clock window. See the module comment above `_INGEST_WATERMARK_QUERY`
    for the rationale and the fallback conditions."""
    if _ingest_safety_force_requested(force):
        return

    watermark_used = False
    conn = None
    try:
        conn = psycopg2.connect(
            host=pg_host,
            port=pg_port,
            dbname=gold_db,
            user=pg_user,
            password=pg_password,
            connect_timeout=15,
        )
    except Exception:
        conn = None  # unreachable -> watermark_used stays False, fall back below

    if conn is not None:
        try:
            watermark_used = refuse_if_ingest_incomplete(conn, force=force)
        finally:
            conn.close()

    if not watermark_used:
        refuse_if_ingest_window_open(force=force)


GOLD_DB_DEFAULT = "marketdata"
GOLD_TABLE = "factors.daily_scores"


@dataclass
class BurstConfig:
    """Everything the driver needs to reach gold + staging, sourced from env."""

    # PostgreSQL gold store (read-only from the burst box).
    pg_host: str
    pg_port: int
    pg_user: str
    pg_password: str
    gold_db: str = GOLD_DB_DEFAULT

    # MinIO staging store (write).
    minio_endpoint: str = "s3.example.com"
    minio_access_key: str = ""
    minio_secret_key: str = ""
    minio_bucket: str = "meridian-parquet"
    minio_use_ssl: bool = True
    minio_region: str = "us-east-1"

    # Run identity.
    run_id: str = "adhoc"
    staging_prefix: str = "burst"
    config_fingerprint: str = ""

    def __post_init__(self) -> None:
        # A run's staging root. Isolated per run_id so concurrent/rerun batches
        # never collide. Sentinels + partition parquet live under here.
        self.run_root = f"{self.staging_prefix}/{self.run_id}"

    @classmethod
    def from_env(
        cls,
        run_id: str,
        *,
        gold_db: str = GOLD_DB_DEFAULT,
        config_fingerprint: str = "",
        force_ingest_window: bool = False,
    ) -> "BurstConfig":
        def _req(name: str) -> str:
            v = os.environ.get(name)
            if not v:
                raise RuntimeError(f"required env var {name} is unset")
            return v

        pg_host = _req("DB_HOST")
        pg_port = int(os.environ.get("DB_PORT", "5432"))
        pg_user = _req("DB_USER")
        pg_password = _req("DB_PASSWORD")

        # Campaign-START preflight (read-during-write hardening): every burst
        # driver constructs its BurstConfig here before doing any read-heavy
        # work, so this is the one place that guards every driver's launch.
        # Prefers LDP's completion watermark over the interim clock window -
        # see `_refuse_if_unsafe_to_start`'s module comment (above
        # `_INGEST_WATERMARK_QUERY`) for the preference order and fallback
        # conditions.
        _refuse_if_unsafe_to_start(
            pg_host=pg_host,
            pg_port=pg_port,
            pg_user=pg_user,
            pg_password=pg_password,
            gold_db=gold_db,
            force=force_ingest_window,
        )

        return cls(
            pg_host=pg_host,
            pg_port=pg_port,
            pg_user=pg_user,
            pg_password=pg_password,
            gold_db=gold_db,
            minio_endpoint=os.environ.get("MERIDIAN_MINIO_ENDPOINT", "s3.example.com"),
            minio_access_key=_req("MERIDIAN_MINIO_ACCESS_KEY"),
            minio_secret_key=_req("MERIDIAN_MINIO_SECRET_KEY"),
            minio_bucket=os.environ.get("MERIDIAN_MINIO_BUCKET", "meridian-parquet"),
            minio_use_ssl=os.environ.get("MERIDIAN_MINIO_USE_SSL", "true").lower() == "true",
            minio_region=os.environ.get("MERIDIAN_MINIO_REGION", "us-east-1"),
            run_id=run_id,
            config_fingerprint=config_fingerprint,
        )

    # Connections are built lazily per-worker (they are not picklable, so the
    # Dask task closure must construct them inside the worker, not ship them).
    def pg_connect(self):
        return psycopg2.connect(
            host=self.pg_host,
            port=self.pg_port,
            dbname=self.gold_db,
            user=self.pg_user,
            password=self.pg_password,
            connect_timeout=15,
        )

    def s3_client(self):
        scheme = "https" if self.minio_use_ssl else "http"
        return boto3.client(
            "s3",
            endpoint_url=f"{scheme}://{self.minio_endpoint}",
            aws_access_key_id=self.minio_access_key,
            aws_secret_access_key=self.minio_secret_key,
            region_name=self.minio_region,
        )


_TRADING_CAL_CTE = (
    "WITH trading_cal AS ("
    "  SELECT ts_event::date AS d"
    "    FROM ohlcv.prices"
    "   WHERE security_type = 'equity'"
    "     AND resolution = 'daily'"
    "     AND ts_event >= %(start)s::date"
    "     AND ts_event < (%(end)s::date + 1)"
    "   GROUP BY 1 HAVING COUNT(*) > 100"
    ") "
)


def discover_partitions(cfg: BurstConfig, start: str, end: str, source: str = "gold") -> list[str]:
    """Trading days within [start, end] (inclusive), per `source`.

    source="gold" (default): trading days present in gold, INTERSECTED with
    the canonical trading calendar. This defines the REQUESTED set for the
    hole-check: a burst run is only "complete" when every one of these has a
    staged result or a named failure. Deriving it from gold keeps parity
    exact (we never demand a day gold does not have), but gold alone is NOT a
    calendar: if any writer seeds a weekend/holiday date, a gold-derived
    discovery faithfully re-scores it forever. That self-amplification loop
    is what refilled factors.daily_scores with non-trading rows soon after
    the #174 purge (#280).

    source="calendar": every trading day in [start, end] per the calendar
    alone, independent of what gold holds. Use this for a FILL rather than a
    re-score: gold-mode is structurally unable to discover dates gold has
    zero rows for at all (a history range where factors.daily_scores has no
    rows whatsoever, even though ohlcv.prices is complete), so a burst run
    intending to backfill that range must discover from the calendar instead
    of the (empty) gold intersection.

    The calendar term is byte-for-byte the definition in db.is_trading_day:
    more than 100 daily equity rows in ohlcv.prices on that date (the
    `resolution = 'daily'` filter, Meridian #275 Phase 0, excludes the
    co-mingled intraday bars ohlcv.prices now also carries). It is bounded to
    [start, end] so this stays an index range scan rather than a full scan of
    ohlcv.prices.
    """
    if source == "gold":
        sql = _TRADING_CAL_CTE + (
            f"SELECT DISTINCT g.score_date FROM {GOLD_TABLE} g "
            "JOIN trading_cal tc ON tc.d = g.score_date "
            "WHERE g.score_date BETWEEN %(start)s AND %(end)s "
            "ORDER BY 1"
        )
    elif source == "calendar":
        sql = _TRADING_CAL_CTE + "SELECT d FROM trading_cal ORDER BY 1"
    else:
        raise ValueError(f"unknown discover_partitions source: {source!r}")

    with cfg.pg_connect() as conn, conn.cursor() as cur:
        cur.execute(sql, {"start": start, "end": end})
        return [r[0].isoformat() for r in cur.fetchall()]


# Kernels (pluggable). PoC ships `verify`; swap the body for real scoring.


def _fingerprint_frame(df: pd.DataFrame) -> dict:
    """Deterministic fingerprint of a partition's gold rows.

    Two boxes reading the same gold rows MUST produce the same fingerprint, so
    an exact match cluster-side proves the fan-out + staging + Pipes plumbing is
    faithful. The z-score checksum is order-independent (sorted) and rounded to
    absorb float print noise across numpy builds.
    """
    z = df["z_score"].dropna().round(9).sort_values().to_numpy()
    checksum = hashlib.sha256(z.tobytes()).hexdigest()[:16]
    per_group = df.groupby("group_name").size().sort_index()
    return {
        "rows": int(len(df)),
        "tickers": int(df["ticker"].nunique()),
        "factors": int(df["factor_name"].nunique()),
        "groups": int(df["group_name"].nunique()),
        "z_checksum": checksum,
        "z_mean": float(df["z_score"].mean()) if len(df) else float("nan"),
        "per_group_json": per_group.to_json(),
    }


def kernel_verify(cfg: BurstConfig, score_date: str) -> dict:
    """PoC kernel: read one day's gold, do representative CPU work, fingerprint.

    The representative CPU work is a real cross-sectional standardization of
    raw_value per factor (a groupby-transform over the whole day's universe) -
    the same *shape* of work a real scoring kernel does - so the fan-out is
    exercised on non-trivial compute, not a metadata peek. The staged artifact
    is the fingerprint (used by hole-check + parity), plus how far our naive
    restandardization lands from the stored z (a diagnostic; sector-neutral
    factors will differ by construction, which is expected and informational).
    """
    sql = (
        f"SELECT score_date, ticker, factor_name, group_name, raw_value, z_score "
        f"FROM {GOLD_TABLE} WHERE score_date = %s"
    )
    with cfg.pg_connect() as conn:
        df = pd.read_sql(sql, conn, params=(score_date,))

    if df.empty:
        raise ValueError(f"no gold rows for {score_date} (should not happen post-discovery)")

    # Representative CPU work: naive cross-sectional z of raw_value per factor.
    grp = df.groupby("factor_name")["raw_value"]
    mu = grp.transform("mean")
    sd = grp.transform("std").replace(0, pd.NA)
    df["z_naive"] = (df["raw_value"] - mu) / sd
    diag = df.dropna(subset=["z_naive", "z_score"])
    max_abs_diff = (
        float((diag["z_naive"] - diag["z_score"]).abs().max()) if len(diag) else float("nan")
    )

    fp = _fingerprint_frame(df)
    fp.update(
        {
            "score_date": score_date,
            "config_fingerprint": cfg.config_fingerprint,
            "diag_max_abs_z_diff": max_abs_diff,
        }
    )
    return fp


def kernel_gpu_smoke(cfg: BurstConfig, score_date: str) -> dict:
    """GPU validator, the counterpart of `verify` for the GPU path.

    Exercises a real CUDA workload (a float64 matmul on the burst box's GPU)
    so the GPU device path is proven end to end -
    device visibility, kernel launch, reduction, and (via the driver) the
    single-GPU-resource Dask topology + CUDA teardown. It ALSO touches the day's
    gold rows so the staging / hole-check / parity plumbing is byte-identical to
    the CPU `verify` kernel. The workload is deterministic (seed derived from
    score_date, fixed float64 inputs), so a re-run on the same GPU reproduces the
    same key -> exact same-box parity. Cross-GPU parity is tolerance-based (a
    real GPU kernel concern, not this smoke).
    """
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available on this box - gpu_smoke needs a GPU")
    dev = torch.device("cuda")
    seed = int(hashlib.sha256(score_date.encode()).hexdigest()[:8], 16) % 997
    idx = (torch.arange(1536, device=dev, dtype=torch.float64) % 10 + 1) / 10.0
    a = torch.outer(idx, idx) + seed / 1000.0  # deterministic 1536x1536 float64
    c = (a @ a.t()).mean().item()  # real GPU matmul + reduction
    gpu_key = hashlib.sha256(f"{round(c, 6)}".encode()).hexdigest()[:16]

    with cfg.pg_connect() as conn, conn.cursor() as cur:
        cur.execute(
            f"SELECT count(*), count(DISTINCT ticker) FROM {GOLD_TABLE} WHERE score_date = %s",
            (score_date,),
        )
        rows, tickers = cur.fetchone()

    return {
        "score_date": score_date,
        "config_fingerprint": cfg.config_fingerprint,
        "rows": int(rows),
        "tickers": int(tickers),
        # Reuse the z_checksum field so hole-check/parity/_run_one are uniform
        # across CPU and GPU kernels (parity compares this key).
        "z_checksum": gpu_key,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_result": round(c, 6),
    }


def _backfill_core_frame(cfg: BurstConfig, score_date: str) -> "pd.DataFrame":
    """Compute one date's CORE backfill score rows via score_partition against
    the authenticated LDP medallion. Shared by the kernel (which stages the
    frame) and the tolerance parity gate (which re-derives it to check
    reproducibility) so both run identical compute. dagster is imported lazily so
    the light kernels don't pull it."""
    import os
    import sys
    from urllib.parse import quote_plus

    from sqlalchemy import create_engine

    repo = os.environ.get("MERIDIAN_REPO_ROOT", "")
    dagster_dir = os.path.join(repo, "dagster")
    if dagster_dir not in sys.path:
        sys.path.insert(0, dagster_dir)
    from meridian_pipeline.partition_scoring import score_partition
    from meridian_pipeline.pg_identity import BURST_APPLICATION_NAME, pg_connect_args

    url = (
        f"postgresql+psycopg2://{cfg.pg_user}:{quote_plus(cfg.pg_password)}"
        f"@{cfg.pg_host}:{cfg.pg_port}/{cfg.gold_db}"
    )
    engine = create_engine(url, connect_args=pg_connect_args(BURST_APPLICATION_NAME))
    try:
        df = score_partition(engine, score_date)  # groups the run requested; reads the medallion
    finally:
        engine.dispose()
    if df.empty:
        raise ValueError(f"score_partition produced 0 rows for {score_date}")
    return df


def kernel_backfill_core(cfg: BurstConfig, score_date: str) -> dict:
    """REAL CPU backfill kernel (proposal 07 "thing 1"): score the CORE
    fundamentals groups for one date via
    meridian_pipeline.partition_scoring.score_partition, reading the authenticated
    LDP medallion (parquet_io, Meridian#266), and STAGE the actual score rows as
    parquet to MinIO under `scores/`. Returns a fingerprint so the driver's
    hole-check + resume plumbing works unchanged; the staged rows are what the
    cluster promote step delete-replaces into gold. Its parity gate is TOLERANCE
    mode (float-OLS z-scores are not bit-reproducible across BLAS threadings /
    driver substrates - an exact z_checksum flags false mismatches at |dz|~1e-15;
    see parity_diff + Kernel.parity_mode).

    Needs the medallion env on the box (LAKEKEEPER_M2M_* + LDP_LAKEHOUSE_ENDPOINT
    + LDP_MINIO_* pointed at the warehouse data endpoint) and a meridian repo
    checkout (MERIDIAN_REPO_ROOT).
    """
    import hashlib
    import io

    import pyarrow as pa
    import pyarrow.parquet as pq

    df = _backfill_core_frame(cfg, score_date)

    # Stage the actual score rows (the promotable backfill output).
    s3 = cfg.s3_client()
    buf = io.BytesIO()
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), buf)
    buf.seek(0)
    scores_key = f"{cfg.run_root}/scores/score_date={score_date}/scores.parquet"
    s3.put_object(Bucket=cfg.minio_bucket, Key=scores_key, Body=buf.getvalue())

    z = df["z_score"].dropna().round(9).sort_values().to_numpy()
    return {
        "score_date": score_date,
        "config_fingerprint": cfg.config_fingerprint,
        "rows": int(len(df)),
        "tickers": int(df["ticker"].nunique()),
        "factors": int(df["factor_name"].nunique()),
        "groups": int(df["group_name"].nunique()),
        "z_checksum": hashlib.sha256(z.tobytes()).hexdigest()[:16],
        "scores_key": scores_key,
    }


@dataclass
class Kernel:
    """A pluggable per-partition compute unit. `device` drives the driver's Dask
    topology: 'cpu' fans out across many workers; 'gpu' runs one GPU-owning
    worker with a Dask GPU resource so partitions queue serially through the
    single device."""

    fn: Callable[["BurstConfig", str], dict]
    device: str  # "cpu" | "gpu"
    # Parity gate mode (parity_diff):
    #   "exact"     - re-derive the fingerprint and require an identical
    #                 z_checksum. Correct for the deterministic PoC kernels that
    #                 read gold and re-standardise (pure function of the inputs).
    #   "tolerance" - re-derive the staged z-score FRAME and require max|dz| < tol
    #                 with zero membership drift. Correct for float-compute
    #                 kernels (OLS neutralisation) whose z-scores are NOT bit-
    #                 reproducible across BLAS threadings / driver substrates - an
    #                 exact checksum flags |dz|~1e-15 noise as a false mismatch.
    parity_mode: str = "exact"
    # For tolerance kernels: re-derive the (ticker, factor_name, z_score) frame so
    # the gate can diff element-wise vs the staged scores.parquet. None for exact.
    frame_fn: "Callable[[BurstConfig, str], object] | None" = None


# Kernel registry - the driver picks by name; adding a real backfill kernel is
# one entry here plus its function. Contract: (cfg, score_date) -> flat dict
# with at least {score_date, rows, z_checksum}.
KERNELS: dict[str, Kernel] = {
    "verify": Kernel(kernel_verify, "cpu"),
    "gpu_smoke": Kernel(kernel_gpu_smoke, "gpu"),
    # Real backfill: CORE score rows from the medallion -> staged parquet (the
    # cluster promotes staging -> factors.daily_scores). Tolerance parity: the
    # float-OLS z-scores are reproducible only within ~1e-9, not bit-exact.
    "backfill_core": Kernel(
        kernel_backfill_core, "cpu", parity_mode="tolerance", frame_fn=_backfill_core_frame
    ),
}


def gpu_teardown() -> None:
    """Critical-rule-19 CUDA teardown so sequential GPU burst runs don't inherit
    a wedged context / XID 43-31 cascade. Best-effort - run via client.run on the
    GPU worker after gather. (The full cuDevicePrimaryCtxRelease-via-ctypes
    template is the cluster's; a burst worker process is torn down by
    cluster.close() right after, so synchronize + empty_cache is the pragmatic
    floor here.)"""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass


# Staging IO (MinIO): per-partition parquet + sentinel, with resume.


def _partition_key(cfg: BurstConfig, score_date: str) -> str:
    return f"{cfg.run_root}/partitions/score_date={score_date}/fingerprint.parquet"


def _sentinel_key(cfg: BurstConfig, score_date: str) -> str:
    return f"{cfg.run_root}/sentinels/{score_date}.json"


def partition_done(cfg: BurstConfig, s3, score_date: str) -> bool:
    """Resume check: a sentinel means this partition already succeeded."""
    try:
        s3.head_object(Bucket=cfg.minio_bucket, Key=_sentinel_key(cfg, score_date))
        return True
    except s3.exceptions.ClientError:
        return False


def stage_partition(cfg: BurstConfig, s3, score_date: str, fingerprint: dict) -> None:
    """Write the partition parquet, then the sentinel. Order matters: the
    sentinel is the commit marker, written last, so a crash between the two
    leaves no false 'done' (resume recomputes) - never a half-committed done."""
    table = pa.Table.from_pylist([fingerprint])
    buf = io.BytesIO()
    pq.write_table(table, buf)
    buf.seek(0)
    s3.put_object(Bucket=cfg.minio_bucket, Key=_partition_key(cfg, score_date), Body=buf.getvalue())
    s3.put_object(
        Bucket=cfg.minio_bucket,
        Key=_sentinel_key(cfg, score_date),
        Body=json.dumps(
            {
                "score_date": score_date,
                "z_checksum": fingerprint["z_checksum"],
                "staged_at": datetime.now(timezone.utc).isoformat(),
            }
        ).encode(),
    )


def load_staged_fingerprints(cfg: BurstConfig, s3) -> pd.DataFrame:
    """Read back every staged partition fingerprint for this run (cluster side
    of the parity gate + the summary)."""
    prefix = f"{cfg.run_root}/partitions/"
    paginator = s3.get_paginator("list_objects_v2")
    rows = []
    for page in paginator.paginate(Bucket=cfg.minio_bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if not obj["Key"].endswith("fingerprint.parquet"):
                continue
            body = s3.get_object(Bucket=cfg.minio_bucket, Key=obj["Key"])["Body"].read()
            rows.append(pq.read_table(io.BytesIO(body)).to_pandas())
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values("score_date").reset_index(drop=True)


def load_staged_scores(cfg: BurstConfig, s3) -> pd.DataFrame:
    """Read back every staged SCORE-ROW parquet for this run (the backfill_core
    output) - what the cluster promote step delete-replaces into
    factors.daily_scores. Columns: score_date, ticker, factor_name, group_name,
    raw_value, z_score, direction."""
    prefix = f"{cfg.run_root}/scores/"
    paginator = s3.get_paginator("list_objects_v2")
    rows = []
    for page in paginator.paginate(Bucket=cfg.minio_bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if not obj["Key"].endswith("scores.parquet"):
                continue
            body = s3.get_object(Bucket=cfg.minio_bucket, Key=obj["Key"])["Body"].read()
            rows.append(pq.read_table(io.BytesIO(body)).to_pandas())
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


# Hole-check + parity (the promotion gates)


@dataclass
class HoleCheck:
    requested: list[str]
    staged: list[str]
    failed: dict = field(default_factory=dict)  # score_date -> error string

    @property
    def missing(self) -> list[str]:
        acct = set(self.staged) | set(self.failed)
        return [d for d in self.requested if d not in acct]

    @property
    def ok(self) -> bool:
        # A batch is only clean when every requested day is staged, with zero
        # named failures and zero unaccounted holes. #238 no-silent-holes.
        return not self.missing and not self.failed


def parity_diff(
    staged: pd.DataFrame, cfg: BurstConfig, kernel_name: str = "verify", tol: float = 1e-9
) -> pd.DataFrame:
    """Re-derive each staged partition via the run's kernel and diff. Returns one
    row per MISMATCHED date (empty = clean); the driver refuses to promote a
    non-empty diff.

    Mode is per-kernel (Kernel.parity_mode):

    - "exact" (PoC kernels): re-derive the fingerprint; require an identical
      z_checksum + row count. The key is a pure function of the inputs, so a
      correct run diffs to ZERO. Any nonzero row means the burst box saw
      different data than the re-derivation.

    - "tolerance" (float-compute kernels, e.g. backfill_core): re-derive the
      z-score FRAME and require max|dz| < tol per (ticker, factor_name) with zero
      membership drift. Float-OLS z-scores are reproducible only to ~1e-9, not
      bit-exact across BLAS threadings / driver substrates, so an exact checksum
      raises false mismatches (observed |dz|~1e-15 flagged as drift). The staged
      scores.parquet is the promotable output; this proves the compute reproduces
      it, not that two BLAS runs are bitwise identical.
    """
    kernel = KERNELS[kernel_name]
    if kernel.parity_mode == "tolerance":
        return _parity_diff_tolerance(staged, cfg, kernel, tol)

    fn = kernel.fn
    mism = []
    for _, r in staged.iterrows():
        fresh = fn(cfg, r["score_date"])
        if fresh["z_checksum"] != r["z_checksum"] or fresh["rows"] != r["rows"]:
            mism.append(
                {
                    "score_date": r["score_date"],
                    "staged_checksum": r["z_checksum"],
                    "fresh_checksum": fresh["z_checksum"],
                    "staged_rows": r["rows"],
                    "fresh_rows": fresh["rows"],
                }
            )
    return pd.DataFrame(mism)


def _parity_diff_tolerance(
    staged: pd.DataFrame, cfg: BurstConfig, kernel: Kernel, tol: float
) -> pd.DataFrame:
    """Tolerance parity for a float-compute kernel: for each staged date, re-run
    the kernel's frame_fn and diff its z-scores vs the staged scores.parquet on
    (ticker, factor_name). A date is a mismatch if any matched |dz| exceeds tol or
    membership drifts (a ticker/factor scored by exactly one side)."""
    if kernel.frame_fn is None:
        raise ValueError(f"tolerance kernel {kernel.fn.__name__} has no frame_fn to re-derive")
    s3 = cfg.s3_client()
    staged_scores = load_staged_scores(cfg, s3)
    mism = []
    for _, r in staged.iterrows():
        d = r["score_date"]
        st = staged_scores[staged_scores["score_date"].astype(str) == str(d)][
            ["ticker", "factor_name", "z_score"]
        ]
        fresh = kernel.frame_fn(cfg, d)[["ticker", "factor_name", "z_score"]]
        m = st.merge(
            fresh,
            on=["ticker", "factor_name"],
            how="outer",
            suffixes=("_st", "_fr"),
            indicator=True,
        )
        both = m[m["_merge"] == "both"].dropna(subset=["z_score_st", "z_score_fr"])
        dz = (both["z_score_st"] - both["z_score_fr"]).abs()
        max_dz = float(dz.max()) if len(dz) else 0.0
        membership_drift = int((m["_merge"] != "both").sum())
        if max_dz > tol or membership_drift:
            mism.append(
                {
                    "score_date": d,
                    "max_dz": max_dz,
                    "tol": tol,
                    "membership_drift": membership_drift,
                    "staged_rows": int(len(st)),
                    "fresh_rows": int(len(fresh)),
                }
            )
    return pd.DataFrame(mism)
