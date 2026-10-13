"""
parquet_io.py - DuckDB analytical read path for LDP data and Meridian factor features.

Primary path: Iceberg REST catalog (Lakekeeper, warehouse 'ldp').
Legacy path:  flat Parquet exports from s3://ldp-parquet/ (retained until
              LDP#131 confirms retirement; see retirement gate in LDP#131 note 6531).

Env vars - Iceberg (primary, in-cluster defaults):
  LDP_LAKEHOUSE_ENDPOINT  REST catalog URL
                          default: http://catalog.example.internal:8181/catalog
  LDP_MINIO_ENDPOINT      MinIO data-file endpoint
                          default: object-store.example.internal:9500
                          (see the history note above the constant definition
                          below for how this default came to be what it is)
  LDP_MINIO_ACCESS_KEY    MinIO access key (same creds as ldp-parquet path)
  LDP_MINIO_SECRET_KEY    MinIO secret key
  LDP_MINIO_USE_SSL       'true' for external minio.example.com; 'false' in-cluster (default)
  LDP_MINIO_REGION        S3 region sent in AWS4 signature scope; default 'us-east-1'
                          set to 'garage' when pointing at a Garage object store

Env vars - flat Parquet (legacy fallback):
  LDP_PARQUET_BUCKET      S3 bucket; default: ldp-parquet
  PARQUET_BASE_PATH       local filesystem root for tests / offline dev (overrides S3)

Env vars - Meridian factor feature table (meridian-parquet bucket, separate warehouse):
  MERIDIAN_MINIO_ENDPOINT    MinIO/S3 endpoint; default: s3.example.com
                             (the external default is intended - bare research
                             boxes use this path too; in-cluster callers
                             should override to
                             object-store.example.internal:9500)
  MERIDIAN_MINIO_ACCESS_KEY  access key (required to enable this path)
  MERIDIAN_MINIO_SECRET_KEY  secret key (required to enable this path)
  MERIDIAN_MINIO_USE_SSL     'true' (default) / 'false'
  MERIDIAN_MINIO_BUCKET      bucket name; default: meridian-parquet

Registered Iceberg tables (namespace.table in warehouse 'ldp'):
  lake.ohlcv.prices                     - daily OHLCV + ret_1d, hidden year-partition
  lake.fundamentals.income_statement    - quarterly/annual income (raw_json excluded)
  lake.fundamentals.balance_sheet       - balance sheet rows
  lake.fundamentals.cash_flow           - cash flow rows

Registered Meridian-owned feature tables (meridian-parquet bucket, flat Parquet):
  factors/daily_scores/year=YYYY/*.parquet
      score_date (date), ticker (str), factor_name (str), z_score (double), year (int)
      Written by scripts/materialize_factor_features.py + daily dual-write.
      Read via fetch_factor_scores_chunks() - chunked generator for ML backfill scatter.

Columns, PIT semantics, and strict_start behaviour are identical across both paths.

Usage (assets.py):
    if parquet_io.parquet_available():
        df = parquet_io.fetch_ohlcv_ret1d(tickers, start, end)
    else:
        df = pl.from_pandas(pd.read_sql(...))

Usage (ML backfill):
    if parquet_io.meridian_parquet_available():
        for chunk in parquet_io.fetch_factor_scores_chunks(factors, start, end):
            scatter(chunk)  # chunk: DataFrame[score_date, ticker, factor_name, z_score]
    else:
        # fall back to PG read
"""

import logging
import os
import re
import time
from datetime import date, timedelta
from typing import Generator

import polars as pl

from meridian_pipeline import adj_close_guard

logger = logging.getLogger(__name__)

# Iceberg catalog settings
LDP_LAKEHOUSE_ENDPOINT: str = os.environ.get(
    "LDP_LAKEHOUSE_ENDPOINT",
    "http://catalog.example.internal:8181/catalog",
)
LDP_MINIO_USE_SSL: bool = os.environ.get("LDP_MINIO_USE_SSL", "false").lower() == "true"

# MinIO / S3 settings (shared by both paths)
#
# LDP_MINIO_ENDPOINT is env-first; the constant below is a last-resort
# fallback for any caller that doesn't set it explicitly.
#
# History (Meridian#239): the old default pointed at a Service that was later
# removed during a storage cutover. A replacement existed under a different
# name, but this code default was never updated to match, so anything that
# inherited no explicit LDP_MINIO_ENDPOINT hard-failed against a dead hostname
# even though a working replacement already existed one env var away. The live
# run-pod env already sets LDP_MINIO_ENDPOINT explicitly, so in practice this
# default is a safety net for anything that doesn't - keep it pointed at
# reality. _resolve_env logs which value won and from where, so the next
# topology change is diagnosable from one startup log line instead of a
# DuckDB IOException three call frames deep in a run pod.
_LDP_MINIO_ENDPOINT_DEFAULT = "object-store.example.internal:9500"


def _resolve_env(var_name: str, default: str) -> tuple[str, str]:
    """Resolve an env var, falling back to `default` if unset or empty.

    Returns (value, source) where source is 'env:<var_name>' when the env
    var won, or 'code-default' when the hardcoded fallback won.
    """
    val = os.environ.get(var_name)
    if val:
        return val, f"env:{var_name}"
    return default, "code-default"


LDP_MINIO_ENDPOINT, _LDP_MINIO_ENDPOINT_SOURCE = _resolve_env(
    "LDP_MINIO_ENDPOINT", _LDP_MINIO_ENDPOINT_DEFAULT
)
logger.info(
    "parquet_io: LDP_MINIO_ENDPOINT resolved to '%s' (source=%s)%s",
    LDP_MINIO_ENDPOINT,
    _LDP_MINIO_ENDPOINT_SOURCE,
    " - using code default, not env-configured; verify this still matches "
    "the current topology (Meridian#239)"
    if _LDP_MINIO_ENDPOINT_SOURCE == "code-default"
    else "",
)
LDP_MINIO_ACCESS_KEY: str = os.environ.get("LDP_MINIO_ACCESS_KEY", "")
LDP_MINIO_SECRET_KEY: str = os.environ.get("LDP_MINIO_SECRET_KEY", "")
LDP_MINIO_REGION: str = os.environ.get("LDP_MINIO_REGION", "us-east-1")

# Legacy flat-Parquet settings
LDP_PARQUET_BUCKET: str = os.environ.get("LDP_PARQUET_BUCKET", "ldp-parquet")
PARQUET_BASE_PATH: str = os.environ.get("PARQUET_BASE_PATH", "")

# Meridian-owned factor feature table settings
# Separate S3 bucket (meridian-parquet) from LDP's ldp-parquet.
# Credentials may differ; endpoint defaults to the public-facing MinIO URL
# because this path also has legitimate callers OUTSIDE the cluster (bare
# research boxes that cannot resolve
# cluster DNS at all, per docs/17-LAKEHOUSE.md). Unlike LDP_MINIO_ENDPOINT
# above, an in-cluster Service hostname would be the WRONG default here, not
# just a stale one - in-cluster callers should override to
# `object-store.example.internal:9500` explicitly via env instead
# (several times the throughput of the `s3.example.com`
# reverse-proxy path for in-cluster reads).
MERIDIAN_MINIO_ENDPOINT: str = os.environ.get("MERIDIAN_MINIO_ENDPOINT", "s3.example.com")
MERIDIAN_MINIO_ACCESS_KEY: str = os.environ.get("MERIDIAN_MINIO_ACCESS_KEY", "")
MERIDIAN_MINIO_SECRET_KEY: str = os.environ.get("MERIDIAN_MINIO_SECRET_KEY", "")
MERIDIAN_MINIO_USE_SSL: bool = os.environ.get("MERIDIAN_MINIO_USE_SSL", "true").lower() == "true"
MERIDIAN_MINIO_BUCKET: str = os.environ.get("MERIDIAN_MINIO_BUCKET", "meridian-parquet")


def _use_s3() -> bool:
    return bool(LDP_MINIO_ACCESS_KEY and LDP_MINIO_SECRET_KEY)


def _scrub_secrets(text: str) -> str:
    """Redact configured S3 credentials from a DuckDB error string.

    CREATE SECRET / SET statements interpolate raw credential values into
    the SQL text (see _build_iceberg_con / _build_con below). A DuckDB
    parser or binder error on one of those statements (e.g. a secret
    containing a single quote) echoes the offending line verbatim in the
    exception message. That message flows into _last_probe_error, which
    availability_reason() returns and which is logged to Dagster run logs
    (shipped to S3ComputeLogManager) - despite availability_reason()'s
    "Never contains secrets" contract. Scrub at the source so no downstream
    caller has to remember to.
    """
    secrets = (
        LDP_MINIO_ACCESS_KEY,
        LDP_MINIO_SECRET_KEY,
        MERIDIAN_MINIO_ACCESS_KEY,
        MERIDIAN_MINIO_SECRET_KEY,
    )
    for secret in secrets:
        if secret:
            text = text.replace(secret, "***REDACTED***")
    return text


# Availability probes (per-process cache)
_iceberg_cache: bool | None = None
_flat_parquet_cache: bool | None = None
_last_probe_error: str | None = None


def _probe_iceberg() -> bool:
    """Try to attach the Iceberg catalog and run a one-row probe. Cached."""
    global _iceberg_cache, _last_probe_error
    if _iceberg_cache is not None:
        return _iceberg_cache
    if not _use_s3():
        _iceberg_cache = False
        _last_probe_error = "iceberg disabled: LDP_MINIO_ACCESS_KEY / LDP_MINIO_SECRET_KEY not set"
        return False
    try:
        con = _iceberg_con()
        con.execute("SELECT 1 FROM lake.ohlcv.prices LIMIT 1")
        _iceberg_cache = True
    except Exception as exc:
        _last_probe_error = f"iceberg probe failed: {type(exc).__name__}: {str(exc)[:300]}"
        _iceberg_cache = False
    return _iceberg_cache


def _probe_flat_parquet() -> bool:
    """Probe legacy flat-Parquet layer. Cached."""
    global _flat_parquet_cache, _last_probe_error
    if _flat_parquet_cache is not None:
        return _flat_parquet_cache
    if _use_s3():
        try:
            _con().execute(
                f"SELECT 1 FROM read_parquet('{_ohlcv_glob()}', union_by_name=true) LIMIT 1"
            )
            _flat_parquet_cache = True
        except Exception as exc:
            _last_probe_error = f"flat-parquet probe failed: {type(exc).__name__}: {str(exc)[:300]}"
            _flat_parquet_cache = False
    else:
        _flat_parquet_cache = bool(
            PARQUET_BASE_PATH and os.path.isdir(os.path.join(PARQUET_BASE_PATH, "ohlcv"))
        )
        if not _flat_parquet_cache:
            _last_probe_error = (
                "S3 disabled and PARQUET_BASE_PATH/ohlcv not found - "
                "set LDP_MINIO_* or PARQUET_BASE_PATH"
            )
    return _flat_parquet_cache


def parquet_available() -> bool:
    """Return True if any fast-path read layer (Iceberg or flat Parquet) is reachable.

    Iceberg is tried first; falls back to flat Parquet. Result is cached per process.
    The reason for the resolved path is available via availability_reason().
    """
    if _probe_iceberg():
        return True
    # Medallion not reachable. Make the degradation LOUD - a silent fall-through
    # to the flat mirror / PG is easy to miss.
    from meridian_pipeline import lakehouse

    if lakehouse._creds_present():
        logger.warning(
            "MEDALLION NOT USED: ldp iceberg unreachable (%s) - falling back to flat "
            "parquet / PostgreSQL. See Meridian#266.",
            _last_probe_error,
        )
    if os.environ.get("MERIDIAN_REQUIRE_MEDALLION", "").lower() in ("1", "true", "yes"):
        raise RuntimeError(
            f"MERIDIAN_REQUIRE_MEDALLION is set but the ldp medallion is unavailable: "
            f"{_last_probe_error}"
        )
    return _probe_flat_parquet()


def availability_reason() -> str:
    """Human-readable explanation of the active read path. Never contains secrets."""
    if _iceberg_cache:
        ssl = "ssl" if LDP_MINIO_USE_SSL else "no-ssl"
        return f"iceberg (endpoint={LDP_LAKEHOUSE_ENDPOINT} minio={LDP_MINIO_ENDPOINT} {ssl})"
    if _flat_parquet_cache:
        if _use_s3():
            ssl = "ssl" if LDP_MINIO_USE_SSL else "no-ssl"
            return (
                f"flat-parquet (s3 endpoint={LDP_MINIO_ENDPOINT} "
                f"bucket={LDP_PARQUET_BUCKET} region={LDP_MINIO_REGION} {ssl})"
            )
        return f"flat-parquet (local PARQUET_BASE_PATH={PARQUET_BASE_PATH})"
    return _last_probe_error or "not resolved yet"


# DuckDB connection factories
#
# Connections are cached per process (#128). The expensive setup - LOAD iceberg,
# CREATE SECRET, and the ATTACH (a Lakekeeper REST round-trip) - ran on every
# fetch call before, ~40 cycles per multi-partition step. A run pod is a single
# process with a bounded lifetime, so one cached connection per backend is reused
# across all that pod's reads. The cached queries are read-only (no temp tables /
# session state), so reuse is result-identical. A build failure does not cache,
# so the availability probes still see the real error and fall back.

_iceberg_con_cache = None
_iceberg_con_minted_at = None  # monotonic seconds when the cached con's bearer was minted
_flat_con_cache = None

# The Authentik m2m bearer baked into the cached Iceberg connection's ICEBERG
# SECRET expires (typically ~1h). A long-running burst backfill (2h+) on one
# process outlives it, after which every catalog read 401s and silently falls
# back to the flat mirror (the exact bypass class Meridian#266 closed). Rebuild
# the connection with a freshly minted token once the cached one is older than
# this TTL. Minting + re-ATTACH is cheap (a few hundred ms) vs the 2.5h run, and
# reads are stateless so a rebuild is transparent. Conservatively below a 1h
# token life; override via MERIDIAN_ICEBERG_TOKEN_TTL_S.
_ICEBERG_TOKEN_TTL_S = int(os.environ.get("MERIDIAN_ICEBERG_TOKEN_TTL_S", "2400"))

# Root for DuckDB spill files. Each process gets its own subdirectory beneath
# this - see _tune_duckdb_memory. Override for a box where /tmp is small or on
# a different device from the data.
DUCKDB_SPILL_ROOT: str = os.environ.get("MERIDIAN_DUCKDB_SPILL_ROOT", "/tmp/duckdb_spill")


def _cgroup_memory_limit_bytes():
    """The container's cgroup memory limit, or None when unlimited/unreadable.

    DuckDB sizes its default memory_limit from the HOST's RAM (~80%), not the
    cgroup - inside a 6Gi-limit pod on a 16GB node it aims for ~12.8GB and gets
    OOM-killed on the first big scan (Meridian#240, runs <run-id>/<run-id>).
    """
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            raw = open(path).read().strip()
        except OSError:
            continue
        if raw == "max":
            return None
        try:
            val = int(raw)
        except ValueError:
            continue
        # v1 reports a huge sentinel when unlimited
        if 0 < val < 1 << 48:
            return val
    return None


def _apply_duckdb_threads(con):
    """Optional DuckDB thread cap. Unset leaves DuckDB's default untouched.

    DuckDB fans S3 reads out across its thread pool, and on a host whose DNS is
    served by a Docker embedded resolver that fan-out can overwhelm the resolver.
    The failure does not look like DNS: the resolver returns a non-UTF8 body,
    DuckDB tries to decode it, and the caller sees

        UnicodeDecodeError: 'utf-8' codec can't decode byte 0x84 in position 0

    from `con.execute(sql)`, with no mention of the network. It is intermittent
    and load-dependent - the same query can read tens of millions of rows and
    then fail at zero rows on the next attempt - which makes it easy to
    misdiagnose as data corruption.

    Capping threads bounds the concurrent lookups. Known-good at 4.
    """
    n = os.environ.get("MERIDIAN_DUCKDB_THREADS", "").strip()
    if not n:
        return con
    try:
        con.execute(f"SET threads={int(n)}")
        logger.info("duckdb threads capped at %s (MERIDIAN_DUCKDB_THREADS)", int(n))
    except Exception as exc:  # a cap is an optimisation, never a hard requirement
        logger.warning("could not set duckdb threads=%s: %s", n, exc)
    return con


def _apply_duckdb_memory_limit(con):
    """Cap DuckDB's memory cgroup-aware: env override wins, else 60 pct of the
    cgroup limit when one exists, else leave DuckDB's host-sized default
    (bare boxes keep full performance)."""
    _apply_duckdb_threads(con)
    override = os.environ.get("MERIDIAN_DUCKDB_MEMORY_LIMIT", "").strip()
    limit = None
    if override:
        limit = override
    else:
        cg = _cgroup_memory_limit_bytes()
        if cg is not None:
            limit = f"{max(1, int(cg * 0.6) >> 20)}MiB"
    if limit:
        try:
            con.execute(f"SET memory_limit='{limit}'")
            # Per-PROCESS spill directory. A shared one is unsafe for the burst
            # fan-out: DuckDB names temp files after the database instance, and
            # an unnamed in-memory DB gets "DEFAULT", so two worker processes
            # both write /tmp/duckdb_spill/duckdb_temp_storage_DEFAULT-0.tmp and
            # truncate each other. Observed live on the #284 re-score:
            #   IOException: Could not read enough bytes from
            #   ".../duckdb_temp_storage_DEFAULT-0.tmp": attempted to read
            #   262144 bytes from location 7340032
            # while a sibling worker held ..._S96K-0.tmp and was unaffected.
            # Only bites once a run actually spills, so a tighter memory_limit
            # (set to protect a shared box) is what surfaces it.
            spill = os.path.join(DUCKDB_SPILL_ROOT, str(os.getpid()))
            os.makedirs(spill, exist_ok=True)
            con.execute(f"SET temp_directory='{spill}'")
            logger.info(
                "duckdb memory_limit set to %s (%s)",
                limit,
                "env:MERIDIAN_DUCKDB_MEMORY_LIMIT" if override else "cgroup-derived",
            )
        except Exception:
            logger.warning("failed to set duckdb memory_limit=%s", limit, exc_info=True)
    return con


def _build_iceberg_con():
    """Build a DuckDB connection with the authenticated `ldp` Iceberg catalog attached.

    Two pieces, both required (Meridian#266):

    1. Catalog auth - an ICEBERG secret carrying a pre-fetched Authentik m2m
       bearer. Lakekeeper requires OIDC since 2026-06-12; the old tokenless
       ATTACH 401'd and every read silently served the flat mirror. DuckDB's
       modern iceberg extension DOES carry a bearer via `CREATE SECRET (TYPE
       ICEBERG, TOKEN ...)` (verified DuckDB 1.5.3). Pre-fetch the token here
       rather than using DuckDB's OAUTH2_SERVER_URI flow (that 405s vs Authentik).

    2. Data-file S3 read - GLOBAL s3 settings (SET s3_*), NOT a named s3 SECRET.
       The catalog vends absolute file URLs; a named secret's scope isn't applied
       to them so the GET 403s, whereas global settings sign correctly (verified
       end-to-end: catalog auth + data files). Uses the LDP_MINIO_* creds/endpoint
       (which must reach the warehouse-registered data endpoint).
    """
    import duckdb

    from meridian_pipeline import lakehouse

    con = _apply_duckdb_memory_limit(duckdb.connect())
    for ext in ("iceberg", "httpfs"):
        try:
            con.execute(f"LOAD {ext}")
        except Exception:
            con.execute(f"INSTALL {ext}; LOAD {ext}")

    ssl_val = "true" if LDP_MINIO_USE_SSL else "false"
    token = lakehouse.mint_lakekeeper_bearer()
    try:
        # (2) global S3 settings for the catalog-vended data-file GETs
        con.execute(f"SET s3_endpoint='{LDP_MINIO_ENDPOINT}'")
        con.execute(f"SET s3_use_ssl={ssl_val}")
        con.execute("SET s3_url_style='path'")
        con.execute(f"SET s3_access_key_id='{LDP_MINIO_ACCESS_KEY}'")
        con.execute(f"SET s3_secret_access_key='{LDP_MINIO_SECRET_KEY}'")
        if LDP_MINIO_REGION:
            con.execute(f"SET s3_region='{LDP_MINIO_REGION}'")
        # (1) catalog bearer + authenticated ATTACH
        con.execute(f"CREATE SECRET ldp_cat (TYPE ICEBERG, TOKEN '{token}')")
        con.execute(
            f"ATTACH 'ldp' AS lake (TYPE iceberg, ENDPOINT '{LDP_LAKEHOUSE_ENDPOINT}', SECRET ldp_cat)"
        )
    except Exception as exc:
        # A parser/binder error echoes the raw SQL (interpolated secret/token)
        # into str(exc); scrub before it reaches _last_probe_error /
        # availability_reason() / Dagster logs.
        exc.args = (_scrub_secrets(str(exc)).replace(token, "***REDACTED***"),)
        raise
    return con


def _iceberg_con():
    """Process-cached Iceberg connection (#128). Builds once on first use, then
    rebuilds with a freshly minted bearer once the cached token passes its TTL so
    a long-running process (burst backfills) never reads with an expired token
    (which 401s -> silent flat-mirror fallback; A5 / Meridian#266)."""
    global _iceberg_con_cache, _iceberg_con_minted_at
    stale = (
        _iceberg_con_minted_at is not None
        and (time.monotonic() - _iceberg_con_minted_at) >= _ICEBERG_TOKEN_TTL_S
    )
    if _iceberg_con_cache is None or stale:
        if stale and _iceberg_con_cache is not None:
            try:
                _iceberg_con_cache.close()
            except Exception:
                pass
            logging.getLogger(__name__).info(
                "iceberg bearer past TTL (%ss) - rebuilding catalog connection with a fresh token",
                _ICEBERG_TOKEN_TTL_S,
            )
        _iceberg_con_cache = _build_iceberg_con()
        _iceberg_con_minted_at = time.monotonic()
    return _iceberg_con_cache


def _build_con():
    """Build a DuckDB connection for legacy flat-Parquet reads (httpfs / local)."""
    import duckdb

    con = _apply_duckdb_memory_limit(duckdb.connect())
    if _use_s3():
        try:
            con.execute("LOAD httpfs")
        except Exception:
            con.execute("INSTALL httpfs")
            con.execute("LOAD httpfs")
        # Honor LDP_MINIO_USE_SSL - external endpoints (e.g. minio.example.com)
        # are HTTPS; in-cluster MinIO and Garage are plain HTTP. Hardcoding false
        # made off-cluster HTTPS reads (e.g. from a burst box) fail and silently fall back to Postgres.
        ssl_val = "true" if LDP_MINIO_USE_SSL else "false"
        try:
            con.execute(f"""
                SET s3_endpoint='{LDP_MINIO_ENDPOINT}';
                SET s3_access_key_id='{LDP_MINIO_ACCESS_KEY}';
                SET s3_secret_access_key='{LDP_MINIO_SECRET_KEY}';
                SET s3_use_ssl={ssl_val};
                SET s3_url_style=path;
                SET s3_region='{LDP_MINIO_REGION}';
            """)
        except Exception as exc:
            # Same rationale as _build_iceberg_con: scrub before this reaches
            # _last_probe_error / availability_reason() / Dagster run logs.
            exc.args = (_scrub_secrets(str(exc)),)
            raise
    return con


def _con():
    """Process-cached flat-Parquet connection (#128). Builds once on first use.

    Retained until LDP#131 retires the ldp-parquet flat exports and
    _probe_flat_parquet() can be removed.
    """
    global _flat_con_cache
    if _flat_con_cache is None:
        _flat_con_cache = _build_con()
    return _flat_con_cache


# Legacy flat-Parquet path helpers


def _ohlcv_glob() -> str:
    if _use_s3():
        return f"s3://{LDP_PARQUET_BUCKET}/ohlcv/**/*.parquet"
    return os.path.join(PARQUET_BASE_PATH, "ohlcv", "*.parquet")


def _fundamentals_path(table: str) -> str:
    if _use_s3():
        return f"s3://{LDP_PARQUET_BUCKET}/fundamentals/{table}.parquet"
    return os.path.join(PARQUET_BASE_PATH, "fundamentals", f"{table}.parquet")


def _in_list(tickers: list[str]) -> str:
    """SQL IN-list literal. Ticker symbols are alphanumeric/dots/hyphens - safe to embed."""
    return "(" + ", ".join(f"'{t}'" for t in tickers) + ")"


# Close-column selection (#125)
# The OHLCV source carries up to three close columns that diverge across splits:
#   close      - split-adjusted to today (LDP's canonical dual-write, always set)
#   raw_close  - the price actually observed on the date (correct for market cap)
#   adj_close  - split + dividend adjusted (correct for returns / momentum)
# The PG path uses COALESCE(raw_close, close) for market cap and
# COALESCE(adj_close, close) for returns (#24 rule). The Parquet/Iceberg path
# must match. LDP#137 added raw_close/adj_close to both exports; until a given
# source is re-materialised the columns may be absent, so we probe once and fall
# back to plain `close` (no worse than today) when they are missing.

_ohlcv_columns_cache: set[str] | None = None


def _ohlcv_columns() -> set[str]:
    """Column names available in the OHLCV source. Cached per process.

    Only a SUCCESSFUL probe is cached. A transient probe failure (network
    blip, catalog hiccup) used to be indistinguishable from "the columns
    genuinely don't exist yet" - both cached set(), which permanently
    degraded _mcap_close_expr()/_return_close_expr() to plain split-adjusted
    `close` (reintroducing the #125 split bias) and emptied
    _equity_scope_clause() (the 2026-07-01 index-leak parity failure) for
    the rest of the process's lifetime - a run pod can span up to 10
    partitions under BackfillPolicy.multi_run - with no log line. On failure
    we now log and leave the cache unset so the NEXT call retries; genuine
    column absence (a real DESCRIBE that just doesn't list raw_close/
    adj_close/security_type/resolution yet) still caches its result and
    degrades gracefully via each caller's own empty-set fallback.
    """
    global _ohlcv_columns_cache
    if _ohlcv_columns_cache is not None:
        return _ohlcv_columns_cache
    try:
        if _iceberg_cache:
            rows = _iceberg_con().execute("DESCRIBE lake.ohlcv.prices").fetchall()
        else:
            rows = (
                _con()
                .execute(
                    f"DESCRIBE SELECT * FROM read_parquet('{_ohlcv_glob()}', union_by_name=true) LIMIT 0"
                )
                .fetchall()
            )
    except Exception as exc:
        logger.warning(
            "_ohlcv_columns probe failed (not cached - will retry next call): %s: %s",
            type(exc).__name__,
            str(exc)[:300],
        )
        return set()
    _ohlcv_columns_cache = {r[0] for r in rows}
    return _ohlcv_columns_cache


_degrade_logged: set[str] = set()


def _log_degrade(column: str, consequence: str) -> None:
    """LOUD log for a read-path degradation caused by a missing column.

    Meridian #295 D1: these fallbacks used to be entirely silent, so a mirror
    that stopped exporting `adj_close` (or a probe that returned a truncated
    column set) quietly downgraded every returns-based factor to plain
    split-adjusted `close` with nothing in the run log to say so. That is its
    own incident class - the check_dq_split_bias lesson applied to reads: a
    degradation nobody can see is indistinguishable from correct behaviour.
    Logged at ERROR, once per column per process (the expression helpers are
    called per query and would otherwise flood the log).

    The blocking enforcement lives in evaluate_ohlcv_contract's
    required_columns leg; this is the log line that makes the degrade
    visible even where the contract is not evaluated.
    """
    if column in _degrade_logged:
        return
    _degrade_logged.add(column)
    logger.error(
        "OHLCV read-boundary degrade: column %r is NOT present in the mirror "
        "(%s). This changes factor VALUES, not just performance - see "
        "evaluate_ohlcv_contract's required_columns leg, which fails "
        "check_ohlcv_read_contract (blocking) on the same condition.",
        column,
        consequence,
    )


def _mcap_close_expr() -> str:
    """Market-cap price column: raw_close (observed) preferred. Mirrors PG
    COALESCE(raw_close, close). Falls back to close if raw_close not exported."""
    if "raw_close" in _ohlcv_columns():
        return "COALESCE(raw_close, close)"
    _log_degrade(
        "raw_close", "market cap falls back to split-adjusted close, reintroducing #125 split bias"
    )
    return "close"


def _return_close_expr() -> str:
    """Returns/momentum price column: adj_close (split+div) preferred. Mirrors PG
    COALESCE(adj_close, close). Falls back to close if adj_close not exported."""
    if "adj_close" in _ohlcv_columns():
        return "COALESCE(adj_close, close)"
    _log_degrade("adj_close", "returns/momentum fall back to close, dropping dividend adjustment")
    return "close"


# Read-boundary contract (Meridian #295, design doc D1)
#
# WHY THIS EXISTS
#   parquet_io is the single chokepoint every LDP read passes through. A
#   contract violation upstream is one that no downstream control can see by
#   looking at its own outputs. Classes of violation covered:
#     * an adj_close column that is systematically inflated relative to close.
#     * a day's bars published with 100% NULL ret_1d.
#     * a set of tickers whose bars stopped while scoring continued.
#   Each would otherwise be defended against, if at all, by individual downstream consumers
#   reasoning about their own outputs. One assertion at the boundary replaces
#   N consumers each defending themselves.
#
# THIS IS THE ONE BLOCKING DETECTIVE CONTROL
#   The repo invariant is that detective checks NEVER fail the run (a
#   data-quality WARN halting scoring starves every downstream consumer). The
#   read boundary is the documented exception: proceeding writes durable
#   corruption into factors.daily_scores, and the cost of that is precisely
#   the multi-year recompute this program exists to stop. Refusing to write is
#   strictly cheaper than writing and recomputing. Note the split within the
#   contract itself - only legs whose violation means "the inputs are wrong"
#   block; legs whose violation can mean "the inputs legitimately changed"
#   (bar-count step, CRSP residual) alert instead. See each leg's comment.
#
# THE adj_close LEG IS SOURCE-AWARE, AND MUST BE
#   `adj_close == close` is NOT the contract. The source classes differ:
#     * Non-CRSP sources carry no dividend adjustment; adj_close never exceeds
#       close.
#     * CRSP rows ARE a total-return (dlyret) chain, so adj_close != close is
#       CORRECT there and drifts further below close the further back you go.
#       An equality assertion would fire on every CRSP row and be switched off
#       within a day.
#     * Residual corruption shows as adj_close ABOVE close. Nothing legitimate
#       pushes it up: dividend back-adjustment and every corporate action
#       adjust historical prices DOWNWARD.
#   So the invariant asserted is one-sided (adj_close <= K * close, K from
#   adj_close_guard so the two cannot drift) and its THRESHOLD is per source.
REQUIRED_OHLCV_COLUMNS = frozenset(
    {"close", "adj_close", "raw_close", "ret_1d", "security_type", "resolution"}
)

# `source` is required for the source-aware adj_close leg but is NOT in
# REQUIRED_OHLCV_COLUMNS: a mirror that does not export it must not block
# scoring. Its absence degrades the leg to the CRSP-tolerant threshold and
# raises a WARN saying so (loudly, not silently).
SOURCE_COLUMN = "source"

# Sources that carry no dividend adjustment at all: adj_close == close.
NON_DIVIDEND_ADJUSTED_SOURCES = frozenset({"fmp", "wrds", "databento"})

# Ratio above which adj_close is implausible. Imported from adj_close_guard so
# the boundary contract and the row-level substitution guard cannot diverge.
ADJ_CLOSE_MAX_RATIO = adj_close_guard.ADJ_CLOSE_MAX_RATIO  # 1.1

# Share of rows allowed above ADJ_CLOSE_MAX_RATIO * close, per source class.
#   non-CRSP: a tight threshold leaves a wide margin over clean data, while a
#     corrupted column sits orders of magnitude above it. Fires on the defect,
#     never on clean data.
#   CRSP: rows above the bound are already handled row-by-row by
#     adj_close_guard (substitute close, WARN), so this leg is a REGRESSION
#     tripwire with a loose threshold, and it ALERTS rather than
#     blocks. Blocking on it would halt every historical partition of a large
#     recompute on an already-mitigated condition.
NON_CRSP_ADJ_ABOVE_MAX_PCT = 0.5
CRSP_ADJ_ABOVE_MAX_PCT = 25.0

# ret_1d non-null share on a trading day. The floor sits well below any
# legitimate dip (even a crash era), so a 90% floor has no false positives on
# healthy data and fires on a broken ret_1d pass with the full margin.
RET1D_NONNULL_MIN_PCT = 90.0

# Bar count vs the prior scored date. On healthy data the per-date ratio
# essentially never falls below the threshold, including crash eras.
# ALERTING, not blocking: a structural source change (the CRSP tape ending) is
# legitimate, and blocking it would wedge the recompute campaign on correct
# data. A large step in the input universe still has to be SEEN.
BAR_COUNT_MIN_RATIO = 0.90

# Below this row count the date is partial / mid-ingestion rather than broken;
# coverage is owned by check_factor_coverage. Same value and rationale as
# ret1d_checks._MIN_ROWS.
CONTRACT_MIN_ROWS = 500


class OhlcvContractError(RuntimeError):
    """Raised by assert_ohlcv_contract when a BLOCKING leg is violated."""


def evaluate_ohlcv_contract(stats: dict) -> dict:
    """Pure evaluator for the read-boundary contract. No DB, no I/O.

    `stats` is the shape ohlcv_contract_stats() produces:

        {
          "score_date":     "2026-07-31",
          "columns":        {"close", "adj_close", ...},   # mirror column set
          "rows":           3000,
          "ret1d_nonnull":  0,
          "prior_date":     "2026-07-30",  # or None
          "prior_rows":     3005,          # or None
          "by_source": [
              {"source": "fmp", "rows": 3000, "adj_above": 0, "max_ratio": 1.0},
          ],
        }

    Returns a dict with `passed`, `blocking` (True when at least one blocking
    leg failed), `violations` (blocking), `warnings` (alerting) and `metrics`.
    Never raises on malformed input legs it can skip - a contract evaluator
    that crashes is a contract that is not evaluated.
    """
    violations: list[dict] = []
    warnings: list[dict] = []
    metrics: dict = {"score_date": str(stats.get("score_date"))}

    columns = set(stats.get("columns") or ())
    rows = int(stats.get("rows") or 0)
    metrics["rows"] = rows
    metrics["source_path"] = stats.get("source_path", "unknown")

    # Leg 1: required columns present (replaces the SILENT DEGRADE)
    missing = sorted(REQUIRED_OHLCV_COLUMNS - columns)
    if missing:
        violations.append(
            {
                "leg": "required_columns",
                "detail": f"OHLCV mirror is missing required column(s): {missing}",
            }
        )
    metrics["missing_columns"] = missing

    # Leg 2: bars exist at all
    if rows == 0:
        violations.append({"leg": "bars_present", "detail": "no OHLCV bars for the score date"})
        # Nothing else is computable on an empty date.
        return {
            "passed": False,
            "blocking": True,
            "violations": violations,
            "warnings": warnings,
            "metrics": metrics,
        }

    partial = rows < CONTRACT_MIN_ROWS
    metrics["partial_day"] = partial

    # Leg 3: ret_1d non-null share
    nonnull = stats.get("ret1d_nonnull")
    if nonnull is not None:
        pct = float(nonnull) / rows * 100.0
        metrics["ret1d_nonnull_pct"] = round(pct, 3)
        if pct < RET1D_NONNULL_MIN_PCT:
            entry = {
                "leg": "ret1d_nonnull_share",
                "detail": (
                    f"ret_1d non-null share {pct:.2f}% < {RET1D_NONNULL_MIN_PCT}% "
                    f"({nonnull}/{rows} rows) - bars published without returns"
                ),
            }
            # A partial day is mid-ingestion; returns legitimately trail bars
            # for minutes. Report it, do not halt on it.
            (warnings if partial else violations).append(entry)

    # Leg 4: adj_close never materially above close, per source
    by_source = stats.get("by_source") or []
    source_aware = SOURCE_COLUMN in columns and bool(by_source)
    if not source_aware:
        warnings.append(
            {
                "leg": "adj_close_source_aware",
                "detail": (
                    f"mirror does not expose {SOURCE_COLUMN!r}; the adj_close leg is "
                    f"degraded to the CRSP-tolerant threshold "
                    f"({CRSP_ADJ_ABOVE_MAX_PCT}%) for the whole cross-section"
                ),
            }
        )
    per_source_metrics = []
    for row in by_source:
        src = str(row.get("source") or "unknown")
        n = int(row.get("rows") or 0)
        if n == 0:
            continue
        above = int(row.get("adj_above") or 0)
        pct = above / n * 100.0
        no_dividend_adj = source_aware and src in NON_DIVIDEND_ADJUSTED_SOURCES
        limit = NON_CRSP_ADJ_ABOVE_MAX_PCT if no_dividend_adj else CRSP_ADJ_ABOVE_MAX_PCT
        per_source_metrics.append(
            {
                "source": src,
                "rows": n,
                "above": above,
                "pct": round(pct, 4),
                "limit_pct": limit,
                "max_ratio": row.get("max_ratio"),
            }
        )
        if pct > limit:
            entry = {
                "leg": "adj_close_above_close",
                "detail": (
                    f"source={src}: {above}/{n} ({pct:.3f}%) rows have adj_close > "
                    f"{ADJ_CLOSE_MAX_RATIO}x close, limit {limit}% "
                    f"(max ratio {row.get('max_ratio')}). Nothing legitimate pushes "
                    f"adj_close above close."
                ),
            }
            # Blocking ONLY for the sources that carry no dividend adjustment,
            # where the correct share is zero. A CRSP residual is already
            # mitigated row-by-row by adj_close_guard - alert, never halt.
            (violations if no_dividend_adj else warnings).append(entry)
    metrics["by_source"] = per_source_metrics

    # Leg 5: bar count vs the prior scored date (ALERTING)
    prior_rows = stats.get("prior_rows")
    if prior_rows:
        ratio = rows / float(prior_rows)
        metrics["bar_count_ratio"] = round(ratio, 4)
        metrics["prior_date"] = str(stats.get("prior_date"))
        if ratio < BAR_COUNT_MIN_RATIO:
            warnings.append(
                {
                    "leg": "bar_count_vs_prior",
                    "detail": (
                        f"{rows} bars vs {prior_rows} on {stats.get('prior_date')} "
                        f"(ratio {ratio:.3f} < {BAR_COUNT_MIN_RATIO}) - the input "
                        f"universe stepped; confirm it is a real source change"
                    ),
                }
            )

    return {
        "passed": not violations and not warnings,
        "blocking": bool(violations),
        "violations": violations,
        "warnings": warnings,
        "metrics": metrics,
    }


def assert_ohlcv_contract(stats: dict, log=None) -> dict:
    """Evaluate the contract and RAISE OhlcvContractError on a blocking leg.

    The raising entry point, for callers that read and then write in the same
    step. `check_ohlcv_read_contract` (checks/ohlcv_contract_checks.py) uses
    the non-raising `evaluate_ohlcv_contract` instead, because a Dagster
    asset check reports the failure through its own result object.
    """
    result = evaluate_ohlcv_contract(stats)
    if log is not None:
        for w in result["warnings"]:
            log.warning(f"[ohlcv contract] {w['leg']}: {w['detail']}")
    if result["blocking"]:
        detail = "; ".join(f"{v['leg']}: {v['detail']}" for v in result["violations"])
        raise OhlcvContractError(
            f"LDP OHLCV read-boundary contract violated for "
            f"{result['metrics'].get('score_date')}: {detail}"
        )
    return result


def ohlcv_contract_stats(score_date: date, prior_date: "date | None" = None) -> "dict | None":
    """Gather contract statistics from the PARQUET/ICEBERG read path.

    Returns None when no fast path is available (the caller then falls back to
    the PostgreSQL stats query in checks/ohlcv_contract_checks.py, so the
    contract is evaluated on whichever path the assets actually read).
    """
    if not parquet_available():
        return None
    cols = _ohlcv_columns()
    if not cols:
        return None

    scope = _equity_scope_clause()
    src_col = SOURCE_COLUMN if SOURCE_COLUMN in cols else "'unknown'"
    ratio_expr = (
        f"CASE WHEN close > 0 AND adj_close > {ADJ_CLOSE_MAX_RATIO} * close THEN 1 ELSE 0 END"
        if {"adj_close", "close"} <= cols
        else "0"
    )
    max_ratio_expr = (
        "MAX(adj_close / NULLIF(close, 0))" if {"adj_close", "close"} <= cols else "NULL"
    )
    ret1d_expr = "COUNT(ret_1d)" if "ret_1d" in cols else "NULL"

    sql = f"""
        SELECT {src_col} AS source,
               COUNT(*) AS rows,
               SUM({ratio_expr}) AS adj_above,
               {max_ratio_expr} AS max_ratio,
               {ret1d_expr} AS ret1d_nonnull
        FROM {{table}}
        WHERE date::DATE = DATE '{score_date}'
          {scope}
        GROUP BY 1
    """
    prior_sql = f"""
        SELECT COUNT(*) AS rows
        FROM {{table}}
        WHERE date::DATE = DATE '{prior_date}'
          {scope}
    """

    def _run(query: str):
        if _iceberg_cache:
            return _iceberg_con().execute(query.format(table="lake.ohlcv.prices")).fetchall()
        return (
            _con()
            .execute(query.format(table=f"read_parquet('{_ohlcv_glob()}', union_by_name=true)"))
            .fetchall()
        )

    try:
        rows = _run(sql)
        prior_rows = None
        if prior_date is not None:
            got = _run(prior_sql)
            prior_rows = int(got[0][0]) if got else None
    except Exception as exc:
        logger.warning(
            "ohlcv_contract_stats parquet path failed (%s: %s) - caller falls back to PG",
            type(exc).__name__,
            str(exc)[:300],
        )
        return None

    by_source = [
        {
            "source": r[0],
            "rows": int(r[1] or 0),
            "adj_above": int(r[2] or 0),
            "max_ratio": float(r[3]) if r[3] is not None else None,
        }
        for r in rows
    ]
    total = sum(s["rows"] for s in by_source)
    ret1d_nonnull = None
    if "ret_1d" in cols:
        ret1d_nonnull = int(sum(int(r[4] or 0) for r in rows))
    return {
        "score_date": score_date,
        "columns": cols,
        "rows": total,
        "ret1d_nonnull": ret1d_nonnull,
        "prior_date": prior_date,
        "prior_rows": prior_rows,
        "by_source": by_source,
        "source_path": "iceberg" if _iceberg_cache else "parquet",
    }


def _adj_close_guard_expr() -> str:
    """Guard predicate for corrupted adj_close rows - see adj_close_guard.py.

    Mirrors the PostgreSQL path in risk_factor_scores_asset (_fetch_ath /
    _fetch_52w_high) EXACTLY by reusing the same predicate builder, so the
    two implementations of the all-time-high / 52-week-high aggregates
    cannot drift apart. Fails OPEN (predicate `TRUE`, i.e. keep every row)
    when the mirror does not export both adj_close and close, since the
    ratio cannot be formed without them.
    """
    cols = _ohlcv_columns()
    if "adj_close" not in cols or "close" not in cols:
        logger.warning(
            "adj_close guard disabled for the Parquet path: mirror exports %s "
            "(needs both adj_close and close). Corrupted adj_close rows will not be rejected.",
            sorted(cols & {"adj_close", "close"}) or "neither adj_close nor close",
        )
        return "TRUE"
    return adj_close_guard.guard_sql()


def _adj_close_guarded_px_expr() -> str:
    """Adjusted-close expression with the guard applied - see adj_close_guard.py.

    Reuses the same builder as the PostgreSQL path, so a rejected row lands
    on the SAME substituted value (its own split-adjusted `close`) in both
    implementations and the two aggregates cannot drift. Degrades to the
    plain adjusted-close expression when the mirror does not export both
    columns (the guard is a no-op there anyway - see _adj_close_guard_expr,
    which owns the WARN for that case).
    """
    cols = _ohlcv_columns()
    if "adj_close" not in cols or "close" not in cols:
        return _return_close_expr()
    return adj_close_guard.guarded_px_sql()


def _adj_close_rejected_expr() -> str:
    """Raw adj_close for WARN reporting, or NULL when the mirror lacks it."""
    return "adj_close" if "adj_close" in _ohlcv_columns() else "NULL"


# Public fetch functions
#
# Each function tries the Iceberg path first (_iceberg_cache set by
# parquet_available() which callers always invoke before reaching here).
# Flat-Parquet SQL is kept as the verbatim fallback until retirement.


def fetch_ohlcv_ret1d(
    tickers: list[str],
    start_date: date,
    end_date: date,
    *,
    strict_start: bool = True,
) -> pl.DataFrame:
    """Fetch daily ret_1d.

    Returns Polars DataFrame: date (Date), ticker (String), ret_1d (Float64).
    strict_start=True: date > start_date (CRITICAL_RULES rule 5 exclusive start).
    strict_start=False: date >= start_date.

    TICKER-SPLICE GUARD (LDP #189 shipped entity_id into the gold export):
    reused ticker symbols (a symbol last traded decades earlier) are the PRIMARY exposure for the row-count-based
    momentum window - see _compute_momentum_pl in assets.py. entity_id is
    now the correct partitioning key. Rather than restructure the rolling
    computation to group by entity_id (assets.py's output is keyed on
    ticker, and factors.daily_scores has no entity_id column), this filters
    the fetch itself: resolve each ticker's CURRENT entity_id (the entity_id
    of its own most recent row on/before end_date) and drop any row whose
    entity_id differs from it - i.e. an old, unrelated company's rows never
    even reach the rolling window. This is equivalent to partitioning by
    entity_id and keeping only the segment matching the ticker's current
    company, but requires no change to the ticker-keyed grouping/window
    code. Rows whose CURRENT entity_id is NULL (unmapped, a small share of rows)
    cannot be filtered this way and pass through unfiltered - the date-gap
    guard in splice_guard.py / _compute_momentum_pl is the remaining
    protection for those.
    """
    op = ">" if strict_start else ">="
    if "entity_id" in _ohlcv_columns():
        sql = f"""
            WITH current_entity AS (
                SELECT ticker, entity_id AS current_eid
                FROM (
                    SELECT ticker, entity_id,
                           row_number() OVER (
                               PARTITION BY ticker ORDER BY date DESC
                           ) AS rn
                    FROM {{table}}
                    WHERE ticker IN {_in_list(tickers)}
                      AND date <= '{end_date}'
                ) ranked
                WHERE rn = 1
            )
            SELECT o.date::DATE AS date, o.ticker, o.ret_1d
            FROM {{table}} o
            JOIN current_entity c ON c.ticker = o.ticker
            WHERE o.ticker IN {_in_list(tickers)}
              AND o.date {op} '{start_date}'
              AND o.date <= '{end_date}'
              AND o.ret_1d IS NOT NULL
              AND (c.current_eid IS NULL OR o.entity_id = c.current_eid)
            ORDER BY o.ticker, o.date
        """
    else:
        sql = f"""
            SELECT date::DATE AS date, ticker, ret_1d
            FROM {{table}}
            WHERE ticker IN {_in_list(tickers)}
              AND date {op} '{start_date}'
              AND date <= '{end_date}'
              AND ret_1d IS NOT NULL
            ORDER BY ticker, date
        """
    if _iceberg_cache:
        return _iceberg_con().execute(sql.format(table="lake.ohlcv.prices")).pl()
    return (
        _con()
        .execute(sql.format(table=f"read_parquet('{_ohlcv_glob()}', union_by_name=true)"))
        .pl()
    )


def _ret1d_pg_fallback_sql(op: str) -> str:
    """The PostgreSQL text for fetch_ohlcv_ret1d_pg_fallback, factored out of
    that function so tests can exercise the exact query text (e.g. against an
    in-memory engine) without duplicating it. `op` is '>' or '>=' per
    strict_start. See fetch_ohlcv_ret1d_pg_fallback's docstring for the
    ticker-splice guard rationale."""
    return f"""
        WITH current_entity AS (
            SELECT DISTINCT ON (ticker) ticker, entity_id AS current_eid
            FROM ohlcv.prices
            WHERE ticker = ANY(%(tickers)s)
              AND resolution = 'daily'
              AND ts_event::date <= %(end_date)s
            ORDER BY ticker, ts_event DESC
        )
        SELECT p.ts_event::date AS date, p.ticker, p.ret_1d
        FROM ohlcv.prices p
        JOIN current_entity c ON c.ticker = p.ticker
        WHERE p.ticker = ANY(%(tickers)s)
          AND p.resolution = 'daily'
          AND p.ts_event::date {op} %(start_date)s
          AND p.ts_event::date <= %(end_date)s
          AND p.ret_1d IS NOT NULL
          AND (c.current_eid IS NULL OR p.entity_id = c.current_eid)
        ORDER BY p.ticker, p.ts_event
    """


def fetch_ohlcv_ret1d_pg_fallback(
    engine,
    tickers: list[str],
    start_date: date,
    end_date: date,
    *,
    strict_start: bool = True,
) -> pl.DataFrame:
    """PostgreSQL fallback for fetch_ohlcv_ret1d, used when the Parquet
    mirror is unavailable or stale (near the live edge).

    Returns Polars DataFrame: date (Date), ticker (String), ret_1d (Float64).
    strict_start=True: date > start_date (CRITICAL_RULES rule 5 exclusive start).
    strict_start=False: date >= start_date.

    TICKER-SPLICE GUARD (Meridian #411, LDP #189): ported from the
    current_entity CTE in assets.py's momentum_scores PG-fallback branch,
    which is the canonical implementation of this guard against ohlcv.prices
    directly (as opposed to the Parquet path above, which reads the flat
    mirror via DuckDB and has its own equivalent CTE). Resolves each
    ticker's CURRENT entity_id (the entity_id of its own most recent row
    on/before end_date) and drops any row whose entity_id differs from it,
    so an old, unrelated company's rows for a reused ticker symbol never
    reach the rolling window. Rows whose CURRENT entity_id is NULL
    (unmapped) are not filterable this way and pass through unfiltered -
    the date-gap guard in the caller's window-completeness check is the
    remaining protection for those, same as the Parquet path.

    Shared by the two burst-compute momentum writers
    (bulk_rescore_ret1d_close.score_momentum,
    partition_scoring._score_momentum) so the guard has one PG-fallback
    implementation instead of two copy-pasted ones.

    COST NOTE (#411): the current_entity CTE's
    unbounded `ts_event::date <= end_date` scan (no lower bound - it needs
    each ticker's most recent row regardless of window start) is materially
    more expensive than the un-guarded query it replaces. The planner falls
    back to a parallel
    sequential scan either way once the ticker list is large (a bound
    ANY(%(tickers)s) array defeats the (ticker, resolution, ts_event) index at
    this cardinality) - the guard adds a second full-table pass for the
    current-entity resolution on top of that. This path only runs when the
    Parquet mirror is stale/unavailable, but a burst campaign that falls back
    for an extended window should budget for roughly 3x the PG-fallback query
    time this used to cost.
    """
    import pandas as pd

    op = ">" if strict_start else ">="
    df = pd.read_sql(
        _ret1d_pg_fallback_sql(op),
        engine,
        params={"tickers": tickers, "start_date": start_date, "end_date": end_date},
    )
    return pl.from_pandas(df)


def fetch_latest_prices(tickers: list[str], score_date: date) -> pl.DataFrame:
    """Latest close per ticker as of score_date.

    Returns Polars DataFrame: ticker (String), close (Float64).
    close = raw_close (observed price) for market cap, matching the PG path (#125).

    NOTE:
    raw_close is the price ACTUALLY OBSERVED on score_date, not rebased for any
    split that happens AFTER score_date. fundamentals.income_statement.
    shares_outstanding, by contrast, is retroactively back-adjusted to TODAY's
    split basis (verified #159: a split-adjusting ticker's shares_outstanding stays flat
    across its split - a fact not yet established when #125/#24 fixed this
    function's basis to raw_close). Multiplying today-basis shares by a
    not-rebased historical price overstates historical market cap by the
    cumulative split factor for any ticker that later splits (e.g. ~10x for a
    name that did a 10:1 split after score_date). Existing callers of THIS
    function (the legacy `_fetch_library_close_range`
    PG-only batch path) are left untouched - do not widen
    this function's basis; use `fetch_latest_mktcap_prices` below for any NEW
    market-cap-denominated consumer.
    """
    px = _mcap_close_expr()
    sql = f"""
        SELECT ticker, close
        FROM (
            SELECT ticker, {px} AS close,
                   ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) AS rn
            FROM {{table}}
            WHERE ticker IN {_in_list(tickers)}
              AND date <= '{score_date}'
              AND close IS NOT NULL
        ) ranked
        WHERE rn = 1
    """
    if _iceberg_cache:
        return _iceberg_con().execute(sql.format(table="lake.ohlcv.prices")).pl()
    return (
        _con()
        .execute(sql.format(table=f"read_parquet('{_ohlcv_glob()}', union_by_name=true)"))
        .pl()
    )


def _mktcap_close_expr() -> str:
    """SPLIT-ADJUSTED market-cap price column: `close` (LDP's canonical
    dual-write column, always set), COALESCE-guarded onto raw_close only for
    the pathological case where `close` itself is null on a row.

    This is the basis for market-cap-denominated consumers (see
    fetch_latest_prices' docstring):
    `close` is split-adjusted to today, which pairs correctly with the
    already-back-adjusted shares_outstanding. PG ohlcv.prices.close is the
    SAME dual-written split-adjusted column, so the PG fallback in
    assets._get_latest_mktcap_prices uses the same COALESCE(close,
    raw_close) expression - the corrected basis holds across a parquet
    outage. It is `split_close` (not `close`) that is deprecated
    (unsuitable as a price basis); it is never used here.
    """
    return "COALESCE(close, raw_close)"


def fetch_latest_mktcap_prices(tickers: list[str], score_date: date) -> pl.DataFrame:
    """Latest SPLIT-ADJUSTED close per ticker as of score_date, for
    market-cap-denominated factors (via assets._get_latest_mktcap_prices).
    See `_mktcap_close_expr`'s docstring for why this
    differs from `fetch_latest_prices` (which intentionally keeps its
    raw_close basis for its existing consumers and the #125 regression test).

    Returns Polars DataFrame: ticker (String), close (Float64).
    """
    px = _mktcap_close_expr()
    sql = f"""
        SELECT ticker, close
        FROM (
            SELECT ticker, {px} AS close,
                   ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) AS rn
            FROM {{table}}
            WHERE ticker IN {_in_list(tickers)}
              AND date <= '{score_date}'
              AND close IS NOT NULL
        ) ranked
        WHERE rn = 1
    """
    if _iceberg_cache:
        return _iceberg_con().execute(sql.format(table="lake.ohlcv.prices")).pl()
    return (
        _con()
        .execute(sql.format(table=f"read_parquet('{_ohlcv_glob()}', union_by_name=true)"))
        .pl()
    )


def fetch_income_pit(tickers: list[str], end_date: date) -> pl.DataFrame:
    """Quarterly income statement rows with filing_date in [end_date-815d, end_date] (PIT).

    The 815-day lower bound is safe because TTM consumers read at most 4 quarters
    at score_date AND 4 quarters at score_date-365d (for YoY growth). The
    earliest filing_date that can contribute is:
        (score_date - 365d) - 450d = score_date - 815d
    (450d is the TTM floor in _compute_ttm_pl). Rows older than 815d are
    provably unused by all current callers.

    DO NOT apply this bound to fetch_balance_pit -- balance-sheet consumers read
    the latest-available row, which can be arbitrarily old.

    Returns Polars DataFrame with columns:
    ticker, date, filing_date, revenue, net_income, ebitda,
    shares_outstanding, eps, gross_profit,
    ebit, income_before_tax, income_tax_expense, interest_expense, eps_diluted.
    """
    lower = end_date - timedelta(days=815)
    sql = f"""
        SELECT ticker, date::DATE AS date, filing_date::DATE AS filing_date,
               revenue, net_income, ebitda, shares_outstanding, eps, gross_profit,
               ebit, income_before_tax, income_tax_expense, interest_expense, eps_diluted
        FROM {{table}}
        WHERE ticker IN {_in_list(tickers)}
          AND period = 'quarter'
          AND filing_date >= '{lower}'
          AND filing_date <= '{end_date}'
        ORDER BY ticker, filing_date
    """
    if _iceberg_cache:
        return _iceberg_con().execute(sql.format(table="lake.fundamentals.income_statement")).pl()
    return (
        _con()
        .execute(sql.format(table=f"read_parquet('{_fundamentals_path('income_statement')}')"))
        .pl()
    )


def fetch_balance_pit(tickers: list[str], end_date: date) -> pl.DataFrame:
    """Balance sheet rows with filing_date <= end_date (PIT). Annual and quarterly.

    Returns Polars DataFrame with columns:
    ticker, date, filing_date, period, total_equity, total_debt,
    cash_and_equivalents, total_assets.
    """
    sql = f"""
        SELECT ticker, date::DATE AS date, filing_date::DATE AS filing_date,
               period, total_equity, total_debt, cash_and_equivalents, total_assets
        FROM {{table}}
        WHERE ticker IN {_in_list(tickers)}
          AND filing_date <= '{end_date}'
        ORDER BY ticker, filing_date
    """
    if _iceberg_cache:
        return _iceberg_con().execute(sql.format(table="lake.fundamentals.balance_sheet")).pl()
    return (
        _con()
        .execute(sql.format(table=f"read_parquet('{_fundamentals_path('balance_sheet')}')"))
        .pl()
    )


def fetch_cashflow_pit(tickers: list[str], end_date: date) -> pl.DataFrame:
    """Quarterly cash flow rows with filing_date in [end_date-815d, end_date] (PIT).

    Same 815-day lower bound as fetch_income_pit -- TTM cashflow consumers never
    read further back than score_date - 815d. See fetch_income_pit docstring for
    the derivation.

    Returns Polars DataFrame with columns:
    ticker, date, filing_date, operating_cash_flow, capital_expenditure,
    interest_paid, net_debt_issuance.
    """
    lower = end_date - timedelta(days=815)
    sql = f"""
        SELECT ticker, date::DATE AS date, filing_date::DATE AS filing_date,
               operating_cash_flow, capital_expenditure,
               interest_paid, net_debt_issuance
        FROM {{table}}
        WHERE ticker IN {_in_list(tickers)}
          AND period = 'quarter'
          AND filing_date >= '{lower}'
          AND filing_date <= '{end_date}'
        ORDER BY ticker, filing_date
    """
    if _iceberg_cache:
        return _iceberg_con().execute(sql.format(table="lake.fundamentals.cash_flow")).pl()
    return (
        _con().execute(sql.format(table=f"read_parquet('{_fundamentals_path('cash_flow')}')")).pl()
    )


def fetch_ohlcv_prices(
    tickers: list[str],
    start_date: date,
    end_date: date,
    *,
    strict_start: bool = False,
) -> pl.DataFrame:
    """Daily close prices.

    Returns Polars DataFrame: date (Date), ticker (String), close (Float64).
    close = adj_close (split+dividend adjusted) for returns, matching the PG path (#125).
    strict_start=False (default) matches the >= semantics of the PostgreSQL fallback.
    """
    op = ">" if strict_start else ">="
    px = _return_close_expr()
    sql = f"""
        SELECT date::DATE AS date, ticker, {px} AS close
        FROM {{table}}
        WHERE ticker IN {_in_list(tickers)}
          AND date {op} '{start_date}'
          AND date <= '{end_date}'
          AND close IS NOT NULL
        ORDER BY ticker, date
    """
    if _iceberg_cache:
        return _iceberg_con().execute(sql.format(table="lake.ohlcv.prices")).pl()
    return (
        _con()
        .execute(sql.format(table=f"read_parquet('{_ohlcv_glob()}', union_by_name=true)"))
        .pl()
    )


def _equity_scope_clause() -> str:
    """WHERE-clause fragment restricting rows to the PG risk-asset scope.

    The OHLCV mirror is NOT pre-filtered: it carries index rows (^DJI, ^GSPC,
    ^IXIC, ^NDX, ^VIX, ...) and non-daily resolutions alongside equity/ETF
    daily bars. The PG aggregate queries in risk_factor_scores_asset filter
    `security_type IN ('equity', 'etf')` and `resolution = 'daily'`; the
    Parquet path must apply the same predicates or index highs leak into the
    ATH/52w cross-section (real-data parity gate failure, 2026-07-01).

    Predicates are emitted only when the mirror actually carries the columns
    (probed once via _ohlcv_columns(), same pattern as the #125 close-column
    fallbacks) so older exports without the columns do not silently filter
    everything to NULL under union_by_name.
    """
    cols = _ohlcv_columns()
    clauses = []
    if "security_type" in cols:
        clauses.append("AND security_type IN ('equity', 'etf')")
    else:
        _log_degrade("security_type", "index/ETF rows leak into the equity cross-section")
    if "resolution" in cols:
        clauses.append("AND resolution = 'daily'")
    else:
        _log_degrade("resolution", "non-daily bars leak into daily aggregates")
    return ("\n          ".join(clauses)) if clauses else ""


def _resolution_clause() -> str:
    """WHERE-clause fragment restricting rows to daily bars, when the mirror
    carries a `resolution` column.

    The Iceberg mirror (lake.ohlcv.prices) is documented (module docstring,
    "Registered Iceberg tables") to carry only daily bars, but the legacy
    flat-Parquet fallback branch is NOT - see _equity_scope_clause above,
    which notes the mirror "carries index rows... and non-daily resolutions".
    Emitted only when the column is actually present (same probe-then-emit
    pattern as _equity_scope_clause / the #125 close-column fallbacks) so
    older exports without the column are not silently filtered to nothing.
    """
    return "AND resolution = 'daily'" if "resolution" in _ohlcv_columns() else ""


_ohlcv_max_date_cache: "date | None" = None
_ohlcv_max_date_probed: bool = False


def ohlcv_max_date() -> "date | None":
    """Max date present in the OHLCV mirror. Probed once per process (cached).

    Freshness primitive for aggregate consumers (fetch_ath / fetch_52w_high):
    the flat Parquet mirror can lag PostgreSQL by several days, so a MAX over
    the mirror silently misses the most recent bars (parity mismatches follow
    when the mirror is several days stale). Callers must
    only take the Parquet path when ohlcv_max_date() >= score_date.

    The probe is a single MAX(date) - answered from Parquet footer statistics /
    Iceberg manifests, so it is cheap. Returns None when the probe fails or no
    fast path is available; callers treat None as "not fresh" and fall back
    to PostgreSQL.
    """
    global _ohlcv_max_date_cache, _ohlcv_max_date_probed
    if _ohlcv_max_date_probed:
        return _ohlcv_max_date_cache
    _ohlcv_max_date_probed = True
    try:
        if _iceberg_cache:
            row = _iceberg_con().execute("SELECT MAX(date::DATE) FROM lake.ohlcv.prices").fetchone()
        else:
            row = (
                _con()
                .execute(
                    f"SELECT MAX(date::DATE) FROM read_parquet('{_ohlcv_glob()}', union_by_name=true)"
                )
                .fetchone()
            )
        _ohlcv_max_date_cache = row[0] if row else None
    except Exception as exc:
        logger.warning("ohlcv_max_date probe failed: %s: %s", type(exc).__name__, str(exc)[:300])
        _ohlcv_max_date_cache = None
    return _ohlcv_max_date_cache


def ohlcv_window_fresh_for(window_end: date, buffer_days: int = 5) -> bool:
    """True when the OHLCV mirror is reachable AND covers window_end with buffer.

    Same freshness discipline as risk_factor_scores_asset._parquet_fresh_for
    (#170 Phase 3b): per-ticker export lag means a plain max-date check is not
    enough to trust a windowed read, so the mirror must extend `buffer_days`
    calendar days past window_end before it is used. Callers with a purely
    historical window (backfill_factor_risk_model.py, spanning_test.py) should
    reliably pass; a window whose end sits within the last ~buffer_days days
    correctly falls back to PostgreSQL.
    """
    if not parquet_available():
        return False
    mirror_max = ohlcv_max_date()
    if mirror_max is None or mirror_max < window_end + timedelta(days=buffer_days):
        return False
    return True


def reset_availability_cache() -> None:
    """Forget every per-process availability/freshness probe result (#424).

    The probes above cache their answer - INCLUDING a False - for the life of
    the process, which is right for a run pod (one batch) and wrong for a
    LONG-LIVED one. Sensor bodies execute in the gRPC code-location server,
    which lives for the life of the deployment and runs two replicas: without
    this, one transient MinIO or Iceberg blip pins `parquet_available()` False
    for that replica forever, and any sensor that gates on lake readiness then
    fails closed and silently stops firing. A poller must
    call this before each probe so "the lake is down" can become "the lake is
    up" again without a pod restart.
    """
    global _iceberg_cache, _flat_parquet_cache, _last_probe_error
    global _ohlcv_max_date_cache, _ohlcv_max_date_probed
    global _iceberg_con_cache, _iceberg_con_minted_at, _flat_con_cache
    _iceberg_cache = None
    _flat_parquet_cache = None
    _last_probe_error = None
    _ohlcv_max_date_cache = None
    _ohlcv_max_date_probed = False
    # The CONNECTIONS go too. A probe that failed because the connection itself
    # was poisoned (a dropped socket, an expired bearer) would otherwise re-probe
    # on the same dead handle and keep answering False, which defeats the point
    # of the reset. Closing is best-effort: a handle we cannot close is a handle
    # we are dropping anyway.
    for con in (_iceberg_con_cache, _flat_con_cache):
        try:
            if con is not None:
                con.close()
        except Exception:  # noqa: BLE001, S110
            pass
    _iceberg_con_cache = None
    _iceberg_con_minted_at = None
    _flat_con_cache = None


class StaleWindowError(RuntimeError):
    """A windowed OHLCV read did not reach the score date (#335)."""


def require_window_ends_on(df: pl.DataFrame, score_date: date, caller: str) -> None:
    """Raise StaleWindowError unless the fetched frame's max `date` == score_date.

    #335: momentum / ETF-feature style consumers derive their trading calendar
    from the rows they are handed, so a source that stops one bar short does
    not fail - it silently produces yesterday's value stamped with today's
    date. Applied to BOTH read paths (parquet and PostgreSQL): the mirror lags
    the live bar by design, and an OHLCV ingest that has not landed yet is the
    same defect on the PG side. Partitions are already trading-day gated, so a
    genuine trading day with no universe-wide bar is an upstream fault worth
    failing on, not a case to score around. Callers skip on an empty frame
    before calling this.
    """
    # PG reads come back as datetime64 while parquet reads are pl.Date; compare
    # on a common Date dtype or the PG path trips the guard on every run.
    fetched_max = df["date"].cast(pl.Date).max()
    if fetched_max is None or fetched_max != score_date:
        raise StaleWindowError(
            f"{caller} {score_date}: fetched OHLCV window ends {fetched_max}, not the score "
            "date - the read path served a lagging window (#335); refusing to score it"
        )


def fetch_ohlcv_window(col: str, start_date: date, end_date: date) -> pl.DataFrame:
    """Full-universe {ticker, date, value} window for `col` in
    ('ret_1d', 'raw_close', 'mktcap_close').

    'mktcap_close' resolves to _mktcap_close_expr() - the SPLIT-ADJUSTED close
    basis that pairs correctly with the back-adjusted
    fundamentals.income_statement.shares_outstanding. It exists so the
    sequential risk-model replay (scripts/backfill_factor_risk_model.py) can
    build its WLS market-cap weights on the SAME basis as the daily asset
    (assets._get_latest_mktcap_prices); 'raw_close' keeps its existing basis
    and existing consumers untouched.

    Mirrors backfill_factor_risk_model.py._load_ohlcv_window's PostgreSQL semantics:
        SELECT ticker, ts_event::date AS d, <expr> AS v
        FROM ohlcv.prices
        WHERE resolution = 'daily'
          AND ts_event >= start_date AND ts_event < end_date + 1 day
          AND <expr> IS NOT NULL

    No ticker filter (full universe) and no security_type filter -- matches the
    PG query exactly (callers intersect against the scored-exposure universe
    downstream, same as the PG path). The `resolution = 'daily'` predicate IS
    applied (via _resolution_clause(), emitted only when the column is
    present) - the Iceberg mirror only carries daily bars so it's a no-op
    there, but the legacy flat-Parquet fallback branch documents (see
    _equity_scope_clause above) that it carries non-daily resolutions
    alongside daily bars, so omitting it there would silently let non-daily
    rows into the window.

    Callers MUST gate on ohlcv_window_fresh_for(end_date) before calling this -
    a lagging mirror silently drops the most recent days in the window.

    Returns Polars DataFrame: ticker (String), date (Date), value (Float64).
    """
    if col not in ("ret_1d", "raw_close", "mktcap_close"):
        raise ValueError(f"fetch_ohlcv_window: unsupported col {col!r}")
    if col == "ret_1d":
        expr = "ret_1d"
    elif col == "mktcap_close":
        expr = _mktcap_close_expr()
    else:
        expr = _mcap_close_expr()
    resolution = _resolution_clause()
    sql = f"""
        SELECT ticker, date::DATE AS date, {expr} AS value
        FROM {{table}}
        WHERE date >= '{start_date}' AND date <= '{end_date}'
          AND {expr} IS NOT NULL
          {resolution}
        ORDER BY ticker, date
    """
    if _iceberg_cache:
        return _iceberg_con().execute(sql.format(table="lake.ohlcv.prices")).pl()
    return (
        _con()
        .execute(sql.format(table=f"read_parquet('{_ohlcv_glob()}', union_by_name=true)"))
        .pl()
    )


def fetch_ath(score_date: date) -> pl.DataFrame:
    """All-time-high adjusted close per ticker up to score_date (DuckDB aggregate).

    Mirrors the PostgreSQL semantics used by risk_factor_scores_asset._fetch_ath:
        SELECT ticker, MAX(COALESCE(adj_close, close)) AS ath
        FROM ohlcv.prices
        WHERE resolution = 'daily'
          AND security_type IN ('equity', 'etf')
          AND date <= score_date
          AND COALESCE(adj_close, close) IS NOT NULL
        GROUP BY ticker

    The mirror is NOT pre-filtered (it carries index rows and non-daily
    resolutions), so the security_type/resolution predicates are applied here
    via _equity_scope_clause() to mirror the PG query exactly.

    Freshness: callers must gate on ohlcv_max_date() >= score_date before
    using this path - a lagging mirror silently understates recent highs.

    Adjusted basis: COALESCE(adj_close, close) via _return_close_expr() (#125).
    Full-history scan (no lower date bound): same semantics as the PG path.

    TICKER-SPLICE GUARD (LDP #189, entity_id now in the gold export):
    mirrors the PG `_fetch_ath` CTEs (see
    risk_factor_scores_asset.py). PRIMARY: `current_entity` resolves each
    ticker's entity_id as of its own most recent row on/before score_date,
    and the MAX only considers rows matching that entity_id - a delisted
    company's genuine all-time-high can no longer survive as the "ATH" for
    an unrelated company that later reused the same ticker symbol. FALLBACK:
    for the small share of tickers whose current entity_id is NULL, `segment_start`
    restricts the MAX to rows on/after the ticker's most recent internal date
    gap larger than splice_guard.SEGMENT_GAP_THRESHOLD_DAYS (the original
    date-based heuristic). Tickers with no gap at all are unrestricted,
    matching prior behavior. Gated on `entity_id` actually being present in
    the mirror (older exports may lack it) - falls back to the pure
    gap-based guard for every ticker when absent.

    ADJ_CLOSE GUARD (adj_close_guard.py): rows whose adj_close exceeds
    ADJ_CLOSE_MAX_RATIO x the same row's split-adjusted close are treated as
    corrupted upstream and enter the MAX on their own `close` instead - substitution,
    not nulling, so the aggregate is not biased LOW by the missing
    observation. They are counted per ticker and logged at WARN. Both the
    predicate and the substituted expression come from the same builders the
    PostgreSQL path uses, so the two implementations cannot drift.

    `rejected_max_date` breaks ties DETERMINISTICALLY (latest date among the
    rows sharing the maximum rejected adj_close, via `arg_max(d, (rej_px, d))`).
    A plain `arg_max(d, rej_px)` leaves ties to scan order, which made the
    equivalent aggregate and running-window forms disagree on a few
    tickers - a diagnostic-only difference, but one that defeats a
    bit-identical parity gate for no reason. Same ordering in the PostgreSQL
    path's `array_agg(d ORDER BY rej_px DESC, d DESC)`.

    Returns Polars DataFrame with columns: ticker (String), ath (Float64),
    rejected_rows (Int64), rejected_max_px (Float64), rejected_max_date
    (Date). Consumers read `ath`; the rejected_* columns are diagnostics.
    """
    from meridian_pipeline.splice_guard import SEGMENT_GAP_THRESHOLD_DAYS

    adj_px = _return_close_expr()
    guarded_px = _adj_close_guarded_px_expr()
    rej_px = _adj_close_rejected_expr()
    scope = _equity_scope_clause()
    has_eid = "entity_id" in _ohlcv_columns()
    eid_col = "entity_id," if has_eid else "NULL AS entity_id,"
    current_entity_join = (
        """
        JOIN current_entity c ON c.ticker = o.ticker
        """
        if has_eid
        else ""
    )
    match_clause = (
        """
            (c.current_eid IS NOT NULL AND o.entity_id = c.current_eid)
            OR (c.current_eid IS NULL AND (s.start_d IS NULL OR o.d >= s.start_d))
        """
        if has_eid
        else "s.start_d IS NULL OR o.d >= s.start_d"
    )
    current_entity_cte = (
        f"""
        current_entity AS (
            SELECT ticker, entity_id AS current_eid
            FROM (
                SELECT ticker, entity_id,
                       row_number() OVER (
                           PARTITION BY ticker ORDER BY date::DATE DESC
                       ) AS rn
                FROM {{table}}
                WHERE date::DATE <= '{score_date}'
                  {scope}
            ) ranked
            WHERE rn = 1
        ),
        """
        if has_eid
        else ""
    )
    sql = f"""
        WITH {current_entity_cte}
        ordered AS (
            SELECT ticker, date::DATE AS d, {guarded_px} AS px,
                   {rej_px} AS rej_px, {eid_col}
                   {_adj_close_guard_expr()} AS px_ok,
                   date::DATE - LAG(date::DATE) OVER (
                       PARTITION BY ticker ORDER BY date::DATE
                   ) AS gap_days
            FROM {{table}}
            WHERE date::DATE <= '{score_date}'
              {scope}
              AND {adj_px} IS NOT NULL
        ),
        segment_start AS (
            SELECT ticker, MAX(d) AS start_d
            FROM ordered
            WHERE gap_days > {SEGMENT_GAP_THRESHOLD_DAYS}
            GROUP BY ticker
        )
        SELECT o.ticker,
               MAX(o.px) AS ath,
               COUNT(*) FILTER (WHERE NOT o.px_ok) AS rejected_rows,
               MAX(o.rej_px) FILTER (WHERE NOT o.px_ok) AS rejected_max_px,
               arg_max(o.d, (o.rej_px, o.d)) FILTER (WHERE NOT o.px_ok) AS rejected_max_date
        FROM ordered o
        {current_entity_join}
        LEFT JOIN segment_start s ON s.ticker = o.ticker
        WHERE {match_clause}
        GROUP BY o.ticker
        ORDER BY o.ticker
    """
    if _iceberg_cache:
        out = _iceberg_con().execute(sql.format(table="lake.ohlcv.prices")).pl()
    else:
        out = (
            _con()
            .execute(sql.format(table=f"read_parquet('{_ohlcv_glob()}', union_by_name=true)"))
            .pl()
        )
    adj_close_guard.log_rejected_aggregate(out, "parquet_io.fetch_ath", score_date, log=logger)
    return out


def fetch_52w_high(score_date: date) -> pl.DataFrame:
    """252-trading-day (380-calendar-day) high adjusted close per ticker (DuckDB aggregate).

    Mirrors the PostgreSQL semantics used by risk_factor_scores_asset._fetch_52w_high:
        SELECT ticker, MAX(COALESCE(adj_close, close)) AS high_52w
        FROM ohlcv.prices
        WHERE resolution = 'daily'
          AND security_type IN ('equity', 'etf')
          AND date > (score_date - 380d)   -- strict lower bound
          AND date <= score_date           -- inclusive upper bound
          AND COALESCE(adj_close, close) IS NOT NULL
        GROUP BY ticker

    The mirror is NOT pre-filtered (it carries index rows and non-daily
    resolutions), so the security_type/resolution predicates are applied here
    via _equity_scope_clause() to mirror the PG query exactly.

    Freshness: callers must gate on ohlcv_max_date() >= score_date before
    using this path - a lagging mirror silently understates recent highs.

    380 calendar days covers the 252-trading-day window with buffer for weekends
    and holidays - same constant as _fetch_52w_high in risk_factor_scores_asset.py.

    Strict lower bound (date >) matches the PG path (`ts_event::date > :start`).

    Adjusted basis: COALESCE(adj_close, close) via _return_close_expr() (#125).

    TICKER-SPLICE GUARD (LDP #189, entity_id now in the gold export):
    mirrors the PG `_fetch_52w_high` CTEs - PRIMARY guard
    restricts the MAX to rows matching the ticker's own current entity_id
    (resolved from its most recent row on/before score_date); FALLBACK guard
    (for the small share of tickers with a NULL current entity_id) restricts to
    rows on/after the most recent internal date gap larger than
    splice_guard.SEGMENT_GAP_THRESHOLD_DAYS within this 380-day window. See
    fetch_ath's TICKER-SPLICE GUARD note for the full rationale.

    ADJ_CLOSE GUARD (adj_close_guard.py): identical to fetch_ath's - see that
    docstring. The 52w window is not immune: affected rows can fall
    in-window.

    Returns Polars DataFrame with columns: ticker (String), high_52w
    (Float64), rejected_rows (Int64), rejected_max_px (Float64),
    rejected_max_date (Date). Consumers read `high_52w`.
    """
    from datetime import timedelta

    from meridian_pipeline.splice_guard import SEGMENT_GAP_THRESHOLD_DAYS

    start = score_date - timedelta(days=380)
    adj_px = _return_close_expr()
    guarded_px = _adj_close_guarded_px_expr()
    rej_px = _adj_close_rejected_expr()
    scope = _equity_scope_clause()
    has_eid = "entity_id" in _ohlcv_columns()
    eid_col = "entity_id," if has_eid else "NULL AS entity_id,"
    current_entity_join = (
        """
        JOIN current_entity c ON c.ticker = o.ticker
        """
        if has_eid
        else ""
    )
    match_clause = (
        """
            (c.current_eid IS NOT NULL AND o.entity_id = c.current_eid)
            OR (c.current_eid IS NULL AND (s.start_d IS NULL OR o.d >= s.start_d))
        """
        if has_eid
        else "s.start_d IS NULL OR o.d >= s.start_d"
    )
    current_entity_cte = (
        f"""
        current_entity AS (
            SELECT ticker, entity_id AS current_eid
            FROM (
                SELECT ticker, entity_id,
                       row_number() OVER (
                           PARTITION BY ticker ORDER BY date::DATE DESC
                       ) AS rn
                FROM {{table}}
                WHERE date::DATE <= '{score_date}'
                  {scope}
            ) ranked
            WHERE rn = 1
        ),
        """
        if has_eid
        else ""
    )
    sql = f"""
        WITH {current_entity_cte}
        ordered AS (
            SELECT ticker, date::DATE AS d, {guarded_px} AS px,
                   {rej_px} AS rej_px, {eid_col}
                   {_adj_close_guard_expr()} AS px_ok,
                   date::DATE - LAG(date::DATE) OVER (
                       PARTITION BY ticker ORDER BY date::DATE
                   ) AS gap_days
            FROM {{table}}
            WHERE date::DATE > '{start}'
              AND date::DATE <= '{score_date}'
              {scope}
              AND {adj_px} IS NOT NULL
        ),
        segment_start AS (
            SELECT ticker, MAX(d) AS start_d
            FROM ordered
            WHERE gap_days > {SEGMENT_GAP_THRESHOLD_DAYS}
            GROUP BY ticker
        )
        SELECT o.ticker,
               MAX(o.px) AS high_52w,
               COUNT(*) FILTER (WHERE NOT o.px_ok) AS rejected_rows,
               MAX(o.rej_px) FILTER (WHERE NOT o.px_ok) AS rejected_max_px,
               arg_max(o.d, (o.rej_px, o.d)) FILTER (WHERE NOT o.px_ok) AS rejected_max_date
        FROM ordered o
        {current_entity_join}
        LEFT JOIN segment_start s ON s.ticker = o.ticker
        WHERE {match_clause}
        GROUP BY o.ticker
        ORDER BY o.ticker
    """
    if _iceberg_cache:
        out = _iceberg_con().execute(sql.format(table="lake.ohlcv.prices")).pl()
    else:
        out = (
            _con()
            .execute(sql.format(table=f"read_parquet('{_ohlcv_glob()}', union_by_name=true)"))
            .pl()
        )
    adj_close_guard.log_rejected_aggregate(out, "parquet_io.fetch_52w_high", score_date, log=logger)
    return out


# Batch cursors: the ATH / 52w-high aggregates for a WINDOW of dates
#
# `fetch_ath` and `fetch_52w_high` are per-date aggregates, and the burst
# re-score kernel calls them once per partition. For a multi-year campaign that
# is thousands of full-history scans of a very large mirror to compute quantities that
# barely move from one trading day to the next - the same per-date whole-table
# pattern the #275 `Precompute` hoist removed everywhere else.
#
# Both cursors below are built ONCE for a contiguous date window and answer
# `for_date(d)` from local state. They are NOT approximations: each reproduces
# its per-date function exactly, including the ticker-splice guard (LDP #189),
# the adj_close corruption guard, the COALESCE(adj_close, close) basis (#125)
# and the equity/etf + daily-resolution scope. See each class for the argument.
#
# Freshness is the caller's job, exactly as for the per-date functions: build a
# cursor only when the mirror is fresh for the window's LAST date. Freshness is
# monotone (`mirror_max >= d + buffer`), so fresh-for-the-last implies
# fresh-for-all-earlier.


def _ohlcv_table_expr() -> str:
    """The FROM expression for the active OHLCV read path."""
    if _iceberg_cache:
        return "lake.ohlcv.prices"
    return f"read_parquet('{_ohlcv_glob()}', union_by_name=true)"


def _ohlcv_con():
    return _iceberg_con() if _iceberg_cache else _con()


def _as_date(value) -> date:
    return value if isinstance(value, date) else date.fromisoformat(str(value))


def _local_con():
    """A private in-memory DuckDB connection for a cursor's cached window.

    NOT the process-cached read connection: `_iceberg_con()` closes
    and rebuilds itself when its bearer token passes its TTL, which would take a
    temp table with it mid-campaign.
    """
    import duckdb

    return _apply_duckdb_memory_limit(duckdb.connect())


class AthCursor:
    """`fetch_ath` for every date in [start, end], from ONE pass over history.

    WHY THIS IS EXACT. The all-time high is a running maximum, so the per-date
    aggregate is a step function of `score_date` that only changes on dates the
    ticker actually printed a bar. This class materialises that step function
    once and looks it up, instead of re-deriving it per date.

    Concretely, `fetch_ath(D)` groups the rows a ticker "owns" as of D and takes
    MAX(px). Which rows it owns is decided by the ticker-splice guard:

      * PRIMARY (entity_id present): `current_eid` is the entity_id of the
        ticker's most recent scoped row on/before D, and the MAX runs over that
        ticker's rows carrying that entity_id. The row that DEFINES current_eid
        is precisely the row this cursor's as-of lookup lands on, so evaluating
        an expanding-frame MAX partitioned by (ticker, entity_id) at that row
        covers exactly the same row set.
      * FALLBACK (current_eid NULL): the MAX runs over rows on/after the most
        recent internal gap larger than SEGMENT_GAP_THRESHOLD_DAYS. A gap-count
        running sum turns that into a partition key, so an expanding-frame MAX
        partitioned by (ticker, segment) is again the same row set - and it
        resets at exactly the same rows the per-date `segment_start` CTE would.

    The remaining details are carried through verbatim:

      * `gap_days` is the difference to the previous NON-NULL-px row (the
        per-date query computes LAG inside the px-filtered `ordered` CTE), which
        `LAG(...) IGNORE NULLS` reproduces while still keeping px-NULL rows in
        the frame - they matter, because `current_entity` is resolved WITHOUT
        the px filter;
      * the adj_close guard SUBSTITUTES the row's own `close` for a rejected
        adj_close (it does not drop the row - see adj_close_guard.py SEMANTICS,
        changed 2026-08-03), so `px` is the guarded expression and EVERY
        non-NULL px row enters the MAX. That is what keeps the running form
        equal to the aggregate: a substituted row can now legitimately WIN the
        maximum, and an expanding `MAX(px)` with no FILTER admits it at exactly
        the date the aggregate would first see it. The step function is still
        monotone non-decreasing, so the as-of lookup is unchanged; what changed
        is only which rows feed it. The raw offending adj_close is carried
        alongside as `rej_px` so `rejected_max_px` / `rejected_max_date` keep
        reporting the offending value verbatim, exactly as the aggregate does;
      * a ticker is emitted for a date only when at least one row is inside its
        match set on that date (`n_match > 0`) - the per-date query drops such
        tickers via the JOIN. `n_match` counts non-NULL px, which is the same
        row set the per-date `ordered` CTE keeps with
        `COALESCE(adj_close, close) IS NOT NULL`: the guarded expression is
        NULL exactly where the unguarded one is, because a rejected row always
        has a non-NULL, positive `close` (the predicate cannot fire otherwise).
        Post-substitution `ath` is never NULL on an emitted row, where the
        nulling form could emit a ticker whose matching rows were all rejected
        with a NULL ath.

    Only the window's own state rows (plus one carry-in row per ticker) are
    kept, so memory is O(tickers x dates-in-window), not O(history).
    """

    _COLUMNS = ("ticker", "ath", "rejected_rows", "rejected_max_px", "rejected_max_date")

    def __init__(self, start, end):
        from meridian_pipeline.splice_guard import SEGMENT_GAP_THRESHOLD_DAYS

        self.start = _as_date(start)
        self.end = _as_date(end)
        if self.end < self.start:
            raise ValueError(f"AthCursor end {self.end} precedes start {self.start}")

        guarded_px = _adj_close_guarded_px_expr()
        rej_px = _adj_close_rejected_expr()
        scope = _equity_scope_clause()
        has_eid = "entity_id" in _ohlcv_columns()
        eid_col = "entity_id" if has_eid else "CAST(NULL AS VARCHAR) AS entity_id"
        table = _ohlcv_table_expr()
        sql = f"""
            WITH scoped AS (
                SELECT ticker, date::DATE AS d, {guarded_px} AS px,
                       {rej_px} AS rej_px, {eid_col},
                       {_adj_close_guard_expr()} AS px_ok
                FROM {table}
                WHERE date::DATE <= '{self.end}'
                  {scope}
            ),
            gapped AS (
                SELECT *,
                       CASE WHEN px IS NULL THEN NULL ELSE
                           d - LAG(CASE WHEN px IS NULL THEN NULL ELSE d END IGNORE NULLS)
                                 OVER (PARTITION BY ticker ORDER BY d)
                       END AS gap_days
                FROM scoped
            ),
            segmented AS (
                SELECT *,
                       SUM(CASE WHEN gap_days > {SEGMENT_GAP_THRESHOLD_DAYS} THEN 1 ELSE 0 END)
                           OVER (PARTITION BY ticker ORDER BY d ROWS UNBOUNDED PRECEDING) AS seg
                FROM gapped
            ),
            cum AS (
                SELECT ticker, d, entity_id,
                       COUNT(px) OVER w_e AS n_e,
                       MAX(px) OVER w_e AS ath_e,
                       COUNT(*) FILTER (WHERE NOT px_ok) OVER w_e AS rej_n_e,
                       MAX(rej_px) FILTER (WHERE NOT px_ok) OVER w_e AS rej_px_e,
                       arg_max(d, (rej_px, d)) FILTER (WHERE NOT px_ok) OVER w_e AS rej_d_e,
                       COUNT(px) OVER w_s AS n_s,
                       MAX(px) OVER w_s AS ath_s,
                       COUNT(*) FILTER (WHERE NOT px_ok) OVER w_s AS rej_n_s,
                       MAX(rej_px) FILTER (WHERE NOT px_ok) OVER w_s AS rej_px_s,
                       arg_max(d, (rej_px, d)) FILTER (WHERE NOT px_ok) OVER w_s AS rej_d_s
                FROM segmented
                WINDOW w_e AS (
                           PARTITION BY ticker, entity_id ORDER BY d ROWS UNBOUNDED PRECEDING
                       ),
                       w_s AS (
                           PARTITION BY ticker, seg ORDER BY d ROWS UNBOUNDED PRECEDING
                       )
            ),
            state AS (
                SELECT ticker, d,
                       CASE WHEN entity_id IS NOT NULL THEN n_e ELSE n_s END AS n_match,
                       CASE WHEN entity_id IS NOT NULL THEN ath_e ELSE ath_s END AS ath,
                       CASE WHEN entity_id IS NOT NULL THEN rej_n_e ELSE rej_n_s END
                           AS rejected_rows,
                       CASE WHEN entity_id IS NOT NULL THEN rej_px_e ELSE rej_px_s END
                           AS rejected_max_px,
                       CASE WHEN entity_id IS NOT NULL THEN rej_d_e ELSE rej_d_s END
                           AS rejected_max_date
                FROM cum
            ),
            carry_in AS (
                SELECT * EXCLUDE (rn) FROM (
                    SELECT *, row_number() OVER (PARTITION BY ticker ORDER BY d DESC) AS rn
                    FROM state WHERE d < '{self.start}'
                ) WHERE rn = 1
            )
            SELECT * FROM state WHERE d >= '{self.start}'
            UNION ALL
            SELECT * FROM carry_in
            ORDER BY ticker, d
        """
        self._state = _ohlcv_con().execute(sql).pl()

    def for_date(self, score_date) -> pl.DataFrame:
        """Exactly `fetch_ath(score_date)` - same columns, same row set, same
        order (ticker), same WARN logging."""
        sd = _as_date(score_date)
        if not (self.start <= sd <= self.end):
            raise ValueError(f"AthCursor covers [{self.start}, {self.end}], asked for {sd}")
        out = (
            self._state.filter(pl.col("d") <= sd)
            .unique(subset=["ticker"], keep="last", maintain_order=True)
            .filter(pl.col("n_match") > 0)
            .select(self._COLUMNS)
            .sort("ticker")
        )
        adj_close_guard.log_rejected_aggregate(out, "parquet_io.fetch_ath", sd, log=logger)
        return out


class High52wCursor:
    """`fetch_52w_high` for every date in [start, end], off ONE window read.

    Unlike the ATH, the 52-week high is NOT cumulative - a high drops out of the
    window as `score_date` advances past it - so there is no step function to
    memoise. What IS hoistable is the READ: every date in the window draws from
    the same (start - 380d, end] slab of the mirror, which is only marginally
    larger than a single date's 380-day window.

    So this cursor caches that slab locally and runs the UNCHANGED per-date
    aggregate against it. Parity is structural rather than argued: the SQL is
    the same query with the same predicates, evaluated over a pre-filtered
    relation that is a strict superset of what each date's own predicates
    select.

    The one predicate that is not window-bounded in the per-date query is
    `current_entity` (`date <= score_date`, no lower bound). It stays exact
    here: a ticker only appears in the result at all if it has a px-non-null row
    inside (D - 380d, D], and its most recent scoped row on/before D is then at
    least as recent as that one - hence also inside the cached slab.

    The adj_close guard is applied when the slab is READ, on the same shared
    builders the per-date function uses, so a rejected row is already carrying
    its substituted `close` as `px` (plus its raw adj_close as `rej_px` for the
    diagnostics) by the time the per-date aggregate runs over it. The slab's
    `px IS NOT NULL` filter is the same row set as the per-date query's
    `COALESCE(adj_close, close) IS NOT NULL`, because the guarded expression is
    NULL exactly where the unguarded one is (a rejected row always has a
    non-NULL, positive `close`).
    """

    _LOOKBACK_DAYS = 380

    def __init__(self, start, end):
        self.start = _as_date(start)
        self.end = _as_date(end)
        if self.end < self.start:
            raise ValueError(f"High52wCursor end {self.end} precedes start {self.start}")

        guarded_px = _adj_close_guarded_px_expr()
        rej_px = _adj_close_rejected_expr()
        scope = _equity_scope_clause()
        has_eid = "entity_id" in _ohlcv_columns()
        self._has_eid = has_eid
        eid_col = "entity_id" if has_eid else "CAST(NULL AS VARCHAR) AS entity_id"
        window_start = self.start - timedelta(days=self._LOOKBACK_DAYS)
        slab = (
            _ohlcv_con()
            .execute(f"""
            SELECT ticker, date::DATE AS d, {guarded_px} AS px,
                   {rej_px} AS rej_px, {eid_col},
                   {_adj_close_guard_expr()} AS px_ok
            FROM {_ohlcv_table_expr()}
            WHERE date::DATE > '{window_start}'
              AND date::DATE <= '{self.end}'
              {scope}
        """)
            .arrow()
        )
        self._con = _local_con()
        self._con.register("_ohlcv_window", slab)
        self._con.execute("CREATE TEMP TABLE ohlcv_window AS SELECT * FROM _ohlcv_window")
        self._con.unregister("_ohlcv_window")
        del slab

    def for_date(self, score_date) -> pl.DataFrame:
        """Exactly `fetch_52w_high(score_date)` - same columns, same row set,
        same order (ticker), same WARN logging."""
        from meridian_pipeline.splice_guard import SEGMENT_GAP_THRESHOLD_DAYS

        sd = _as_date(score_date)
        if not (self.start <= sd <= self.end):
            raise ValueError(f"High52wCursor covers [{self.start}, {self.end}], asked for {sd}")
        start = sd - timedelta(days=self._LOOKBACK_DAYS)
        current_entity_cte = (
            f"""
            current_entity AS (
                SELECT ticker, entity_id AS current_eid
                FROM (
                    SELECT ticker, entity_id,
                           row_number() OVER (PARTITION BY ticker ORDER BY d DESC) AS rn
                    FROM ohlcv_window
                    WHERE d <= '{sd}'
                ) ranked
                WHERE rn = 1
            ),
            """
            if self._has_eid
            else ""
        )
        current_entity_join = (
            "JOIN current_entity c ON c.ticker = o.ticker" if self._has_eid else ""
        )
        match_clause = (
            """
                (c.current_eid IS NOT NULL AND o.entity_id = c.current_eid)
                OR (c.current_eid IS NULL AND (s.start_d IS NULL OR o.d >= s.start_d))
            """
            if self._has_eid
            else "s.start_d IS NULL OR o.d >= s.start_d"
        )
        out = self._con.execute(f"""
            WITH {current_entity_cte}
            ordered AS (
                SELECT ticker, d, px, rej_px, entity_id, px_ok,
                       d - LAG(d) OVER (PARTITION BY ticker ORDER BY d) AS gap_days
                FROM ohlcv_window
                WHERE d > '{start}'
                  AND d <= '{sd}'
                  AND px IS NOT NULL
            ),
            segment_start AS (
                SELECT ticker, MAX(d) AS start_d
                FROM ordered
                WHERE gap_days > {SEGMENT_GAP_THRESHOLD_DAYS}
                GROUP BY ticker
            )
            SELECT o.ticker,
                   MAX(o.px) AS high_52w,
                   COUNT(*) FILTER (WHERE NOT o.px_ok) AS rejected_rows,
                   MAX(o.rej_px) FILTER (WHERE NOT o.px_ok) AS rejected_max_px,
                   arg_max(o.d, (o.rej_px, o.d)) FILTER (WHERE NOT o.px_ok) AS rejected_max_date
            FROM ordered o
            {current_entity_join}
            LEFT JOIN segment_start s ON s.ticker = o.ticker
            WHERE {match_clause}
            GROUP BY o.ticker
            ORDER BY o.ticker
        """).pl()
        adj_close_guard.log_rejected_aggregate(out, "parquet_io.fetch_52w_high", sd, log=logger)
        return out

    def close(self) -> None:
        con, self._con = getattr(self, "_con", None), None
        if con is not None:
            con.close()


# Meridian-owned feature table: DuckDB connection + helpers
#
# The meridian-parquet bucket is a separate S3 warehouse from LDP's ldp-parquet.
# It uses a dedicated DuckDB connection so its S3 credentials never bleed into
# the LDP connection's session state and LDP reads are unaffected.
#
# Layout: s3://<bucket>/factors/daily_scores/year=YYYY/*.parquet
# Columns: score_date (date), ticker (str), factor_name (str), z_score (double), year (int)
# DuckDB hive_partitioning=true unpacks the year= directory component automatically.

_meridian_con_cache = None


def _use_meridian_s3() -> bool:
    return bool(MERIDIAN_MINIO_ACCESS_KEY and MERIDIAN_MINIO_SECRET_KEY)


def _build_meridian_con():
    """Build a DuckDB connection configured for the meridian-parquet bucket.

    Mirrors _build_con() for the LDP flat-Parquet path but uses the Meridian
    S3 credentials and endpoint. The connection is httpfs-only (no Iceberg);
    hive_partitioning on the glob handles year= directory partitioning.
    """
    import duckdb

    con = _apply_duckdb_memory_limit(duckdb.connect())
    try:
        con.execute("LOAD httpfs")
    except Exception:
        con.execute("INSTALL httpfs")
        con.execute("LOAD httpfs")

    ssl_val = "true" if MERIDIAN_MINIO_USE_SSL else "false"
    con.execute(f"SET s3_endpoint='{MERIDIAN_MINIO_ENDPOINT}'")
    con.execute(f"SET s3_use_ssl={ssl_val}")
    con.execute("SET s3_url_style='path'")
    con.execute(f"SET s3_access_key_id='{MERIDIAN_MINIO_ACCESS_KEY}'")
    con.execute(f"SET s3_secret_access_key='{MERIDIAN_MINIO_SECRET_KEY}'")
    return con


def _meridian_con():
    """Process-cached DuckDB connection for the meridian-parquet bucket.

    Builds once on first call; lifetime matches the run pod. Read-only queries
    only - no temp tables or session mutations after setup.
    """
    global _meridian_con_cache
    if _meridian_con_cache is None:
        _meridian_con_cache = _build_meridian_con()
    return _meridian_con_cache


def _factor_glob() -> str:
    """Glob that covers all year-partitioned factor score Parquet files.

    DuckDB hive_partitioning=true picks up the year= component; callers must
    pass hive_partitioning=true to read_parquet when using this path.
    """
    return f"s3://{MERIDIAN_MINIO_BUCKET}/factors/daily_scores/**/*.parquet"


_DUALWRITE_NAME_RE = re.compile(
    r"^dualwrite__(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})__([0-9a-f]+)\.parquet$"
)


def _meridian_s3_client():
    """Lazy boto3 S3 client for the meridian-parquet bucket.

    DuckDB's httpfs can COPY objects but cannot LIST or DELETE them, so the
    dual-write prune step (`_prune_superseded_dualwrite`) needs a real S3
    client. boto3 arrives transitively via dagster-aws[s3]; the import is lazy
    so environments that never write to the mirror need not have it.
    """
    import boto3  # noqa: PLC0415 - intentional lazy import, mirrors panel_cache._s3_client

    scheme = "https" if MERIDIAN_MINIO_USE_SSL else "http"
    return boto3.client(
        "s3",
        endpoint_url=f"{scheme}://{MERIDIAN_MINIO_ENDPOINT}",
        aws_access_key_id=MERIDIAN_MINIO_ACCESS_KEY,
        aws_secret_access_key=MERIDIAN_MINIO_SECRET_KEY,
    )


def superseded_dualwrite_keys(
    keys: list[str],
    min_date: date,
    max_date: date,
    batch_factors: set,
    factor_reader,
    keep_key: str,
) -> list[str]:
    """Return the dual-write object keys fully superseded by the current batch.

    A prior `dualwrite__<d1>_<d2>__<hash>.parquet` fragment is superseded only
    when BOTH hold:
      1. its filename date span [d1, d2] lies inside the batch span
         [min_date, max_date] - so every row it carries is a date the batch is
         rewriting; and
      2. its factor set is a subset of the batch's factor set - so every row it
         carries is a (date, factor) the batch is rewriting.

    Both conditions are required. Sibling assets legitimately write different
    factor subsets for the SAME date (value_scores vs quality_scores), and
    those fragments must survive: they fail condition 2. Only a fragment whose
    every (date, factor) cell is re-covered by the new file can be dropped, and
    dropping it is exactly what stops generations layering on top of each other
    (Meridian #275: a re-score wrote corrected z_scores into new files while
    the pre-correction files stayed, so a plain glob read both).

    Materialised `data_*` files are never candidates - they are the
    consolidated full-year snapshot written from canonical PG and are pruned
    only by scripts/materialize_factor_features.py's own year pre-delete.

    Parameters
    ----------
    keys:
        Object keys under the year prefix (full keys, not basenames).
    min_date, max_date:
        Inclusive date span of the batch just written.
    batch_factors:
        Set of factor_name values in the batch just written.
    factor_reader:
        Callable key -> set of factor_name values in that object. Only invoked
        for keys that already pass the (cheap, name-only) date-span test.
    keep_key:
        The key just written; never a deletion candidate even if it matches.
    """
    superseded: list[str] = []
    for key in keys:
        if key == keep_key:
            continue
        name = key.rsplit("/", 1)[-1]
        m = _DUALWRITE_NAME_RE.match(name)
        if m is None:
            continue  # data_* materialisation, or an unrecognised name: leave it
        try:
            d1 = date.fromisoformat(m.group(1))
            d2 = date.fromisoformat(m.group(2))
        except ValueError:
            continue
        if d1 < min_date or d2 > max_date:
            continue
        try:
            file_factors = factor_reader(key)
        except Exception:
            continue  # unreadable: never delete on a failed read
        if file_factors and set(file_factors).issubset(batch_factors):
            superseded.append(key)
    return superseded


def _prune_superseded_dualwrite(
    con,
    year: int,
    min_date: date,
    max_date: date,
    batch_factors: set,
    keep_key: str,
) -> int:
    """Delete dual-write fragments the just-written file fully supersedes.

    Runs AFTER the new file is written, so there is never a window in which
    the glob would miss the affected rows. Best effort: any failure is logged
    and swallowed (the mirror is an accelerator, never the source of truth).
    Returns the number of objects deleted.
    """
    deleted = 0
    try:
        client = _meridian_s3_client()
        prefix = f"factors/daily_scores/year={year}/"
        keys: list[str] = []
        token = None
        while True:
            kwargs = {"Bucket": MERIDIAN_MINIO_BUCKET, "Prefix": prefix}
            if token:
                kwargs["ContinuationToken"] = token
            resp = client.list_objects_v2(**kwargs)
            keys.extend(o["Key"] for o in resp.get("Contents", []))
            if not resp.get("IsTruncated"):
                break
            token = resp.get("NextContinuationToken")

        def _read_factors(key: str) -> set:
            rows = con.execute(
                "SELECT DISTINCT factor_name FROM read_parquet(?)",
                [f"s3://{MERIDIAN_MINIO_BUCKET}/{key}"],
            ).fetchall()
            return {r[0] for r in rows}

        stale = superseded_dualwrite_keys(
            keys, min_date, max_date, batch_factors, _read_factors, keep_key
        )
        # Per-key delete_object, NOT batch delete_objects: MinIO requires a
        # Content-MD5 header on DeleteObjects and botocore >= 1.36 stopped
        # sending it (CRC checksums), so the batch call fails with
        # MissingContentMD5 (same bug as panel_asset, Meridian #240).
        # Observed live during the #332 promote on 2026-09-02: every year
        # logged "dual-write prune failed" and superseded fragments piled up.
        for key in stale:
            client.delete_object(Bucket=MERIDIAN_MINIO_BUCKET, Key=key)
            deleted += 1
        if deleted:
            logger.info(
                "dual-write prune: removed %d superseded fragment(s) under year=%d "
                "(batch %s..%s, %d factors)",
                deleted,
                year,
                min_date,
                max_date,
                len(batch_factors),
            )
    except Exception as exc:
        logger.warning(
            "dual-write prune failed for year=%d: %s: %s",
            year,
            type(exc).__name__,
            str(exc)[:300],
        )
    return deleted


_factor_computed_at_cache: "dict[str, bool]" = {}


def _factor_glob_has_computed_at(glob: str) -> bool:
    """True when at least one file under `glob` carries a `computed_at` column.

    The mirror gained `computed_at` in Meridian #275. `union_by_name=true` fills
    the column with NULL for any file that predates it, but DuckDB still cannot
    BIND a column that NO file defines - so the dedup ORDER BY has to include
    the term conditionally. Probed once per glob per process (DESCRIBE reads
    footers only); a failed probe answers False, which degrades to the legacy
    (tier, filename) ordering rather than breaking the read.
    """
    cached = _factor_computed_at_cache.get(glob)
    if cached is not None:
        return cached
    present = False
    try:
        rows = (
            _meridian_con()
            .execute(
                "DESCRIBE SELECT * FROM read_parquet("
                f"'{glob}', hive_partitioning=true, union_by_name=true) LIMIT 0"
            )
            .fetchall()
        )
        present = any(r[0] == "computed_at" for r in rows)
    except Exception as exc:
        logger.debug("computed_at probe failed for %s: %s", glob, exc)
    _factor_computed_at_cache[glob] = present
    return present


def _year_range_clause(start: date, end: date) -> str:
    """WHERE-clause fragment pruning the hive `year=YYYY` partitions.

    _factor_glob() is a fixed all-years glob; filtering only on score_date
    (as the three readers below used to) cannot trigger DuckDB's
    year-partition pruning - every file's footer must be opened before
    score_date zonemaps apply, even though the docstrings claimed pruning
    was already happening. This turns the file set DuckDB has to touch back
    into just the queried window's year partitions.
    """
    return f"AND year >= {start.year} AND year <= {end.year}"


def factor_dedup_qualify(glob: str) -> str:
    """QUALIFY clause that resolves duplicate flat-mirror generations.

    Single source of truth for the mirror's duplicate-key precedence. Every
    reader of the flat `factors/daily_scores` glob MUST apply it: the mirror
    can legitimately carry the SAME (score_date, ticker, factor_name) key in
    more than one file (see `fetch_factor_scores_chunks` for the full
    write-side story), and a reader that dedups with ANY_VALUE/GROUP BY picks
    an arbitrary generation - Meridian #381, where research-side readers
    returned the stale pre-re-score z_score on most
    DuckDB runs on a two-generation fixture.

    Order (recency first):
      1. `computed_at` DESC NULLS LAST - the generation stamp added in #275.
         The newest write for a key wins outright, whichever file it sits in.
         Only emitted when at least one file under the glob defines the column
         (`_factor_glob_has_computed_at`); DuckDB cannot bind a column no file
         declares, even with union_by_name=true.
      2. `data_*` tier over `dualwrite__*`. Legacy fallback for unstamped
         rows: a materialisation reads all of factors.daily_scores at run
         time, so pre-#275 it was the best available proxy for "consolidated".
      3. `filename` DESC. Stable, deterministic last resort; dualwrite names
         carry a factor-set hash, not a timestamp, so this is not a recency
         guarantee.

    The caller must read with `filename=true` and `union_by_name=true` for
    terms 1 and 2 to bind.
    """
    recency_term = (
        "computed_at DESC NULLS LAST,\n                "
        if _factor_glob_has_computed_at(glob)
        else ""
    )
    return f"""QUALIFY ROW_NUMBER() OVER (
            PARTITION BY score_date, ticker, factor_name
            ORDER BY
                {recency_term}CASE WHEN contains(filename, '/data_')
                    OR contains(filename, chr(92) || 'data_') THEN 1 ELSE 0 END DESC,
                filename DESC
        ) = 1"""


def factor_dedup_sql(select_expr: str, where: str, glob: str | None = None) -> str:
    """Build a generation-correct SELECT over the flat factor mirror.

    `select_expr` is the projection (may alias, e.g. `score_date AS date`);
    `where` is the full WHERE body WITHOUT the leading keyword. The QUALIFY
    window references the base columns, so aliasing the projection is safe.

    Readers should compose `where` from `_year_range_clause(start, end)` so
    the hive year partitions are pruned - note a window extended backwards
    (the weight panel's as-of lookback) can cross a year boundary, so build
    the clause from the EXTENDED start, not the caller's start.
    """
    glob = glob or _factor_glob()
    return f"""
        SELECT {select_expr}
        FROM read_parquet(
            '{glob}', hive_partitioning=true, filename=true, union_by_name=true
        )
        WHERE {where}
        {factor_dedup_qualify(glob)}
    """


# Meridian-parquet availability probe (per-process cache)
_meridian_parquet_cache: bool | None = None
_meridian_last_probe_error: str | None = None


def meridian_parquet_available() -> bool:
    """Return True if the meridian-parquet factor feature table is reachable.

    Cached per process - probes once then returns the cached result. Returns
    False immediately when credentials are absent (no network call). Errors
    are caught and logged so callers can fall back to PG without crashing.

    Mirrors the parquet_available() caching + _last_probe_error pattern used
    for the LDP read path.
    """
    global _meridian_parquet_cache, _meridian_last_probe_error
    if _meridian_parquet_cache is not None:
        return _meridian_parquet_cache

    if not _use_meridian_s3():
        _meridian_last_probe_error = (
            "meridian-parquet disabled: "
            "MERIDIAN_MINIO_ACCESS_KEY / MERIDIAN_MINIO_SECRET_KEY not set"
        )
        _meridian_parquet_cache = False
        return False

    try:
        glob = _factor_glob()
        _meridian_con().execute(
            f"SELECT 1 FROM read_parquet('{glob}', hive_partitioning=true, union_by_name=true) LIMIT 1"
        )
        _meridian_parquet_cache = True
    except Exception as exc:
        _meridian_last_probe_error = (
            f"meridian-parquet probe failed: {type(exc).__name__}: {str(exc)[:300]}"
        )
        logger.warning(_meridian_last_probe_error)
        _meridian_parquet_cache = False

    return _meridian_parquet_cache


def meridian_parquet_availability_reason() -> str:
    """Human-readable explanation of the meridian-parquet probe result."""
    if _meridian_parquet_cache:
        ssl = "ssl" if MERIDIAN_MINIO_USE_SSL else "no-ssl"
        return (
            f"meridian-parquet available "
            f"(endpoint={MERIDIAN_MINIO_ENDPOINT} bucket={MERIDIAN_MINIO_BUCKET} {ssl})"
        )
    return _meridian_last_probe_error or "meridian-parquet not probed yet"


# Chunked factor score reader


def _year_pages(start: date, end: date) -> Generator[tuple[date, date], None, None]:
    """Split an inclusive [start, end] date range into per-calendar-year
    sub-ranges, each clipped to [start, end].

    e.g. (2022-06-01, 2024-03-15) yields:
        (2022-06-01, 2022-12-31), (2023-01-01, 2023-12-31), (2024-01-01, 2024-03-15)

    A range that fits inside a single year yields exactly one page equal to
    the original range - so single-year callers see identical read_daily_scores
    calls to the pre-paging implementation.
    """
    year = start.year
    while year <= end.year:
        page_start = start if year == start.year else date(year, 1, 1)
        page_end = end if year == end.year else date(year, 12, 31)
        yield page_start, page_end
        year += 1


def _fetch_factor_scores_chunks_iceberg(
    factors: list[str],
    start: date,
    end: date,
    chunk_rows: int,
    *,
    tickers: list[str] | None = None,
) -> Generator["pd.DataFrame", None, None]:  # type: ignore[name-defined]  # noqa: F821
    """Iceberg-primary implementation backing fetch_factor_scores_chunks().

    Reads meridian.factors.daily_scores via lakehouse.read_daily_scores() with
    the same (date range, factor_names, tickers) filters as the flat-parquet
    path, then pages the result into pandas chunks of the same shape/columns
    the flat path yields: [score_date, ticker, factor_name, z_score].

    Pages the underlying read one calendar year at a time (_year_pages) rather
    than materialising the whole [start, end] range in one
    lakehouse.read_daily_scores() call - a multi-year research read (the
    full table) would otherwise pull the entire range into memory
    before any chunk_rows slicing happened, defeating the chunk contract's
    "peak RSS proportional to chunk_rows" guarantee and reproducing the
    #98/#188 whole-population OOM bug class. Peak memory is now bounded by one
    year's slice (plus a small leftover carried across a page boundary so
    yielded chunk sizes still line up on chunk_rows, matching the pre-paging
    behaviour byte-for-byte within a single year).

    write_daily_scores() replaces rows by (score_date, factor_name[, group])
    on every write, so the Iceberg table has no dualwrite-fragment duplicate
    problem - only the z_score IS NOT NULL filter (applied client-side here,
    since read_daily_scores() has no null predicate) mirrors the flat path's
    behaviour of dropping null-z_score rows.
    """
    from meridian_pipeline import lakehouse

    leftover: "pl.DataFrame | None" = None  # noqa: F821
    for page_start, page_end in _year_pages(start, end):
        frame = lakehouse.read_daily_scores(
            start=page_start,
            end=page_end,
            tickers=tickers,
            factor_names=factors,
            columns=["score_date", "ticker", "factor_name", "z_score"],
        )
        frame = frame.filter(pl.col("z_score").is_not_null())
        if leftover is not None:
            frame = pl.concat([leftover, frame])
            leftover = None
        if frame.height == 0:
            continue

        n_rows = frame.height
        offset = 0
        while offset + chunk_rows <= n_rows:
            yield frame.slice(offset, chunk_rows).to_pandas()
            offset += chunk_rows
        if offset < n_rows:
            # Carry the remainder into the next page instead of yielding an
            # undersized chunk mid-range - keeps chunking behaviour identical
            # to the unpaged implementation for ranges that span years.
            leftover = frame.slice(offset, n_rows - offset)

    if leftover is not None and leftover.height > 0:
        yield leftover.to_pandas()


def fetch_factor_scores_chunks(
    factors: list[str],
    start: date,
    end: date,
    chunk_rows: int = 2_000_000,
    *,
    tickers: list[str] | None = None,
) -> Generator["pd.DataFrame", None, None]:  # type: ignore[name-defined]  # noqa: F821
    """Yield pandas DataFrames from the meridian-parquet factor feature table.

    Each yielded chunk has exactly these columns (matching a pd.read_sql of
    the same query against factors.daily_scores):
        score_date  - date or datetime (as returned by DuckDB for a date column)
        ticker      - str
        factor_name - str
        z_score     - float64

    Column projection and WHERE predicates are pushed down into DuckDB so only
    the requested (factor_name, date-range) slice is read from S3. An explicit
    `year` predicate (_year_range_clause) additionally limits which Parquet
    files are opened via the hive year= partitions.

    Why a chunked generator?
    This mirrors the ml_data streaming-scatter chunk contract used in the
    ML backfill (#98). The consumer scatters each chunk's
    (score_date, ticker, factor_name, z_score) rows into dense in-memory arrays
    keyed by (score_date, ticker). Row and chunk ordering are irrelevant because
    daily_scores keys are unique (one z_score per ticker/factor/date) ONCE the
    QUALIFY dedup below has resolved any duplicate (score_date, ticker,
    factor_name) rows across dual-write files, so last-write-wins across
    chunks is result-identical. Yielding in bounded chunks instead of one
    fetch keeps peak RSS proportional to chunk_rows rather than the full date
    range (a very large table).

    Duplicate-row precedence (audit batch C #171, recency-first since #275):
    `append_factor_scores_partition` writes one `dualwrite__<dates>__<hash>`
    file per distinct non-null factor set per batch, and never deletes a
    prior file when a later re-score changes which factors are non-null - so
    two files can legitimately carry the SAME (score_date, ticker,
    factor_name) key with DIFFERENT z_score values (e.g. a corrected re-run
    after a factor-math fix). `factor_dedup_qualify` breaks that tie
    deterministically, RECENCY FIRST:
      1. Newest `computed_at` wins, whichever file the row sits in. A
         re-score is by construction the corrected generation, so a later
         `dualwrite__*` fragment BEATS an earlier materialised `data_*`
         file - not the other way round. (This ordering is the shipped
         behaviour; the docstring here claimed the inverse until #381.)
      2. Rows with no stamp (files written before #275) fall back to the
         tier rule: a materialised full-rebuild file (basename `data_*`,
         written by scripts/materialize_factor_features.py from canonical PG)
         beats any `dualwrite__*` fragment, because a materialisation reads
         all of factors.daily_scores at run time.
      3. Among unstamped files of the same tier, ties break on `filename`
         DESC. This is a STABLE, deterministic fallback, not a true
         write-time ordering - dualwrite filenames carry a factorset hash,
         not a timestamp, precisely so idempotent same-day/same-factor-set
         reruns overwrite in place instead of accumulating files. A
         dualwrite-vs-dualwrite collision between two unstamped rows is
         therefore resolved deterministically but not provably by recency;
         the durable fix for that residual case is to materialise the
         affected year, which collapses it to one generation.

    Parameters
    ----------
    factors:
        Factor names to include (column values of factor_name). An empty list
        returns no rows.
    start, end:
        Inclusive date range [start, end] for score_date.
    chunk_rows:
        Target rows per yielded chunk. DuckDB's fetch_df_chunk works in
        vector-batch units of ~2 048 rows; we calculate vectors_per_chunk to
        approximate this target (minimum 1). The actual chunk size may differ
        slightly - consumers must not rely on exact size.
    tickers:
        Optional ticker shard. When None (default), all tickers are returned
        (behaviour identical to before this parameter existed). When a non-empty
        list, adds ``AND ticker IN (...)`` to the WHERE clause so DuckDB only
        reads rows for that shard - useful for the Phase 2 sharded dataloader
        that processes one ticker-shard at a time. An empty list returns no rows.

    Yields
    ------
    pd.DataFrame with columns [score_date, ticker, factor_name, z_score].
    Raises StopIteration when all rows are exhausted.
    Propagates DuckDB exceptions to the caller (no silent swallow here; callers
    should guard with meridian_parquet_available() and handle exceptions to
    fall back to PG).
    """
    import pandas as pd  # noqa: F401 - used implicitly via DuckDB fetch_df_chunk

    if not factors:
        return
    if tickers is not None and not tickers:
        return

    from meridian_pipeline import lakehouse

    if lakehouse.iceberg_read_primary() and lakehouse.daily_scores_available():
        yield from _fetch_factor_scores_chunks_iceberg(
            factors, start, end, chunk_rows, tickers=tickers
        )
        return

    glob = _factor_glob()

    # Build a safe IN-list for factor names. Factor names come from FACTOR_MAP /
    # FACTOR_GROUPS - they are alphanumeric with underscores; no injection risk,
    # but we quote defensively anyway.
    factor_in = ", ".join(f"'{f}'" for f in factors)

    # Optional ticker shard pushed down into DuckDB. _in_list() quotes each
    # symbol with single quotes (same style as factor_in above). Ticker symbols
    # are alphanumeric with dots/hyphens - no injection risk.
    ticker_clause = f"  AND ticker IN {_in_list(tickers)}" if tickers is not None else ""
    year_clause = _year_range_clause(start, end)

    sql = factor_dedup_sql(
        "score_date, ticker, factor_name, z_score",
        f"""score_date >= '{start}'
          AND score_date <= '{end}'
          {year_clause}
          AND factor_name IN ({factor_in})
          AND z_score IS NOT NULL
{ticker_clause}""",
        glob=glob,
    )

    con = _meridian_con()
    rel = con.execute(sql)

    # DuckDB vector size is 2048 rows; choose how many vectors to fetch per
    # chunk to approximate chunk_rows. Minimum 1 to avoid an infinite loop on
    # very small chunk_rows values.
    _DUCKDB_VECTOR_SIZE = 2048
    vectors_per_chunk = max(1, chunk_rows // _DUCKDB_VECTOR_SIZE)

    while True:
        chunk = rel.fetch_df_chunk(vectors_per_chunk)
        if chunk.empty:
            break
        yield chunk


def append_factor_scores_partition(df: "pd.DataFrame", factors: list[str]) -> int:  # type: ignore[name-defined]  # noqa: F821
    """Best-effort append of ALL factor rows from `df` to the feature table.

    Returns rows written (0 if skipped/failed). NEVER raises.

    `df` is the upsert payload with at least columns:
        score_date, ticker, factor_name, z_score
    (plus group_name, raw_value, direction which are ignored here).

    The `factors` argument is accepted for backward compatibility but is NOT used
    as the filter set. The function mirrors EVERY factor_name in the payload -
    no catalogue filter. The mirror is a faithful full copy of what lands in
    factors.daily_scores, including factors that are not registered in
    FACTOR_GROUPS. Consumers (research-side readers) rely on the mirror
    matching PG content per date; filtering here silently starves them
    (a parity gate found many rows per date missing under the old
    FACTOR_GROUPS-only filter).

    We keep only rows where z_score is not null.

    CONSEQUENCE: a per-date BROADCAST constant (one value shared by every ticker
    on a date) is stored without a z_score by convention, so the null filter
    drops it and it is NOT in the mirror at all. Such constants are PG-ONLY:
    read them from factors.daily_scores, never from the mirror. The mirror
    returns an empty result rather than an error for them, which is the failure
    mode the "silently starves them" warning above describes, just arriving by
    a different route.

    Rows are then written to s3://<bucket>/factors/daily_scores/year=YYYY/
    as a per-batch Parquet file. Layout is consistent with the materialisation
    script (year=YYYY/*.parquet, hive year-partition only).

    File naming for idempotency + no cross-asset collision:
        year=YYYY/dualwrite__<mindate>_<maxdate>__<factorset_hash>.parquet
    where factorset_hash is a short SHA-1 of the sorted factor list.
    - Same asset re-running the same day OVERWRITES its own file (same hash).
    - Different assets writing different factor subsets -> different hashes ->
      different files; DuckDB globs them all on read.
    - A `year` column is added to match the hive-partition layout.
    """
    rows_written = 0
    try:
        if not _use_meridian_s3():
            return 0

        import datetime as _dt
        import hashlib

        import pandas as _pd

        # No catalogue filter: mirror every factor row with a non-null z_score
        # so the mirror stays a faithful full copy of factors.daily_scores.
        mask = df["z_score"].notna()
        subset = df.loc[mask, ["score_date", "ticker", "factor_name", "z_score"]].copy()
        if subset.empty:
            return 0

        # Deterministic factorset hash: SHA-1 of the batch's actual factor set.
        # Same asset re-running the same day rewrites the same file (idempotent);
        # different assets writing different factor subsets get different files.
        _batch_factors = sorted(subset["factor_name"].unique())
        factorset_hash = hashlib.sha1(",".join(_batch_factors).encode()).hexdigest()[:12]

        # Add year column for hive layout; convert score_date to Python date
        # so DuckDB serialises it as a Parquet date column.
        subset["score_date"] = _pd.to_datetime(subset["score_date"]).dt.date
        subset["year"] = subset["score_date"].apply(lambda d: d.year)

        # Generation stamp: lets a reader break a duplicate-key tie by recency
        # and makes a layered mirror diagnosable after the fact (Meridian #275).
        computed_at = _dt.datetime.now(_dt.timezone.utc).replace(tzinfo=None)
        subset["computed_at"] = computed_at

        con = _meridian_con()
        bucket = MERIDIAN_MINIO_BUCKET

        # Group by year so cross-year batches each land in the correct partition.
        for yr, grp in subset.groupby("year"):
            min_date = grp["score_date"].min()
            max_date = grp["score_date"].max()
            fname = f"dualwrite__{min_date}_{max_date}__{factorset_hash}.parquet"
            key = f"factors/daily_scores/year={yr}/{fname}"
            s3_path = f"s3://{bucket}/{key}"

            # Register the slice as a DuckDB relation then COPY to S3.
            # OVERWRITE_OR_IGNORE makes the write idempotent: same file
            # name -> same content on re-run (same date, same factors).
            # DROP the `year` column: the materialisation uses PARTITION_BY (year)
            # which keeps `year` ONLY in the path (year=YYYY/), not in the data. We
            # must match that schema exactly so a mixed read (a year holding both a
            # materialised file and dualwrite files) sees one consistent schema -
            # hive_partitioning reconstructs `year` from the path for all files.
            grp_clean = grp.drop(columns=["year"]).reset_index(drop=True)
            con.register("_dualwrite_tmp", grp_clean)
            try:
                con.execute(
                    f"COPY _dualwrite_tmp TO '{s3_path}' (FORMAT PARQUET, OVERWRITE_OR_IGNORE)"
                )
                rows_written += len(grp_clean)
            finally:
                try:
                    con.unregister("_dualwrite_tmp")
                except Exception:
                    pass

            # Delete-replace, not append-alongside: drop any older fragment
            # this write fully supersedes so re-scores cannot layer a second
            # generation of z_scores on top of the first (Meridian #275).
            _prune_superseded_dualwrite(
                con,
                int(yr),
                min_date,
                max_date,
                set(_batch_factors),
                key,
            )

    except Exception as exc:
        logger.warning(
            "append_factor_scores_partition: dual-write failed after %d rows: %s: %s",
            rows_written,
            type(exc).__name__,
            str(exc)[:400],
        )

    return rows_written


def distinct_factor_tickers(start: date, end: date) -> list[str]:
    """Return the sorted list of distinct tickers in the factor feature table.

    Queries ``factors/daily_scores`` in the meridian-parquet bucket over the
    inclusive date range [start, end] for non-null z_score rows. Intended for
    the Phase 2 sharded dataloader to enumerate the scoring universe before
    splitting it into ticker shards.

    Returns [] when the meridian-parquet bucket is unreachable (credentials
    absent or S3 error). Propagates DuckDB exceptions only when the connection
    is established but the query itself fails.

    Parameters
    ----------
    start, end:
        Inclusive score_date range. Same semantics as fetch_factor_scores_chunks.

    Returns
    -------
    list[str]
        Sorted distinct ticker symbols present in [start, end].
    """
    # TODO(M14.3): add an Iceberg-first branch here (lakehouse.read_daily_scores()
    # + polars .select("ticker").unique()) once a scan-cost-conscious approach is
    # worked out - a naive full-column-scan-then-unique is not obviously cheaper
    # than the DuckDB path for a wide date range. Left on flat parquet for now.
    if not _use_meridian_s3():
        logger.warning(
            "distinct_factor_tickers: meridian-parquet credentials not set - returning []"
        )
        return []

    glob = _factor_glob()
    year_clause = _year_range_clause(start, end)
    sql = f"""
        SELECT DISTINCT ticker
        FROM read_parquet('{glob}', hive_partitioning=true, union_by_name=true)
        WHERE score_date >= '{start}'
          AND score_date <= '{end}'
          {year_clause}
          AND z_score IS NOT NULL
        ORDER BY ticker
    """
    try:
        rows = _meridian_con().execute(sql).fetchall()
        return [r[0] for r in rows]
    except Exception as exc:
        logger.warning("distinct_factor_tickers query failed: %s: %s", type(exc).__name__, exc)
        return []


def mirrored_factor_names(start: date, end: date) -> set[str]:
    """Return the distinct factor_name values present in the mirror for [start, end].

    Runs a single DuckDB query against the hive-partitioned factor glob with
    date pushdown so only the relevant year partitions are read. Year-partition
    pruning additionally limits which Parquet files are opened.

    Intended as a completeness-check primitive: callers compare the returned set
    against the full FACTOR_GROUPS catalogue to decide whether to trust the
    mirror or fall back to Postgres.

        from meridian_pipeline.factor_definitions import FACTOR_GROUPS
        catalogue = {f for grp in FACTOR_GROUPS.values() for f in grp}
        present = mirrored_factor_names(start, end)
        missing = catalogue - present  # factors not yet in the mirror

    Guards with meridian_parquet_available(); returns an empty set immediately
    when credentials are absent or the bucket is unreachable (no exception
    propagated to callers).

    Parameters
    ----------
    start, end:
        Inclusive score_date range. Same semantics as fetch_factor_scores_chunks.

    Returns
    -------
    set[str]
        Distinct factor names present in the mirror for the given date window.
        Empty set when the mirror is unreachable or contains no matching rows.
    """
    # TODO(M14.3): add an Iceberg-first branch here (lakehouse.read_daily_scores()
    # + polars .select("factor_name").unique()) once a scan-cost-conscious approach
    # is worked out - same reasoning as distinct_factor_tickers() above. Left on
    # flat parquet for now.
    if not meridian_parquet_available():
        logger.debug("mirrored_factor_names: meridian-parquet unavailable - returning empty set")
        return set()

    glob = _factor_glob()
    year_clause = _year_range_clause(start, end)
    sql = f"""
        SELECT DISTINCT factor_name
        FROM read_parquet('{glob}', hive_partitioning=true, union_by_name=true)
        WHERE score_date >= '{start}'
          AND score_date <= '{end}'
          {year_clause}
          AND z_score IS NOT NULL
        ORDER BY factor_name
    """
    try:
        rows = _meridian_con().execute(sql).fetchall()
        return {r[0] for r in rows}
    except Exception as exc:
        logger.warning("mirrored_factor_names query failed: %s: %s", type(exc).__name__, exc)
        return set()
