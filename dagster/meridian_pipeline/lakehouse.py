"""
lakehouse.py
------------
Authenticated pyiceberg access to Meridian's own Iceberg warehouse (`meridian`)
in the shared Lakekeeper catalog. See docs/17-LAKEHOUSE.md.

Why this is separate from parquet_io.py: this module owns Meridian's own
warehouse (reads and writes) via pyiceberg. parquet_io reads
LDP data through the DuckDB iceberg extension; that path IS authenticated now
(an ICEBERG TOKEN secret carrying the m2m bearer - Meridian#266; the earlier
"DuckDB ATTACH cannot send a bearer token" assumption was outdated and had it
silently falling back to flat Parquet after the catalog turned on Authentik OIDC
2026-06-12). parquet_io reuses `mint_lakekeeper_bearer()` below for that token.

Two-layer auth (mirrors LDP `iceberg_resource.py`):
  * catalog access  -> Authentik OIDC bearer (LAKEKEEPER_M2M_CLIENT_ID/SECRET,
    exchanged at LAKEKEEPER_OAUTH2_SERVER_URI with scope=openid - MANDATORY).
  * S3 data I/O     -> direct MinIO keys (MERIDIAN_MINIO_*); remote-signing is
    disabled on the warehouse so clients read/write MinIO themselves.

The `meridian` catalog runs authz-backend=allow-all today, so the read-only
boundary on `ldp` is enforced by the MinIO credential, not the catalog. This
module never writes `ldp`.

Everything is best-effort + lazy: pyiceberg is imported inside functions so the
module imports fine in the slim CI image, and `lakehouse_available()` lets
callers degrade gracefully when creds are absent or the catalog is unreachable.
"""

from __future__ import annotations

import logging
import os
import random
import time
from datetime import date, datetime, timezone
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Endpoint defaults to the in-cluster REST catalog; bare boxes override to
# https://lakehouse.example.com/catalog (they cannot resolve cluster DNS).
#
# Meridian#299 considered changing this default (either to the external URL,
# resolvable from both sides, or to no default at all so a missing config is a
# loud error). Both were rejected:
#   * external default: k8s_config.py documents that in-cluster step pods carry
#     NO explicit MERIDIAN_LAKEHOUSE_ENDPOINT override today - "endpoints
#     default to the correct in-cluster URLs ... so no extra env is needed".
#     Flipping the default would silently route every in-cluster write through
#     the public ingress instead of the direct in-cluster Service, adding a
#     hairpin network hop + external-DNS/cert dependency to the hot path with
#     no test coverage of that route, for zero benefit to in-cluster callers.
#   * required, no default: dagster_k8s strips literal `env` entries from step
#     configs (see iceberg_dual_write_enabled()'s docstring below), so there is
#     no mechanism for an in-cluster step pod to supply this var at all even if
#     the operator wanted to - the default IS the only value in-cluster pods
#     can ever see. Requiring it with no default would hard-break every
#     in-cluster dual-write on the next deploy.
# So the default stays in-cluster, unchanged, and the loud-failure requirement
# is met a different way: assert_dual_write_ready() (below) now preflights
# actual catalog REACHABILITY as well as credential presence, so an off-cluster
# run with the wrong (in-cluster) endpoint dies at job start with a clear
# "set MERIDIAN_LAKEHOUSE_ENDPOINT" error instead of silently skipping every
# partition's mirror write for hours (the Meridian#299 incident).
MERIDIAN_LAKEHOUSE_ENDPOINT: str = os.environ.get(
    "MERIDIAN_LAKEHOUSE_ENDPOINT",
    "http://catalog.example.internal:8181/catalog",
)
MERIDIAN_WAREHOUSE: str = os.environ.get("MERIDIAN_WAREHOUSE", "meridian")

# OIDC (catalog bearer)
# Shared lakekeeper-m2m client (same one LDP + Trino use); k8s secret
# dagster-secrets. scope=openid is required - an empty scope yields an opaque
# token Lakekeeper's JWKS validation rejects.
LAKEKEEPER_M2M_CLIENT_ID: str = os.environ.get("LAKEKEEPER_M2M_CLIENT_ID", "")
LAKEKEEPER_M2M_CLIENT_SECRET: str = os.environ.get("LAKEKEEPER_M2M_CLIENT_SECRET", "")
LAKEKEEPER_OAUTH2_SERVER_URI: str = os.environ.get(
    "LAKEKEEPER_OAUTH2_SERVER_URI", "https://auth.example.com/application/o/token/"
)
LAKEKEEPER_OAUTH2_SCOPE: str = os.environ.get("LAKEKEEPER_OAUTH2_SCOPE", "openid")

# S3 data I/O (MinIO)
# Reuses the MERIDIAN_MINIO_* keys parquet_io already defines. The warehouse was
# registered with region "local"; s3.endpoint needs a scheme so we build it from
# the host + USE_SSL flag.
MERIDIAN_MINIO_ENDPOINT: str = os.environ.get("MERIDIAN_MINIO_ENDPOINT", "s3.example.com")
MERIDIAN_MINIO_ACCESS_KEY: str = os.environ.get("MERIDIAN_MINIO_ACCESS_KEY", "")
MERIDIAN_MINIO_SECRET_KEY: str = os.environ.get("MERIDIAN_MINIO_SECRET_KEY", "")
MERIDIAN_MINIO_USE_SSL: bool = os.environ.get("MERIDIAN_MINIO_USE_SSL", "true").lower() == "true"
MERIDIAN_MINIO_REGION: str = os.environ.get("MERIDIAN_MINIO_REGION", "local")

FACTORS_NAMESPACE: str = "factors"


def _s3_endpoint_url() -> str:
    scheme = "https" if MERIDIAN_MINIO_USE_SSL else "http"
    return f"{scheme}://{MERIDIAN_MINIO_ENDPOINT}"


def _scrub_secrets(text: str) -> str:
    """Redact catalog + S3 credentials from an error string before it is logged.

    pyiceberg can echo the `credential="<id>:<secret>"` or S3 keys into an
    exception message; scrub at the source so no caller has to remember to.
    """
    for secret in (
        LAKEKEEPER_M2M_CLIENT_SECRET,
        MERIDIAN_MINIO_SECRET_KEY,
        MERIDIAN_MINIO_ACCESS_KEY,
    ):
        if secret:
            text = text.replace(secret, "***REDACTED***")
    return text


def _creds_present() -> bool:
    return bool(
        LAKEKEEPER_M2M_CLIENT_ID
        and LAKEKEEPER_M2M_CLIENT_SECRET
        and MERIDIAN_MINIO_ACCESS_KEY
        and MERIDIAN_MINIO_SECRET_KEY
    )


def _catalog_config(warehouse: str) -> dict[str, str]:
    return {
        "uri": MERIDIAN_LAKEHOUSE_ENDPOINT,
        "warehouse": warehouse,
        "credential": f"{LAKEKEEPER_M2M_CLIENT_ID}:{LAKEKEEPER_M2M_CLIENT_SECRET}",
        "oauth2-server-uri": LAKEKEEPER_OAUTH2_SERVER_URI,
        "scope": LAKEKEEPER_OAUTH2_SCOPE,
        "py-io-impl": "pyiceberg.io.pyarrow.PyArrowFileIO",
        "s3.endpoint": _s3_endpoint_url(),
        "s3.access-key-id": MERIDIAN_MINIO_ACCESS_KEY,
        "s3.secret-access-key": MERIDIAN_MINIO_SECRET_KEY,
        "s3.region": MERIDIAN_MINIO_REGION,
        "s3.path-style-access": "true",
    }


# Process-cached catalogs (the OIDC token exchange + REST round-trip is not free).
_catalog_cache: dict[str, Any] = {}


def _catalog(warehouse: str):
    if warehouse in _catalog_cache:
        return _catalog_cache[warehouse]
    if not _creds_present():
        raise RuntimeError(
            "lakehouse credentials absent: set LAKEKEEPER_M2M_CLIENT_ID/SECRET and "
            "MERIDIAN_MINIO_ACCESS_KEY/SECRET_KEY"
        )
    from pyiceberg.catalog.rest import RestCatalog

    try:
        cat = RestCatalog(warehouse, **_catalog_config(warehouse))
    except Exception as exc:  # scrub before it reaches logs
        exc.args = (_scrub_secrets(str(exc)),)
        raise
    _catalog_cache[warehouse] = cat
    return cat


def meridian_catalog():
    """Authenticated pyiceberg RestCatalog for the `meridian` warehouse (r/w)."""
    return _catalog(MERIDIAN_WAREHOUSE)


def ldp_catalog():
    """Authenticated pyiceberg RestCatalog for the `ldp` warehouse (READ-ONLY).

    An OIDC-authenticated pyiceberg reader for `ldp.*`. (parquet_io's DuckDB path
    is now ALSO authenticated via an ICEBERG TOKEN secret - Meridian#266 - so this
    is a convenience/fallback reader, not the only authenticated path.) Read-only.
    """
    return _catalog("ldp")


def mint_lakekeeper_bearer(timeout: int = 20) -> str:
    """Mint an Authentik m2m bearer for the Lakekeeper catalog via the
    client-credentials exchange (scope=openid mandatory). Same exchange pyiceberg
    does internally, but pre-fetched so a caller can hand the raw token to
    DuckDB's `CREATE SECRET (TYPE ICEBERG, TOKEN ...)` - which DOES authenticate
    (Meridian#266; the old "DuckDB can't send a bearer" assumption was outdated).
    Passing DuckDB the OAUTH2_SERVER_URI flow instead 405s against Authentik, so
    always pre-fetch here.
    """
    import json as _json
    import urllib.parse
    import urllib.request

    data = urllib.parse.urlencode(
        {
            "grant_type": "client_credentials",
            "client_id": LAKEKEEPER_M2M_CLIENT_ID,
            "client_secret": LAKEKEEPER_M2M_CLIENT_SECRET,
            "scope": LAKEKEEPER_OAUTH2_SCOPE,
        }
    ).encode()
    req = urllib.request.Request(
        LAKEKEEPER_OAUTH2_SERVER_URI,
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    return _json.load(urllib.request.urlopen(req, timeout=timeout))["access_token"]


def read_ldp_table(table: str, row_filter=None, columns=None, limit=None):
    """Authenticated pyiceberg read of an `ldp.*` medallion table -> pyarrow.Table.

    The medallion equivalent of parquet_io's flat/DuckDB reads, but over the
    OIDC-authenticated catalog (the DuckDB ATTACH cannot send a bearer token, so
    it silently falls back to the flat mirror - Meridian#266). `table` is the
    `ldp` namespace.table, e.g. 'fundamentals.income_statement' or 'ohlcv.prices'.
    `row_filter` is a pyiceberg predicate string (pushed down to the scan);
    `columns` selects fields. Returns Arrow so the caller can register it into
    DuckDB and reuse its existing SELECT/cast SQL unchanged.
    """
    tbl = ldp_catalog().load_table(table)
    kwargs = {}
    if row_filter:
        kwargs["row_filter"] = row_filter
    if columns:
        kwargs["selected_fields"] = tuple(columns)
    if limit is not None:
        kwargs["limit"] = limit
    return tbl.scan(**kwargs).to_arrow()


def ldp_available() -> bool:
    """True if creds present and the `ldp` medallion catalog is reachable."""
    if not _creds_present():
        return False
    try:
        ldp_catalog().list_namespaces()
        return True
    except Exception as exc:
        logger.warning(
            "ldp medallion unavailable: %s",
            _scrub_secrets(f"{type(exc).__name__}: {exc}")[:300],
        )
        return False


_available_cache: bool | None = None


def lakehouse_available() -> bool:
    """True if creds are present and the meridian catalog is reachable. Cached."""
    global _available_cache
    if _available_cache is not None:
        return _available_cache
    if not _creds_present():
        _available_cache = False
        return False
    try:
        meridian_catalog().list_namespaces()
        _available_cache = True
    except Exception as exc:
        logger.warning(
            "lakehouse unavailable: %s", _scrub_secrets(f"{type(exc).__name__}: {exc}")[:300]
        )
        _available_cache = False
    return _available_cache


def _log_replace_scope(op: str, table: str, **scope: Any) -> None:
    """INFO preview of a destructive Iceberg overwrite filter, before it commits
    (QoL 12): table, date span, factor count and the expected row bound."""
    logger.info(
        "%s: about to overwrite table=%s %s",
        op,
        table,
        " ".join(f"{k}={v}" for k, v in scope.items()),
    )


# ══════════════════════════════════════════════════════════════════════════════
# factors.daily_scores  (Iceberg lab mirror of the PG gold table)
# ══════════════════════════════════════════════════════════════════════════════
# meridian.factors.daily_scores mirrors the Postgres factors.daily_scores table,
# which remains gold-serving-primary and authoritative (see
# docs/18-ICEBERG_MIGRATION.md, the medallion reframe). The write path
# (db.upsert_daily_scores) dual-writes here behind MERIDIAN_ICEBERG_DUAL_WRITE,
# which is ON by default since 2026-07-09: step pods can't reliably receive a
# literal env var (dagster_k8s excludes `env` from step configs - see
# k8s_config.py), so the gold -> lab mirror is enabled in the image; set
# MERIDIAN_ICEBERG_DUAL_WRITE=false to disable.
# This remains the live writer until separately approved checked-mirror
# cutover. The stopped checked sensor/sweep do not replace it on deployment.
# Meridian's own bulk/research reads in parquet_io.py prefer this Iceberg mirror behind
# MERIDIAN_ICEBERG_READ_PRIMARY (a research-substrate switch only, default
# OFF). factor_store.py (the API) is gated by the separate
# iceberg_read_primary_api() / MERIDIAN_ICEBERG_API_READ flag and stays
# dormant by design - see docs/18 section 6, "Superseded decisions".

DAILY_SCORES_TABLE: str = f"{FACTORS_NAMESPACE}.daily_scores"
GROUP_SCORES_TABLE: str = f"{FACTORS_NAMESPACE}.group_scores"


def group_scores_schema():
    """Schema for an explicitly approved group mirror creation (no auto-create)."""
    from pyiceberg.schema import Schema
    from pyiceberg.types import DateType, DoubleType, NestedField, StringType, TimestamptzType

    return Schema(
        NestedField(1, "score_date", DateType(), required=False),
        NestedField(2, "ticker", StringType(), required=False),
        NestedField(3, "group_name", StringType(), required=False),
        NestedField(4, "group_score", DoubleType(), required=False),
        NestedField(5, "computed_at", TimestamptzType(), required=False),
    )


def group_scores_partition_spec():
    from pyiceberg.partitioning import PartitionField, PartitionSpec
    from pyiceberg.transforms import MonthTransform

    return PartitionSpec(
        PartitionField(
            source_id=1, field_id=1000, transform=MonthTransform(), name="score_date_month"
        )
    )


_DAILY_SCORES_COLUMNS: tuple[str, ...] = (
    "score_date",
    "ticker",
    "factor_name",
    "group_name",
    "raw_value",
    "z_score",
    "direction",
    "computed_at",
)


def _flag(name: str, default: bool = False) -> bool:
    """Read a boolean rollout flag from the env at call time (so an operator's
    pod-env change or a test monkeypatch takes effect without a re-import)."""
    return os.environ.get(name, "true" if default else "false").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def iceberg_dual_write_enabled() -> bool:
    """Gate for db.upsert_daily_scores mirroring each batch into Iceberg (write).

    Mirrors each production factors.daily_scores batch into the Iceberg lab
    (the gold -> lab mirror). ON by default (2026-07-09) because step pods
    can't reliably receive a literal env var - dagster_k8s excludes `env`
    from step configs, see k8s_config.py - so the mirror is enabled in the
    image rather than via a cluster env var. Set
    MERIDIAN_ICEBERG_DUAL_WRITE=false to disable. It is best-effort
    regardless and can never affect the PG gold write.
    """
    return _flag("MERIDIAN_ICEBERG_DUAL_WRITE", default=True)


def assert_dual_write_ready(context: str) -> None:
    """Loud, up-front preflight for batch/monitored callers that intend an
    Iceberg dual-write (e.g. scripts/burst/promote_backfill.py).

    db.upsert_daily_scores() treats a missing-credential OR unreachable-catalog
    dual-write as best-effort and logs-and-continues per call - correct for an
    in-cluster Dagster step pod (creds are always present there via the
    `dagster-secrets` k8s secret, and the in-cluster catalog endpoint is
    reachable by construction) but wrong for an off-cluster batch script: a
    1000+ partition promote would log the same warning 1000+ times and the
    meridian.factors Iceberg mirror would silently drift out of sync with PG
    gold for the entire promoted range, with no loud signal anywhere in the
    run. This is exactly what happened in Meridian#299: two promoted ranges
    landed in gold with no
    Iceberg mirror because the run had valid credentials but the catalog
    endpoint was still the in-cluster default, unreachable from off-cluster.

    Call this ONCE, before the first partition is written (see
    scripts/burst/promote_backfill.py) - not per-partition or per-write. It is
    a real network round-trip (an OIDC token exchange + a REST list_namespaces
    call via lakehouse_available()), and the result is process-cached, so
    calling it again in the same process is free but still correct.

    Distinguishes three states:
      1. dual-write disabled (MERIDIAN_ICEBERG_DUAL_WRITE=false, e.g.
         --no-iceberg) -> not configured at all; this is a legitimate,
         intended PG-only run. Returns immediately, no network call.
      2. dual-write enabled, credentials present, catalog reachable -> the
         intended path. Returns after one successful probe.
      3. dual-write enabled but credentials are absent, OR credentials are
         present but the catalog is unreachable or rejects auth -> configured
         to dual-write and unable to. Raises RuntimeError (once, before any
         writes happen) so the run fails in seconds, not after hours of
         half-mirrored work. Set MERIDIAN_ICEBERG_OPTIONAL=1 to downgrade
         either failure mode to a single warning log line for a planned
         PG-only run (the caller is then responsible for mirroring the range
         afterward, e.g. via scripts/load_daily_scores_iceberg.py).
    """
    if not iceberg_dual_write_enabled():
        return  # state 1: not configured at all - legitimate PG-only context

    if not _creds_present():
        msg = (
            f"{context}: MERIDIAN_ICEBERG_DUAL_WRITE defaults to true but lakehouse "
            "credentials are absent - LAKEKEEPER_M2M_CLIENT_ID/LAKEKEEPER_M2M_CLIENT_SECRET "
            "and/or MERIDIAN_MINIO_ACCESS_KEY/MERIDIAN_MINIO_SECRET_KEY are unset. "
            "LAKEKEEPER_M2M_CLIENT_ID/SECRET live in the infra repo's .env, "
            "not this repo's .env - source it before running off-cluster. "
            "Without it every partition this "
            "run writes will silently skip the Iceberg mirror, drifting "
            "meridian.factors.daily_scores out of sync with gold for the whole range. "
            "Either fix the environment, or explicitly accept a PG-only run (pass "
            "--no-iceberg / set MERIDIAN_ICEBERG_OPTIONAL=1) and mirror the range "
            "afterward with scripts/load_daily_scores_iceberg.py --replace."
        )
        if _flag("MERIDIAN_ICEBERG_OPTIONAL", default=False):
            logger.warning(msg)
            return
        raise RuntimeError(msg)

    if lakehouse_available():
        return  # state 2: configured and reachable

    # state 3: credentials are present but the catalog is unreachable or
    # rejected auth (bad/unresolvable endpoint, network block, expired
    # secret). This is the Meridian#299 gap: credentials alone are NOT
    # sufficient off-cluster. lakehouse_available() already logged the
    # underlying exception at WARNING just above this call.
    msg = (
        f"{context}: MERIDIAN_ICEBERG_DUAL_WRITE defaults to true and lakehouse "
        "credentials are present, but the meridian Iceberg catalog at "
        f"{MERIDIAN_LAKEHOUSE_ENDPOINT!r} is unreachable or rejected authentication "
        "(see the warning logged just above this error for the underlying exception). "
        "Setting LAKEKEEPER_M2M_CLIENT_ID/SECRET and MERIDIAN_MINIO_ACCESS_KEY/SECRET_KEY "
        "is NOT sufficient off-cluster: MERIDIAN_LAKEHOUSE_ENDPOINT must also be "
        "overridden, e.g. export MERIDIAN_LAKEHOUSE_ENDPOINT=https://lakehouse.example.com/catalog "
        "- the in-cluster default cannot resolve from a workstation or burst box. "
        "See docs/17-LAKEHOUSE.md. Without this fix every partition this run writes "
        "will silently skip the Iceberg mirror, drifting meridian.factors.daily_scores "
        "out of sync with gold for the whole range (the Meridian#299 incident). Either "
        "fix the environment, or explicitly accept a PG-only run (pass --no-iceberg / "
        "set MERIDIAN_ICEBERG_OPTIONAL=1) and mirror the range afterward with "
        "scripts/load_daily_scores_iceberg.py --replace."
    )
    if _flag("MERIDIAN_ICEBERG_OPTIONAL", default=False):
        logger.warning(msg)
        return
    raise RuntimeError(msg)


def iceberg_read_primary() -> bool:
    """Gate for Meridian's own bulk/research reads (parquet_io: panels, evidence,
    IC, weights, grading) preferring the Iceberg lab over the flat parquet
    mirror. This is a RESEARCH-SUBSTRATE switch only - it must never gate the
    serving API. See docs/18-ICEBERG_MIGRATION.md section 3/6 (the API-primary
    direction was explored and explicitly rejected: a large point-lookup
    regression). The API's read path is gated by the separate, structurally
    independent iceberg_read_primary_api() below, so flipping this flag on a
    research/backfill deployment can never accidentally serve Iceberg-first
    from the API."""
    return _flag("MERIDIAN_ICEBERG_READ_PRIMARY")


def iceberg_read_primary_api() -> bool:
    """Gate for factor_store.py's (API) Iceberg-first/Postgres-fallback shim.

    A SEPARATE env var from MERIDIAN_ICEBERG_READ_PRIMARY (which
    only controls Meridian's own research reads in parquet_io.py) so that
    enabling the research substrate can never, even by accident, route the
    serving API to Iceberg. Default OFF.

    docs/18-ICEBERG_MIGRATION.md section 6 records that API-primary Iceberg
    serving was explored and REJECTED (a large point-lookup regression vs PG;
    downstream tenants have no OIDC path to the catalog). Do not set
    MERIDIAN_ICEBERG_API_READ in any deployment without a fresh decision
    superseding that rejection - docs/06-CRITICAL_RULES.md Absolute Rule 1 requires the API
    stay read-only against PostgreSQL.
    """
    return _flag("MERIDIAN_ICEBERG_API_READ")


def _daily_scores_schema():
    from pyiceberg.schema import Schema
    from pyiceberg.types import (
        DateType,
        DoubleType,
        IntegerType,
        NestedField,
        StringType,
        TimestamptzType,
    )

    # Mirrors Postgres factors.daily_scores exactly. direction is smallint in PG
    # (values -1/0/1); Iceberg's narrowest integer is 32-bit. computed_at is
    # timestamptz in PG -> TimestamptzType (tz-aware) for round-trip fidelity.
    return Schema(
        NestedField(1, "score_date", DateType(), required=False),
        NestedField(2, "ticker", StringType(), required=False),
        NestedField(3, "factor_name", StringType(), required=False),
        NestedField(4, "group_name", StringType(), required=False),
        NestedField(5, "raw_value", DoubleType(), required=False),
        NestedField(6, "z_score", DoubleType(), required=False),
        NestedField(7, "direction", IntegerType(), required=False),
        NestedField(8, "computed_at", TimestamptzType(), required=False),
    )


def _daily_scores_partition_spec():
    from pyiceberg.partitioning import PartitionField, PartitionSpec
    from pyiceberg.transforms import IdentityTransform, MonthTransform

    # identity(factor_name) + month(score_date): prunes the two dominant query
    # shapes (single-date all-factor reads; date-range single-factor panels) and
    # aligns the per-(date, factor) overwrite the write path issues.
    return PartitionSpec(
        PartitionField(
            source_id=3, field_id=1000, transform=IdentityTransform(), name="factor_name"
        ),
        PartitionField(
            source_id=1, field_id=1001, transform=MonthTransform(), name="score_date_month"
        ),
    )


def ensure_daily_scores_table():
    """Idempotently create the `factors` namespace + the daily_scores table."""
    cat = meridian_catalog()
    try:
        cat.create_namespace_if_not_exists(FACTORS_NAMESPACE)
    except Exception as exc:
        exc.args = (_scrub_secrets(str(exc)),)
        raise
    return cat.create_table_if_not_exists(
        DAILY_SCORES_TABLE,
        schema=_daily_scores_schema(),
        partition_spec=_daily_scores_partition_spec(),
    )


def _to_daily_scores_arrow(df):
    """Coerce a polars/pandas DataFrame (or pyarrow Table) to the daily_scores
    Arrow schema. Non-finite raw_value/z_score are nulled to match the PG write
    path's NaN/inf sanitisation (so parity checksums line up); computed_at is
    filled with now() when absent (PG sets it server-side via NOW())."""
    import pyarrow as pa
    import pyarrow.compute as pc

    if hasattr(df, "to_arrow"):  # polars
        tbl = df.to_arrow()
    elif isinstance(df, pa.Table):
        tbl = df
    else:  # pandas
        tbl = pa.Table.from_pandas(df, preserve_index=False)

    cols = {name: tbl.column(name) for name in tbl.column_names}
    n = tbl.num_rows
    now = datetime.now(timezone.utc)

    missing = [c for c in ("score_date", "ticker", "factor_name", "group_name") if c not in cols]
    if missing:
        raise ValueError(f"daily_scores missing required columns: {missing}")
    if "raw_value" not in cols:
        cols["raw_value"] = pa.nulls(n, pa.float64())
    if "z_score" not in cols:
        cols["z_score"] = pa.nulls(n, pa.float64())
    if "direction" not in cols:
        cols["direction"] = pa.nulls(n, pa.int32())
    if "computed_at" not in cols:
        cols["computed_at"] = pa.array([now] * n, pa.timestamp("us", tz="UTC"))

    out_schema = pa.schema(
        [
            ("score_date", pa.date32()),
            ("ticker", pa.string()),
            ("factor_name", pa.string()),
            ("group_name", pa.string()),
            ("raw_value", pa.float64()),
            ("z_score", pa.float64()),
            ("direction", pa.int32()),
            ("computed_at", pa.timestamp("us", tz="UTC")),
        ]
    )

    def _finite(arr):
        a = pa.array(arr).cast(pa.float64())
        return pc.if_else(pc.is_finite(a), a, pa.scalar(None, pa.float64()))

    arrays = []
    for name in _DAILY_SCORES_COLUMNS:
        src = cols[name]
        if name in ("raw_value", "z_score"):
            arrays.append(_finite(src))
        else:
            arrays.append(pa.array(src).cast(out_schema.field(name).type))
    return pa.table(arrays, schema=out_schema)


# Concurrent daily assets (cpu_heavy pool = 4) each mirror their own factor
# group for the same score_date, so parallel commits to the one daily_scores
# table are ROUTINE. Iceberg commits are optimistic table-level CAS, and
# pyiceberg's REST catalog retries only auth errors - CommitFailedException is
# NOT retried upstream. The groups' delete filters are disjoint, so a conflict
# loser succeeds cleanly on refresh + retry; without this, the loser's mirror
# slice would silently go missing (db.py swallows the dual-write error).
_COMMIT_ATTEMPTS = 5


def _commit_with_retry(tbl, op):
    """Run an Iceberg commit callable, refreshing + retrying through
    optimistic-concurrency conflicts (CommitFailedException) with jittered
    backoff. Re-raises the final failure; all other exceptions pass through."""
    from pyiceberg.exceptions import CommitFailedException

    for attempt in range(_COMMIT_ATTEMPTS):
        try:
            return op()
        except CommitFailedException:
            if attempt == _COMMIT_ATTEMPTS - 1:
                raise
            time.sleep(random.uniform(0.2, 0.6) * (attempt + 1))
            tbl.refresh()


def write_daily_scores(
    df,
    replace: bool = True,
    owned_factors: "list[str] | None" = None,
    replace_date_range=None,
) -> int:
    """Mirror a factors.daily_scores batch into meridian.factors.daily_scores.

    Matches db.upsert_daily_scores's delete-replace semantics exactly, per
    score_date present in `df` (a batch may span several dates on a backfill):
      * replace=True + owned_factors -> replace only those factors' rows for the
        date (factor-scoped rerun);
      * replace=True, no owned_factors -> replace the (date, group_name) pairs
        present in df (group-scoped rerun);
      * replace=False -> pure append.
    Each per-date overwrite is a single Iceberg snapshot; commits retry through
    concurrent-writer conflicts (see _commit_with_retry). Returns the new
    snapshot_id (-1 if the table has no snapshot, e.g. empty df).
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    from pyiceberg.expressions import And, EqualTo, GreaterThanOrEqual, In, LessThanOrEqual

    if replace_date_range is not None:
        lo, hi = replace_date_range
        if not replace or not owned_factors or lo > hi:
            raise ValueError("range replacement requires an ordered interval and owned factors")

    arrow = _to_daily_scores_arrow(df)
    if replace_date_range is not None and arrow.num_rows:
        valid = pc.and_(
            pc.and_(
                pc.greater_equal(arrow["score_date"], lo), pc.less_equal(arrow["score_date"], hi)
            ),
            pc.is_in(arrow["factor_name"], value_set=pa.array(list(owned_factors))),
        )
        if not pc.all(pc.fill_null(valid, False)).as_py():
            raise ValueError("payload falls outside declared replacement scope")
    tbl = ensure_daily_scores_table()
    if arrow.num_rows == 0 and replace_date_range is None:
        tbl.refresh()
        snap = tbl.current_snapshot()
        return snap.snapshot_id if snap is not None else -1

    try:
        if replace_date_range is not None:
            _log_replace_scope(
                "write_daily_scores(replace_date_range)",
                "factors.daily_scores",
                date_min=lo,
                date_max=hi,
                span_days=(hi - lo).days + 1,
                owned_factors=len(owned_factors),
                payload_rows=arrow.num_rows,
                key_space_days_x_factors=((hi - lo).days + 1) * len(owned_factors),
            )
            del_filter = And(
                And(GreaterThanOrEqual("score_date", lo), LessThanOrEqual("score_date", hi)),
                In("factor_name", list(owned_factors)),
            )
            _commit_with_retry(tbl, lambda: tbl.overwrite(arrow, overwrite_filter=del_filter))
        elif not replace:
            _commit_with_retry(tbl, lambda: tbl.append(arrow))
        else:
            if owned_factors:
                days = pc.unique(arrow.column("score_date")).to_pylist()
                _log_replace_scope(
                    "write_daily_scores(owned_factors)",
                    "factors.daily_scores",
                    date_min=min(days),
                    date_max=max(days),
                    payload_dates=len(days),
                    owned_factors=len(owned_factors),
                    payload_rows=arrow.num_rows,
                    key_space_days_x_factors=len(days) * len(owned_factors),
                )
            for d in pc.unique(arrow.column("score_date")).to_pylist():
                day = arrow.filter(pc.equal(arrow.column("score_date"), d))
                if owned_factors:
                    del_filter = And(
                        EqualTo("score_date", d), In("factor_name", list(owned_factors))
                    )
                else:
                    groups = pc.unique(day.column("group_name")).to_pylist()
                    del_filter = And(EqualTo("score_date", d), In("group_name", groups))
                # overwrite = delete-matching + append, in one commit
                _commit_with_retry(
                    tbl, lambda day=day, f=del_filter: tbl.overwrite(day, overwrite_filter=f)
                )
    except Exception as exc:
        exc.args = (_scrub_secrets(str(exc)),)
        raise
    tbl.refresh()
    snap = tbl.current_snapshot()
    return snap.snapshot_id if snap is not None else -1


def month_spans(start: date, end: date) -> list[tuple[date, date]]:
    """Split [start, end] into one (lo, hi) pair per calendar month, clipped.

    Canonical home for this helper (Meridian #420 recurrence fix). It used to
    live only in scripts/bulk_replace_daily_scores_iceberg.py; that script now
    re-exports this name so its existing CLI and tests are unaffected, and
    reconcile_daily_scores_months (below) shares the same implementation the
    scheduled reconcile asset uses.

    The spans are contiguous, non-overlapping, cover [start, end] exactly, and
    every interior boundary lands on a calendar month edge - the point being
    that the Iceberg delete filter built from each span aligns with
    DAILY_SCORES_TABLE's MonthTransform(score_date) partition field, so a
    commit drops a whole partition instead of copy-on-writing surviving rows.
    Far cheaper per unit of data than a per-date overwrite (see
    the script's module docstring for the original benchmark).
    """
    from datetime import timedelta

    if end < start:
        raise ValueError(f"end ({end}) before start ({start})")
    out: list[tuple[date, date]] = []
    cur = start.replace(day=1)
    while cur <= end:
        nxt = (cur + timedelta(days=32)).replace(day=1)
        lo = max(cur, start)
        hi = min(nxt - timedelta(days=1), end)
        out.append((lo, hi))
        cur = nxt
    return out


def _commit_month_with_rebuild(tbl, build_payload):
    """Commit a whole-month overwrite, rebuilding the payload from scratch on
    every attempt including retries (Meridian#420 follow-up).

    `_commit_with_retry` (above) replays the SAME closure - and therefore the
    SAME Arrow payload - on every attempt after a `tbl.refresh()`. Its own
    docstring is explicit that this is safe only when concurrent writers'
    delete filters are DISJOINT from this call's: true for
    `write_daily_scores`' per-date, per-factor (or per-date, per-group)
    filters, NOT true for `replace_daily_scores_months`' whole-month,
    every-ticker filter. If a routine per-date dual-write lands on a date
    inside this month between our PG read and our commit, a
    `CommitFailedException` retry that replays the STALE payload would
    delete that concurrent write on the retried commit and put back exactly
    what PG held BEFORE it landed - silent, mirror-only data loss (PG stays
    correct; the next scheduled reconcile run would eventually re-heal it,
    but silently, which defeats the point of a control that is supposed to
    make staleness observable).

    `build_payload()` is called fresh before every attempt (including the
    first) and must return `(arrow, overwrite_filter)` built from a CURRENT
    read of PostgreSQL, or `None` if there is nothing to write (e.g. the
    month is empty and the caller opted to skip empty months - re-checked on
    every attempt too, in case PG's state for this range changed between
    reads). Returns the Arrow table actually committed, or `None` if
    `build_payload` ever returns `None`.
    """
    from pyiceberg.exceptions import CommitFailedException

    for attempt in range(_COMMIT_ATTEMPTS):
        built = build_payload()
        if built is None:
            return None
        arrow, del_filter = built
        try:
            tbl.overwrite(arrow, overwrite_filter=del_filter)
            return arrow
        except CommitFailedException:
            if attempt == _COMMIT_ATTEMPTS - 1:
                raise
            time.sleep(random.uniform(0.2, 0.6) * (attempt + 1))
            tbl.refresh()


def replace_daily_scores_months(
    engine,
    start: date,
    end: date,
    skip_empty: bool = True,
    on_month: "Callable[[date, date, int], None] | None" = None,
) -> dict[str, Any]:
    """Whole-month-partition Iceberg mirror refresh over [start, end] (#420
    recurrence fix - the scheduled control, not the one-off repair).

    Per calendar month in the range: read everything PostgreSQL currently
    holds for that month and overwrite the matching Iceberg month partition
    wholesale. This is the ONLY mechanism that removes an orphan row a
    correctness fix deleted from PG through a path that never called
    db.upsert_daily_scores / db.copy_replace_daily_scores - a direct SQL fix,
    an ops migration, or a historical rerun script commits straight to
    PostgreSQL and never reaches `_dual_write_daily_scores`, so there is no
    delete signal for the Iceberg mirror to consume no matter how the
    dual-write path itself is written. Meridian#420 found orphaned rows
    this way, with the
    retraction comment on that issue establishing the general shape: the
    mirror is stale wherever a partition has not been whole-partition-replaced
    since the last PG delete that touched it, which is not a one-time
    transition artefact but a condition every future correctness fix
    reintroduces. Running this on a schedule over a rolling window (see
    mirror_reconcile_asset.py) is what makes replacement a standing,
    self-healing control instead of something that only happens when someone
    repairs an incident by hand.

    Returns {"months_written": <n>, "rows": <total rows written>,
    "spans": [(lo, hi), ...]} for the caller to log or attach as run
    metadata. A month with zero PG rows is left untouched when
    skip_empty=True (default) - deleting its Iceberg partition with nothing
    to replace it would turn a genuinely empty month into data loss on the
    mirror, mirroring the ad-hoc script's --skip-empty default.

    `on_month`, if given, is called `(lo, hi, rows_written)` once per span in
    the range, immediately after that month's outcome is known (rows_written
    is 0 for a skipped-empty month). This function is the ONE place that
    calls `month_spans` for an actual (non-dry-run) replacement - callers
    that want per-month progress visibility (e.g.
    `scripts/bulk_replace_daily_scores_iceberg.py`'s log lines) use this
    callback instead of re-splitting the range themselves and calling this
    function once per month, which would compute `month_spans` twice for the
    same range from two layers that could silently drift apart.
    """
    import pandas as pd
    from pyiceberg.expressions import And, GreaterThanOrEqual, LessThanOrEqual
    from sqlalchemy import text

    spans = month_spans(start, end)
    tbl = ensure_daily_scores_table()
    sql = text(
        "SELECT score_date, ticker, factor_name, group_name, raw_value, "
        "z_score, direction, computed_at FROM factors.daily_scores "
        "WHERE score_date >= :lo AND score_date <= :hi"
    )

    total_rows = 0
    months_written = 0
    for lo, hi in spans:

        def _build(lo=lo, hi=hi):
            # Read PG FRESH on every call, including retries after a
            # commit conflict - see _commit_month_with_rebuild's docstring
            # for why replaying a payload read before the conflict is unsafe
            # for this whole-month, every-ticker filter.
            d = pd.read_sql(sql, engine, params={"lo": lo, "hi": hi})
            if d.empty and skip_empty:
                return None
            a = _to_daily_scores_arrow(d)
            # Raw date objects, not .isoformat() strings - matches
            # write_daily_scores' replace_date_range filter above (both bind
            # to the same pyiceberg DateLiteral), so one convention is used
            # for every date-range filter in this module instead of two that
            # happen to coerce the same way today.
            f = And(
                GreaterThanOrEqual("score_date", lo),
                LessThanOrEqual("score_date", hi),
            )
            return a, f

        arrow = _commit_month_with_rebuild(tbl, _build)
        if arrow is None:
            if on_month is not None:
                on_month(lo, hi, 0)
            continue
        total_rows += arrow.num_rows
        months_written += 1
        if on_month is not None:
            on_month(lo, hi, arrow.num_rows)

    return {"months_written": months_written, "rows": total_rows, "spans": spans}


def read_daily_scores(
    score_date=None,
    start=None,
    end=None,
    tickers: "list[str] | None" = None,
    factor_names: "list[str] | None" = None,
    columns: "list[str] | None" = None,
    as_pandas: bool = False,
):
    """Read meridian.factors.daily_scores as polars (or pandas with as_pandas).

    Filters compose: `score_date` (single day), `start`/`end` (inclusive range),
    `tickers`, `factor_names`. Callers keep a Postgres fallback for latency-
    sensitive point lookups (see M14.4 / factor_store)."""
    from functools import reduce

    import polars as pl
    from pyiceberg.expressions import (
        AlwaysTrue,
        And,
        EqualTo,
        GreaterThanOrEqual,
        In,
        LessThanOrEqual,
    )

    conds = []
    if score_date is not None:
        conds.append(EqualTo("score_date", score_date))
    if start is not None:
        conds.append(GreaterThanOrEqual("score_date", start))
    if end is not None:
        conds.append(LessThanOrEqual("score_date", end))
    if tickers:
        conds.append(In("ticker", list(tickers)))
    if factor_names:
        conds.append(In("factor_name", list(factor_names)))
    row_filter = reduce(And, conds) if conds else AlwaysTrue()

    tbl = meridian_catalog().load_table(DAILY_SCORES_TABLE)
    arrow = tbl.scan(
        row_filter=row_filter,
        selected_fields=tuple(columns) if columns else ("*",),
    ).to_arrow()
    return arrow.to_pandas() if as_pandas else pl.from_arrow(arrow)


_ds_available_cache: bool | None = None


def daily_scores_available() -> bool:
    """True if creds are present and meridian.factors.daily_scores is loadable.
    Cached per process; independent of the rollout flags."""
    global _ds_available_cache
    if _ds_available_cache is not None:
        return _ds_available_cache
    if not _creds_present():
        _ds_available_cache = False
        return False
    try:
        meridian_catalog().load_table(DAILY_SCORES_TABLE)
        _ds_available_cache = True
    except Exception as exc:
        logger.warning(
            "daily_scores iceberg unavailable: %s",
            _scrub_secrets(f"{type(exc).__name__}: {exc}")[:200],
        )
        _ds_available_cache = False
    return _ds_available_cache


def daily_scores_parity_stats(score_date) -> dict[str, Any]:
    """Cheap per-date comparator for PG-vs-Iceberg parity (M14.2/M14.6).

    Returns {rows, factors, tickers, z_checksum, raw_checksum}. Checksums sum the
    finite values only (nulls skipped) - the PG side of the monitor must apply the
    same finite filter so NaN/inf rows don't diverge the two sums. Float summation
    order can differ, so compare checksums with a tolerance, not for equality."""
    import pyarrow.compute as pc
    from pyiceberg.expressions import EqualTo

    tbl = meridian_catalog().load_table(DAILY_SCORES_TABLE)
    arrow = tbl.scan(
        row_filter=EqualTo("score_date", score_date),
        selected_fields=("factor_name", "ticker", "z_score", "raw_value"),
    ).to_arrow()

    def _sum(col: str) -> float:
        v = pc.sum(arrow.column(col)).as_py()
        return float(v) if v is not None else 0.0

    return {
        "rows": arrow.num_rows,
        "factors": len(pc.unique(arrow.column("factor_name"))) if arrow.num_rows else 0,
        "tickers": len(pc.unique(arrow.column("ticker"))) if arrow.num_rows else 0,
        "z_checksum": round(_sum("z_score"), 6),
        "raw_checksum": round(_sum("raw_value"), 6),
    }


def daily_scores_max_date(floor_date) -> date | None:
    """Bounded MAX(score_date) over the Iceberg mirror (Meridian #415).

    Scans only rows with score_date >= floor_date - DAILY_SCORES_TABLE is
    partitioned by score_date (_daily_scores_partition_spec), so this is a
    partition-pruned scan, not a full-table one, provided the caller passes a
    reasonably recent floor. This exists to catch the failure shape a
    per-date row-count comparison over a fixed trailing window can miss: a
    mirror that is perfectly consistent for every date it already holds and
    simply has not received the newest one(s) - see
    lakehouse_parity.staleness_check, the caller.

    Returns None if the scanned window has no rows at all (empty mirror or
    floor_date set past the mirror's actual data).
    """
    import pyarrow.compute as pc
    from pyiceberg.expressions import GreaterThanOrEqual

    tbl = meridian_catalog().load_table(DAILY_SCORES_TABLE)
    arrow = tbl.scan(
        row_filter=GreaterThanOrEqual("score_date", floor_date),
        selected_fields=("score_date",),
    ).to_arrow()
    if arrow.num_rows == 0:
        return None
    return pc.max(arrow.column("score_date")).as_py()


# ══════════════════════════════════════════════════════════════════════════════
# factors.gold_promotion_scores  (Iceberg gold authoring surface -> PG promotion)
# ══════════════════════════════════════════════════════════════════════════════
# meridian.factors.gold_promotion_scores is the ONE-WAY Iceberg-gold -> PG
# promotion staging table for Meridian#265. Large recompute / backfill /
# research batches are computed on the cheap columnar Iceberg substrate and
# their FINALIZED output is written here; the iceberg_gold_sync_sensor then
# promotes each new snapshot into PostgreSQL factors.daily_scores (the
# authoritative, point-lookup-fast serving gold - see
# docs/18-ICEBERG_MIGRATION.md section 6, decision ACCEPTED 2026-07-17).
#
# This is a separate table from meridian.factors.daily_scores
# (the best-effort mirror the daily pipeline dual-writes on every partition).
# If the sensor watched the mirror it would fire a PG sync on every routine
# daily-pipeline commit - a pointless round-trip that re-writes PG from a
# mirror of PG. A dedicated promotion table means a snapshot appears ONLY when
# a batch/recompute/research job writes finalized rows to promote,
# so a snapshot-id change is an unambiguous "promote this" signal.
#
# Direction of truth is unchanged: PG is authoritative-once-written. Promotion
# is a SECOND write path (batch) alongside the daily pipeline's direct PG
# write; both are scoped by owned_factors so neither clobbers the other's
# factors (db.upsert_daily_scores owned_factors delete-replace). PG is never a
# fallback DERIVED from Iceberg for serving; the API is never pointed at
# Iceberg (docs/06-CRITICAL_RULES.md Absolute Rule 1).

GOLD_PROMOTION_TABLE: str = f"{FACTORS_NAMESPACE}.gold_promotion_scores"

# Snapshot-summary property keys carrying the promotion SCOPE. The sync asset
# reads these back off the target snapshot to know exactly which factors and
# date range to delete-replace in PG - the scope is stamped on the snapshot at
# write time so it travels with the data, not out-of-band.
GOLD_SCOPE_OWNED_FACTORS: str = "meridian.owned_factors"  # CSV of factor names
GOLD_SCOPE_DATE_MIN: str = "meridian.score_date_min"  # ISO date (inclusive)
GOLD_SCOPE_DATE_MAX: str = "meridian.score_date_max"  # ISO date (inclusive)
GOLD_SCOPE_SOURCE: str = "meridian.promotion_source"  # free-text provenance tag


def ensure_gold_promotion_table():
    """Idempotently create the `factors` namespace + the gold_promotion_scores
    table. Same schema + partitioning as the daily_scores mirror (it holds the
    identical 8-column factors.daily_scores shape), so the two tables stay
    column-compatible and a promoted batch round-trips through PG unchanged."""
    cat = meridian_catalog()
    try:
        cat.create_namespace_if_not_exists(FACTORS_NAMESPACE)
    except Exception as exc:
        exc.args = (_scrub_secrets(str(exc)),)
        raise
    return cat.create_table_if_not_exists(
        GOLD_PROMOTION_TABLE,
        schema=_daily_scores_schema(),
        partition_spec=_daily_scores_partition_spec(),
    )


def _scope_properties(owned_factors, score_date_min, score_date_max, source):
    """Build the snapshot_properties dict stamping the promotion scope."""
    props = {
        GOLD_SCOPE_OWNED_FACTORS: ",".join(owned_factors),
        GOLD_SCOPE_DATE_MIN: str(score_date_min),
        GOLD_SCOPE_DATE_MAX: str(score_date_max),
    }
    if source:
        props[GOLD_SCOPE_SOURCE] = str(source)
    return props


def write_gold_scores(
    df,
    owned_factors: "list[str]",
    replace: bool = True,
    source: "str | None" = None,
) -> int:
    """Write a FINALIZED batch of factor scores to the Iceberg gold promotion
    table (meridian.factors.gold_promotion_scores) and stamp the promotion
    SCOPE onto the resulting snapshot's summary properties.

    `df` is a polars/pandas DataFrame (or pyarrow Table) carrying the
    factors.daily_scores columns (score_date, ticker, factor_name, group_name,
    raw_value, z_score, direction; computed_at is filled when absent).

    `owned_factors` is the EXCLUSIVE factor-name set this batch owns. It is
    stamped as a CSV snapshot property (GOLD_SCOPE_OWNED_FACTORS) alongside the
    batch's score_date min/max (GOLD_SCOPE_DATE_MIN/MAX) so the downstream sync
    asset (gold_scores_pg_sync) can scope its PG delete-replace to exactly these
    factors and dates - it never touches a factor it does not own.

    replace=True (default) delete-replaces, per score_date in the batch, the
    rows for owned_factors before appending - so a partial-failure re-run of the
    same batch never double-counts and a shrunk universe leaves no orphan rows
    (mirrors db.upsert_daily_scores / write_daily_scores semantics). replace=False
    is a pure append. Each call produces exactly one new snapshot carrying the
    scope. Returns the new snapshot_id (-1 for an empty df / no snapshot).

    Callers MUST pass a non-empty owned_factors even though every row also
    carries its own factor_name: the scope is the AUTHORITY on what the sync
    promotes, decoupled from whichever rows happen to be present in this df.
    """
    import pyarrow.compute as pc
    from pyiceberg.expressions import And, GreaterThanOrEqual, In, LessThanOrEqual

    if not owned_factors:
        raise ValueError("write_gold_scores: owned_factors must be a non-empty list")
    owned_factors = list(dict.fromkeys(owned_factors))  # de-dup, preserve order

    tbl = ensure_gold_promotion_table()
    arrow = _to_daily_scores_arrow(df)
    if arrow.num_rows == 0:
        tbl.refresh()
        snap = tbl.current_snapshot()
        return snap.snapshot_id if snap is not None else -1

    dates = pc.unique(arrow.column("score_date")).to_pylist()
    score_date_min = min(dates)
    score_date_max = max(dates)
    props = _scope_properties(owned_factors, score_date_min, score_date_max, source)

    try:
        if not replace:
            _commit_with_retry(tbl, lambda: tbl.append(arrow, snapshot_properties=props))
        else:
            # One delete-replace commit spanning the whole batch: delete the
            # owned factors across the batch's date range, then append. Scoped
            # to (score_date range, owned_factors) so a re-run is idempotent and
            # a factor whose entire group is absent from this df is still
            # cleared (owned_factors is the exclusive authority, exactly the PG
            # side's owned_factors delete contract).
            del_filter = And(
                And(
                    GreaterThanOrEqual("score_date", score_date_min),
                    LessThanOrEqual("score_date", score_date_max),
                ),
                In("factor_name", owned_factors),
            )
            _commit_with_retry(
                tbl,
                lambda: tbl.overwrite(
                    arrow, overwrite_filter=del_filter, snapshot_properties=props
                ),
            )
    except Exception as exc:
        exc.args = (_scrub_secrets(str(exc)),)
        raise
    tbl.refresh()
    snap = tbl.current_snapshot()
    return snap.snapshot_id if snap is not None else -1


def _read_snapshot_scope(snap) -> dict[str, Any]:
    """Extract the promotion scope stamped on a snapshot's summary properties.

    Returns {owned_factors: list[str], score_date_min, score_date_max, source,
    snapshot_id}. owned_factors is parsed from the CSV property; empty/missing
    yields an empty list (the sync asset treats that as 'nothing to promote')."""
    summary = getattr(snap, "summary", None)

    def _get(key: str):
        if summary is None:
            return None
        # pyiceberg Summary supports .get(); fall back to additional_properties.
        try:
            val = summary.get(key)
        except Exception:
            val = None
        if val is None:
            val = getattr(summary, "additional_properties", {}).get(key)
        return val

    raw_owned = _get(GOLD_SCOPE_OWNED_FACTORS) or ""
    owned = [f for f in (s.strip() for s in raw_owned.split(",")) if f]
    return {
        "snapshot_id": snap.snapshot_id,
        "owned_factors": owned,
        "score_date_min": _get(GOLD_SCOPE_DATE_MIN),
        "score_date_max": _get(GOLD_SCOPE_DATE_MAX),
        "source": _get(GOLD_SCOPE_SOURCE),
    }


def read_gold_snapshot(snapshot_id: "int | None" = None, as_pandas: bool = True):
    """Read a gold promotion snapshot's rows + its promotion scope.

    snapshot_id=None reads the CURRENT snapshot (the latest promotion). The rows
    are the table state AT that snapshot, filtered by the snapshot's own scope
    (owned_factors + score_date_min/max) so the returned frame is exactly the
    batch that snapshot promoted - even if later snapshots added other factors.

    Returns (df, scope) where scope is the dict from _read_snapshot_scope. df is
    a pandas DataFrame by default (ready for db.upsert_daily_scores); pass
    as_pandas=False for polars. Returns (empty df, scope-with-empty-owned) when
    the table has no snapshot or the target snapshot carries no owned_factors
    scope (nothing to promote)."""
    from functools import reduce

    import polars as pl
    from pyiceberg.expressions import And, GreaterThanOrEqual, In, LessThanOrEqual

    tbl = meridian_catalog().load_table(GOLD_PROMOTION_TABLE)
    if snapshot_id is None:
        snap = tbl.current_snapshot()
    else:
        snap = tbl.snapshot_by_id(snapshot_id)
    if snap is None:
        empty = pl.DataFrame(schema={c: pl.Utf8 for c in _DAILY_SCORES_COLUMNS})
        empty_scope = {"snapshot_id": snapshot_id, "owned_factors": []}
        return (empty.to_pandas() if as_pandas else empty), empty_scope

    scope = _read_snapshot_scope(snap)
    owned = scope["owned_factors"]
    if not owned:
        empty = pl.DataFrame(schema={c: pl.Utf8 for c in _DAILY_SCORES_COLUMNS})
        return (empty.to_pandas() if as_pandas else empty), scope

    conds = [In("factor_name", owned)]
    if scope.get("score_date_min"):
        conds.append(GreaterThanOrEqual("score_date", scope["score_date_min"]))
    if scope.get("score_date_max"):
        conds.append(LessThanOrEqual("score_date", scope["score_date_max"]))
    row_filter = reduce(And, conds)

    arrow = tbl.scan(row_filter=row_filter, snapshot_id=snap.snapshot_id).to_arrow()
    df = pl.from_arrow(arrow)
    return (df.to_pandas() if as_pandas else df), scope


def gold_promotion_current_snapshot_id() -> "int | None":
    """Current snapshot id of the gold promotion table, or None if the table is
    unreachable / has no snapshot yet. Cheap poll for iceberg_gold_sync_sensor;
    never raises (returns None on any error so a transient catalog blip just
    yields a no-op sensor tick)."""
    if not _creds_present():
        return None
    try:
        tbl = meridian_catalog().load_table(GOLD_PROMOTION_TABLE)
        snap = tbl.current_snapshot()
        return snap.snapshot_id if snap is not None else None
    except Exception as exc:
        logger.warning(
            "gold_promotion_current_snapshot_id: %s",
            _scrub_secrets(f"{type(exc).__name__}: {exc}")[:200],
        )
        return None


_gold_available_cache: bool | None = None


def gold_promotion_available() -> bool:
    """True if creds are present and the gold_promotion_scores table is loadable.
    Cached per process. The table not existing yet (no recompute has written it)
    is an expected pre-cutover state, so the sync asset SKIPS rather than fails
    when this is False."""
    global _gold_available_cache
    if _gold_available_cache is not None:
        return _gold_available_cache
    if not _creds_present():
        _gold_available_cache = False
        return False
    try:
        meridian_catalog().load_table(GOLD_PROMOTION_TABLE)
        _gold_available_cache = True
    except Exception as exc:
        logger.warning(
            "gold_promotion table unavailable: %s",
            _scrub_secrets(f"{type(exc).__name__}: {exc}")[:200],
        )
        _gold_available_cache = False
    return _gold_available_cache
