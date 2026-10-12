"""
Tests for lakehouse.py - the authenticated pyiceberg access layer to Meridian's
`meridian` Iceberg warehouse.

The config/scrub/coercion tests are dependency-light and run everywhere. The
schema/partition tests need pyiceberg and skip on the slim CI image. A live
round-trip test runs only when real creds are present + the catalog is reachable
(never in CI).
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pytest

_DAGSTER_DIR = Path(__file__).resolve().parents[1] / "dagster"
if str(_DAGSTER_DIR) not in sys.path:
    sys.path.insert(0, str(_DAGSTER_DIR))

from meridian_pipeline import lakehouse  # noqa: E402


def test_oauth_scope_openid_default():
    """scope=openid is mandatory (an empty scope yields an opaque token that
    Lakekeeper's JWKS validation rejects)."""
    assert lakehouse.LAKEKEEPER_OAUTH2_SCOPE == "openid"
    assert lakehouse.LAKEKEEPER_OAUTH2_SERVER_URI.startswith("https://auth.")


def test_s3_endpoint_url_builds_scheme(monkeypatch):
    monkeypatch.setattr(lakehouse, "MERIDIAN_MINIO_ENDPOINT", "minio.internal:9500")
    monkeypatch.setattr(lakehouse, "MERIDIAN_MINIO_USE_SSL", False)
    assert lakehouse._s3_endpoint_url() == "http://minio.internal:9500"
    monkeypatch.setattr(lakehouse, "MERIDIAN_MINIO_USE_SSL", True)
    assert lakehouse._s3_endpoint_url() == "https://minio.internal:9500"


def test_scrub_secrets_redacts(monkeypatch):
    monkeypatch.setattr(lakehouse, "LAKEKEEPER_M2M_CLIENT_SECRET", "topsecretclient")
    monkeypatch.setattr(lakehouse, "MERIDIAN_MINIO_SECRET_KEY", "minio-secret-xyz")
    msg = "auth failed for credential=id:topsecretclient using minio-secret-xyz"
    scrubbed = lakehouse._scrub_secrets(msg)
    assert "topsecretclient" not in scrubbed
    assert "minio-secret-xyz" not in scrubbed
    assert "***REDACTED***" in scrubbed


def test_catalog_config_carries_oidc_and_s3(monkeypatch):
    """The catalog config must carry both auth layers: OIDC credential + s3 keys."""
    monkeypatch.setattr(lakehouse, "LAKEKEEPER_M2M_CLIENT_ID", "cid")
    monkeypatch.setattr(lakehouse, "LAKEKEEPER_M2M_CLIENT_SECRET", "csec")
    monkeypatch.setattr(lakehouse, "MERIDIAN_MINIO_ACCESS_KEY", "ak")
    monkeypatch.setattr(lakehouse, "MERIDIAN_MINIO_SECRET_KEY", "sk")
    cfg = lakehouse._catalog_config("meridian")
    assert cfg["warehouse"] == "meridian"
    assert cfg["credential"] == "cid:csec"
    assert cfg["scope"] == "openid"
    assert cfg["s3.access-key-id"] == "ak"
    assert cfg["s3.path-style-access"] == "true"


def test_creds_present_and_unavailable_without_creds(monkeypatch):
    """No creds -> _creds_present False and lakehouse_available False WITHOUT any
    network call (must not raise)."""
    for name in (
        "LAKEKEEPER_M2M_CLIENT_ID",
        "LAKEKEEPER_M2M_CLIENT_SECRET",
        "MERIDIAN_MINIO_ACCESS_KEY",
        "MERIDIAN_MINIO_SECRET_KEY",
    ):
        monkeypatch.setattr(lakehouse, name, "")
    monkeypatch.setattr(lakehouse, "_available_cache", None)
    assert lakehouse._creds_present() is False
    assert lakehouse.lakehouse_available() is False


def test_catalog_raises_clear_error_without_creds(monkeypatch):
    monkeypatch.setattr(lakehouse, "LAKEKEEPER_M2M_CLIENT_ID", "")
    monkeypatch.setattr(lakehouse, "_catalog_cache", {})
    with pytest.raises(RuntimeError, match="lakehouse credentials absent"):
        lakehouse.meridian_catalog()


def test_assert_dual_write_ready_raises_when_creds_absent(monkeypatch):
    """Batch preflight must fail loudly (not log-and-continue) when dual-write
    is enabled but lakehouse creds are absent - the #275 promote gap."""
    monkeypatch.setattr(lakehouse, "LAKEKEEPER_M2M_CLIENT_ID", "")
    monkeypatch.setattr(lakehouse, "LAKEKEEPER_M2M_CLIENT_SECRET", "")
    monkeypatch.setattr(lakehouse, "MERIDIAN_MINIO_ACCESS_KEY", "ak")
    monkeypatch.setattr(lakehouse, "MERIDIAN_MINIO_SECRET_KEY", "sk")
    monkeypatch.delenv("MERIDIAN_ICEBERG_DUAL_WRITE", raising=False)
    monkeypatch.delenv("MERIDIAN_ICEBERG_OPTIONAL", raising=False)
    with pytest.raises(RuntimeError, match="LAKEKEEPER_M2M_CLIENT_ID"):
        lakehouse.assert_dual_write_ready("test_context")


def test_assert_dual_write_ready_ok_when_creds_present_and_catalog_reachable(monkeypatch):
    """State 2 (configured and reachable): creds present + the catalog probe
    succeeds -> proceed with no error."""
    monkeypatch.setattr(lakehouse, "LAKEKEEPER_M2M_CLIENT_ID", "cid")
    monkeypatch.setattr(lakehouse, "LAKEKEEPER_M2M_CLIENT_SECRET", "csec")
    monkeypatch.setattr(lakehouse, "MERIDIAN_MINIO_ACCESS_KEY", "ak")
    monkeypatch.setattr(lakehouse, "MERIDIAN_MINIO_SECRET_KEY", "sk")
    monkeypatch.delenv("MERIDIAN_ICEBERG_DUAL_WRITE", raising=False)
    monkeypatch.setattr(lakehouse, "lakehouse_available", lambda: True)
    lakehouse.assert_dual_write_ready("test_context")  # must not raise


def test_assert_dual_write_ready_noop_when_dual_write_disabled(monkeypatch):
    """State 1 (not configured at all): MERIDIAN_ICEBERG_DUAL_WRITE=false (e.g.
    --no-iceberg) is a legitimate, deliberate PG-only run - proceed, no network
    probe at all (assert lakehouse_available is never even consulted)."""
    monkeypatch.setattr(lakehouse, "LAKEKEEPER_M2M_CLIENT_ID", "")
    monkeypatch.setattr(lakehouse, "LAKEKEEPER_M2M_CLIENT_SECRET", "")
    monkeypatch.setattr(lakehouse, "MERIDIAN_MINIO_ACCESS_KEY", "")
    monkeypatch.setattr(lakehouse, "MERIDIAN_MINIO_SECRET_KEY", "")
    monkeypatch.setenv("MERIDIAN_ICEBERG_DUAL_WRITE", "false")

    def _boom():
        raise AssertionError("lakehouse_available must not be called when dual-write is off")

    monkeypatch.setattr(lakehouse, "lakehouse_available", _boom)
    lakehouse.assert_dual_write_ready("test_context")  # must not raise


def test_assert_dual_write_ready_downgrades_to_warning_when_optional(monkeypatch):
    """MERIDIAN_ICEBERG_OPTIONAL=1 downgrades the creds-absent failure mode to
    a single warning, for an operator who explicitly accepts a PG-only run."""
    monkeypatch.setattr(lakehouse, "LAKEKEEPER_M2M_CLIENT_ID", "")
    monkeypatch.setattr(lakehouse, "LAKEKEEPER_M2M_CLIENT_SECRET", "")
    monkeypatch.setattr(lakehouse, "MERIDIAN_MINIO_ACCESS_KEY", "ak")
    monkeypatch.setattr(lakehouse, "MERIDIAN_MINIO_SECRET_KEY", "sk")
    monkeypatch.delenv("MERIDIAN_ICEBERG_DUAL_WRITE", raising=False)
    monkeypatch.setenv("MERIDIAN_ICEBERG_OPTIONAL", "1")
    lakehouse.assert_dual_write_ready("test_context")  # must not raise


def test_assert_dual_write_ready_raises_when_creds_present_but_catalog_unreachable(monkeypatch):
    """THE Meridian#299 regression test: credentials alone are not sufficient.
    State 3 (configured but unreachable) - creds present, but the catalog
    probe fails (e.g. the in-cluster endpoint default cannot resolve from a
    workstation) - must raise, not silently skip the mirror for the whole run.
    """
    monkeypatch.setattr(lakehouse, "LAKEKEEPER_M2M_CLIENT_ID", "cid")
    monkeypatch.setattr(lakehouse, "LAKEKEEPER_M2M_CLIENT_SECRET", "csec")
    monkeypatch.setattr(lakehouse, "MERIDIAN_MINIO_ACCESS_KEY", "ak")
    monkeypatch.setattr(lakehouse, "MERIDIAN_MINIO_SECRET_KEY", "sk")
    monkeypatch.delenv("MERIDIAN_ICEBERG_DUAL_WRITE", raising=False)
    monkeypatch.delenv("MERIDIAN_ICEBERG_OPTIONAL", raising=False)
    monkeypatch.setattr(lakehouse, "lakehouse_available", lambda: False)
    with pytest.raises(RuntimeError, match="unreachable or rejected authentication"):
        lakehouse.assert_dual_write_ready("test_context")


def test_assert_dual_write_ready_downgrades_to_warning_when_unreachable_and_optional(monkeypatch):
    """MERIDIAN_ICEBERG_OPTIONAL=1 also downgrades the unreachable-catalog
    failure mode (not just creds-absent) to a warning."""
    monkeypatch.setattr(lakehouse, "LAKEKEEPER_M2M_CLIENT_ID", "cid")
    monkeypatch.setattr(lakehouse, "LAKEKEEPER_M2M_CLIENT_SECRET", "csec")
    monkeypatch.setattr(lakehouse, "MERIDIAN_MINIO_ACCESS_KEY", "ak")
    monkeypatch.setattr(lakehouse, "MERIDIAN_MINIO_SECRET_KEY", "sk")
    monkeypatch.delenv("MERIDIAN_ICEBERG_DUAL_WRITE", raising=False)
    monkeypatch.setenv("MERIDIAN_ICEBERG_OPTIONAL", "1")
    monkeypatch.setattr(lakehouse, "lakehouse_available", lambda: False)
    lakehouse.assert_dual_write_ready("test_context")  # must not raise


def test_assert_dual_write_ready_probes_reachability_once_per_process(monkeypatch):
    """Preflight-once-at-job-start, not per-write: assert_dual_write_ready must
    reuse lakehouse_available()'s process cache rather than re-probing the
    catalog on every call (a 1000+ partition promote must not open 1000+ OIDC
    token exchanges)."""
    monkeypatch.setattr(lakehouse, "LAKEKEEPER_M2M_CLIENT_ID", "cid")
    monkeypatch.setattr(lakehouse, "LAKEKEEPER_M2M_CLIENT_SECRET", "csec")
    monkeypatch.setattr(lakehouse, "MERIDIAN_MINIO_ACCESS_KEY", "ak")
    monkeypatch.setattr(lakehouse, "MERIDIAN_MINIO_SECRET_KEY", "sk")
    monkeypatch.delenv("MERIDIAN_ICEBERG_DUAL_WRITE", raising=False)
    monkeypatch.setattr(lakehouse, "_available_cache", None)

    probe_calls = {"n": 0}

    class _FakeCatalog:
        def list_namespaces(self):
            probe_calls["n"] += 1
            return ["factors", "research"]

    monkeypatch.setattr(lakehouse, "meridian_catalog", _FakeCatalog)

    for _ in range(5):  # simulate the preflight plus several "per-write" calls
        lakehouse.assert_dual_write_ready("test_context")

    assert probe_calls["n"] == 1, "the catalog must be probed once, not per call"


def test_lakehouse_available_false_when_creds_present_but_catalog_raises(monkeypatch):
    """lakehouse_available() must distinguish 'no creds' from 'creds present
    but the catalog rejects the connection' - both are False, but only the
    latter is the Meridian#299 unreachable-endpoint failure mode that
    assert_dual_write_ready must raise on."""
    monkeypatch.setattr(lakehouse, "LAKEKEEPER_M2M_CLIENT_ID", "cid")
    monkeypatch.setattr(lakehouse, "LAKEKEEPER_M2M_CLIENT_SECRET", "csec")
    monkeypatch.setattr(lakehouse, "MERIDIAN_MINIO_ACCESS_KEY", "ak")
    monkeypatch.setattr(lakehouse, "MERIDIAN_MINIO_SECRET_KEY", "sk")
    monkeypatch.setattr(lakehouse, "_available_cache", None)

    class _FakeCatalog:
        def list_namespaces(self):
            raise TimeoutError("connect timed out")

    monkeypatch.setattr(lakehouse, "meridian_catalog", _FakeCatalog)
    assert lakehouse.lakehouse_available() is False


class _FakeCandidateTable:
    """Minimal stand-in for a pyiceberg Table, just enough to observe how
    write_candidate_factor_scores() calls append()/overwrite()."""

    def __init__(self):
        self.append_calls = []
        self.overwrite_calls = []  # list of (arrow_table, overwrite_filter)
        self._snapshot_id = 1

    def append(self, arrow):
        self.append_calls.append(arrow)
        self._snapshot_id += 1

    def overwrite(self, arrow, overwrite_filter=None):
        self.overwrite_calls.append((arrow, overwrite_filter))
        self._snapshot_id += 1

    def refresh(self):
        pass

    def current_snapshot(self):
        class _Snap:
            def __init__(self, sid):
                self.snapshot_id = sid

        return _Snap(self._snapshot_id)


# ── daily_scores (M14) ────────────────────────────────────────────────────────


def test_daily_scores_rollout_flags(monkeypatch):
    """DUAL_WRITE defaults ON (2026-07-09: step pods can't take a literal env
    var, so the gold -> lab mirror is enabled in the image) but is still
    overridable to off. Both read flags (research READ_PRIMARY + the separate
    API_READ) still default OFF."""
    monkeypatch.delenv("MERIDIAN_ICEBERG_DUAL_WRITE", raising=False)
    monkeypatch.delenv("MERIDIAN_ICEBERG_READ_PRIMARY", raising=False)
    monkeypatch.delenv("MERIDIAN_ICEBERG_API_READ", raising=False)
    assert lakehouse.iceberg_dual_write_enabled() is True
    assert lakehouse.iceberg_read_primary() is False
    assert lakehouse.iceberg_read_primary_api() is False
    monkeypatch.setenv("MERIDIAN_ICEBERG_DUAL_WRITE", "false")
    assert lakehouse.iceberg_dual_write_enabled() is False
    monkeypatch.delenv("MERIDIAN_ICEBERG_DUAL_WRITE", raising=False)
    for truthy in ("1", "true", "YES", "on"):
        monkeypatch.setenv("MERIDIAN_ICEBERG_DUAL_WRITE", truthy)
        assert lakehouse.iceberg_dual_write_enabled() is True
    monkeypatch.setenv("MERIDIAN_ICEBERG_READ_PRIMARY", "false")
    assert lakehouse.iceberg_read_primary() is False


def test_api_read_flag_is_independent_of_research_flag(monkeypatch):
    """MERIDIAN_ICEBERG_API_READ and MERIDIAN_ICEBERG_READ_PRIMARY are separate
    env vars gating separate readers (API vs. Meridian's own research reads in
    parquet_io.py) - flipping one must never flip the other's return value."""
    monkeypatch.setenv("MERIDIAN_ICEBERG_READ_PRIMARY", "true")
    monkeypatch.delenv("MERIDIAN_ICEBERG_API_READ", raising=False)
    assert lakehouse.iceberg_read_primary() is True
    assert lakehouse.iceberg_read_primary_api() is False

    monkeypatch.delenv("MERIDIAN_ICEBERG_READ_PRIMARY", raising=False)
    monkeypatch.setenv("MERIDIAN_ICEBERG_API_READ", "true")
    assert lakehouse.iceberg_read_primary() is False
    assert lakehouse.iceberg_read_primary_api() is True


def test_to_daily_scores_arrow_schema_and_sanitisation():
    """Coercion yields the exact 8-column daily_scores schema, fills computed_at,
    and nulls non-finite raw_value/z_score (matching the PG write sanitisation)."""
    pytest.importorskip("polars")

    import polars as pl

    df = pl.DataFrame(
        {
            "score_date": [dt.date(2026, 7, 2), dt.date(2026, 7, 2)],
            "ticker": ["AAPL", "MSFT"],
            "factor_name": ["pe_ratio", "pe_ratio"],
            "group_name": ["value", "value"],
            "raw_value": [1.5, float("inf")],
            "z_score": [0.9, float("nan")],
            "direction": [1, -1],
        }
    )
    arrow = lakehouse._to_daily_scores_arrow(df)
    assert arrow.column_names == list(lakehouse._DAILY_SCORES_COLUMNS)
    assert arrow.column("computed_at").to_pylist()[0] is not None
    # non-finite -> null
    assert arrow.column("raw_value").to_pylist() == [1.5, None]
    zs = arrow.column("z_score").to_pylist()
    assert zs[0] == 0.9 and zs[1] is None
    assert arrow.column("direction").to_pylist() == [1, -1]


def test_to_daily_scores_arrow_requires_core_columns():
    pytest.importorskip("polars")
    import polars as pl

    df = pl.DataFrame({"score_date": [dt.date(2026, 7, 2)], "ticker": ["AAPL"]})
    with pytest.raises(ValueError, match="missing required columns"):
        lakehouse._to_daily_scores_arrow(df)


def test_daily_scores_schema_and_partition_spec():
    """8-column PG-mirroring schema; partitioning identity(factor_name) +
    month(score_date). Skips without pyiceberg (slim CI image)."""
    pytest.importorskip("pyiceberg")
    schema = lakehouse._daily_scores_schema()
    assert [f.name for f in schema.fields] == list(lakehouse._DAILY_SCORES_COLUMNS)
    spec = lakehouse._daily_scores_partition_spec()
    assert {f.name for f in spec.fields} == {"factor_name", "score_date_month"}


def test_daily_scores_available_false_without_creds(monkeypatch):
    monkeypatch.setattr(lakehouse, "_ds_available_cache", None)
    monkeypatch.setattr(lakehouse, "LAKEKEEPER_M2M_CLIENT_ID", "")
    monkeypatch.setattr(lakehouse, "MERIDIAN_MINIO_ACCESS_KEY", "")
    assert lakehouse.daily_scores_available() is False


def test_commit_with_retry_survives_concurrent_conflicts(monkeypatch):
    """Concurrent daily assets commit to the one daily_scores table in parallel
    (disjoint factor groups, same date); pyiceberg does NOT retry
    CommitFailedException, so _commit_with_retry must: two conflicts then
    success -> op runs 3 times, table refreshed twice, no sleep in tests."""
    pytest.importorskip("pyiceberg")
    from pyiceberg.exceptions import CommitFailedException

    monkeypatch.setattr(lakehouse.time, "sleep", lambda s: None)

    class FakeTable:
        refreshes = 0

        def refresh(self):
            self.refreshes += 1

    tbl = FakeTable()
    calls = {"n": 0}

    def op():
        calls["n"] += 1
        if calls["n"] < 3:
            raise CommitFailedException("concurrent update")
        return "snap"

    assert lakehouse._commit_with_retry(tbl, op) == "snap"
    assert calls["n"] == 3
    assert tbl.refreshes == 2


def test_commit_with_retry_exhausts_and_reraises(monkeypatch):
    pytest.importorskip("pyiceberg")
    from pyiceberg.exceptions import CommitFailedException

    monkeypatch.setattr(lakehouse.time, "sleep", lambda s: None)

    class FakeTable:
        def refresh(self):
            pass

    def always_conflict():
        raise CommitFailedException("concurrent update")

    with pytest.raises(CommitFailedException):
        lakehouse._commit_with_retry(FakeTable(), always_conflict)


# ─────────────────────────────────────────────
# month_spans / replace_daily_scores_months (#420 recurrence fix)
# ─────────────────────────────────────────────
#
# These moved here from scripts/bulk_replace_daily_scores_iceberg.py so the
# scheduled mirror_reconcile_asset.py and the ad-hoc repair script share one
# implementation. tests/test_bulk_replace_daily_scores_iceberg.py still
# covers month_spans' boundary contract via the script's re-export; the tests
# below cover the reusable Iceberg-writing half that script never had unit
# tests for (it always ran against a live table).


def test_month_spans_single_month_partial_range():
    lo, hi = dt.date(2024, 1, 15), dt.date(2024, 1, 20)
    assert lakehouse.month_spans(lo, hi) == [(lo, hi)]


def test_month_spans_splits_on_calendar_boundaries():
    spans = lakehouse.month_spans(dt.date(2024, 1, 15), dt.date(2024, 3, 10))
    assert spans == [
        (dt.date(2024, 1, 15), dt.date(2024, 1, 31)),
        (dt.date(2024, 2, 1), dt.date(2024, 2, 29)),  # 2024 is a leap year
        (dt.date(2024, 3, 1), dt.date(2024, 3, 10)),
    ]


def test_month_spans_rejects_reversed_range():
    with pytest.raises(ValueError):
        lakehouse.month_spans(dt.date(2024, 3, 1), dt.date(2024, 1, 1))


def test_replace_daily_scores_months_one_overwrite_per_nonempty_month(monkeypatch):
    """#420: the reconcile control's core contract. One whole-month Iceberg
    overwrite per calendar month that has PG rows, with a delete filter that
    lands on the REQUESTED range's own boundary (not necessarily the
    calendar-month boundary, when the range starts/ends mid-month) - so a
    row PostgreSQL no longer carries for that month (an orphan left behind by
    any write path that never touched the mirror) is removed by the
    overwrite's delete-then-replace semantics, not merely appended around."""
    pytest.importorskip("pyiceberg")
    pytest.importorskip("pandas")
    import pandas as pd
    from pyiceberg.expressions import And, GreaterThanOrEqual, LessThanOrEqual

    fake_tbl = _FakeCandidateTable()
    monkeypatch.setattr(lakehouse, "ensure_daily_scores_table", lambda: fake_tbl)

    def _fake_read_sql(sql, engine, params=None):
        lo = params["lo"]
        cols = [
            "score_date",
            "ticker",
            "factor_name",
            "group_name",
            "raw_value",
            "z_score",
            "direction",
            "computed_at",
        ]
        if lo.month == 1:
            # Stands in for PG's post-correctness-fix state: only ONE row
            # survives where the mirror previously held more (the orphan
            # case #420 measured).
            return pd.DataFrame(
                {
                    "score_date": [lo],
                    "ticker": ["ASML"],
                    "factor_name": ["earnings_yield"],
                    "group_name": ["value"],
                    "raw_value": [1.0],
                    "z_score": [0.5],
                    "direction": [1],
                    "computed_at": [dt.datetime(2026, 1, 1)],
                }
            )
        return pd.DataFrame(columns=cols)  # February: nothing in PG

    monkeypatch.setattr(pd, "read_sql", _fake_read_sql)

    result = lakehouse.replace_daily_scores_months(
        engine=object(), start=dt.date(2024, 1, 15), end=dt.date(2024, 2, 15)
    )

    assert result["months_written"] == 1
    assert result["rows"] == 1
    assert len(fake_tbl.overwrite_calls) == 1
    arrow, del_filter = fake_tbl.overwrite_calls[0]
    assert arrow.num_rows == 1
    assert del_filter == And(
        GreaterThanOrEqual("score_date", dt.date(2024, 1, 15)),
        LessThanOrEqual("score_date", dt.date(2024, 1, 31)),
    )


def test_replace_daily_scores_months_on_month_callback_fires_per_span(monkeypatch):
    """#420 follow-up: on_month lets a caller (the ad-hoc CLI script) get
    per-month progress without re-splitting the range itself and calling
    replace_daily_scores_months once per month - this function is the ONE
    place month_spans is called for an actual run."""
    pytest.importorskip("pyiceberg")
    pytest.importorskip("pandas")
    import pandas as pd

    fake_tbl = _FakeCandidateTable()
    monkeypatch.setattr(lakehouse, "ensure_daily_scores_table", lambda: fake_tbl)
    cols = [
        "score_date",
        "ticker",
        "factor_name",
        "group_name",
        "raw_value",
        "z_score",
        "direction",
        "computed_at",
    ]

    def _fake_read_sql(sql, engine, params=None):
        lo = params["lo"]
        if lo.month == 1:
            return pd.DataFrame(
                {
                    "score_date": [lo],
                    "ticker": ["AAPL"],
                    "factor_name": ["earnings_yield"],
                    "group_name": ["value"],
                    "raw_value": [1.0],
                    "z_score": [0.5],
                    "direction": [1],
                    "computed_at": [dt.datetime(2026, 1, 1)],
                }
            )
        return pd.DataFrame(columns=cols)  # February: empty

    monkeypatch.setattr(pd, "read_sql", _fake_read_sql)

    calls = []
    lakehouse.replace_daily_scores_months(
        engine=object(),
        start=dt.date(2024, 1, 15),
        end=dt.date(2024, 2, 15),
        on_month=lambda lo, hi, rows: calls.append((lo, hi, rows)),
    )

    assert calls == [
        (dt.date(2024, 1, 15), dt.date(2024, 1, 31), 1),
        (dt.date(2024, 2, 1), dt.date(2024, 2, 15), 0),  # skipped-empty -> 0
    ]


def test_replace_daily_scores_months_skips_empty_month_by_default(monkeypatch):
    """A month with zero PG rows must NOT delete the Iceberg partition when
    skip_empty=True (default): an empty read could mean a genuinely empty
    month, or a transient PG issue, and either way overwriting with an empty
    arrow table would wipe the mirror's existing data for nothing to replace
    it - the same footgun the ad-hoc script's --skip-empty guards against."""
    pytest.importorskip("pyiceberg")
    pytest.importorskip("pandas")
    import pandas as pd

    fake_tbl = _FakeCandidateTable()
    monkeypatch.setattr(lakehouse, "ensure_daily_scores_table", lambda: fake_tbl)
    cols = [
        "score_date",
        "ticker",
        "factor_name",
        "group_name",
        "raw_value",
        "z_score",
        "direction",
        "computed_at",
    ]
    monkeypatch.setattr(pd, "read_sql", lambda sql, engine, params=None: pd.DataFrame(columns=cols))

    result = lakehouse.replace_daily_scores_months(
        engine=object(), start=dt.date(2024, 1, 1), end=dt.date(2024, 1, 31)
    )

    assert result["months_written"] == 0
    assert result["rows"] == 0
    assert fake_tbl.overwrite_calls == []


def test_replace_daily_scores_months_wipes_empty_month_when_opted_in(monkeypatch):
    """skip_empty=False is the deliberate opposite: a genuinely-empty month
    IS replaced (wiped), matching the ad-hoc script's non-default
    behaviour."""
    pytest.importorskip("pyiceberg")
    pytest.importorskip("pandas")
    import pandas as pd

    fake_tbl = _FakeCandidateTable()
    monkeypatch.setattr(lakehouse, "ensure_daily_scores_table", lambda: fake_tbl)
    cols = [
        "score_date",
        "ticker",
        "factor_name",
        "group_name",
        "raw_value",
        "z_score",
        "direction",
        "computed_at",
    ]
    monkeypatch.setattr(pd, "read_sql", lambda sql, engine, params=None: pd.DataFrame(columns=cols))

    result = lakehouse.replace_daily_scores_months(
        engine=object(), start=dt.date(2024, 1, 1), end=dt.date(2024, 1, 31), skip_empty=False
    )

    assert result["months_written"] == 1
    assert result["rows"] == 0
    assert len(fake_tbl.overwrite_calls) == 1


def test_replace_daily_scores_months_rebuilds_payload_on_commit_conflict(monkeypatch):
    """Meridian#420 follow-up: replace_daily_scores_months must NOT replay a
    payload read before a CommitFailedException. A whole-month, every-ticker
    delete filter is not provably disjoint from a concurrent per-date
    dual-write the way write_daily_scores' filters are (see
    _commit_with_retry's docstring) - if a routine daily dual-write to a
    date inside this month lands between our PG read and our commit, retrying
    with the STALE payload would delete that write on the retried commit and
    put back exactly what PG held before it landed (silent mirror-only data
    loss). This proves the retried commit reflects a FRESH read of PG - the
    row count actually committed matches the SECOND read, not the first."""
    pytest.importorskip("pyiceberg")
    pytest.importorskip("pandas")
    import pandas as pd
    from pyiceberg.exceptions import CommitFailedException

    monkeypatch.setattr(lakehouse.time, "sleep", lambda s: None)

    class _ConflictOnceTable(_FakeCandidateTable):
        def __init__(self):
            super().__init__()
            self._raised = False

        def overwrite(self, arrow, overwrite_filter=None):
            if not self._raised:
                self._raised = True
                raise CommitFailedException("concurrent dual-write landed")
            super().overwrite(arrow, overwrite_filter=overwrite_filter)

    fake_tbl = _ConflictOnceTable()
    monkeypatch.setattr(lakehouse, "ensure_daily_scores_table", lambda: fake_tbl)

    reads = {"n": 0}

    def _fake_read_sql(sql, engine, params=None):
        reads["n"] += 1
        # First read (pre-conflict): 1 row. Second read (post-conflict,
        # standing in for the concurrent dual-write that landed in between):
        # 2 rows.
        n_rows = 1 if reads["n"] == 1 else 2
        return pd.DataFrame(
            {
                "score_date": [dt.date(2024, 1, 15)] * n_rows,
                "ticker": ["AAPL", "MSFT"][:n_rows],
                "factor_name": ["earnings_yield"] * n_rows,
                "group_name": ["value"] * n_rows,
                "raw_value": [1.0, 2.0][:n_rows],
                "z_score": [0.5, -0.5][:n_rows],
                "direction": [1, 1][:n_rows],
                "computed_at": [dt.datetime(2026, 1, 1)] * n_rows,
            }
        )

    monkeypatch.setattr(pd, "read_sql", _fake_read_sql)

    result = lakehouse.replace_daily_scores_months(
        engine=object(), start=dt.date(2024, 1, 15), end=dt.date(2024, 1, 15)
    )

    assert reads["n"] == 2  # one pre-conflict read, one rebuilt post-conflict read
    assert result["rows"] == 2  # the SECOND read's row count, not the stale first
    assert len(fake_tbl.overwrite_calls) == 1  # the raising attempt is not recorded
