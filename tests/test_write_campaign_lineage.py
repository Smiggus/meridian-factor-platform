"""Write-campaign lineage (Meridian #294 / #295).

Two things are being protected here, and they pull in opposite directions:

  1. Lineage must actually be recorded - a write that lands in gold with no
     campaign is the exact failure that forces a full-history recompute.
  2. Lineage must NEVER change a factor value, and must NEVER fail a write.
     It is additive bookkeeping bolted onto the hot path for every one of
     a very large table.

The parity tests are the half of (2) that matters most: the same input frame is
written with and without a campaign, and the score payload the database
receives must be byte-identical. If a future change makes a lineage column
perturb a value, these fail.
"""

import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from tests._stage_copy import record_stage_copy

_DAGSTER_DIR = Path(__file__).resolve().parents[1] / "dagster"
if str(_DAGSTER_DIR) not in sys.path:
    sys.path.insert(0, str(_DAGSTER_DIR))

os.environ.setdefault("PARTITION_START_DATE", "2010-01-02")
os.environ.setdefault("PGPASS", "test-pg-pass")


def _sqlalchemy_complete() -> bool:
    try:
        import sqlalchemy  # noqa: F401
    except ImportError:
        return False
    return True


if not _sqlalchemy_complete():  # slim CI image
    pytest.skip("SQLAlchemy not installed (slim CI test image)", allow_module_level=True)


OWNED = ["earnings_yield", "book_to_market"]

# _df() carries earnings_yield only, so OWNED is a superset of the payload - the
# canonical-superset writer shape. These tests are about lineage stamping, not
# delete scope, so they opt out of the #378 delete-without-replace check the way
# value_scores and friends do.
_OWNED_OPTOUT = {"allow_unwritten_owned_factors": True}


@pytest.fixture(autouse=True)
def _clean_lineage_state():
    """Every test starts with no ambient/implicit campaign bound."""
    from meridian_pipeline import db, lineage

    lineage.reset_for_tests()
    db._campaign_column_cache.clear()
    yield
    lineage.reset_for_tests()
    db._campaign_column_cache.clear()


def _df(n_rows: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "score_date": [date(2026, 5, 1)] * n_rows,
            "ticker": [f"T{i}" for i in range(n_rows)],
            "factor_name": ["earnings_yield"] * n_rows,
            "group_name": ["value"] * n_rows,
            "raw_value": [0.1 * i for i in range(n_rows)],
            "z_score": [0.05 * i for i in range(n_rows)],
            "direction": [1] * n_rows,
        }
    )


def _mock_copy_engine():
    cur = MagicMock()
    raw = MagicMock()
    raw.cursor.return_value = cur
    engine = MagicMock()
    engine.raw_connection.return_value = raw
    return engine, raw, cur


def _mock_upsert_engine():
    conn = MagicMock()
    conn.stage_copies = []  # decoded stage COPYs of the #432 daily/group write path
    record_stage_copy(conn, conn.stage_copies)
    engine = MagicMock()
    engine.begin.return_value.__enter__.return_value = conn
    engine.begin.return_value.__exit__.return_value = False
    return engine, conn


def _copy_sql_and_payload(cur):
    """(COPY statement, list-of-CSV-rows) from a recorded copy_expert call."""
    stmt, buf = cur.copy_expert.call_args[0]
    return stmt, [line for line in buf.getvalue().splitlines() if line]


def _insert_chunks(conn):
    """Row dicts passed to every executemany-style INSERT on a mock conn, plus
    the rows of every stage-table COPY (the daily/group writers' row path)."""
    out = []
    for copied in conn.stage_copies:
        out.extend(copied["params"])
    for call in conn.execute.call_args_list:
        if len(call.args) > 1 and isinstance(call.args[1], list):
            out.extend(call.args[1])
    return out


# ─────────────────────────────────────────────
# input_vintage: the field that matters most
# ─────────────────────────────────────────────


class TestInputVintage:
    def test_source_vintage_renders_dates_as_iso_and_drops_unknowns(self):
        from meridian_pipeline import lineage

        v = lineage.source_vintage(
            "ohlcv.prices",
            max_inserted_at=datetime(2026, 7, 31, 23, 4, 11, tzinfo=timezone.utc),
            reader="medallion:silver",
            n_rows=4123456,
            snapshot_id=None,  # unknown must not be recorded as a value
        )
        assert v == {
            "ohlcv.prices": {
                "max_inserted_at": "2026-07-31T23:04:11+00:00",
                "reader": "medallion:silver",
                "n_rows": 4123456,
            }
        }

    def test_merge_is_per_source_not_a_collapse(self):
        """The whole argument for jsonb: several sources, independent vintages,
        none of which may be silently dropped."""
        from meridian_pipeline import lineage

        merged = lineage.merge_vintages(
            lineage.source_vintage("ohlcv.prices", max_inserted_at="2026-07-31T00:00:00+00:00"),
            lineage.source_vintage("fundamentals.income_statement", max_filing_date="2026-07-30"),
            lineage.source_vintage("ohlcv.prices", reader="medallion:silver"),
        )
        assert set(merged) == {"ohlcv.prices", "fundamentals.income_statement"}
        # Same-source merges accumulate keys rather than replacing the entry.
        assert merged["ohlcv.prices"] == {
            "max_inserted_at": "2026-07-31T00:00:00+00:00",
            "reader": "medallion:silver",
        }

    def test_scalar_is_the_max_over_every_source(self):
        """input_vintage_max drives `WHERE input_vintage_max < '<fix date>'`.
        MAX because the question is "could this campaign have seen the fix"."""
        from meridian_pipeline import lineage

        merged = lineage.merge_vintages(
            lineage.source_vintage("a", ts="2026-01-01T00:00:00+00:00"),
            lineage.source_vintage("b", ts="2026-07-31T23:04:11+00:00", note="not a timestamp"),
            lineage.source_vintage("c", d=date(2026, 3, 1)),
        )
        assert lineage.vintage_scalar(merged) == datetime(
            2026, 7, 31, 23, 4, 11, tzinfo=timezone.utc
        )

    def test_scalar_is_none_when_nothing_parses(self):
        from meridian_pipeline import lineage

        assert lineage.vintage_scalar({}) is None
        assert lineage.vintage_scalar({"a": {"reader": "postgresql", "n": 5}}) is None

    def test_input_campaign_manifest_is_exact_and_fail_closed(self):
        from meridian_pipeline import lineage

        manifest = lineage.input_campaign_manifest(pd.DataFrame({"campaign_id": [12, 12, 19]}))
        assert manifest == {
            "campaigns": [12, 19],
            "row_count": 3,
            "stamped_row_count": 3,
            "unstamped_row_count": 0,
            "pre_lineage_row_count": 0,
            "complete": True,
        }

        incomplete = lineage.input_campaign_manifest(pd.DataFrame({"campaign_id": [12, None, 0]}))
        assert incomplete["campaigns"] == [12]
        assert incomplete["complete"] is False
        assert incomplete["unstamped_row_count"] == 1
        assert incomplete["pre_lineage_row_count"] == 1

    def test_attach_input_vintage_updates_an_implicit_output_campaign(self, monkeypatch):
        from meridian_pipeline import lineage

        recorded = []
        monkeypatch.setattr(
            "meridian_pipeline.db.add_write_campaign_input_vintage",
            lambda engine, campaign_id, input_vintage: recorded.append(
                (campaign_id, input_vintage)
            ),
        )
        manifest = lineage.input_campaign_manifest(pd.DataFrame({"campaign_id": [12, 19]}))

        lineage.attach_input_vintage(
            object(), 77, lineage.source_vintage("factors.daily_scores", **manifest)
        )

        assert recorded == [(77, {"factors.daily_scores": manifest})]


# ─────────────────────────────────────────────
# Bit-identical parity: lineage changes no VALUE
# ─────────────────────────────────────────────


class TestParityWithAndWithoutCampaign:
    """The hard constraint: adding lineage must not change a single factor
    value. Written twice from the same frame, once stamped and once not, the
    score payload must be identical."""

    def test_copy_path_score_fields_are_byte_identical(self):
        from meridian_pipeline import db

        df = _df(4)
        df.loc[1, "z_score"] = float("nan")
        df.loc[2, "raw_value"] = float("inf")

        with patch("meridian_pipeline.db._dual_write_daily_scores"):
            eng_plain, _r, cur_plain = _mock_copy_engine()
            with patch.object(db, "resolve_campaign_id", return_value=None):
                db.copy_replace_daily_scores(eng_plain, df, owned_factors=OWNED, **_OWNED_OPTOUT)

            eng_stamped, _r2, cur_stamped = _mock_copy_engine()
            with patch.object(db, "resolve_campaign_id", return_value=4242):
                db.copy_replace_daily_scores(eng_stamped, df, owned_factors=OWNED, **_OWNED_OPTOUT)

        plain_sql, plain_rows = _copy_sql_and_payload(cur_plain)
        stamped_sql, stamped_rows = _copy_sql_and_payload(cur_stamped)

        assert "campaign_id" not in plain_sql
        assert plain_sql.endswith("FROM STDIN WITH (FORMAT csv, NULL '')")
        assert "campaign_id" in stamped_sql
        assert len(plain_rows) == len(stamped_rows) == len(df)
        for plain, stamped in zip(plain_rows, stamped_rows):
            # The stamped row is the plain row plus exactly one trailing field.
            assert stamped == f"{plain},4242"

    def test_copy_path_unstamped_sql_is_unchanged_from_pre_294(self):
        """Second check: with no campaign the COPY statement is character
        -for-character what it was before lineage existed."""
        from meridian_pipeline import db

        eng, _raw, cur = _mock_copy_engine()
        with (
            patch("meridian_pipeline.db._dual_write_daily_scores"),
            patch.object(db, "resolve_campaign_id", return_value=None),
        ):
            db.copy_replace_daily_scores(eng, _df(), owned_factors=OWNED, **_OWNED_OPTOUT)

        stmt, _rows = _copy_sql_and_payload(cur)
        assert stmt == (
            "COPY factors.daily_scores "
            "(score_date, ticker, factor_name, group_name, raw_value, z_score, direction) "
            "FROM STDIN WITH (FORMAT csv, NULL '')"
        )

    def test_upsert_path_score_fields_are_identical(self):
        from meridian_pipeline import db

        df = _df(3)

        with patch("meridian_pipeline.db._dual_write_daily_scores"):
            eng_plain, conn_plain = _mock_upsert_engine()
            with patch.object(db, "resolve_campaign_id", return_value=None):
                db.upsert_daily_scores(eng_plain, df, owned_factors=OWNED, **_OWNED_OPTOUT)

            eng_stamped, conn_stamped = _mock_upsert_engine()
            with (
                patch.object(db, "resolve_campaign_id", return_value=99),
                # the stamp only reaches the stage when the column exists
                patch.object(db, "_campaign_column_available", return_value=True),
            ):
                db.upsert_daily_scores(eng_stamped, df, owned_factors=OWNED, **_OWNED_OPTOUT)

        plain = _insert_chunks(conn_plain)
        stamped = _insert_chunks(conn_stamped)
        assert len(plain) == len(stamped) == len(df)
        score_keys = ("score_date", "ticker", "factor_name", "group_name", "raw_value", "z_score")
        for p, s in zip(plain, stamped):
            assert {k: p[k] for k in score_keys} == {k: s[k] for k in score_keys}
            assert p["direction"] == s["direction"]
            assert "campaign_id" not in p
            assert s["campaign_id"] == 99

    def test_mirror_never_receives_campaign_id(self):
        """PG is the lineage authority for now; the flat-parquet / Iceberg
        mirrors have fixed schemas and must not silently gain a column."""
        from meridian_pipeline import db

        engine, _conn = _mock_upsert_engine()
        with (
            patch("meridian_pipeline.db._dual_write_daily_scores") as mock_dw,
            patch.object(db, "resolve_campaign_id", return_value=7),
        ):
            db.upsert_daily_scores(engine, _df(), owned_factors=OWNED, **_OWNED_OPTOUT)

        mirrored = mock_dw.call_args.args[0]
        assert "campaign_id" not in mirrored.columns

    def test_row_counts_are_unaffected(self):
        from meridian_pipeline import db

        df = _df(5)
        with patch("meridian_pipeline.db._dual_write_daily_scores"):
            eng_a, _ = _mock_upsert_engine()
            with patch.object(db, "resolve_campaign_id", return_value=None):
                n_plain = db.upsert_daily_scores(eng_a, df, owned_factors=OWNED, **_OWNED_OPTOUT)
            eng_b, _ = _mock_upsert_engine()
            with patch.object(db, "resolve_campaign_id", return_value=1):
                n_stamped = db.upsert_daily_scores(eng_b, df, owned_factors=OWNED, **_OWNED_OPTOUT)
        assert n_plain == n_stamped == 5


# ─────────────────────────────────────────────
# computed_at: compute time, not write time
# ─────────────────────────────────────────────


class TestComputedAtDefect:
    """The pre-#294 defect: copy_replace_daily_scores omitted computed_at from
    the COPY column list, so the table DEFAULT stamped WRITE time. On the burst
    path that is hours-to-days after the scores were computed - and clustering
    that column is how write campaigns were (wrongly) reconstructed."""

    def test_copy_writes_computed_at_when_the_frame_carries_it(self):
        from meridian_pipeline import db

        df = _df(2)
        df["computed_at"] = pd.Timestamp("2026-07-15T12:00:00Z")

        eng, _raw, cur = _mock_copy_engine()
        with (
            patch("meridian_pipeline.db._dual_write_daily_scores"),
            patch.object(db, "resolve_campaign_id", return_value=None),
        ):
            db.copy_replace_daily_scores(eng, df, owned_factors=OWNED, **_OWNED_OPTOUT)

        stmt, rows = _copy_sql_and_payload(cur)
        assert "computed_at" in stmt
        assert all(r.endswith("2026-07-15T12:00:00+00:00") for r in rows)

    def test_copy_handles_mixed_and_missing_compute_stamps(self):
        """The constant-fold fast path (one stamp per partition, rendered once)
        must not be the only correct path: a frame with differing or missing
        stamps still round-trips, with NaT becoming a SQL NULL."""
        from meridian_pipeline import db

        df = _df(3)
        df["computed_at"] = [
            pd.Timestamp("2026-07-15T12:00:00Z"),
            pd.Timestamp("2026-07-16T12:00:00Z"),
            pd.NaT,
        ]
        eng, _raw, cur = _mock_copy_engine()
        with (
            patch("meridian_pipeline.db._dual_write_daily_scores"),
            patch.object(db, "resolve_campaign_id", return_value=None),
        ):
            db.copy_replace_daily_scores(eng, df, owned_factors=OWNED, **_OWNED_OPTOUT)

        _stmt, rows = _copy_sql_and_payload(cur)
        assert rows[0].endswith("2026-07-15T12:00:00.000000+0000")
        assert rows[1].endswith("2026-07-16T12:00:00.000000+0000")
        assert rows[2].endswith(","), "NaT must render as an empty field (SQL NULL)"

    def test_copy_omits_computed_at_when_the_frame_does_not(self):
        """No compute stamp available means fall back to the column DEFAULT -
        the pre-#294 behaviour, not a fabricated timestamp."""
        from meridian_pipeline import db

        eng, _raw, cur = _mock_copy_engine()
        with (
            patch("meridian_pipeline.db._dual_write_daily_scores"),
            patch.object(db, "resolve_campaign_id", return_value=None),
        ):
            db.copy_replace_daily_scores(eng, _df(), owned_factors=OWNED, **_OWNED_OPTOUT)

        stmt, _rows = _copy_sql_and_payload(cur)
        assert "computed_at" not in stmt

    def test_copy_still_deletes_before_copying(self):
        """Delete-replace semantics must survive the extra columns: computed_at
        is not part of the PK and the DELETE runs first, so which rows exist is
        unchanged."""
        from meridian_pipeline import db

        df = _df(2)
        df["computed_at"] = pd.Timestamp("2026-07-15T12:00:00Z")
        eng, _raw, cur = _mock_copy_engine()
        with (
            patch("meridian_pipeline.db._dual_write_daily_scores"),
            patch.object(db, "resolve_campaign_id", return_value=11),
        ):
            db.copy_replace_daily_scores(eng, df, owned_factors=OWNED, **_OWNED_OPTOUT)

        deletes = [c for c in cur.execute.call_args_list if "DELETE" in c.args[0]]
        assert len(deletes) == 1
        assert "factor_name = ANY(%s)" in deletes[0].args[0]

    def test_upsert_writes_the_frames_computed_at(self):
        from meridian_pipeline import db

        df = _df(2)
        df["computed_at"] = pd.Timestamp("2026-07-15T12:00:00Z")
        engine, conn = _mock_upsert_engine()
        with (
            patch("meridian_pipeline.db._dual_write_daily_scores"),
            patch.object(db, "resolve_campaign_id", return_value=None),
        ):
            db.upsert_daily_scores(engine, df, owned_factors=OWNED, **_OWNED_OPTOUT)

        inserts = [c for c in conn.execute.call_args_list if "INSERT" in str(c.args[0])]
        sql = str(inserts[0].args[0])
        assert "ON CONFLICT" not in sql  # Meridian#432
        assert sql[: sql.index(") SELECT")].endswith("computed_at")
        assert conn.stage_copies[0]["sql"].endswith("computed_at) FROM STDIN")
        assert conn.stage_copies[0]["params"][0]["computed_at"].startswith("2026-07-15 12:00:00")

    def test_upsert_defaults_to_now_when_the_frame_has_no_stamp(self):
        from meridian_pipeline import db

        engine, conn = _mock_upsert_engine()
        with (
            patch("meridian_pipeline.db._dual_write_daily_scores"),
            patch.object(db, "resolve_campaign_id", return_value=None),
        ):
            db.upsert_daily_scores(engine, _df(), owned_factors=OWNED, **_OWNED_OPTOUT)

        sql = str([c for c in conn.execute.call_args_list if "INSERT" in str(c.args[0])][0].args[0])
        # column DEFAULT now() stamps the fresh row (#432): not in the INSERT column list
        assert "computed_at" not in sql[: sql.index(") SELECT")]


# ─────────────────────────────────────────────
# Never fail a write
# ─────────────────────────────────────────────


class TestNeverFailsAWrite:
    def test_missing_campaign_column_falls_back_silently(self):
        """The migration ships unapplied. Until it is applied the probe is
        False and every writer emits its pre-#294 SQL."""
        from meridian_pipeline import db

        engine, _raw, cur = _mock_copy_engine()
        with (
            patch("meridian_pipeline.db._dual_write_daily_scores"),
            patch("meridian_pipeline.db._table_and_columns_exist", return_value=False),
        ):
            n = db.copy_replace_daily_scores(engine, _df(), owned_factors=OWNED, **_OWNED_OPTOUT)

        assert n == 3
        stmt, _rows = _copy_sql_and_payload(cur)
        assert "campaign_id" not in stmt

    def test_campaign_probe_exception_is_reraised_and_not_cached(self, caplog):
        """Meridian#432: the daily/group writers delete and re-insert, so a probe
        error answered with False would re-insert every row with campaign_id NULL
        and silently wipe its lineage. The error is logged at WARNING and
        re-raised; only a SUCCESSFUL probe is cached."""
        from meridian_pipeline import db

        engine = MagicMock()
        with (
            caplog.at_level("WARNING"),
            patch(
                "meridian_pipeline.db._table_and_columns_exist",
                side_effect=RuntimeError("connection refused"),
            ),
            pytest.raises(RuntimeError, match="connection refused"),
        ):
            db.resolve_campaign_id(engine, None, "factors.daily_scores")
        assert "campaign_id probe failed" in caplog.text
        assert not db._campaign_column_cache  # the failure was not remembered
        # the next call probes again; a successful answer is then cached
        with patch("meridian_pipeline.db._table_and_columns_exist", return_value=True) as probe:
            assert db._campaign_column_available(engine, "factors.daily_scores") is True
            assert db._campaign_column_available(engine, "factors.daily_scores") is True
        probe.assert_called_once()

    def test_probe_error_aborts_the_write_before_any_sql(self):
        from meridian_pipeline import db

        engine, conn = _mock_upsert_engine()
        with (
            patch("meridian_pipeline.db._table_and_columns_exist", side_effect=OSError("down")),
            pytest.raises(OSError),
        ):
            db.upsert_daily_scores(engine, _df(), owned_factors=OWNED, **_OWNED_OPTOUT)
        conn.execute.assert_not_called()
        engine.begin.assert_not_called()

    def test_record_write_campaign_returns_none_on_db_error(self):
        from meridian_pipeline import db

        engine = MagicMock()
        engine.begin.side_effect = RuntimeError("db down")
        assert db.record_write_campaign(engine, writer="dagster_asset") is None

    def test_finish_write_campaign_swallows_errors(self):
        from meridian_pipeline import db

        engine = MagicMock()
        engine.begin.side_effect = RuntimeError("db down")
        db.finish_write_campaign(engine, 5, gate_result="pass")  # must not raise

    def test_ensure_bootstrap_never_raises(self):
        from meridian_pipeline import db

        engine = MagicMock()
        with patch(
            "meridian_pipeline.db._table_and_columns_exist", side_effect=RuntimeError("no db")
        ):
            db.ensure_write_campaigns_table(engine)  # must not raise

    def test_open_campaign_yields_even_when_the_insert_fails(self):
        """A campaign that could not be recorded must still let the work run -
        unstamped rows beat no rows."""
        from meridian_pipeline import lineage

        engine = MagicMock()
        with patch("meridian_pipeline.db.record_write_campaign", return_value=None):
            with lineage.open_campaign(engine, lineage.WRITER_BURST_PROMOTE) as campaign:
                assert campaign.campaign_id is None

    def test_kill_switch_disables_everything(self, monkeypatch):
        from meridian_pipeline import db, lineage

        monkeypatch.setenv("MERIDIAN_LINEAGE_DISABLED", "1")
        engine = MagicMock()
        assert lineage.current_campaign_id(engine) is None
        with patch("meridian_pipeline.db._table_and_columns_exist", return_value=True):
            assert db.resolve_campaign_id(engine, None, "factors.daily_scores") is None


# ─────────────────────────────────────────────
# Campaign binding
# ─────────────────────────────────────────────


class TestCampaignBinding:
    def test_explicit_campaign_wins_over_ambient(self):
        from meridian_pipeline import db, lineage

        engine = MagicMock()
        with patch("meridian_pipeline.db._table_and_columns_exist", return_value=True):
            with patch("meridian_pipeline.db.record_write_campaign", return_value=100):
                with lineage.open_campaign(engine, lineage.WRITER_MANUAL_SCRIPT):
                    assert db.resolve_campaign_id(engine, None, "factors.daily_scores") == 100
                    assert db.resolve_campaign_id(engine, 555, "factors.daily_scores") == 555

    def test_ambient_campaign_is_picked_up_by_the_writer(self):
        from meridian_pipeline import db, lineage

        engine, _raw, cur = _mock_copy_engine()
        with (
            patch("meridian_pipeline.db._dual_write_daily_scores"),
            patch("meridian_pipeline.db._table_and_columns_exist", return_value=True),
            patch("meridian_pipeline.db.record_write_campaign", return_value=321),
            patch("meridian_pipeline.db.finish_write_campaign"),
        ):
            with lineage.open_campaign(engine, lineage.WRITER_BURST_PROMOTE):
                db.copy_replace_daily_scores(engine, _df(1), owned_factors=OWNED, **_OWNED_OPTOUT)

        _stmt, rows = _copy_sql_and_payload(cur)
        assert rows[0].endswith(",321")

    def test_implicit_campaign_is_minted_once_per_process(self):
        """No campaign bound: the first write still gets a real one, so a row
        is never anonymous - and a second write reuses it rather than minting
        one campaign per partition."""
        from meridian_pipeline import lineage

        engine = MagicMock()
        with patch("meridian_pipeline.db.record_write_campaign", return_value=17) as rec:
            assert lineage.current_campaign_id(engine) == 17
            assert lineage.current_campaign_id(engine) == 17
        assert rec.call_count == 1
        assert rec.call_args.kwargs["input_vintage"] == {}, (
            "an implicit campaign must record NO vintage rather than guessing one"
        )

    def test_implicit_campaign_carries_the_bound_dagster_run_id(self):
        from meridian_pipeline import lineage

        engine = MagicMock()
        context = MagicMock()
        context.run_id = "run-abc-123"
        lineage.bind_dagster_run(context)
        with patch("meridian_pipeline.db.record_write_campaign", return_value=18) as rec:
            lineage.current_campaign_id(engine)
        assert rec.call_args.kwargs["run_id"] == "run-abc-123"
        assert rec.call_args.kwargs["writer"] == lineage.WRITER_DAGSTER_ASSET

    def test_auto_off_leaves_writes_unstamped(self, monkeypatch):
        from meridian_pipeline import lineage

        monkeypatch.setenv("MERIDIAN_LINEAGE_AUTO", "0")
        engine = MagicMock()
        with patch("meridian_pipeline.db.record_write_campaign") as rec:
            assert lineage.current_campaign_id(engine) is None
        rec.assert_not_called()

    def test_gate_result_and_failure_are_recorded_on_close(self):
        from meridian_pipeline import lineage

        engine = MagicMock()
        with (
            patch("meridian_pipeline.db.record_write_campaign", return_value=9),
            patch("meridian_pipeline.db.finish_write_campaign") as fin,
        ):
            with lineage.open_campaign(engine, lineage.WRITER_BURST_PROMOTE):
                lineage.set_gate_result(lineage.GATE_PASS, verify_failures=0)
        assert fin.call_args.kwargs["gate_result"] == lineage.GATE_PASS
        assert fin.call_args.kwargs["detail"]["verify_failures"] == 0

        with (
            patch("meridian_pipeline.db.record_write_campaign", return_value=10),
            patch("meridian_pipeline.db.finish_write_campaign") as fin_fail,
            pytest.raises(RuntimeError),
        ):
            with lineage.open_campaign(engine, lineage.WRITER_BURST_PROMOTE):
                raise RuntimeError("promote blew up")
        assert fin_fail.call_args.kwargs["gate_result"] == lineage.GATE_FAIL


@pytest.mark.parametrize("surface_name", ["group_scores", "composite_scores"])
def test_derived_writer_persists_exact_manifest_for_each_output_partition_date(surface_name):
    from meridian_pipeline import db

    engine, conn = _mock_upsert_engine()
    first = date(2026, 5, 1)
    second = date(2026, 5, 2)
    if surface_name == "group_scores":
        output = pd.DataFrame(
            {
                "score_date": [first, second],
                "ticker": ["T0", "T0"],
                "group_name": ["value", "value"],
                "group_score": [0.5, 0.6],
            }
        )
    else:
        output = pd.DataFrame(
            {
                "score_date": [first, second],
                "ticker": ["T0", "T0"],
                "composite_score": [0.5, 0.6],
                "quintile": [3, 4],
            }
        )
    inputs = pd.DataFrame(
        {
            "score_date": [first, first, second, second],
            "ticker": ["T0", "T1", "T0", "T1"],
            "factor_name": ["book_yield", "book_yield", "momentum_12_1", "momentum_12_1"],
            "campaign_id": [12, 12, 19, 19],
        }
    )

    with patch.object(db, "resolve_campaign_id", return_value=64):
        if surface_name == "group_scores":
            db.upsert_group_scores(engine, output, input_campaign_rows=inputs)
        else:
            with patch.object(db, "_composite_has_ngroups", return_value=False):
                db.upsert_composite_scores(engine, output, input_campaign_rows=inputs)

    manifest_rows = next(
        call.args[1]
        for call in conn.execute.call_args_list
        if len(call.args) > 1
        and isinstance(call.args[1], list)
        and call.args[1]
        and "surface_name" in call.args[1][0]
    )
    assert {
        (row["score_date"], row["factor_name"], row["campaign_id"]) for row in manifest_rows
    } == {
        (first, "book_yield", 12),
        (second, "momentum_12_1", 19),
    }
    assert all(row["surface_name"] == surface_name for row in manifest_rows)
    assert all(row["output_campaign_id"] == 64 for row in manifest_rows)
    assert engine.begin.call_count == 1


def test_derived_manifest_binds_each_input_tuple_to_its_output_campaign():
    """Republishing one output campaign must not erase another's provenance."""
    from meridian_pipeline import db

    engine, conn = _mock_upsert_engine()
    score_date = date(2026, 5, 1)
    inputs = pd.DataFrame(
        {
            "score_date": [score_date],
            "factor_name": ["book_yield"],
            "campaign_id": [12],
        }
    )

    with engine.begin() as transaction:
        db._replace_derived_surface_input_campaigns(
            transaction, "group_scores", 64, [score_date], inputs
        )

    delete_sql, delete_params = conn.execute.call_args_list[0].args
    assert "output_campaign_id = :output_campaign_id" in str(delete_sql)
    assert delete_params["output_campaign_id"] == 64
    insert_sql, manifest_rows = conn.execute.call_args_list[1].args
    assert "output_campaign_id" in str(insert_sql)
    assert (
        "ON CONFLICT (surface_name, output_campaign_id, score_date, factor_name, campaign_id)"
        in str(insert_sql)
    )
    assert manifest_rows == [
        {
            "surface_name": "group_scores",
            "output_campaign_id": 64,
            "score_date": score_date,
            "factor_name": "book_yield",
            "campaign_id": 12,
        }
    ]
