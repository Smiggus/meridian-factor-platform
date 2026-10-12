"""
Tests for db.upsert_* functions - chunked-executemany pattern.

These three upserts used to do row-by-row INSERTs (~one round-trip per
row to db.internal). Refactored to chunked-executemany matching the proven
`risk_metrics_asset._upsert_risk_metrics` idiom. Same `ON CONFLICT
DO UPDATE` semantics so re-running a backfill still overwrites cleanly.
(Meridian#432: factors.daily_scores / group_scores no longer use ON CONFLICT or
executemany; their payload goes to a temp stage table in ONE COPY, see
tests/test_upsert_replace.py. composite_scores and the rest still chunk.)

Test contract:
- A DataFrame of N rows triggers ceil(N / chunk_size) INSERT execute() calls,
  each receiving a list of dict rows (NOT a single dict per call).
- The INSERT SQL retains the `ON CONFLICT ... DO UPDATE` clause (re-run safety).
- Empty DataFrame is a no-op (no execute() calls).
- Return value is len(df) regardless of chunking.

Since #108 the upserts also delete-replace per (score_date[, group_name]) by
default, so each call emits a leading DELETE before the chunked INSERTs. Tests
filter executes into DELETE vs INSERT and assert on each.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from tests._stage_copy import record_stage_copy

_DAGSTER_DIR = Path(__file__).resolve().parents[1] / "dagster"
if str(_DAGSTER_DIR) not in sys.path:
    sys.path.insert(0, str(_DAGSTER_DIR))

os.environ.setdefault("PARTITION_START_DATE", "2010-01-02")
os.environ.setdefault("PGPASS", "test-pg-pass")


def _mock_engine_capture():
    """Return (engine_mock, executed_calls_list) where each entry is the
    second positional arg passed to conn.execute() (the param payload)."""
    captured = []
    conn = MagicMock()

    def execute(stmt, params=None):
        captured.append({"sql": str(stmt), "params": params})
        return MagicMock()  # .fetchall() -> [] (nothing deleted / carried)

    conn.execute.side_effect = execute
    record_stage_copy(conn, captured)  # the stage COPY, decoded back to row dicts
    engine = MagicMock()
    engine.begin.return_value.__enter__.return_value = conn
    engine.begin.return_value.__exit__.return_value = False
    return engine, captured


def _inserts(captured):
    # Row-carrying statements: the ONE stage COPY for daily/group scores (#432
    # routes the target insert through a single `INSERT ... SELECT FROM stage`)
    # and plain INSERTs for every other table.
    return [
        c
        for c in captured
        if c["sql"].startswith("COPY _m432_stage")
        or ("INSERT" in c["sql"] and "_m432_" not in c["sql"])
    ]


def _deletes(captured):
    """Scoped replace DELETEs (#108); the Meridian#432 key-exact pass is separate."""
    return [c for c in captured if "DELETE" in c["sql"] and "_m432_stage" not in c["sql"]]


def _make_daily_scores_df(n_rows: int) -> pd.DataFrame:
    from datetime import date

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


def test_empty_df_is_noop():
    """No SQL emitted when input DataFrame is empty."""
    from meridian_pipeline import db

    engine, captured = _mock_engine_capture()
    n = db.upsert_daily_scores(engine, pd.DataFrame())
    assert n == 0
    assert captured == []


def test_stage_is_loaded_by_one_copy_not_chunked_executemany():
    """Meridian#432: N rows -> ONE stage COPY (not ceil(N/chunk) INSERTs)."""
    from meridian_pipeline import db

    n_rows = 2500
    df = _make_daily_scores_df(n_rows)
    engine, captured = _mock_engine_capture()
    returned = db.upsert_daily_scores(engine, df)

    assert returned == n_rows
    ins = _inserts(captured)
    assert len(ins) == 1, f"expected a single stage COPY, got {len(ins)}"
    assert ins[0]["sql"].startswith("COPY _m432_stage (score_date, ticker, factor_name, ")
    assert len(ins[0]["params"]) == n_rows
    assert isinstance(ins[0]["params"][0], dict)
    assert len(_deletes(captured)) == 1  # #108: one DELETE for the single (date, group)


def test_no_on_conflict_clause_cold_tier_safe():
    """Meridian#432: the cold-tier parent has no unique index, so the insert must
    not name a conflict target. Re-run safety now comes from a key-exact delete
    under a per-score_date advisory lock (see tests/test_upsert_replace.py)."""
    from meridian_pipeline import db

    engine, captured = _mock_engine_capture()
    db.upsert_daily_scores(engine, _make_daily_scores_df(10))

    sql = _inserts(captured)[0]["sql"]
    assert "ON CONFLICT" not in sql.upper()
    assert sql.startswith("COPY _m432_stage")  # staged, then one INSERT ... SELECT


def test_group_scores_chunked_without_on_conflict():
    """Same single-COPY stage contract for upsert_group_scores (no ON CONFLICT, #432)."""
    from datetime import date

    from meridian_pipeline import db

    df = pd.DataFrame(
        {
            "score_date": [date(2026, 5, 1)] * 1500,
            "ticker": [f"T{i}" for i in range(1500)],
            "group_name": ["value"] * 1500,
            "group_score": [0.1 * i for i in range(1500)],
        }
    )
    engine, captured = _mock_engine_capture()
    n = db.upsert_group_scores(engine, df)

    assert n == 1500
    assert len(_inserts(captured)) == 1  # one stage COPY for all 1500 rows
    assert len(_inserts(captured)[0]["params"]) == 1500
    sql = _inserts(captured)[0]["sql"]
    assert "ON CONFLICT" not in sql.upper()  # Meridian#432


def test_composite_scores_chunked_with_on_conflict():
    """Same chunked / ON CONFLICT contract for upsert_composite_scores."""
    from datetime import date

    from meridian_pipeline import db

    df = pd.DataFrame(
        {
            "score_date": [date(2026, 5, 1)] * 800,
            "ticker": [f"T{i}" for i in range(800)],
            "composite_score": [0.1 * i for i in range(800)],
            "quintile": [(i % 5) + 1 for i in range(800)],
        }
    )
    engine, captured = _mock_engine_capture()
    n = db.upsert_composite_scores(engine, df)

    assert n == 800
    assert len(_inserts(captured)) == 1
    sql = _inserts(captured)[0]["sql"]
    assert "ON CONFLICT (score_date, ticker)" in sql
    assert "DO UPDATE SET" in sql
    assert "composite_score = EXCLUDED.composite_score" in sql
    assert "quintile = EXCLUDED.quintile" in sql
    assert "computed_at = NOW()" in sql


def test_composite_scores_nan_sanitized_to_none():
    """A zero-coverage ticker (NaN composite -> NaN quintile) must NOT reach the
    smallint quintile column as NaN - psycopg2 raises "smallint out of range"
    and wedges the whole partition. pandas re-coerces a mixed int/None column
    back to float64 (None -> NaN), so upsert_composite_scores sanitizes NaN ->
    None at the DB boundary. Regression for the #120 composite NaN path."""
    import math
    from datetime import date

    from meridian_pipeline import db

    df = pd.DataFrame(
        {
            "score_date": [date(2026, 5, 1)] * 3,
            "ticker": ["AAA", "BBB", "CCC"],
            "composite_score": [0.5, float("nan"), -0.3],  # BBB zero-coverage
            "quintile": [4.0, float("nan"), 2.0],  # float dtype incl. NaN
        }
    )
    engine, captured = _mock_engine_capture()
    n = db.upsert_composite_scores(engine, df)

    assert n == 3
    payload = _inserts(captured)[0]["params"]
    by_ticker = {r["ticker"]: r for r in payload}

    # NaN quintile -> None (NULL), finite quintiles -> int (not float)
    assert by_ticker["BBB"]["quintile"] is None
    assert by_ticker["AAA"]["quintile"] == 4 and isinstance(by_ticker["AAA"]["quintile"], int)
    assert by_ticker["CCC"]["quintile"] == 2

    # NaN composite_score -> None; no non-finite float survives anywhere
    assert by_ticker["BBB"]["composite_score"] is None
    for r in payload:
        q = r["quintile"]
        assert q is None or (isinstance(q, int) and 1 <= q <= 5)
        cs = r["composite_score"]
        assert cs is None or (isinstance(cs, float) and math.isfinite(cs))


def test_replace_delete_is_scoped_to_dfs_factors_not_whole_group():
    """Regression for #160: the delete-replace must target only the factor_names
    present in the df, not the whole (score_date, group_name). Multiple assets
    write the same group in sequence (each writing its own factors); a
    group-wide delete made each writer clobber the others, leaving only the
    last writer's factors."""
    from datetime import date

    from meridian_pipeline import db

    df = pd.DataFrame(
        {
            "score_date": [date(2026, 5, 1)] * 2,
            "ticker": ["AAA", "AAA"],
            "factor_name": ["macro_a", "macro_b"],
            "group_name": ["macro", "macro"],
            "raw_value": [0.1, 0.2],
            "z_score": [0.3, -0.4],
            "direction": [1, 1],
        }
    )
    engine, captured = _mock_engine_capture()
    db.upsert_daily_scores(engine, df)

    deletes = _deletes(captured)
    assert len(deletes) == 1
    sql = deletes[0]["sql"]
    assert "d.factor_name = k.factor_name" in sql  # factor-scoped, NOT a whole-group wipe
    params = deletes[0]["params"]
    # one statement keyed by the (score_date, group_name, factor_name) triples,
    # with a literal date array for partition pruning
    assert set(params["k_groups"]) == {"macro"}
    assert set(params["k_factors"]) == {"macro_a", "macro_b"}
    assert params["score_dates"] == [date(2026, 5, 1)]
    assert "d.score_date = ANY(CAST(:score_dates AS date[]))" in sql


def test_round_trip_count_is_independent_of_row_count():
    """A typical daily partition writes thousands of rows for value_scores. Round-trips
    are O(1): one stage COPY, never one statement per chunk (Meridian#432)."""
    from meridian_pipeline import db

    df = _make_daily_scores_df(16000)
    engine, captured = _mock_engine_capture()
    db.upsert_daily_scores(engine, df)

    assert len(_inserts(captured)) == 1
    assert len(_inserts(captured)[0]["params"]) == 16000


def test_daily_scores_nonfinite_sanitized_to_none():
    """Regression for #161: upsert_daily_scores must null-out NaN and inf
    raw_value/z_score before reaching Postgres float8. NaN/inf survive
    dropna() and break downstream consumers (QC Insights decimal, risk model)."""
    import math

    from meridian_pipeline import db

    df = _make_daily_scores_df(3)
    df["raw_value"] = [0.42, float("nan"), float("inf")]
    df["z_score"] = [0.1, float("-inf"), float("nan")]

    engine, captured = _mock_engine_capture()
    n = db.upsert_daily_scores(engine, df)

    assert n == 3
    payload = _inserts(captured)[0]["params"]
    by_ticker = {r["ticker"]: r for r in payload}

    # T0: finite values pass through unchanged
    assert by_ticker["T0"]["raw_value"] == pytest.approx(0.42)
    assert by_ticker["T0"]["z_score"] == pytest.approx(0.1)

    # T1: NaN -> None
    assert by_ticker["T1"]["raw_value"] is None
    assert by_ticker["T1"]["z_score"] is None

    # T2: inf/-inf -> None
    assert by_ticker["T2"]["raw_value"] is None
    assert by_ticker["T2"]["z_score"] is None

    # No non-finite float survives anywhere
    for r in payload:
        for col in ("raw_value", "z_score"):
            v = r[col]
            assert v is None or (isinstance(v, float) and math.isfinite(v)), (
                f"{col} in {r['ticker']} is non-finite: {v}"
            )


def test_parquet_dual_write_receives_sanitized_rows():
    """Regression (2026-07 audit, db.py:437): the best-effort parquet dual-write
    must receive the SANITIZED rows (NaN/inf already coerced to None), not the
    raw df - otherwise non-finite values reach the parquet mirror even though
    PG never sees them, a silent divergence between the authoritative store
    and the ML/evidence-engine read path."""
    import math
    from unittest.mock import patch

    from meridian_pipeline import db

    df = _make_daily_scores_df(3)
    df["raw_value"] = [0.42, float("nan"), float("inf")]
    df["z_score"] = [0.1, float("-inf"), float("nan")]

    engine, _captured = _mock_engine_capture()

    with patch("meridian_pipeline.parquet_io.append_factor_scores_partition") as mock_dual_write:
        db.upsert_daily_scores(engine, df)

    assert mock_dual_write.called
    dual_write_df = mock_dual_write.call_args[0][0]
    # No +/-inf may reach the mirror (NaN is an acceptable "missing" marker -
    # pandas silently re-coerces a sanitized None back to NaN in a float64
    # column, same as every other float column in this codebase - but a raw
    # inf/-inf surviving through is the real defect this guards against).
    assert dual_write_df["raw_value"].apply(lambda v: pd.isna(v) or math.isfinite(v)).all()
    assert dual_write_df["z_score"].apply(lambda v: pd.isna(v) or math.isfinite(v)).all()
    by_ticker = dual_write_df.set_index("ticker")
    assert pd.isna(by_ticker.loc["T1", "raw_value"])
    assert pd.isna(by_ticker.loc["T2", "z_score"])


def _macro_partial_df():
    """A macro payload carrying macro_a only - macro_b produced no
    rows this run. The #378 shape."""
    from datetime import date

    return pd.DataFrame(
        {
            "score_date": [date(2026, 5, 1)] * 2,
            "ticker": ["AAA", "BBB"],
            "factor_name": ["macro_a", "macro_a"],
            "group_name": ["macro", "macro"],
            "raw_value": [0.1, 0.2],
            "z_score": [0.3, -0.4],
            "direction": [1, 1],
        }
    )


def test_owned_factors_deletes_all_owned_even_if_absent_from_df():
    """Fix #5 (#161): a canonical-superset writer's delete must cover an owned
    factor that produced no rows this run, so its stale row for that date is
    cleared. Since #378 that is the OPT-IN behaviour
    (allow_unwritten_owned_factors=True) - see the test below for the default.

    The delete is scoped to (score_date, factor_name) only, NOT group_name - see the
    multi-group-writer regression test below for why group_name scoping is unsafe."""
    from meridian_pipeline import db

    df = _macro_partial_df()
    engine, captured = _mock_engine_capture()
    db.upsert_daily_scores(
        engine,
        df,
        owned_factors=["macro_a", "macro_b"],
        allow_unwritten_owned_factors=True,
    )

    deletes = _deletes(captured)
    assert len(deletes) == 1
    params = deletes[0]["params"]
    assert "group_name" not in params
    # owned_factors used for delete, not just df's factors
    assert set(params["factor_names"]) == {"macro_a", "macro_b"}


def test_owned_factors_absent_from_df_raises_by_default():
    """#378: naming an owned factor the payload has no rows for is a
    delete-WITHOUT-replace - the DELETE runs first, nothing replaces it, and the
    run reports SUCCESS. Rows can be lost this way. The default
    must refuse the write, and refuse it BEFORE issuing any SQL."""
    from meridian_pipeline import db

    df = _macro_partial_df()
    engine, captured = _mock_engine_capture()
    with pytest.raises(ValueError) as exc:
        db.upsert_daily_scores(engine, df, owned_factors=["macro_a", "macro_b"])

    assert "macro_b" in str(exc.value)
    assert "macro_a" not in str(exc.value)  # only the missing one is named
    assert not _deletes(captured), "the refusal must happen before any DELETE is issued"


def test_owned_factors_fully_covered_by_df_does_not_raise():
    """The check must not fire when every owned name is in the payload - order
    and duplicates in owned_factors are irrelevant."""
    from meridian_pipeline import db

    df = _macro_partial_df()
    engine, captured = _mock_engine_capture()
    db.upsert_daily_scores(engine, df, owned_factors=["macro_a", "macro_a"])
    assert len(_deletes(captured)) == 1


def _macro_multi_date_df(b_dates):
    """A two-date macro payload. macro_a is present on both dates;
    macro_b only on the dates in `b_dates`."""
    from datetime import date

    d1, d2 = date(2026, 5, 1), date(2026, 5, 4)
    recs = []
    for d in (d1, d2):
        recs.append((d, "AAA", "macro_a"))
        if d in b_dates:
            recs.append((d, "AAA", "macro_b"))
    return pd.DataFrame(
        {
            "score_date": [r[0] for r in recs],
            "ticker": [r[1] for r in recs],
            "factor_name": [r[2] for r in recs],
            "group_name": ["macro"] * len(recs),
            "raw_value": [0.1] * len(recs),
            "z_score": [0.3] * len(recs),
            "direction": [1] * len(recs),
        }
    )


def test_owned_factors_missing_on_one_date_of_a_multi_date_frame_raises():
    """#378 review follow-up: the check is PER score_date, matching the DELETE.

    A whole-frame check passes this frame - macro_b IS present, on 2026-05-01 -
    while 2026-05-04 takes the delete-without-replace and loses macro_b
    entirely. That is #378 again. Today only the opted-out batch promoters batch
    dates, but that is a fact about the current callers, not an invariant."""
    from datetime import date

    from meridian_pipeline import db

    df = _macro_multi_date_df(b_dates={date(2026, 5, 1)})
    engine, captured = _mock_engine_capture()
    with pytest.raises(ValueError) as exc:
        db.upsert_daily_scores(engine, df, owned_factors=["macro_a", "macro_b"])

    assert "macro_b" in str(exc.value)
    assert "2026-05-04" in str(exc.value), "the offending date must be named"
    assert not _deletes(captured), "the refusal must happen before any DELETE is issued"


def test_owned_factors_covered_on_every_date_of_a_multi_date_frame_passes():
    """The per-date check must not fire when every date carries every owned
    name - ONE DELETE covering every distinct score_date, no refusal."""
    from datetime import date

    from meridian_pipeline import db

    df = _macro_multi_date_df(b_dates={date(2026, 5, 1), date(2026, 5, 4)})
    engine, captured = _mock_engine_capture()
    db.upsert_daily_scores(engine, df, owned_factors=["macro_a", "macro_b"])

    deletes = _deletes(captured)
    assert len(deletes) == 1
    assert deletes[0]["params"]["score_dates"] == [date(2026, 5, 1), date(2026, 5, 4)]
    assert "score_date = ANY(CAST(:score_dates AS date[]))" in deletes[0]["sql"]


def test_owned_factors_check_skipped_when_replace_false():
    """replace=False issues no DELETE at all, so there is nothing to guard."""
    from meridian_pipeline import db

    df = _macro_partial_df()
    engine, captured = _mock_engine_capture()
    db.upsert_daily_scores(engine, df, replace=False, owned_factors=["macro_a", "macro_b"])
    assert not _deletes(captured)


def test_owned_factors_multi_group_writer_orphan_deleted():
    """Regression for the multi-group-writer orphan (2026-07 audit, db.py:394).

    A writer assigns each of its factors its OWN one-factor group. A run that
    writes fewer factors because one piece was empty this time must still
    delete the prior run's stale rows for the missing factor - even though its
    group never appears in this payload's group_name column at all.
    """
    from datetime import date

    from meridian_pipeline import db

    # This run's payload has 1 factor in its own group - 'group_b' (home of
    # factor_b) is entirely absent, simulating the empty-piece case.
    df = pd.DataFrame(
        {
            "score_date": [date(2026, 5, 1)] * 2,
            "ticker": ["AAA", "BBB"],
            "factor_name": ["factor_a", "factor_a"],
            "group_name": ["group_a", "group_a"],
            "raw_value": [0.1, 0.2],
            "z_score": [0.3, -0.4],
            "direction": [1, 1],
        }
    )
    engine, captured = _mock_engine_capture()
    owned = ["factor_a", "factor_b", "factor_c"]
    # A canonical-superset writer opts out of the #378 delete-without-replace
    # check precisely so this orphan clear happens.
    db.upsert_daily_scores(engine, df, owned_factors=owned, allow_unwritten_owned_factors=True)

    deletes = _deletes(captured)
    assert len(deletes) == 1
    params = deletes[0]["params"]
    # No group_name scoping - stale factor_b rows (in the absent
    # 'group_b' group) must still be targeted by this delete.
    assert "group_name" not in params
    assert set(params["factor_names"]) == set(owned)


def test_owned_factors_none_falls_back_to_df_factors():
    """When owned_factors=None, behaviour matches previous: delete only df's factors."""
    from datetime import date

    from meridian_pipeline import db

    df = pd.DataFrame(
        {
            "score_date": [date(2026, 5, 1)] * 2,
            "ticker": ["AAA", "BBB"],
            "factor_name": ["macro_a", "macro_a"],
            "group_name": ["macro", "macro"],
            "raw_value": [0.1, 0.2],
            "z_score": [0.3, -0.4],
            "direction": [1, 1],
        }
    )
    engine, captured = _mock_engine_capture()
    db.upsert_daily_scores(engine, df)  # owned_factors=None by default

    deletes = _deletes(captured)
    assert len(deletes) == 1
    params = deletes[0]["params"]
    # With owned_factors=None, only df's factors in the delete
    assert set(params["k_factors"]) == {"macro_a"}


def test_direction_rewritten_on_overwrite():
    """Fix #16 (#161), restated for #432: direction is part of the INSERT column
    list, and the existing key is removed by the key-exact delete, so an
    overwrite replaces direction (it used to be in DO UPDATE SET)."""
    from meridian_pipeline import db

    engine, captured = _mock_engine_capture()
    db.upsert_daily_scores(engine, _make_daily_scores_df(5), replace=False)

    sql = _inserts(captured)[0]["sql"]
    assert "direction" in sql.split("VALUES")[0]
    assert any("_m432_stage" in c["sql"] for c in captured if "DELETE" in c["sql"])


def test_is_trading_day_uses_sargable_half_open_range():
    """Perf fix (2026-07 audit, db.py:288): is_trading_day must filter with
    the sargable half-open range `ts_event >= :d AND ts_event < :d_next`, NOT
    the non-sargable `ts_event::date = :d` cast (which forces a sequential
    scan - no btree index on ts_event can be used through that cast)."""
    from datetime import date, timedelta

    from meridian_pipeline import db

    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = (150,)
    engine = MagicMock()
    engine.connect.return_value.__enter__.return_value = conn
    engine.connect.return_value.__exit__.return_value = False

    d = date(2026, 5, 1)
    result = db.is_trading_day(engine, d)

    assert result is True
    sql, params = conn.execute.call_args[0]
    sql_str = str(sql)
    assert "ts_event::date" not in sql_str
    assert "ts_event >= :d" in sql_str
    assert "ts_event < :d_next" in sql_str
    assert params["d"] == d
    assert params["d_next"] == d + timedelta(days=1)
