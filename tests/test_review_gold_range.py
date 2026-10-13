"""Gold snapshot replacement includes dates absent from its payload."""

import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dagster"))
from meridian_pipeline import db, lakehouse

from tests._stage_copy import record_stage_copy

LO, HI = date(2025, 9, 1), date(2025, 9, 3)


def payload():
    return pd.DataFrame(
        [
            dict(
                score_date=d,
                ticker="A",
                factor_name="roe",
                group_name="quality",
                raw_value=0.1,
                z_score=0.5,
                direction=1,
            )
            for d in (LO, HI)
        ]
    )


@pytest.mark.parametrize("empty", [False, True])
def test_promoter_replaces_full_scope_in_one_pg_transaction(empty):
    import dagster

    if not hasattr(dagster, "asset"):
        pytest.skip("Complete Dagster required")
    from meridian_pipeline import gold_pg_sync_asset as sync

    df = payload().iloc[:0] if empty else payload()
    scope = {
        "snapshot_id": 123,
        "owned_factors": ["roe"],
        "score_date_min": LO.isoformat(),
        "score_date_max": HI.isoformat(),
    }
    conn, engine = MagicMock(), MagicMock()
    engine.begin.return_value.__enter__.return_value = conn
    stage_copies = []
    record_stage_copy(conn, stage_copies)
    with (
        patch.object(lakehouse, "gold_promotion_available", return_value=True),
        patch.object(lakehouse, "read_gold_snapshot", return_value=(df, scope)),
        patch.object(db, "engine_from_env", return_value=engine),
        patch.object(db, "resolve_campaign_id", return_value=None),
        patch.object(db, "_dual_write_daily_scores") as mirror,
        patch.object(sync.lineage, "bind_dagster_run"),
        # The declared range's only trading days are LO and HI (per this
        # fixture). empty=True legitimately covers zero of them, so it needs
        # the #401 explicit override below - it is not testing that guard.
        # side_effect rather than return_value so the mock keeps the real
        # semantics (gaps = calendar minus payload) instead of hardcoding an
        # answer the caller's own logic would never have produced.
        patch.object(
            db,
            "missing_trading_days",
            side_effect=lambda engine, lo, hi, payload_days: sorted({LO, HI} - payload_days),
        ),
    ):
        sync.gold_scores_pg_sync.op.compute_fn.decorated_fn(
            SimpleNamespace(log=MagicMock()),
            sync.GoldPgSyncConfig(snapshot_id=123, allow_interior_gaps=empty),
        )
    engine.begin.assert_called_once()
    deletes = [c for c in conn.execute.call_args_list if str(c.args[0]).startswith("DELETE")]
    assert len(deletes) == 1
    assert "BETWEEN :date_min AND :date_max" in str(deletes[0].args[0])
    assert deletes[0].args[1] == {"date_min": LO, "date_max": HI, "factor_names": ["roe"]}
    # #432: payload rows go to a temp stage in ONE COPY, then one target INSERT.
    assert len(stage_copies) == (0 if empty else 1)
    assert mirror.call_args.kwargs["replace_date_range"] == (LO, HI)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"replace": False},
        {"owned_factors": None},
        {"allow_unwritten_owned_factors": False},
        {"replace_date_range": (HI, LO)},
        {"replace_date_range": (LO, LO)},
        {"owned_factors": ["another_factor"]},
    ],
)
def test_invalid_scope_is_rejected_before_database_access(kwargs):
    engine = MagicMock()
    options = dict(
        replace=True,
        owned_factors=["roe"],
        allow_unwritten_owned_factors=True,
        replace_date_range=(LO, HI),
    )
    options.update(kwargs)
    with pytest.raises(ValueError):
        db.upsert_daily_scores(engine, payload(), **options)
    engine.begin.assert_not_called()


INTERIOR_LO, INTERIOR_MID, INTERIOR_HI = date(2025, 9, 2), date(2025, 9, 3), date(2025, 9, 4)


def interior_gap_payload():
    """Covers the range's ENDS but is silently missing the middle trading day
    - the #401 hole. A pure containment check cannot see this: every row IS
    inside [INTERIOR_LO, INTERIOR_HI], the payload is just missing one whole
    date from the interior."""
    return pd.DataFrame(
        [
            dict(
                score_date=d,
                ticker="A",
                factor_name="roe",
                group_name="quality",
                raw_value=0.1,
                z_score=0.5,
                direction=1,
            )
            for d in (INTERIOR_LO, INTERIOR_HI)  # INTERIOR_MID silently missing
        ]
    )


def test_interior_date_gap_is_refused_by_default():
    """#401: on main this passes silently (containment holds - every row is
    inside the range) and INTERIOR_MID's rows get deleted with nothing to
    replace them. Must now raise before any DB access."""
    engine = MagicMock()
    with patch.object(
        db,
        "missing_trading_days",
        side_effect=lambda engine, lo, hi, payload_days: sorted(
            {INTERIOR_LO, INTERIOR_MID, INTERIOR_HI} - payload_days
        ),
        create=True,
    ):
        with pytest.raises(ValueError, match="interior-date gap|in-scope trading day"):
            db.upsert_daily_scores(
                engine,
                interior_gap_payload(),
                replace=True,
                owned_factors=["roe"],
                allow_unwritten_owned_factors=True,
                replace_date_range=(INTERIOR_LO, INTERIOR_HI),
            )
    engine.begin.assert_not_called()


def test_interior_date_gap_allowed_with_explicit_override():
    """A human who has reviewed the gap and says so explicitly can still
    promote the full declared range - the override is opt-in, not implicit."""
    conn, engine = MagicMock(), MagicMock()
    engine.begin.return_value.__enter__.return_value = conn
    with (
        patch.object(
            db,
            "missing_trading_days",
            side_effect=lambda engine, lo, hi, payload_days: sorted(
                {INTERIOR_LO, INTERIOR_MID, INTERIOR_HI} - payload_days
            ),
            create=True,
        ),
        patch.object(db, "resolve_campaign_id", return_value=None),
        patch.object(db, "_dual_write_daily_scores"),
    ):
        n = db.upsert_daily_scores(
            engine,
            interior_gap_payload(),
            replace=True,
            owned_factors=["roe"],
            allow_unwritten_owned_factors=True,
            replace_date_range=(INTERIOR_LO, INTERIOR_HI),
            allow_interior_gaps=True,
        )
    assert n == 2
    engine.begin.assert_called_once()
    deletes = [c for c in conn.execute.call_args_list if str(c.args[0]).startswith("DELETE")]
    assert len(deletes) == 1
    assert deletes[0].args[1] == {
        "date_min": INTERIOR_LO,
        "date_max": INTERIOR_HI,
        "factor_names": ["roe"],
    }


@pytest.mark.parametrize("empty", [False, True])
def test_iceberg_mirror_uses_same_range_and_factor_scope(empty):
    pytest.importorskip("pyiceberg")
    from pyiceberg.expressions import And, GreaterThanOrEqual, In, LessThanOrEqual

    table = MagicMock()
    df = payload().iloc[:0] if empty else payload()
    with patch.object(lakehouse, "ensure_daily_scores_table", return_value=table):
        lakehouse.write_daily_scores(df, owned_factors=["roe"], replace_date_range=(LO, HI))
    table.overwrite.assert_called_once()
    assert table.overwrite.call_args.args[0].num_rows == len(df)
    assert table.overwrite.call_args.kwargs["overwrite_filter"] == And(
        And(GreaterThanOrEqual("score_date", LO), LessThanOrEqual("score_date", HI)),
        In("factor_name", ["roe"]),
    )


@pytest.mark.parametrize(
    "column,value",
    [
        ("score_date", date(2025, 8, 31)),
        ("score_date", date(2025, 9, 4)),
        ("score_date", None),
        ("factor_name", "unowned"),
    ],
)
def test_direct_iceberg_range_write_rejects_out_of_scope_rows(column, value):
    pytest.importorskip("pyiceberg")
    df = payload()
    df.loc[0, column] = value
    with patch.object(lakehouse, "ensure_daily_scores_table") as table:
        with pytest.raises(ValueError, match="scope"):
            lakehouse.write_daily_scores(df, owned_factors=["roe"], replace_date_range=(LO, HI))
    table.assert_not_called()


def test_missing_trading_days_skips_the_database_when_nothing_is_suspect():
    """#401 review: cost must scale with how suspicious the payload looks,
    not with how wide the range is. A weekend can never be a trading day and
    a weekday the payload covers needs no answer, so a payload covering every
    weekday in the range must issue NO query at all.

    This is the property that makes the guard affordable on a decade-scale
    gold promotion. The range-scan form this replaced ran as a slow seq scan
    over a decade-scale range, and chunking it by year bought nothing."""
    from datetime import date as _date
    from datetime import timedelta as _td

    lo, hi = _date(2013, 1, 1), _date(2026, 9, 16)
    every_weekday = set()
    d = lo
    while d <= hi:
        if d.weekday() < 5:
            every_weekday.add(d)
        d += _td(days=1)

    engine = MagicMock()
    out = db.missing_trading_days(engine, lo, hi, every_weekday)
    assert out == []
    engine.connect.assert_not_called(), "a fully covered payload must not touch the DB"


def test_missing_trading_days_only_asks_about_uncovered_weekdays():
    """Weekends are never suspects, and a covered weekday is never asked
    about. The query must receive exactly the uncovered weekdays."""
    from datetime import date as _date

    # 2026-09-14 Mon .. 2026-09-20 Sun. Payload covers Mon/Tue/Thu/Fri.
    lo, hi = _date(2026, 9, 14), _date(2026, 9, 20)
    payload = {_date(2026, 9, 14), _date(2026, 9, 15), _date(2026, 9, 17), _date(2026, 9, 18)}

    conn, engine = MagicMock(), MagicMock()
    engine.connect.return_value.__enter__.return_value = conn
    # 09-18 is the canary (max payload date) and must come back, else the
    # calendar probe raises. 09-16 is the genuine gap.
    conn.execute.return_value.fetchall.return_value = [
        (_date(2026, 9, 16),),
        (_date(2026, 9, 18),),
    ]

    out = db.missing_trading_days(engine, lo, hi, payload)

    assert out == [_date(2026, 9, 16)]
    params = conn.execute.call_args.args[1]
    # The uncovered WEEKDAY (Wed 09-16) plus the canary. Sat 09-19 and Sun
    # 09-20 are excluded without a query, and the covered days are not
    # asked about as suspects.
    assert params["suspects"] == [_date(2026, 9, 16), _date(2026, 9, 18)]


def test_missing_trading_days_raises_if_the_calendar_probe_comes_back_empty():
    """#401 follow-up: the ts_event equality only matches while daily equity
    rows are midnight-stamped. If that ever stops holding the query matches
    nothing, the caller reads zero gaps, and replace_date_range deletes
    in-scope dates with nothing to replace them - a silent FALSE PASS on a
    delete path. A date the payload itself covers is sent along as a canary
    and must come back."""
    from datetime import date as _date

    lo, hi = _date(2026, 9, 14), _date(2026, 9, 18)
    payload = {_date(2026, 9, 14), _date(2026, 9, 15), _date(2026, 9, 17)}

    conn, engine = MagicMock(), MagicMock()
    engine.connect.return_value.__enter__.return_value = conn
    # Simulate the broken invariant: the equality matches nothing at all.
    conn.execute.return_value.fetchall.return_value = []

    with pytest.raises(ValueError, match="calendar probe failed"):
        db.missing_trading_days(engine, lo, hi, payload)


def test_missing_trading_days_does_not_report_the_canary_as_a_gap():
    """The canary is a date the payload COVERS, so it must never be returned
    as missing even though it is sent to the query."""
    from datetime import date as _date

    lo, hi = _date(2026, 9, 14), _date(2026, 9, 18)
    payload = {_date(2026, 9, 14), _date(2026, 9, 15), _date(2026, 9, 17)}

    conn, engine = MagicMock(), MagicMock()
    engine.connect.return_value.__enter__.return_value = conn
    # Canary (09-17, the max payload date) comes back, plus a real gap 09-16.
    conn.execute.return_value.fetchall.return_value = [
        (_date(2026, 9, 16),),
        (_date(2026, 9, 17),),
    ]

    out = db.missing_trading_days(engine, lo, hi, payload)
    assert out == [_date(2026, 9, 16)], "canary must not be reported as a gap"
