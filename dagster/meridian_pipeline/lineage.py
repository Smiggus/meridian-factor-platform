"""Write-campaign lineage: who wrote a factor row, from what input vintage, with what code
version (Meridian #294 / #295).

On 2026-08-01 nobody could say what had written a given row. `factors.daily_scores` had one
column for it, `computed_at timestamptz DEFAULT now()`, and `db.copy_replace_daily_scores` left
that column out of its COPY list, so the DEFAULT stamped write time and not compute time.
Working out which historical write produced which rows meant clustering `computed_at` values,
which is unreliable. When a bad input turned up there was no record of which rows had consumed
it, so the only safe recompute was the full history. Lineage is what lets that become a scoped
recompute.

A campaign is one write episode: a Dagster asset run, a burst promote, a manual backfill script.
It is recorded once in `factors.write_campaigns`, and each row it wrote carries a `campaign_id
bigint`, so full provenance is a join away.

A write gets its campaign in one of three ways, tried in this order. The caller can pass
`campaign_id=` to a `db.upsert_*` or `db.copy_replace_*` helper, as the batch scripts do. Or it
can wrap its work in `open_campaign(...)`, which binds the campaign to a ContextVar so every
write inside the block picks it up without a change at the call site. Scripts and assets that
know their own input vintage usually do this. If neither happened, the first write in the process
opens a campaign from the environment (writer, code version, run id, host, pid, argv) with an
empty `input_vintage`, so no write is anonymous.

A lineage failure never fails a write. Every public entry point catches its own exceptions and
returns None, and the row then goes in with a NULL campaign_id, as it did before this module
existed. `MERIDIAN_LINEAGE_DISABLED=1` turns the whole thing off. `MERIDIAN_LINEAGE_AUTO=0`
keeps explicit and ambient campaigns and disables only the implicit one.

`input_vintage` is jsonb keyed by source dataset:

    {
      "ohlcv.prices": {"max_inserted_at": "2026-07-31T23:04:11+00:00",
                       "reader": "medallion:silver", "n_rows": 4123456},
      "fundamentals.income_statement": {"max_filing_date": "2026-07-30"},
      "factors.daily_scores": {"campaigns": [412, 413]}
    }

A campaign reads several datasets with independent vintages, and a single scalar would hide
which one was stale. The reader is recorded too ("medallion:silver", "flat-parquet",
"postgresql"), because a silent fallback between read paths does not show in the values. The
incident asked "which scores were computed from OHLCV read before the adj_close fix landed", so
the writer also derives a scalar `input_vintage_max timestamptz`, the latest parseable timestamp
anywhere in the jsonb. `WHERE input_vintage_max < '2026-08-01'` then works and is B-tree
indexable. `vintage_scalar()` computes it at write time from the same jsonb, so the two agree.

`source_vintage()` builds one source's entry and `merge_vintages()` combines entries. Both are
pure and unit tested.
"""

import contextlib
import datetime as _dt
import logging
import os
import socket
import sys
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Optional

_log = logging.getLogger(__name__)

# Campaign id for every row written before lineage existed. The migration adds `campaign_id
# bigint DEFAULT 0` and drops the default, so old rows read back as 0 without a table rewrite
# (dagster/db/migrations/2026-08-02_write_campaigns.sql). Their real provenance cannot be
# recovered, and 0 at least makes "unknown" something you can query for.
PRE_LINEAGE_CAMPAIGN_ID = 0

WRITER_DAGSTER_ASSET = "dagster_asset"
WRITER_BURST_PROMOTE = "burst_promote"
WRITER_MANUAL_SCRIPT = "manual_script"
WRITER_PRE_LINEAGE = "pre_lineage"

# Results a campaign's admission gate can record. The column is free text; these are the values
# the shipped writers use.
GATE_PASS = "pass"
GATE_FAIL = "fail"
GATE_FORCED = "forced"
GATE_UNVERIFIED = "unverified"
GATE_ABORTED = "aborted"


# Environment / kill switches


def _truthy(value: Optional[str]) -> bool:
    return (value or "").strip().lower() in ("1", "true", "yes", "on")


def _falsy(value: Optional[str]) -> bool:
    return (value or "").strip().lower() in ("0", "false", "no", "off")


def lineage_enabled() -> bool:
    """False when MERIDIAN_LINEAGE_DISABLED is set - the global kill switch.

    With lineage off, every helper here returns None and the db writers fall
    back to their exact pre-#294 SQL. Provided so a lineage bug can never be
    the reason a nightly scoring run fails.
    """
    return not _truthy(os.environ.get("MERIDIAN_LINEAGE_DISABLED"))


def auto_campaign_enabled() -> bool:
    """False when MERIDIAN_LINEAGE_AUTO=0 - disables only the IMPLICIT campaign.

    Explicit `campaign_id=` and ambient `open_campaign()` still work. Useful for
    tests, and for a process that would otherwise mint a meaningless campaign.
    """
    return lineage_enabled() and not _falsy(os.environ.get("MERIDIAN_LINEAGE_AUTO"))


def resolve_code_version() -> str:
    """The code version this process is running, best effort, never raising.

    Order: MERIDIAN_CODE_VERSION (set it explicitly if you care) -> the image
    tag the k3s deployment was rolled with -> the installed/declared package
    version -> "unknown". NOT a git call: run pods have no git
    checkout, and shelling out on a hot path to learn something the image
    already knows is the wrong trade.
    """
    for env_var in ("MERIDIAN_CODE_VERSION", "MERIDIAN_IMAGE_TAG", "IMAGE_TAG"):
        value = (os.environ.get(env_var) or "").strip()
        if value:
            return value
    try:
        from importlib.metadata import version

        return version("meridian")
    except Exception:
        pass
    # Fall back to the declared version in pyproject.toml (dev checkouts, where
    # the package is not pip-installed).
    try:
        import tomllib
        from pathlib import Path

        pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
        if pyproject.exists():
            with pyproject.open("rb") as fh:
                declared = tomllib.load(fh).get("project", {}).get("version")
            if declared:
                return str(declared)
    except Exception:
        pass
    return "unknown"


def resolve_run_id() -> Optional[str]:
    """The orchestrator run id, if the environment advertises one.

    Dagster does NOT export the run id into the step-pod environment (verified
    against dagster 1.13 / dagster-k8s), so the in-cluster asset path binds it
    explicitly via `bind_dagster_run(context)`. These env vars cover the burst
    and manual paths, which do set them.
    """
    for env_var in ("MERIDIAN_RUN_ID", "DAGSTER_RUN_ID", "BURST_RUN_ID"):
        value = (os.environ.get(env_var) or "").strip()
        if value:
            return value
    return None


def _process_detail() -> dict:
    """Cheap process fingerprint stored on every campaign's `detail` jsonb.

    Not provenance in the strict sense, but it is what lets an operator find
    the pod/log of a campaign they are suspicious of, which is most of the job.
    """
    detail: dict[str, Any] = {"pid": os.getpid()}
    try:
        detail["host"] = socket.gethostname()
    except Exception:
        pass
    try:
        detail["argv0"] = os.path.basename(sys.argv[0]) if sys.argv else None
    except Exception:
        pass
    return detail


# input_vintage construction (pure helpers)


def _iso(value: Any) -> Any:
    """Render dates/datetimes as ISO-8601 strings; pass anything else through.

    Naive datetimes are treated as UTC - every timestamp Meridian reads from PG
    is timestamptz and every one it constructs is UTC, so the alternative
    (dropping the offset) loses information for no benefit.
    """
    if isinstance(value, _dt.datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=_dt.timezone.utc)
        return value.astimezone(_dt.timezone.utc).isoformat()
    if isinstance(value, _dt.date):
        return value.isoformat()
    return value


def source_vintage(source: str, **fields: Any) -> dict:
    """One source dataset's vintage entry, e.g.

        source_vintage("ohlcv.prices",
                       max_inserted_at=ts, reader="medallion:silver", n_rows=n)
        -> {"ohlcv.prices": {"max_inserted_at": "...", "reader": "...",
                             "n_rows": 4123456}}

    Keys are free-form on purpose: what identifies a vintage differs per source
    (an ingest timestamp for ohlcv.prices, a filing_date for fundamentals, a set
    of upstream campaign ids for a derived table). Locking it to one column name
    would force the callers with a better answer to throw it away. Fields whose
    value is None are dropped so an unknown never masquerades as a recorded one.
    """
    payload = {k: _iso(v) for k, v in fields.items() if v is not None}
    return {source: payload}


def merge_vintages(*parts: Optional[dict]) -> dict:
    """Shallow-merge per-source vintage dicts. Later parts win on conflict."""
    merged: dict[str, Any] = {}
    for part in parts:
        if not part:
            continue
        for key, value in part.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = {**merged[key], **value}
            else:
                merged[key] = value
    return merged


def input_campaign_manifest(rows: Any) -> dict:
    """Exact, fail-closed manifest of campaign stamps in a derived input frame.

    Derived writers attach this payload under their source-table key in
    ``input_vintage``.  ``campaigns`` alone is insufficient: a
    downstream hard gate must also see whether any consumed row was NULL-stamped or
    pre-lineage (0), because either makes the input set non-auditable.
    """
    try:
        values = list(rows["campaign_id"].tolist())
    except Exception:
        try:
            n_rows = len(rows)
        except Exception:
            n_rows = 0
        return {
            "campaigns": [],
            "row_count": n_rows,
            "stamped_row_count": 0,
            "unstamped_row_count": n_rows,
            "pre_lineage_row_count": 0,
            "complete": False,
        }

    campaigns: set[int] = set()
    unstamped = 0
    pre_lineage = 0
    for value in values:
        if value is None or (isinstance(value, float) and value != value):
            unstamped += 1
            continue
        try:
            campaign_id = int(value)
        except (TypeError, ValueError):
            unstamped += 1
            continue
        if campaign_id <= PRE_LINEAGE_CAMPAIGN_ID:
            pre_lineage += 1
            continue
        campaigns.add(campaign_id)
    stamped = len(values) - unstamped - pre_lineage
    return {
        "campaigns": sorted(campaigns),
        "row_count": len(values),
        "stamped_row_count": stamped,
        "unstamped_row_count": unstamped,
        "pre_lineage_row_count": pre_lineage,
        "complete": bool(values) and unstamped == 0 and pre_lineage == 0,
    }


def _parse_ts(value: Any) -> Optional[_dt.datetime]:
    """Best-effort ISO-8601 timestamp/date parse, UTC-normalised, else None."""
    if isinstance(value, _dt.datetime):
        return value if value.tzinfo else value.replace(tzinfo=_dt.timezone.utc)
    if isinstance(value, _dt.date):
        return _dt.datetime(value.year, value.month, value.day, tzinfo=_dt.timezone.utc)
    if not isinstance(value, str):
        return None
    text_value = value.strip()
    if len(text_value) < 8:
        return None
    try:
        parsed = _dt.datetime.fromisoformat(text_value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=_dt.timezone.utc)
    return parsed.astimezone(_dt.timezone.utc)


def vintage_scalar(vintage: Optional[dict]) -> Optional[_dt.datetime]:
    """The single most-recent timestamp anywhere in an input_vintage dict.

    Denormalised into `factors.write_campaigns.input_vintage_max` so the
    "computed before or after the upstream fix?" question is a one-line,
    indexable predicate. MAX (not MIN) because the question is always "could
    this campaign possibly have seen the fix" - a campaign whose newest input
    predates the fix definitely could not have, which is the sound direction to
    be conservative in.
    """
    best: Optional[_dt.datetime] = None
    for entry in (vintage or {}).values():
        candidates = entry.values() if isinstance(entry, dict) else [entry]
        for candidate in candidates:
            parsed = _parse_ts(candidate)
            if parsed is not None and (best is None or parsed > best):
                best = parsed
    return best


# The ambient campaign


@dataclass
class Campaign:
    """A bound write campaign. `campaign_id` is None when the DB insert failed
    or lineage is disabled - callers must treat that as "write without a stamp",
    never as an error."""

    campaign_id: Optional[int]
    writer: str
    code_version: Optional[str] = None
    run_id: Optional[str] = None
    input_vintage: dict = field(default_factory=dict)
    started_at: Optional[_dt.datetime] = None
    gate_result: Optional[str] = None
    detail: dict = field(default_factory=dict)


_current: ContextVar[Optional[Campaign]] = ContextVar("meridian_write_campaign", default=None)

# Per-process implicit campaign, memoised per engine URL so a process writing
# through one engine mints exactly one campaign row no matter how many
# partitions or assets it writes.
_implicit: dict[str, Optional[int]] = {}

# Run metadata bound by bind_dagster_run(), consumed when the implicit campaign
# is minted. Kept separate from _current: binding a run id must not itself
# create a campaign row (an asset run that writes nothing should leave no
# trace).
_bound_run: dict[str, Optional[str]] = {"run_id": None, "writer": None}


def bind_dagster_run(context, writer: str = WRITER_DAGSTER_ASSET) -> None:
    """Record the Dagster run id for this process so implicit campaigns carry it.

    Call once near the top of an asset, right after the engine is obtained.
    Idempotent, never raises, and does NOT touch the database - it only seeds
    the metadata the first write will use. Dagster does not export the run id
    into the step-pod environment, which is why this explicit call exists.
    """
    if not lineage_enabled():
        return
    try:
        run_id = getattr(context, "run_id", None)
        _bound_run["run_id"] = str(run_id) if run_id else None
        _bound_run["writer"] = writer
    except Exception:  # pragma: no cover - defensive
        pass


def current_campaign() -> Optional[Campaign]:
    """The ambient campaign bound by `open_campaign`, or None."""
    if not lineage_enabled():
        return None
    return _current.get()


def current_campaign_id(engine=None) -> Optional[int]:
    """The campaign id to stamp on a write happening right now.

    Resolution order: the ambient campaign -> the process's implicit campaign
    (minted on first use, needs `engine`) -> None. Never raises; None simply
    means the row is written with a NULL campaign_id, exactly as it was before
    lineage existed.
    """
    if not lineage_enabled():
        return None
    try:
        bound = _current.get()
        if bound is not None:
            return bound.campaign_id
        if engine is None or not auto_campaign_enabled():
            return None
        return _implicit_campaign_id(engine)
    except Exception as exc:  # pragma: no cover - defensive
        _log.debug("lineage: current_campaign_id degraded to None: %s", exc)
        return None


def _engine_key(engine) -> str:
    try:
        return str(getattr(engine, "url", None) or id(engine))
    except Exception:  # pragma: no cover - defensive
        return str(id(engine))


def _implicit_campaign_id(engine) -> Optional[int]:
    """Mint (once per process+engine) the low-fidelity fallback campaign.

    Records NO input_vintage: a path that never told us what it
    read gets an empty jsonb, not a guess. The value here is `writer` +
    `code_version` + `run_id` + the process fingerprint, which is already
    strictly more than the zero bits a pre-#294 row carried.
    """
    key = _engine_key(engine)
    if key in _implicit:
        return _implicit[key]
    campaign_id: Optional[int] = None
    try:
        from meridian_pipeline import db

        writer = (os.environ.get("MERIDIAN_WRITER") or "").strip() or (
            _bound_run["writer"] or WRITER_MANUAL_SCRIPT
        )
        campaign_id = db.record_write_campaign(
            engine,
            writer=writer,
            code_version=resolve_code_version(),
            run_id=_bound_run["run_id"] or resolve_run_id(),
            input_vintage={},
            detail={**_process_detail(), "campaign_origin": "implicit"},
        )
    except Exception as exc:  # pragma: no cover - defensive
        _log.debug("lineage: implicit campaign not recorded: %s", exc)
        campaign_id = None
    _implicit[key] = campaign_id
    if campaign_id is not None:
        _register_implicit_close(engine, campaign_id)
    return campaign_id


def _register_implicit_close(engine, campaign_id: int) -> None:
    """Close the implicit campaign when the process exits.

    An implicit campaign's lifetime IS the process's: it is minted on the first
    write and memoised per engine, and a Dagster step pod has no other natural
    close point. Without this every implicit row stayed open forever:
    `finished_at` was never populated, so the table recorded that writes had
    STARTED and never that any finished. That is most of the value the lineage
    work was supposed to add: you could not tell a running write from a
    completed one from a crashed one, and a reader scanning the open rows had
    no way to know whether to be alarmed.

    Registering at exit rather than after each write is on purpose. It keeps the
    one-row-per-process shape (the whole point of memoising), and it makes
    `finished_at IS NULL` MEAN something afterwards: the process died before it
    could close, since atexit does not run on SIGKILL or a hard OOM. An
    always-null column carries no signal; a null that implies a kill does.

    Never raises: lineage is observability, and it must not be able to fail the
    write it is observing.
    """
    import atexit

    def _close() -> None:
        try:
            from meridian_pipeline import db

            db.finish_write_campaign(engine, campaign_id, gate_result=GATE_UNVERIFIED)
        except Exception as exc:  # pragma: no cover - defensive
            _log.debug("lineage: implicit campaign %s not closed: %s", campaign_id, exc)

    try:
        atexit.register(_close)
    except Exception as exc:  # pragma: no cover - defensive
        _log.debug("lineage: could not register implicit close: %s", exc)


@contextlib.contextmanager
def open_campaign(
    engine,
    writer: str,
    *,
    code_version: Optional[str] = None,
    run_id: Optional[str] = None,
    input_vintage: Optional[dict] = None,
    detail: Optional[dict] = None,
    gate_result: Optional[str] = None,
):
    """Open a campaign, bind it ambiently, and close it out on exit.

    Every `db.upsert_*` / `db.copy_replace_*` write inside the block is stamped
    with this campaign without any callsite change. On exit the row's
    `finished_at` is set, plus `gate_result` (from `set_gate_result()` inside
    the block, or the `gate_result=` argument) and any `detail` accumulated.

    An exception inside the block closes the campaign with `gate_result='fail'`
    and re-raises - a half-written campaign is exactly the thing an operator
    needs to be able to find later, so it must not vanish.

    Yields the `Campaign`, whose `campaign_id` may be None if the insert failed.
    That is not an error: the block still runs and its writes are simply
    unstamped.
    """
    campaign = Campaign(
        campaign_id=None,
        writer=writer,
        code_version=code_version or resolve_code_version(),
        run_id=run_id or resolve_run_id(),
        input_vintage=dict(input_vintage or {}),
        started_at=_dt.datetime.now(_dt.timezone.utc),
        gate_result=gate_result,
        detail={**_process_detail(), **(detail or {}), "campaign_origin": "explicit"},
    )
    if lineage_enabled():
        try:
            from meridian_pipeline import db

            campaign.campaign_id = db.record_write_campaign(
                engine,
                writer=campaign.writer,
                code_version=campaign.code_version,
                run_id=campaign.run_id,
                input_vintage=campaign.input_vintage,
                started_at=campaign.started_at,
                gate_result=campaign.gate_result,
                detail=campaign.detail,
            )
        except Exception as exc:  # pragma: no cover - defensive
            _log.warning("lineage: could not record campaign for %s: %s", writer, exc)

    token = _current.set(campaign)
    failed = False
    try:
        yield campaign
    except BaseException:
        failed = True
        raise
    finally:
        _current.reset(token)
        if campaign.campaign_id is not None:
            try:
                from meridian_pipeline import db

                db.finish_write_campaign(
                    engine,
                    campaign.campaign_id,
                    gate_result=(GATE_FAIL if failed else campaign.gate_result),
                    input_vintage=campaign.input_vintage or None,
                    detail=campaign.detail or None,
                )
            except Exception as exc:  # pragma: no cover - defensive
                _log.warning("lineage: could not close campaign %s: %s", campaign.campaign_id, exc)


def begin_campaign(
    engine,
    writer: str,
    *,
    code_version: Optional[str] = None,
    run_id: Optional[str] = None,
    input_vintage: Optional[dict] = None,
    detail: Optional[dict] = None,
) -> Campaign:
    """Non-nesting form of `open_campaign` for a script with a flat `main()`.

    Binds the campaign for the rest of the process (no token to reset) and
    returns it; pair with `end_campaign()`. Prefer `open_campaign` wherever a
    `with` block is natural - it closes the campaign even on an exception,
    which this cannot. Exists so a long procedural backfill does not have to be
    re-indented (and silently re-reviewed) just to gain lineage.
    """
    campaign = Campaign(
        campaign_id=None,
        writer=writer,
        code_version=code_version or resolve_code_version(),
        run_id=run_id or resolve_run_id(),
        input_vintage=dict(input_vintage or {}),
        started_at=_dt.datetime.now(_dt.timezone.utc),
        detail={**_process_detail(), **(detail or {}), "campaign_origin": "explicit"},
    )
    if lineage_enabled():
        try:
            from meridian_pipeline import db

            campaign.campaign_id = db.record_write_campaign(
                engine,
                writer=campaign.writer,
                code_version=campaign.code_version,
                run_id=campaign.run_id,
                input_vintage=campaign.input_vintage,
                started_at=campaign.started_at,
                detail=campaign.detail,
            )
        except Exception as exc:  # pragma: no cover - defensive
            _log.warning("lineage: could not record campaign for %s: %s", writer, exc)
    _current.set(campaign)
    return campaign


def end_campaign(engine, campaign: Optional[Campaign], gate_result: Optional[str] = None) -> None:
    """Close a `begin_campaign()` campaign. Never raises."""
    if campaign is None or campaign.campaign_id is None:
        return
    try:
        from meridian_pipeline import db

        db.finish_write_campaign(
            engine,
            campaign.campaign_id,
            gate_result=gate_result or campaign.gate_result,
            input_vintage=campaign.input_vintage or None,
            detail=campaign.detail or None,
        )
    except Exception as exc:  # pragma: no cover - defensive
        _log.warning("lineage: could not close campaign %s: %s", campaign.campaign_id, exc)
    finally:
        _current.set(None)


def set_gate_result(result: str, **detail: Any) -> None:
    """Record the admission gate's verdict on the ambient campaign.

    "Did anything gate this campaign into gold?" is the question that makes
    `final` a property a row can have. Call this from inside `open_campaign`
    once the parity / coverage / verification gate has run; the value is
    persisted when the campaign closes.
    """
    campaign = current_campaign()
    if campaign is None:
        return
    campaign.gate_result = result
    if detail:
        campaign.detail = {**campaign.detail, **{k: _iso(v) for k, v in detail.items()}}


def add_input_vintage(*parts: Optional[dict]) -> None:
    """Merge more source vintages into the ambient campaign after it opened.

    For campaigns that only learn a source's vintage mid-flight (a promote that
    discovers the staged partitions' stamps as it streams them, for instance).
    Persisted when the campaign closes.
    """
    campaign = current_campaign()
    if campaign is None:
        return
    campaign.input_vintage = merge_vintages(campaign.input_vintage, *parts)


def attach_input_vintage(engine, campaign_id: Optional[int], *parts: Optional[dict]) -> None:
    """Persist late-discovered input provenance for an already-resolved campaign.

    This is the implicit-campaign counterpart to ``add_input_vintage``.  It is
    used by derived writers after resolving their output campaign but before
    writing rows, so their input manifest is durable without falsely closing
    the campaign.  Failures are intentionally swallowed by the DB helper;
    consumers treat a missing/incomplete manifest as a hard-gate denial.
    """
    if campaign_id is None:
        return
    merged = merge_vintages(*parts)
    if not merged:
        return
    try:
        from meridian_pipeline import db

        db.add_write_campaign_input_vintage(engine, campaign_id, merged)
    except Exception as exc:  # pragma: no cover - defensive
        _log.warning("lineage: could not attach input vintage to campaign %s: %s", campaign_id, exc)


def reset_for_tests() -> None:
    """Drop all process-level lineage state. Test-support only."""
    _implicit.clear()
    _bound_run["run_id"] = None
    _bound_run["writer"] = None
    _current.set(None)
