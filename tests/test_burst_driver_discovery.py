"""Guard the burst Dask driver's partition-discovery wiring (#280 follow-up, audit E1).

`burst_common.discover_partitions` grew a `source` parameter in !446 so a burst run
can discover from the trading CALENDAR rather than from gold. Gold-sourced discovery
is an inner join against `factors.daily_scores`, so it is structurally unable to see
a range gold has zero rows for (a history gap where the calendar still has trading days).

`scripts/burst/rescore_ret1d_close.py` exposed the flag immediately. The Dask driver
did NOT: it called `discover_partitions(cfg, start, end)`, took the "gold" default,
and on an empty result logged "nothing to do" and returned 0. A history fill became a
silent no-op with a success exit code - the fail-open shape this repo has hit
repeatedly (#266 preflight-vs-action, #280 gold-derived calendar, the FORMAT csv NULL
crash, the NULL delisting_date guard).

These are source-level assertions rather than an end-to-end run because the driver's
main() builds a real Dask LocalCluster and opens a Pipes session. The point is to fail
loudly if the wiring regresses, which is cheap and exactly what was missing.
"""

import re
from pathlib import Path

import pytest

_DRIVER = Path(__file__).resolve().parents[1] / "scripts" / "burst" / "pipes_dask_driver.py"
_COMMON = Path(__file__).resolve().parents[1] / "scripts" / "burst" / "burst_common.py"

if not _DRIVER.exists():  # pragma: no cover - layout guard
    pytest.skip("burst driver not present", allow_module_level=True)

_DRIVER_SRC = _DRIVER.read_text()


def test_driver_exposes_discover_from_flag():
    """The driver must offer the same discovery choice as the rescore CLI."""
    assert "--discover-from" in _DRIVER_SRC, (
        "pipes_dask_driver.py must expose --discover-from; without it a burst fill "
        "cannot target a range gold has no rows for."
    )
    # Both choices must be selectable, not just documented.
    assert '"gold"' in _DRIVER_SRC and '"calendar"' in _DRIVER_SRC


def test_driver_passes_source_through_to_discover_partitions():
    """Regression guard: the call must not silently take the gold default."""
    calls = re.findall(r"discover_partitions\((.*?)\)", _DRIVER_SRC, flags=re.S)
    assert calls, "expected at least one discover_partitions call in the driver"
    # The primary discovery call must forward the CLI choice. An explicit
    # source="calendar" (the disambiguation probe) is also acceptable.
    assert any("source=args.discover_from" in c for c in calls), (
        "the driver's discovery call must pass source=args.discover_from; taking the "
        "gold default is what makes a fill over an empty gold range a silent no-op"
    )
    for c in calls:
        assert "source=" in c, (
            f"every discover_partitions call must be explicit about its source, got: {c.strip()!r}"
        )


def test_driver_aborts_loudly_when_gold_empty_but_calendar_is_not():
    """An ambiguous empty result must not exit 0 with 'nothing to do'."""
    assert "ABORT" in _DRIVER_SRC, (
        "the gold-empty-but-calendar-populated case must abort loudly, not log a "
        "benign 'nothing to do' and return 0"
    )
    assert "--discover-from calendar" in _DRIVER_SRC, (
        "the abort message must name the remedy so the operator knows what to do"
    )
    # The abort must be a non-zero exit. Find the abort block and assert a return 1
    # follows it before the benign return 0.
    idx = _DRIVER_SRC.index("ABORT")
    tail = _DRIVER_SRC[idx : idx + 800]
    assert "return 1" in tail, "the abort path must return a non-zero exit code"


def test_discover_partitions_still_defaults_to_gold():
    """Parity semantics must not shift: existing callers keep gold behaviour."""
    src = _COMMON.read_text()
    sig = re.search(r"def discover_partitions\((.*?)\)\s*->", src, flags=re.S)
    assert sig, "could not locate discover_partitions signature"
    assert 'source: str = "gold"' in sig.group(1), (
        "discover_partitions must keep defaulting to gold so the re-score parity "
        "gate is unchanged; calendar mode is strictly opt-in"
    )
