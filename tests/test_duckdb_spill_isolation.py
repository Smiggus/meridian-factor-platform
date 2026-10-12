"""Each process needs its own DuckDB spill directory (#284 burst run).

DuckDB names temp files after the database instance; an unnamed in-memory DB
gets "DEFAULT". With a SHARED temp_directory two worker processes both write
duckdb_temp_storage_DEFAULT-0.tmp and truncate each other, which surfaces as
"Could not read enough bytes" on read-back. Observed live mid-run.
"""

import re
import sys
from pathlib import Path

_DAGSTER_DIR = Path(__file__).resolve().parents[1] / "dagster"
if str(_DAGSTER_DIR) not in sys.path:
    sys.path.insert(0, str(_DAGSTER_DIR))

_SRC = (_DAGSTER_DIR / "meridian_pipeline" / "parquet_io.py").read_text()


def _tune_body() -> str:  # body of _apply_duckdb_memory_limit
    i = _SRC.index("def _apply_duckdb_memory_limit")
    return _SRC[i : _SRC.index("\ndef ", i + 1)]


def test_spill_dir_is_not_a_bare_shared_constant():
    """The literal that caused the collision must not come back."""
    assert "SET temp_directory='/tmp/duckdb_spill'" not in _SRC


def test_spill_dir_is_process_scoped():
    body = _tune_body()
    assert "getpid()" in body, "spill directory is not per-process"
    assert "temp_directory=" in body


def test_spill_dir_is_created_before_use():
    """DuckDB will not create a missing temp_directory; it errors instead."""
    body = _tune_body()
    assert "makedirs" in body
    assert body.index("makedirs") < body.index("SET temp_directory="), (
        "the directory must be created BEFORE it is handed to DuckDB"
    )


def test_root_is_overridable():
    """A box with a small or wrong-device /tmp needs to redirect this."""
    assert "MERIDIAN_DUCKDB_SPILL_ROOT" in _SRC
    assert re.search(r"DUCKDB_SPILL_ROOT[^\n]*os\.environ\.get", _SRC)


def test_two_processes_would_get_different_paths():
    """Simulate the collision: the path must be a function of the PID."""
    import os

    from meridian_pipeline.parquet_io import DUCKDB_SPILL_ROOT

    p1 = os.path.join(DUCKDB_SPILL_ROOT, str(1234))
    p2 = os.path.join(DUCKDB_SPILL_ROOT, str(5678))
    assert p1 != p2
    assert p1.startswith(DUCKDB_SPILL_ROOT)


def test_rationale_records_the_live_failure():
    """Keep the observed error in the file so this is not re-litigated."""
    body = _tune_body()
    assert "Could not read enough bytes" in body
    assert "DEFAULT" in body
