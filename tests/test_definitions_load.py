"""The code location loads and its wiring is self-consistent."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dagster"))

from meridian_pipeline.definitions import defs  # noqa: E402


def test_definitions_resolve_without_a_database():
    repo = defs.get_repository_def()
    assert {j.name for j in repo.get_all_jobs()} >= {
        "daily_factor_core_job",
        "gold_pg_sync_job",
        "backfill_verification_job",
    }
    assert len(list(repo.sensor_defs)) == 11


def test_every_sensor_targets_a_registered_job_or_has_none():
    repo = defs.get_repository_def()
    job_names = {j.name for j in repo.get_all_jobs()}
    for sensor in repo.sensor_defs:
        for target in sensor.targets or []:
            assert target.job_name in job_names, (sensor.name, target.job_name)


def test_asset_checks_are_registered():
    repo = defs.get_repository_def()
    names = {k.name for k in repo.asset_checks_defs_by_key}
    assert {"check_ohlcv_read_contract", "check_trading_day_invariant"} <= names
