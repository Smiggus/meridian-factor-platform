"""
pg_identity.py - the one place a process names itself to Postgres.

Meridian#428 item 4 gave every engine a libpq ``application_name`` so load on
the shared db-host Postgres can be attributed in ``pg_stat_activity``. Meridian#467
item 5 removed the seven copies of the expression that did it.

Stdlib only, on purpose: the API (which does not install dagster), the Dagster
code location and the burst kernels all import this module, the first two via
the ``dagster/`` sys.path entry that ``factor_store`` and ``api/factor_catalogue``
already add.

Defaults are per role so replay load is separable from nightly load:

    meridian-api      FastAPI process
    meridian-factors  Dagster code location (nightly and sensor runs)
    meridian-burst    burst kernels (scripts/burst/*), replay and re-score

``MERIDIAN_PG_APPLICATION_NAME`` overrides the default for the whole process
whatever its role; an empty or blank override is ignored.
"""

import os

APPLICATION_NAME_ENV = "MERIDIAN_PG_APPLICATION_NAME"

API_APPLICATION_NAME = "meridian-api"
PIPELINE_APPLICATION_NAME = "meridian-factors"
BURST_APPLICATION_NAME = "meridian-burst"


def pg_application_name(default: str) -> str:
    """libpq application_name for an engine: env override, else ``default``.

    Postgres truncates the value at 63 bytes; an empty override is ignored.
    """
    return (os.environ.get(APPLICATION_NAME_ENV) or "").strip() or default


def pg_connect_args(default: str) -> dict:
    """``connect_args`` for ``create_engine`` carrying the application_name."""
    return {"application_name": pg_application_name(default)}


def adopt_default_application_name(default: str) -> None:
    """Make ``default`` this process's name for every engine, unless already set.

    Burst entry points call this from their ``__main__`` block: the Dagster
    assets they run build their own engine (``assets._get_engine``) with the
    pipeline role, so naming only the engines the script builds itself would
    leave the heaviest replay connections labelled ``meridian-factors``. An
    explicit ``MERIDIAN_PG_APPLICATION_NAME`` (for example one name per
    campaign) is kept.
    """
    if not (os.environ.get(APPLICATION_NAME_ENV) or "").strip():
        os.environ[APPLICATION_NAME_ENV] = default
