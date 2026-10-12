"""
api_db.py

Single source of truth for the serving API's PostgreSQL connection settings
(API-002). factor_store, risk_metrics and the dcf / risk / registry / admin
routers all build their URL here, so one process can never read scores,
weights, risk metrics and DCF inputs from different databases.

Precedence per setting, highest first:

    FACTOR_DB_<X>  ->  DB_<X>  ->  PG<X>  ->  built-in default

for X in HOST / USER / PORT. Password: FACTOR_DB_PASSWORD, DB_PASSWORD, PGPASS
(KeyError if none is set, as before). Database name: FACTOR_DB_NAME, else
marketdata. FACTOR_DB_* is an opt-in override and is unset in production,
where DB_* / PG* behave as they did before this module existed.
"""

from __future__ import annotations

import os

_DEFAULT_HOST = "db.internal"
_DEFAULT_USER = "dbuser"
_DEFAULT_PORT = "5432"
_DEFAULT_DB = "marketdata"


def _first(*names: str, default: str = "") -> str:
    for name in names:
        value = os.environ.get(name)
        if value is not None:
            return value
    return default


def db_configured() -> bool:
    """True when any host variable is set (FACTOR_DB_HOST, DB_HOST or PGHOST)."""
    return bool(_first("FACTOR_DB_HOST", "DB_HOST", "PGHOST"))


def engine_url() -> str:
    """SQLAlchemy URL for the marketdata database, per the precedence above."""
    return "postgresql://{user}:{password}@{host}:{port}/{db}".format(
        user=_first("FACTOR_DB_USER", "DB_USER", "PGUSER", default=_DEFAULT_USER),
        password=os.environ.get("FACTOR_DB_PASSWORD")
        or os.environ.get("DB_PASSWORD")
        or os.environ["PGPASS"],
        host=_first("FACTOR_DB_HOST", "DB_HOST", "PGHOST", default=_DEFAULT_HOST),
        port=_first("FACTOR_DB_PORT", "DB_PORT", "PGPORT", default=_DEFAULT_PORT),
        db=os.environ.get("FACTOR_DB_NAME", _DEFAULT_DB),
    )
