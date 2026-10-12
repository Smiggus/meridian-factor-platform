"""
api/pg_identity.py

Thin re-export shim: surfaces the shared libpq ``application_name`` helper from
the Dagster package (Meridian#467 item 5) so the API's engine builders, and the
root-level ``factor_store`` / ``risk_metrics`` modules, share one implementation
instead of seven copies of the same expression.

Same sys.path trick as api/factor_catalogue.py; ``meridian_pipeline.pg_identity``
is stdlib-only so this never needs dagster installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PKG = Path(__file__).resolve().parent.parent / "dagster"
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from meridian_pipeline.pg_identity import (  # noqa: E402
    API_APPLICATION_NAME,
    pg_application_name,
    pg_connect_args,
)

__all__ = ["API_APPLICATION_NAME", "pg_application_name", "pg_connect_args"]
