"""
api/factor_catalogue.py

Thin re-export shim: surfaces FACTOR_GROUPS / FACTOR_MAP / GROUP_WEIGHTS
from the Dagster pipeline (the canonical source of truth per Meridian #59)
so API routers can import them without dragging in any live-fallback code.

The Dagster code-server is the writer; the API is read-only. Both share
the same factor catalogue, which lives under dagster/meridian_pipeline/
so the code-server image can import it without depending on this file.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the dagster/ subtree importable without installing it as a package.
# Same trick the deleted feature_builder.py used; centralised here so the
# rest of the API doesn't need to know about the sys.path manipulation.
_PKG = Path(__file__).resolve().parent.parent / "dagster"
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from meridian_pipeline.factor_definitions import (  # noqa: E402
    COMPUTED_FACTORS,
    FACTOR_GROUPS,
    FACTOR_MAP,
    GROUP_WEIGHTS,
)

__all__ = [
    "COMPUTED_FACTORS",
    "FACTOR_GROUPS",
    "FACTOR_MAP",
    "GROUP_WEIGHTS",
]
