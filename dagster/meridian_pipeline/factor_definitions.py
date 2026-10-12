"""
factor_definitions.py - SHOWCASE STUB.

The private repository keeps the factor catalogue here: which factors exist,
how they are grouped and how the groups are weighted into the composite. That
catalogue is the research output of the program and is not published.

This stub carries the SHAPE of the catalogue so the API and the write path
import and run against a tiny, obviously synthetic example:

    FACTOR_GROUPS   group name -> list of factor names
    FACTOR_MAP      factor name -> (source column, direction, transform)
    GROUP_WEIGHTS   group name -> composite weight (equal weights here)
    COMPUTED_FACTORS  factors a pipeline computes (as opposed to ingests)

Replace the three literals with your own catalogue to point the service at
your own factors.
"""

from __future__ import annotations

FACTOR_GROUPS: dict[str, list[str]] = {
    "value": ["example_value_a", "example_value_b"],
    "quality": ["example_quality_a", "example_quality_b"],
    "growth": ["example_growth_a"],
    "momentum": ["example_momentum_a"],
}

FACTOR_MAP: dict[str, tuple[str, int, str]] = {
    name: (name, 1, "zscore") for names in FACTOR_GROUPS.values() for name in names
}

GROUP_WEIGHTS: dict[str, float] = {g: 1.0 / len(FACTOR_GROUPS) for g in FACTOR_GROUPS}

COMPUTED_FACTORS: list[str] = sorted(FACTOR_MAP)
