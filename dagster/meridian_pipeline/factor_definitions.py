"""Factor catalogue: groups, factor definitions and group weights.

This is a minimal placeholder holding a small synthetic catalogue, enough for
the API and the write path to import and run. The module has four names:

    FACTOR_GROUPS     group name -> list of factor names
    FACTOR_MAP        factor name -> (source column, direction, transform)
    GROUP_WEIGHTS     group name -> composite weight (equal weights here)
    COMPUTED_FACTORS  factors the pipeline computes rather than ingests

Edit the first three to point the service at a different set of factors.
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
