"""
partition_scoring.py - SHOWCASE STUB.

In the private repository this module scores one date's factor groups without
a Dagster context, so the burst box and the in-cluster assets run the SAME
compute. The factor bodies it composes are the research output of the program
and are not published.

What the stub keeps is the INTERFACE the burst ship-back and promote
machinery depends on:

    FUNDAMENTALS_GROUPS         the groups a core backfill owns (illustrative)
    GROUP_FACTORS               group -> the factors it OWNS (delete-replace scope)
    LIVE_ONLY_DEFAULT           factors with no historical data, never erased
    owned_factors_for_groups()  the ONLY sanctioned way to build a promote scope
    score_partition()           the compute; raises here

so `scripts/burst/promote_backfill.py` and its tests exercise the real scope
logic against the example catalogue in `factor_definitions.py`.
"""

from __future__ import annotations

from meridian_pipeline.factor_definitions import FACTOR_GROUPS

FUNDAMENTALS_GROUPS = ("value", "quality")

ALL_GROUPS = tuple(FACTOR_GROUPS)

GROUP_FACTORS: dict[str, list[str]] = {g: list(fs) for g, fs in FACTOR_GROUPS.items()}

# Factors that exist only from the day they were first captured; a historical
# delete-replace scope must never include them (it would erase data that cannot
# be recomputed).
LIVE_ONLY_DEFAULT: "tuple[str, ...]" = ("live_a", "live_b")

GROUP_FACTORS["growth"].extend(LIVE_ONLY_DEFAULT)


def resolve_groups(groups) -> "tuple[str, ...]":
    """Validate and normalise a group selection (None means every group)."""
    if groups is None:
        return ALL_GROUPS
    selected = tuple(groups)
    unknown = [g for g in selected if g not in GROUP_FACTORS]
    if unknown:
        raise ValueError(f"unknown factor group(s): {unknown}")
    return selected


def owned_factors_for_groups(groups) -> "list[str]":
    """The delete-replace scope implied by a group selection: the sorted union
    of each selected group's owned factors. This is the ONLY sanctioned way to
    build an owned_factors list for a burst promote."""
    out: set = set()
    for g in resolve_groups(groups):
        out.update(GROUP_FACTORS[g])
    return sorted(out)


def score_partition(engine, score_date, groups=None):
    """Score one date. Not published: the factor constructions are withheld."""
    raise NotImplementedError(
        "score_partition is a showcase stub: the factor constructions are not published"
    )
