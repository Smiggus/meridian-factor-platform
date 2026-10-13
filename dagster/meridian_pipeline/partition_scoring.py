"""Scoring of one date's factor groups, without a Dagster context.

This is a minimal placeholder. It keeps the names the burst ship-back and
promote code depends on, so `scripts/burst/promote_backfill.py` and its tests
run the real scope logic against the example catalogue in `factor_definitions.py`:

    FUNDAMENTALS_GROUPS         the groups a core backfill owns (illustrative)
    GROUP_FACTORS               group -> the factors it owns (delete-replace scope)
    LIVE_ONLY_DEFAULT           factors with no historical data, never erased
    owned_factors_for_groups()  the only supported way to build a promote scope
    score_partition()           the compute; raises NotImplementedError here
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
    of each selected group's owned factors. Burst promotes build their
    owned_factors list this way and no other."""
    out: set = set()
    for g in resolve_groups(groups):
        out.update(GROUP_FACTORS[g])
    return sorted(out)


def score_partition(engine, score_date, groups=None):
    """Score one date. Not implemented in this copy."""
    raise NotImplementedError("score_partition is not implemented in this copy")
