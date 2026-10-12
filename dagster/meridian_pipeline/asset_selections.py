"""
asset_selections.py - SHOWCASE STUB.

Canonical AssetSelection groupings shared by the schedules and sensors. The
private repository lists every factor asset of the program here; the stub
lists the generic ones the shipped sensors and jobs need.

Two-phase split: CORE assets are CPU-only and parallelise freely across
partitions; the GPU/composite phase is serialised by a cluster-wide GPU pool
cap and runs once all CORE assets have landed for the same partition.
"""

from __future__ import annotations

from dagster import AssetSelection

CORE_FACTOR_ASSET_KEYS: tuple[str, ...] = (
    "value_scores",
    "quality_scores",
    "growth_scores",
    "momentum_scores",
)

GPU_COMPOSITE_ASSET_KEYS: tuple[str, ...] = (
    "risk_metrics",
    "composite_scores",
    "factor_group_scores",
)

ALL_FACTOR_ASSET_KEYS: tuple[str, ...] = CORE_FACTOR_ASSET_KEYS + GPU_COMPOSITE_ASSET_KEYS

CORE_FACTOR_ASSETS = AssetSelection.keys(*CORE_FACTOR_ASSET_KEYS)

PHASE1_TRIGGER_ASSETS = CORE_FACTOR_ASSETS | AssetSelection.keys("market_vol")

GPU_COMPOSITE_ASSETS = AssetSelection.keys(*GPU_COMPOSITE_ASSET_KEYS)

DAILY_FACTOR_ASSETS = CORE_FACTOR_ASSETS | GPU_COMPOSITE_ASSETS

_GROUP_SCORES_EXCLUDED_KEYS = ("risk_metrics", "composite_scores", "factor_group_scores")
GROUP_SCORES_ASSET_KEYS: tuple[str, ...] = tuple(
    k for k in ALL_FACTOR_ASSET_KEYS if k not in _GROUP_SCORES_EXCLUDED_KEYS
)
