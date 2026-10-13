"""AssetSelection groupings shared by the jobs and sensors.

This is a minimal placeholder: it lists only the asset keys the jobs and
sensors in this copy refer to.

The daily run is split in two. The core assets are CPU-only and run in parallel
across partitions. The GPU and composite phase is serialised by a cluster-wide
GPU pool cap and starts once every core asset has landed for the same partition.
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
