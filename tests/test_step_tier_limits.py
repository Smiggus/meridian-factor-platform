"""
No step tier may set a memory LIMIT below the untagged default.

A step pod with no `op_tags` receives the chart's default block:
request 1024Mi, limit 8192Mi. A tier whose limit is lower therefore makes the
assets carrying it strictly WORSE off than assets nobody bothered to size.

That inversion is what #513 was. A pool member did not OOM because it was
unsized; it OOMed because CPU_HEAVY's limit had been tightened to 4Gi (!503)
while the untagged default it would otherwise have inherited was 8Gi.

Requests are NOT constrained here. They drive scheduling and
tightening them is the point of the tiers; the #305 measurements behind LIGHT's
512Mi and CPU_HEAVY's 1Gi are sound. Only limits are floored, because a limit
buys no placement accuracy - setting one below the default trades an OOM for
nothing.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

_DAGSTER_DIR = Path(__file__).resolve().parents[1] / "dagster"
if str(_DAGSTER_DIR) not in sys.path:
    sys.path.insert(0, str(_DAGSTER_DIR))

_K8S_CONFIG = _DAGSTER_DIR / "meridian_pipeline" / "k8s_config.py"

# The untagged default, read off a live pod - not inferred from the chart.
UNTAGGED_DEFAULT_LIMIT_MI = 8192


def _to_mi(mem: str) -> int:
    m = re.fullmatch(r"(\d+)(Mi|Gi)", mem)
    assert m, f"unparseable memory value {mem!r}"
    return int(m.group(1)) * (1024 if m.group(2) == "Gi" else 1)


def _tier_limits() -> dict[str, str]:
    """Parse every *_STEP_CONFIG's memory limit straight from the source.

    Source-level rather than importing the dicts: this must hold
    for every tier that exists, including one added tomorrow by someone who
    never reads this file.
    """
    src = _K8S_CONFIG.read_text()
    out: dict[str, str] = {}
    for m in re.finditer(r"^(_[A-Z_]*STEP_CONFIG)\s*:.*?=\s*\{(.*?)\n\}", src, re.S | re.M):
        lim = re.search(r'"limits":\s*\{[^}]*"memory":\s*"([^"]+)"', m.group(2))
        if lim:
            out[m.group(1)] = lim.group(1)
    return out


@pytest.mark.parametrize("tier", sorted(_tier_limits()))
def test_tier_limit_not_below_untagged_default(tier):
    """A tier may sit below the default ONLY via a recorded measurement.

    "Below 8Gi" on its own is a false positive - a measured cap is a legitimate
    design choice (LDP runs two). What made #513 a defect was that the number
    had no measurement behind it for the workload that hit it. So the exemption
    is explicit and carries its justification, rather than the test being
    weakened by whoever next needs a small tier.
    """
    from meridian_pipeline.k8s_config import MEASURED_SUB_FLOOR_LIMITS

    if tier in MEASURED_SUB_FLOOR_LIMITS:
        assert MEASURED_SUB_FLOOR_LIMITS[tier].strip(), (
            f"{tier} claims a sub-floor exemption with an empty justification"
        )
        return

    limit = _tier_limits()[tier]
    assert _to_mi(limit) >= UNTAGGED_DEFAULT_LIMIT_MI, (
        f"{tier} sets a memory limit of {limit}, below the {UNTAGGED_DEFAULT_LIMIT_MI}Mi "
        f"an UNTAGGED step pod gets. Assets on this tier would have less headroom than "
        f"assets nobody sized - the #513 inversion. Raise the limit, or if the tier is "
        f"deliberately smaller, justify it against a measurement taken on the WIDEST "
        f"partition its assets run (not a daily window - that is what missed #513) "
        f"and record it in MEASURED_SUB_FLOOR_LIMITS."
    )


def test_every_exemption_names_a_real_tier():
    """An exemption for a tier that no longer exists silently protects nothing."""
    from meridian_pipeline.k8s_config import MEASURED_SUB_FLOOR_LIMITS

    tiers = set(_tier_limits())
    for name in MEASURED_SUB_FLOOR_LIMITS:
        assert name in tiers, f"{name} is exempted but is not a real tier: {sorted(tiers)}"


def test_the_two_sub_floor_tiers_are_the_expected_ones():
    """Exactly the two 4Gi tiers are exempted, and nothing else drifts under.

    Pins the SET as well as the rule: a third tier appearing here should be a
    review decision, not something that arrives unnoticed.
    """
    from meridian_pipeline.k8s_config import MEASURED_SUB_FLOOR_LIMITS

    assert set(MEASURED_SUB_FLOOR_LIMITS) == {
        "_CPU_HEAVY_STEP_CONFIG",
        "_LIGHT_STEP_CONFIG",
    }


def test_each_exemption_cites_a_measurement():
    """A justification must contain a number and a source; prose alone will not do."""
    from meridian_pipeline.k8s_config import MEASURED_SUB_FLOOR_LIMITS

    for tier, why in MEASURED_SUB_FLOOR_LIMITS.items():
        assert "MEASURED" in why, f"{tier}'s justification does not claim a measurement"
        assert "canary" in why or "run " in why, (
            f"{tier}'s justification names no run the figure came from"
        )


def test_requests_are_not_constrained_by_this_rule():
    """Documents the asymmetry so nobody 'fixes' the requests too.

    LIGHT keeps a 512Mi request against an 8Gi limit; that gap is intended.
    """
    src = _K8S_CONFIG.read_text()
    m = re.search(r"^_LIGHT_STEP_CONFIG\s*:.*?=\s*\{(.*?)\n\}", src, re.S | re.M)
    assert m
    req = re.search(r'"requests":\s*\{[^}]*"memory":\s*"([^"]+)"', m.group(1))
    assert req and _to_mi(req.group(1)) < UNTAGGED_DEFAULT_LIMIT_MI


def test_exempted_tiers_keep_headroom_over_their_measured_peak():
    """The exemption is only honest while the ceiling still clears the peak.

    CPU_HEAVY's figure is the FULL-BREADTH daily peak, NOT the
    lower figure seen on the widest-universe canaries - those canaries run at
    a lower factor breadth, so guarding against them would let the
    tier be tuned down to a ceiling the real daily workload can exceed. LIGHT's
    canary figure IS usable: its members compute fixed market series that do not
    scale with library breadth. Requiring 2x headroom means a future downtune
    toward the observed peak fails here rather than in a pod.
    """
    from meridian_pipeline.k8s_config import CPU_HEAVY_OP_TAGS, LIGHT_OP_TAGS

    measured_peak_mi = {"CPU_HEAVY": 1800, "LIGHT": 300}  # illustrative peaks, not measurements
    for name, tags in (("LIGHT", LIGHT_OP_TAGS), ("CPU_HEAVY", CPU_HEAVY_OP_TAGS)):
        cfg = json.loads(tags["dagster-k8s/config"])
        lim = _to_mi(cfg["container_config"]["resources"]["limits"]["memory"])
        assert lim >= 2 * measured_peak_mi[name], (
            f"{name} limit {lim}Mi leaves under 2x headroom over its measured "
            f"peak {measured_peak_mi[name]}Mi"
        )
