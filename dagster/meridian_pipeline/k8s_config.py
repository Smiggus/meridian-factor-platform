"""
k8s_config.py
-------------
Shared K8s configuration for Meridian assets under two different routing
paths:

  K8sRunLauncher (run pod)
      Reads `dagster/run_k8s_config` from run-level tags. Used by
      the GPU backfill jobs, which set GPU_OP_TAGS at the job level via `tags=`.

  k8s_job_executor (step pods)
      Reads `dagster-k8s/config` from per-op tags. Used by GPU assets such as
      `risk_metrics`, which set GPU_OP_TAGS at the asset level via `op_tags=`. With k8s_job_executor each step gets its own K8s Job pod;
      the executor merges `dagster-k8s/config` from op_tags into the step
      pod spec.

Both keys are carried in GPU_OP_TAGS so the same constant works in both
contexts without callers caring which executor is active.

GPU_K8S_TAG_VALUE / dagster/run_k8s_config
    Routes the run pod to a GPU node, tolerates the nvidia taint, and
    requests nvidia.com/gpu:1. Read by K8sRunLauncher for job-level GPU
    routing (backfill jobs).

GPU_K8S_STEP_CONFIG / dagster-k8s/config
    Same routing + resource request, but read by k8s_job_executor for
    per-step pod GPU routing (risk_metrics).

Node labels used: `workload=gpu` + `accelerator=gpu_pool`. With more than
one GPU node type, `workload=gpu` alone would select between them by luck, so
the `accelerator` selector below pins GPU batch work to one node class.
GPU taint: the dedicated GPU node also keeps `nvidia.com/gpu=present:NoSchedule`. The
toleration below is harmless when the taint is absent and correct when present.
Single-GPU exclusivity holds regardless: the device plugin advertises
`nvidia.com/gpu=1`, so at most one GPU-requesting pod schedules at a time.
RuntimeClass: `nvidia` - required so containerd invokes the NVIDIA Container
Runtime and CUDA is visible inside the pod. Without it torch.cuda.is_available()
returns False even though nvidia.com/gpu:1 is requested.
"""

from __future__ import annotations

import json
from typing import Any, Final

# k8s_job_executor step pods otherwise schedule at PriorityClass priority 0
# (dagster-platform#88 / LDP#147): higher-priority pods (other runs at
# dagster-run-low=100/default, GitLab CI at 10000) preempt them mid-fan-out, and
# on a packed node the smallest bucket loses. Spread this into each step tier's
# pod_spec_config so steps schedule at the cluster-managed `dagster-run-default`
# PriorityClass (ArgoCD-managed in k3s-cluster; already used by LDP's
# NFS_HEAVY step config, !214). Run-launcher configs are a separate scheduling
# path (launcher-managed) and are intentionally left out of scope here.
_STEP_PRIORITY_POD_SPEC: Final[dict[str, str]] = {"priority_class_name": "dagster-run-default"}

GPU_K8S_CONFIG: Final[dict[str, Any]] = {
    "pod_spec_config": {
        # Both selectors are LABELS (not hostname pins). `workload=gpu` alone
        # matches every GPU node, so the `accelerator` label is what actually
        # pins this to one node class (Meridian#296).
        #
        # THIS IS THE SINGLE SOURCE OF THE SELECTOR. Every GPU tier below
        # spreads this dict, so editing it here moves all of them together:
        #   GPU_OP_TAGS             (risk_metrics, backfill jobs)
        # Changing only one tier would leave the others behind, silently.
        #
        # The device plugin caps each node at nvidia.com/gpu=1 -> single GPU pod.
        # Caveat the scheduler cannot see: a host-level service (not a pod)
        # consumes no device-plugin resource while still holding part of the
        # card's VRAM, so k8s can hand us an "exclusive" card with less free
        # VRAM than it appears.
        # The VRAM logging in gpu_asset / risk_metrics exists to catch
        # that; if peaks approach the free VRAM, shrink the reservation window
        # (#296 option 2).
        "node_selector": {"workload": "gpu", "accelerator": "gpu_pool"},
        "tolerations": [
            {
                "key": "nvidia.com/gpu",
                "operator": "Exists",
                "effect": "NoSchedule",
            }
        ],
        # Required: containerd uses runc by default; the nvidia runtime class
        # hooks in the NVIDIA Container Runtime so CUDA is actually available
        # inside the pod. Without this, NVIDIA_VISIBLE_DEVICES is set but
        # torch.cuda.is_available() returns False.
        "runtime_class_name": "nvidia",
    },
    "container_config": {
        "resources": {
            # 1 GPU + 8Gi host RAM. nvidia.com/gpu: 1 = whole device (no MIG).
            #
            # Host-memory request: size only against a measured peak.
            "limits": {"nvidia.com/gpu": "1", "memory": "8Gi"},
            "requests": {"memory": "2Gi"},
        },
        # Keep CUDA managed-memory allocations on the device rather than in
        # system memory.
        "env": [{"name": "CUDA_MANAGED_FORCE_DEVICE_ALLOC", "value": "1"}],
    },
}

# Serialised forms. Dagster tag values must be strings.
GPU_K8S_TAG_VALUE: Final[str] = json.dumps(GPU_K8S_CONFIG)

# dagster-k8s/config is the tag key read by k8s_job_executor for per-step
# pod spec overrides. Same structure as GPU_K8S_CONFIG but read from op_tags
# by the step handler, not the run launcher.
#
# env is intentionally excluded from the step config: dagster_k8s's model
# parser calls getattr(kubernetes.client.models, attr_type) on every field
# of V1EnvVar, and newer kubernetes Python clients annotate some fields as
# "dict[str, str]" (lowercase generic) which the parser cannot resolve.
# The CUDA_MANAGED_FORCE_DEVICE_ALLOC env var is instead set at module level
# in risk_metrics_asset.py and assets.py before any CUDA context is created.
_GPU_K8S_STEP_CONFIG_BASE: Final[dict[str, Any]] = {
    "pod_spec_config": {**GPU_K8S_CONFIG["pod_spec_config"], **_STEP_PRIORITY_POD_SPEC},
    "container_config": {k: v for k, v in GPU_K8S_CONFIG["container_config"].items() if k != "env"},
}
GPU_K8S_STEP_CONFIG: Final[str] = json.dumps(_GPU_K8S_STEP_CONFIG_BASE)

# GPU op_tags bundle. Spread on any GPU asset:
#   @asset(op_tags=GPU_OP_TAGS)
#   def my_gpu_asset(): ...
#
# Three keys:
#   dagster/run_k8s_config  - K8sRunLauncher reads this for run-pod routing
#                             (used by backfill jobs via job-level tags=).
#   dagster-k8s/config      - k8s_job_executor reads this for step-pod routing
#                             (used by GPU assets such as risk_metrics via op_tags=).
#   dagster/gpu: "true"     - opts into the platform GPU concurrency cap (1
#                             cluster-wide). Without this, multiple GPU steps
#                             pile up Pending on the GPU node.
#                             See dagster-platform docs/TENANT_VALUES.md GPU tag contract.
GPU_OP_TAGS: Final[dict[str, str]] = {
    "dagster/run_k8s_config": GPU_K8S_TAG_VALUE,
    "dagster-k8s/config": GPU_K8S_STEP_CONFIG,
    "dagster/gpu": "true",
}


# ─────────────────────────────────────────────
# Right-sized step-pod resources per asset tier (Meridian #80, 2026-05-14)
# ─────────────────────────────────────────────
#
# Background: Dagster's k8s_job_executor copies the code-server pod's
# resource block as the template for step pods unless dagster-k8s/config
# op_tags override it. Our code-server requests 500m / 1Gi, which is the
# right number for the code-server itself but ~28x over-request on CPU
# for the lighter asset materialisations.
#
# The light steps sit far below their old request.
#
# Limits are kept generous (2cpu / 4Gi) so the steps can still burst if
# any partition is larger than typical. Only requests change - this is a
# scheduler reservation fix, not a behaviour change.
#
# composite_scores / risk_metrics are NOT in LIGHT_OP_TAGS
# yet because they weren't observed during the test (they run after the
# parallel fanout). Tune them after the next backfill produces telemetry.

# ─────────────────────────────────────────────
# MinIO secret injection for the LDP Parquet read path (Meridian #101) +
# the meridian Iceberg warehouse write path (docs/17-LAKEHOUSE.md)
# ─────────────────────────────────────────────
#
# The factor assets execute in run/step pods spawned by K8sRunLauncher /
# k8s_job_executor, NOT in the code-server pod. STEP pods (k8s_job_executor) get
# their env only from this op_tags env_from + the platform run_launcher config
# (dagster-platform/dagster_home/dagster.yaml), which injects `dagster-secrets`
# (-> LAKEKEEPER_M2M_* for lakehouse OIDC) + the `dagster-env` ConfigMap. The
# tenant code-server envSecrets (`meridian-ldp-minio`, `meridian-minio`) do NOT
# reach step pods (they DO reach single run pods via includeConfigInLaunchedRuns,
# but step pods are built from op_tags), so both MinIO secrets are injected here:
#   * meridian-ldp-minio (LDP_MINIO_*)     - read ldp Parquet / Iceberg (#101/#119)
#   * meridian-minio     (MERIDIAN_MINIO_*) - read+WRITE the meridian warehouse
#     (lakehouse.py); without it lakehouse_available() is False in step pods.
# Endpoints default to the correct in-cluster URLs (LDP_LAKEHOUSE_ENDPOINT /
# MERIDIAN_LAKEHOUSE_ENDPOINT), and meridian-minio pins MERIDIAN_MINIO_ENDPOINT to
# the in-cluster minio.internal:9500 (USE_SSL=false), so no extra env is needed.
#
# env_from with a secret_ref adds the whole Secret as envFrom on the pod. We use
# env_from (V1EnvFromSource -> V1SecretEnvSource, only simple fields) rather than
# env (V1EnvVar), so it sidesteps the k8s_snake_case_dict parser bug that forces
# `env` to be excluded from step configs (see GPU_K8S_STEP_CONFIG comment above).
# See docs/11-DAGSTER_PIPELINE.md "Parquet/DuckDB read path".
_MINIO_ENV_FROM: Final[list[dict[str, Any]]] = [
    {"secret_ref": {"name": "meridian-ldp-minio"}},
    {"secret_ref": {"name": "meridian-minio"}},
]

# Run-pod config for meridian_backfill_job: its single multiprocess run pod
# hosts BOTH a GPU asset (which needs GPU routing) AND the parquet
# readers (which need LDP_MINIO_*), so the
# run pod needs the union - GPU pod_spec + GPU container_config + MinIO env_from.
_GPU_WITH_MINIO_CONFIG: Final[dict[str, Any]] = {
    "pod_spec_config": GPU_K8S_CONFIG["pod_spec_config"],
    "container_config": {
        **GPU_K8S_CONFIG["container_config"],
        "env_from": _MINIO_ENV_FROM,
    },
}
GPU_WITH_MINIO_RUN_K8S_CONFIG: Final[str] = json.dumps(_GPU_WITH_MINIO_CONFIG)

# ─────────────────────────────────────────────
# Step memory LIMITS must never fall below the untagged default
# ─────────────────────────────────────────────
#
# A step pod with NO op_tags gets the chart's default block (1Gi request /
# 8Gi limit):
#
#     request 1024Mi   limit 8192Mi
#
# So a TIER whose limit is below 8Gi makes the assets carrying it strictly worse
# off than assets nobody bothered to size. That inversion is not theoretical: it
# is the failure this rule prevents: a sized step OOMs where an unsized one
# would not, because the tier's limit is tighter than the untagged default.
#
# REQUESTS are the half that earns the tiering. They drive scheduling, and the
# #305 measurement behind LIGHT's 512Mi is sound for the daily workload. CPU_
# HEAVY's request does NOT stay at #305's 1Gi - #397 raised it once the same
# measurement was read against USAGE-OVER-REQUEST rather than p50 comfort: the
# heaviest pool member's own recorded peak already exceeded the old request,
# which is the eviction-ranking mechanism this file's #397
# note (near the CPU_HEAVY tier below) explains in full.
#
# LIMITS are the half that can cause an inversion. A limit only caps; it buys no
# placement accuracy, so setting one BELOW the default trades an OOM for nothing.
# Both 4Gi ceilings are restored to the 8Gi default here.
#
# A measurement taken on DAILY partitions is not enough. Historical
# partitions are wider (older dates carry a larger universe), so the sample
# could not contain the peak the #284 backfill will produce. Many assets ride
# these two tiers and Phase 1 runs all of them.
#
# RULE: do not set a step limit below LIMIT_FLOOR without a measurement taken on
# the WIDEST partition the asset actually runs. tests/test_step_tier_limits.py
# enforces it.
LIMIT_FLOOR_MEMORY: Final[str] = "8Gi"

# Deliberate, MEASURED exceptions to the floor above.
#
# "Below the untagged default" is not automatically wrong - a cap can be a real
# design choice, e.g. failing a runaway step fast rather than letting it consume
# a node. What makes an inverted tier a defect is not the number, it is that the
# number had no measurement behind it for the workload that hit it.
#
# So the floor is not absolute: a tier may sit below it if it appears here with
# the measurement that justifies it. That keeps the escape hatch explicit and
# reviewable instead of forcing whoever needs one to weaken the test - which is
# how an invariant quietly stops being an invariant.
#
# (Credit for the distinction: the k3s-cluster session, comparing this rule
# against LDP's NFS_HEAVY 4Gi and RUN_POD 7Gi tiers - both below the default and
# both correct, because both record the footprint they were sized against.)
#
# Format: tier name -> the measurement that justifies going below the floor.
MEASURED_SUB_FLOOR_LIMITS: Final[dict[str, str]] = {
    "_CPU_HEAVY_STEP_CONFIG": (
        "4Gi is MEASURED-safe. The primary evidence is a long step-pod memory "
        "window at FULL factor breadth, the only window that observes these "
        "assets at the breadth they run at in production. Secondary evidence "
        "is widest-universe canary runs, which stress universe WIDTH, not "
        "factor BREADTH, so canary figures must be read narrowly: a member "
        "that wrote no rows on a canary date measures an empty run and bounds "
        "nothing. Anyone re-sizing this tier must use a full-breadth daily "
        "window, not a historical canary. "
        "Raising this ceiling to the default was considered and rejected: the "
        "limit is not binding for any remaining member on either body of "
        "evidence, and a limit far above the request lets a step balloon past "
        "what the scheduler reserved on a small node, trading a ceiling "
        "nothing touches for real node-OOM exposure."
    ),
    "_LIGHT_STEP_CONFIG": (
        "4Gi is MEASURED-safe, and this tier's members are the ones for which "
        "the canary caveat above does NOT bite: they compute a fixed set of "
        "market/ETF series that is not a function of factor-library breadth, "
        "so a historical measurement transfers, with orders of magnitude of "
        "headroom."
    ),
}

_LIGHT_STEP_CONFIG: Final[dict[str, Any]] = {
    "pod_spec_config": {**_STEP_PRIORITY_POD_SPEC},
    "container_config": {
        # MinIO creds for the Parquet read path - see _MINIO_ENV_FROM above.
        # Harmless on the light assets that don't read Parquet (macro,
        # expectations, composite): the env vars are present but unused.
        "env_from": _MINIO_ENV_FROM,
        "resources": {
            "requests": {"cpu": "100m", "memory": "512Mi"},
            "limits": {"cpu": "2", "memory": "4Gi"},
        },
    },
}

# Apply via op_tags=LIGHT_OP_TAGS on assets that are read-aggregate-write
# patterns over a single partition's pandas frame. See the tier table in
# `docs/02-ARCHITECTURE.md` / Meridian#80 for which assets qualify.
LIGHT_OP_TAGS: Final[dict[str, str]] = {
    "dagster-k8s/config": json.dumps(_LIGHT_STEP_CONFIG),
}

# ─────────────────────────────────────────────
# CPU_HEAVY tier (Meridian#138; re-tuned + split #305)
# ─────────────────────────────────────────────
#
# Factor assets that read LDP Parquet and run a Polars/numpy compute over a
# full partition (several input tables, or a trailing price-history window) peak well above the 512Mi LIGHT
# request. With only a 512Mi reservation the scheduler bin-packs several heavy
# step pods onto one small node and overcommits it. (The original fix
# reserved 2Gi; re-tuned to 1Gi below once measured RSS confirmed this
# pool's actual peak - see #305 note.)
#
# #305 SPLIT (2026-08-19): a long window of step-pod peak-RSS showed one pool
# member far above every other, sharing this pool and this resource tier with
# assets that stayed much smaller, and three monthly/weekly batch assets rode
# this same op_tags constant for their resource block. Each group now has its
# own tier sized off its own peaks. Nothing left on this one has been observed
# anywhere near the old ceiling, so the limit drops 8Gi -> 4Gi (matching
# LIGHT_OP_TAGS' own 4Gi ceiling, one fewer distinct number to reason about).
#
# Cap this tier is sized for: 4 (unchanged) - the pool cap itself is not
# touched by this MR. The request/limit arithmetic is written so it stays
# correct if the cap is later raised: request x cap is the reserved total and
# limit x cap is the ceiling if every step bursts to its limit at once, both
# checked against a single current node's free memory. dagster-platform#87
# registered cpu_heavy's default_limit=4 explicitly as provisional ("tune
# up/down once we have telemetry") and it was set for MEMORY pressure
# specifically (bounding peak node RSS from the daily CPU fan-out), not CPU
# contention. We now have that telemetry,
# which is the justification for revisiting it - record that so a future
# reader does not mistake this for ignoring a hard-won limit. Raising the cap
# itself is a platform-side change and is NOT made in this MR (Meridian#305
# coordination, see MR description; not filed from here per docs/06-CRITICAL_RULES.md rule 17).
#
# #397 (2026-09-17): request raised back 1Gi -> 2Gi. The #305 request drop
# reasoned from p50s - but eviction under node memory pressure ranks
# Burstable pods on how far USAGE exceeds REQUEST, not on the p50, and the
# heaviest member's own recorded peak already exceeded the un-tuned 1Gi
# request by a wide margin. That is not a tail risk, it is the tier's own
# documented normal case for its heaviest member: its steps ran
# Burstable-over-request the whole time, i.e. it was ALREADY reproducing the
# under-request shape this tier exists to prevent, just measured in
# eviction-queue position rather than a bin-pack freeze. 2Gi clears the known
# ceiling with margin; the limit is untouched (4Gi = 2x the new request, and
# still the MEASURED_SUB_FLOOR_LIMITS-exempted floor value). Pool-cap
# arithmetic (see cpu_heavy pool worst-case note near the per-asset
# tiers below) does not move: the pool's binding request was already 2Gi via
# a sibling tier, so bringing this tier's other members up to the
# same 2Gi changes nothing about the pool's reservation ceiling.
#
# DO NOT add a hostname topologySpreadConstraint here. The anti-storm
# spread is already applied PLATFORM-side: dagster-platform's runK8sConfig
# (helm/platform/values.yaml) sets a {kubernetes.io/hostname, ScheduleAnyway}
# topologySpreadConstraint whose labelSelector (app.kubernetes.io/part-of=dagster)
# covers BOTH run_worker AND step_worker pods (#95) - so our cpu_heavy step pods
# are already spread. A tenant-added constraint was
# redundant AND fatal: k8s requires topologySpreadConstraints be unique on
# {topologyKey, whenUnsatisfiable}, and when dagster-k8s merges our op_tags
# pod_spec_config onto the platform base the two identical {hostname,
# ScheduleAnyway} entries collide -> the step-pod Job create fails with HTTP 422
# "Duplicate value {kubernetes.io/hostname, ScheduleAnyway}" (FieldValueDuplicate).
# This trap is invisible to unit tests (the collision only happens at the k8s
# API after the platform-base merge), so guard it with test_cpu_heavy_no_topology_
# spread instead. If we ever need a DIFFERENT spread axis, it must differ from the
# platform's {hostname, ScheduleAnyway} pair (e.g. a zone key) to avoid the dup.
# The same rule applies to every new tier below (LIBRARY_HEAVY_OP_TAGS,
# CPU_HEAVY_BATCH_OP_TAGS) - none of them add their own spread either.
_CPU_HEAVY_STEP_CONFIG: Final[dict[str, Any]] = {
    "pod_spec_config": {**_STEP_PRIORITY_POD_SPEC},
    "container_config": {
        "env_from": _MINIO_ENV_FROM,  # Parquet read path creds (#101)
        "resources": {
            "requests": {"cpu": "500m", "memory": "2Gi"},  # #397: 1Gi -> 2Gi, see note above
            "limits": {"cpu": "2", "memory": "4Gi"},
        },
    },
}

CPU_HEAVY_OP_TAGS: Final[dict[str, str]] = {
    "dagster-k8s/config": json.dumps(_CPU_HEAVY_STEP_CONFIG),
}

# NOTE: a bespoke LIBRARY tier was folded back into the central CPU_HEAVY tag (now
# 2Gi/8Gi) - see the CPU_HEAVY sizing note above. Keep memory-heavy CPU factor
# assets on the shared cpu_heavy tag rather than proliferating per-asset tiers.

# ─────────────────────────────────────────────
# CPU multiprocess backfill run-pod config (Meridian #107)
# ─────────────────────────────────────────────
#
# Run-pod config for meridian_cpu_backfill_job: the CPU phase of the two-phase
# backfill. Carried as a job-level `dagster/run_k8s_config` tag.
#
# Deliberately carries NO `dagster/gpu` tag and NO `nvidia.com/gpu` request, so
# the run stays OFF the cluster-wide gpu pool (cap=1) and schedules on ANY node.
# That decouples the CPU bulk from the GPU bottleneck and avoids the
# multiprocess+gpu-pool self-deadlock (the GPU phase stays on
# step pods via daily_gpu_composite_job - see #107).
#
# Fat CPU so multiprocess_executor (one subprocess per CORE asset) + Polars'
# internal threading actually parallelise instead of fighting over ~2 cores.
# POLARS_MAX_THREADS bounds per-subprocess threading so 8 subprocesses don't
# oversubscribe the 8-core limit. env (not env_from) is fine here: this is the
# run-launcher path, not the step-executor path that trips the V1EnvVar parser
# (see the GPU_K8S_STEP_CONFIG note above). Resources are a starting point -
# tune after the first benchmark.
#
# #397 (2026-09-17) audit note: NOT resized. No peak-RSS figure has ever been
# recorded against this run pod - the comment above has said "tune after the
# first benchmark" since this config was written, and no benchmark appears to
# have landed since. No measurement, so no change, per #397's rule. Needs: a
# real meridian_cpu_backfill_job run with peak-RSS captured.
_CPU_BACKFILL_CONFIG: Final[dict[str, Any]] = {
    "container_config": {
        "env_from": _MINIO_ENV_FROM,
        "env": [{"name": "POLARS_MAX_THREADS", "value": "4"}],
        "resources": {
            "requests": {"cpu": "4", "memory": "4Gi"},
            "limits": {"cpu": "8", "memory": "12Gi"},
        },
    },
}
CPU_BACKFILL_RUN_K8S_CONFIG: Final[str] = json.dumps(_CPU_BACKFILL_CONFIG)

# ─────────────────────────────────────────────
# In-cluster bulk rescore run-pod config (Meridian #275 follow-up)
# ─────────────────────────────────────────────
#
# Run-pod config for incluster_bulk_rescore_job (bulk_rescore_ret1d_close.py
# kernel run in-cluster instead of on a burst box). Carried as a job-level
# `dagster/run_k8s_config` tag - this job is a single non-partitioned op that
# loops a whole date range in-process (parquet-first bulk pattern, docs/07),
# so it gets ONE run pod rather than per-partition step pods.
#
# Sizing targets the planned fat in-cluster workers rather than the existing
# node pool - the whole point of this job is that a fat in-cluster pod becomes
# viable once those workers exist. Concretely:
#
#   requests: cpu=2, memory=8Gi   - reservation the scheduler uses to place the
#     pod. Kept well under the node total (roughly 1/6) so the pod does
#     NOT reserve the whole worker - there must be headroom left for the
#     kubelet, node-level daemonsets, and any other pod the scheduler wants to
#     co-locate. A request sized to "the whole node" is exactly what leaves a
#     multi-year backfill run Pending indefinitely if anything else is
#     already scheduled there.
#   limits: cpu=6, memory=24Gi    - ceiling. This kernel's working
#     set is whole-table hoists via bulk_rescore_ret1d_
#     close.Precompute, plus per-date frames that are staged and dropped
#     immediately rather than accumulated, and it runs over a long date
#     range in one process, so the ceiling is set high enough to absorb a
#     dense multi-year run without OOM while still leaving ample node memory
#     for the kubelet + co-scheduled pods.
#
# No node_selector yet: the fat-worker label doesn't exist until the
# capacity lands. Add a node_selector here (mirroring GPU_K8S_CONFIG's
# `workload: gpu` pattern) once that label is defined - until then the pod
# schedules on any node with capacity, which is correct on the current
# (pre-expansion) node pool too.
#
# CPU-only: no `dagster/gpu` tag, no nvidia.com/gpu request. MinIO env_from
# is required - the value group reads the LDP medallion via parquet_io
# (Meridian#101), and a missing LDP_MINIO_* credential must fail loudly
# rather than silently falling back to the (much slower, and for a bulk
# multi-year run, prod-PG-hostile) per-date pd.read_sql path. See
# incluster_bulk_rescore.py's _require_medallion_creds().
#
# #397 (2026-09-17) audit note: NOT resized. The 8Gi request/24Gi limit above
# are reasoned by analogy to a similar workload, not by a peak-RSS measurement of this actual
# kernel. No measurement, so no change, per #397's rule. Needs: a real
# incluster_bulk_rescore_job run with peak-RSS captured before this tier's
# request can be sized against evidence rather than analogy.
_BULK_RESCORE_CONFIG: Final[dict[str, Any]] = {
    "pod_spec_config": {**_STEP_PRIORITY_POD_SPEC},
    "container_config": {
        "env_from": _MINIO_ENV_FROM,
        "resources": {
            "requests": {"cpu": "2", "memory": "8Gi"},
            "limits": {"cpu": "6", "memory": "24Gi"},
        },
    },
}
BULK_RESCORE_RUN_K8S_CONFIG: Final[str] = json.dumps(_BULK_RESCORE_CONFIG)
