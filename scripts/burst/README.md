# Burst compute (Dagster Pipes + Dask)

Elastic CPU offload for whole-population jobs the k3s cluster struggles with.
Design notes: [`docs/ARCHITECTURE.md`](../../docs/ARCHITECTURE.md).

The cluster's Dagster asset opens a Pipes session whose messages travel through MinIO, and Dask
on the burst box does the compute. The box writes a MinIO staging prefix and never writes gold.
The cluster hole-checks, parity-gates and promotes.

## Files

| File | Role | Runs on |
|---|---|---|
| `burst_common.py` | shared config / partitions / kernel / staging IO / gates | anywhere |
| `pipes_dask_driver.py` | Dask fan-out and Pipes reporting | burst box |
| `dagster/meridian_pipeline/burst_backfill.py` | parked Pipes asset | cluster (unregistered until validated) |

## Standalone run (PoC - no cluster asset needed)

`open_dagster_pipes()` degrades gracefully with no orchestrator, so the driver
runs on its own today:

```bash
ssh user@<burst-box>
tmux new -s burst_poc
cd <repo> && git pull
source <venv>/bin/activate
set -a && . .env && set +a
export MERIDIAN_REPO_ROOT=$PWD
python scripts/burst/pipes_dask_driver.py \
    --range 2025-06-02:2025-06-13 --run-id poc-2025-06 \
    --workers 10 --mem-limit 4GiB --parity
```

Exit 0 = clean (hole-check ok, parity zero mismatches). Exit 2 = holes or parity
mismatch (do not promote). Sizing: 1 thread/worker (the kernel is the unit of
work), `--mem-limit` sized to `free -h` at launch.

## Orchestrated run (after the asset is merged + deployed)

1. Arm `burst_backfill` from the Dagster UI with `range_start/range_end/run_id`.
   It writes `s3://<bucket>/burst/<run_id>/bootstrap.env` and parks, waiting.
2. On the burst box, fetch the bootstrap, export it, launch the driver (same
   command, it now reports materializations home through MinIO):
   ```bash
   aws --endpoint https://s3.example.com s3 cp \
       s3://<bucket>/burst/<run_id>/bootstrap.env /tmp/bs.env
   set -a && . /tmp/bs.env && set +a
   python scripts/burst/pipes_dask_driver.py --range ... --run-id <run_id> --parity
   ```
3. The parked asset unblocks when `run_summary.json` appears; it fails the run
   if the batch is not clean. Promotion (staging -> gold) is a separate cluster
   step.

## Resume after a crash

Relaunch the driver with the same `--run-id` and `--range`. Per-partition
sentinels (`burst/<run_id>/sentinels/<date>.json`) make it skip completed days
and recompute only the gap. No Dagster restart needed.

## Kernels

Kernels declare a `device` in the `KERNELS` registry; the driver picks its Dask
topology from it.

| Kernel | Device | What it proves |
|---|---|---|
| `verify` (default) | cpu | CPU path: reads a day's gold, cross-sectional standardization, fingerprint. Exact parity (pure function of gold). Fans out across `--workers`. |
| `gpu_smoke` | gpu | GPU path: real float64 matmul on the box's GPU (cu128), deterministic, + touches gold. One GPU-owning worker, `resources={"GPU": 1}`, CUDA teardown after gather. `--workers` is ignored (single device). |

Swap in a real per-partition scoring kernel (one entry in `KERNELS` with its
device) to turn this into a production backfill - the driver, staging, gates,
and every launcher are unchanged. CPU kernel = the `score_partition` hoist;
GPU kernel = e.g. the ML backfill (#98).

## GPU run

```bash
# same env setup as above, then:
python scripts/burst/pipes_dask_driver.py \
    --range 2025-06-02:2025-06-13 --run-id gpu-smoke-2025-06 \
    --kernel gpu_smoke --mem-limit 8GiB --parity
```

The driver runs a single GPU-resource worker so partitions serialize through the
one GPU device; it tears down the CUDA context (synchronize + empty_cache)
before closing the cluster. Same hole-check + parity + staging as the CPU path.

## Leave the box clean

```bash
pkill -f "pipes_dask_driver" ; pkill -f "dask" ; tmux kill-session -t burst_poc
cd ~/Meridian && git checkout main
pgrep -f "dask (scheduler|worker)"   # must be empty
```

