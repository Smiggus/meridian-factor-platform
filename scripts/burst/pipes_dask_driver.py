#!/usr/bin/env python3
"""
pipes_dask_driver.py - the MUSCLE plane of the elastic burst-compute pattern.

Runs on a burst box (a dev machine, or an EC2 spot node). Fans a partition
range across a local Dask cluster, computes each partition with a pluggable
kernel, stages results to MinIO, reports progress home to a parked Dagster
asset through Dagster Pipes, and hole-checks the batch. It NEVER writes gold.

Design: docs/proposals/07-elastic-burst-compute-pipes.md.

Standalone vs orchestrated
--------------------------
`open_dagster_pipes()` degrades gracefully when no orchestrator launched us
(no DAGSTER_PIPES_CONTEXT in env): report_* calls no-op with a warning and the
driver runs identically. So this same file is:
  - the PoC's standalone run (no cluster asset needed yet), and
  - the production external process a parked `burst_backfill_*` asset drives
    once merged and deployed.

Launch (standalone PoC)
-----------------------
    cd <repo> && source <venv>/bin/activate
    set -a && . .env && set +a
    export MERIDIAN_REPO_ROOT=$PWD
    python scripts/burst/pipes_dask_driver.py \
        --range 2025-06-02:2025-06-13 --run-id poc-2025-06 \
        --workers 10 --mem-limit 4GiB --parity

Exit codes: 0 = clean (hole-check ok; parity clean if --parity). 2 = holes or
parity mismatch (the batch must not be promoted).
"""

import argparse
import json
import os
import sys
import traceback
from datetime import datetime, timezone

# Make `scripts.burst.*` importable on every Dask worker (cloudpickle ships the
# task closure by value, but the module it imports must resolve on the worker).
_REPO_ROOT = os.environ.get("MERIDIAN_REPO_ROOT") or os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dagster_pipes import (  # noqa: E402
    PipesS3ContextLoader,
    PipesS3MessageWriter,
    open_dagster_pipes,
)
from dask.distributed import Client, LocalCluster  # noqa: E402

from scripts.burst.burst_common import (  # noqa: E402
    KERNELS,
    BurstConfig,
    HoleCheck,
    discover_partitions,
    gpu_teardown,
    load_staged_fingerprints,
    parity_diff,
)


def _open_pipes(cfg: BurstConfig):
    """Open the Pipes context. When an orchestrator launched us (a parked
    burst_backfill asset -> DAGSTER_PIPES_CONTEXT present), pair with the asset's
    S3 injector/reader via the S3 loader/writer so messages route home through
    MinIO. Standalone (no orchestrator), open_dagster_pipes degrades gracefully
    with the default loader/writer and report_* no-op with a warning."""
    if os.environ.get("DAGSTER_PIPES_CONTEXT"):
        client = cfg.s3_client()
        return open_dagster_pipes(
            context_loader=PipesS3ContextLoader(client=client),
            message_writer=PipesS3MessageWriter(client=client),
        )
    return open_dagster_pipes()


def _run_one(cfg: BurstConfig, score_date: str, kernel_name: str) -> dict:
    """Worker task. ALWAYS returns a status dict; never raises into the
    scheduler (a raise would fail the gather; a status dict lets the hole-check
    account for every partition as ok / named-failure)."""
    # Re-establish sys.path on the worker (spawned processes lose the driver's).
    repo = os.environ.get("MERIDIAN_REPO_ROOT", _REPO_ROOT)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from scripts.burst.burst_common import KERNELS as K
    from scripts.burst.burst_common import partition_done as _done
    from scripts.burst.burst_common import stage_partition as _stage

    try:
        s3 = cfg.s3_client()
        if _done(cfg, s3, score_date):
            return {"score_date": score_date, "status": "skipped_resume"}
        fp = K[kernel_name].fn(cfg, score_date)
        _stage(cfg, s3, score_date, fp)
        return {
            "score_date": score_date,
            "status": "ok",
            "z_checksum": fp["z_checksum"],
            "rows": fp["rows"],
        }
    except Exception as exc:  # noqa: BLE001 - loud, accounted failure, not a crash
        return {
            "score_date": score_date,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "trace": traceback.format_exc()[-800:],
        }


def main() -> int:
    ap = argparse.ArgumentParser(description="Burst partition driver (Pipes + Dask).")
    ap.add_argument(
        "--range", required=True, help="START:END inclusive, e.g. 2025-06-02:2025-06-13"
    )
    ap.add_argument("--kernel", default="verify", choices=sorted(KERNELS))
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--gold-db", default="marketdata")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--mem-limit", default="4GiB")
    ap.add_argument("--config-fingerprint", default="")
    ap.add_argument("--parity", action="store_true", help="run the parity gate after staging")
    ap.add_argument(
        "--discover-from",
        choices=["gold", "calendar"],
        default="gold",
        help="discover_partitions source (default: gold). 'gold' intersects the trading "
        "calendar with dates factors.daily_scores already holds - correct for a re-score "
        "parity gate. 'calendar' returns every trading day in [--start,--end] independent "
        "of gold - required to FILL history gold has zero rows for at all. "
        "Mirrors the identical flag on scripts/burst/rescore_ret1d_close.py.",
    )
    ap.add_argument(
        "--parity-tol",
        type=float,
        default=1e-9,
        help="max |dz| tolerated per (ticker,factor) for tolerance-mode kernels (float compute)",
    )
    args = ap.parse_args()

    start, end = args.range.split(":")
    cfg = BurstConfig.from_env(
        args.run_id, gold_db=args.gold_db, config_fingerprint=args.config_fingerprint
    )

    device = KERNELS[args.kernel].device

    with _open_pipes(cfg) as pipes:

        def log(msg: str) -> None:
            pipes.log.info(msg)
            print(msg, flush=True)

        log(
            f"[burst] run={cfg.run_id} kernel={args.kernel} device={device} range={start}:{end} "
            f"workers={args.workers} mem={args.mem_limit} staging=s3://{cfg.minio_bucket}/{cfg.run_root}"
        )

        requested = discover_partitions(cfg, start, end, source=args.discover_from)
        log(
            f"[burst] discovered {len(requested)} trading-day partitions "
            f"(source={args.discover_from})"
        )
        if not requested:
            # An empty result under source="gold" is AMBIGUOUS, and this used to
            # exit 0 with a benign "nothing to do". That hid the one case this
            # driver could not express at all: a range gold has zero rows for
            # (a history range) where the trading calendar does
            # have days. A fill silently became a no-op with a success exit code.
            # Disambiguate, and fail loudly with the remedy named.
            if args.discover_from == "gold":
                cal = discover_partitions(cfg, start, end, source="calendar")
                if cal:
                    log(
                        f"[burst] ABORT: gold holds 0 partitions in {start}:{end} but the "
                        f"trading calendar has {len(cal)}. Gold-sourced discovery is "
                        f"structurally unable to see this range; re-run with "
                        f"--discover-from calendar to FILL it."
                    )
                    return 1
            log(f"[burst] nothing to do - no trading days in {start}:{end}")
            return 0

        # Topology by device. GPU: ONE GPU-owning worker advertising a Dask GPU
        # resource, and every task requires GPU=1, so partitions queue serially
        # through the single 12 GB device (with retry/resume/hole-check intact).
        # CPU: fan out across many workers.
        if device == "gpu":
            cluster = LocalCluster(
                n_workers=1,
                threads_per_worker=1,
                memory_limit=args.mem_limit,
                processes=True,
                resources={"GPU": 1},
            )
            submit_kwargs = {"resources": {"GPU": 1}}
        else:
            cluster = LocalCluster(
                n_workers=args.workers,
                threads_per_worker=1,
                memory_limit=args.mem_limit,
                processes=True,
            )
            submit_kwargs = {}
        client = Client(cluster)
        try:
            log(f"[burst] dask dashboard: {client.dashboard_link}")
            futures = [
                client.submit(_run_one, cfg, d, args.kernel, pure=False, retries=2, **submit_kwargs)
                for d in requested
            ]
            results = client.gather(futures)
            if device == "gpu":
                # Rule-19 CUDA teardown on the GPU worker before it is closed.
                try:
                    client.run(gpu_teardown)
                except Exception as exc:  # noqa: BLE001 - teardown is best-effort
                    log(f"[burst] gpu_teardown warning: {exc}")
        finally:
            client.close()
            cluster.close()

        ok = [r for r in results if r["status"] in ("ok", "skipped_resume")]
        failed = {r["score_date"]: r.get("error", "?") for r in results if r["status"] == "error"}
        for sd, err in failed.items():
            log(f"[burst][FAIL] {sd}: {err}")

        hc = HoleCheck(requested=requested, staged=[r["score_date"] for r in ok], failed=failed)
        log(f"[burst] staged={len(hc.staged)} failed={len(hc.failed)} missing={len(hc.missing)}")

        parity_mismatch = 0
        if args.parity and hc.ok:
            _mode = KERNELS[args.kernel].parity_mode
            _how = (
                f"tolerance |dz|<{args.parity_tol:.0e}"
                if _mode == "tolerance"
                else "exact z_checksum"
            )
            log(f"[burst] running parity gate (re-derive via kernel, {_how})...")
            staged_fps = load_staged_fingerprints(cfg, cfg.s3_client())
            diff = parity_diff(staged_fps, cfg, args.kernel, tol=args.parity_tol)
            parity_mismatch = len(diff)
            log(f"[burst] parity mismatches: {parity_mismatch}")
            if parity_mismatch:
                log(diff.to_string())

        summary = {
            "run_id": cfg.run_id,
            "kernel": args.kernel,
            "range": [start, end],
            "requested": len(requested),
            "staged": len(hc.staged),
            "failed": len(hc.failed),
            "missing": len(hc.missing),
            "parity_mismatch": parity_mismatch,
            "clean": bool(hc.ok and parity_mismatch == 0),
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
        s3 = cfg.s3_client()
        s3.put_object(
            Bucket=cfg.minio_bucket,
            Key=f"{cfg.run_root}/run_summary.json",
            Body=json.dumps(summary, indent=2).encode(),
        )

        # Single summary materialization home to Dagster (per-partition detail is
        # in the streamed logs above; a partitioned-asset per-partition report is
        # the productionization step).
        pipes.report_asset_materialization(
            metadata={
                k: {"raw_value": v, "type": "json"}
                if isinstance(v, (list, dict))
                else {"raw_value": v}
                for k, v in summary.items()
            },
            data_version=cfg.config_fingerprint or None,
        )
        log(f"[burst] summary: {json.dumps(summary)}")

        if not hc.ok:
            log("[burst] HOLES present - batch is NOT promotable")
            return 2
        if parity_mismatch:
            log("[burst] PARITY mismatch - batch is NOT promotable")
            return 2
        log("[burst] clean - safe to promote (cluster side)")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
