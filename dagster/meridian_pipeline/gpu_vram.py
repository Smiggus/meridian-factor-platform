"""Peak-VRAM reporting for the GPU assets.

Why this exists (Meridian#296): GPU batch work is pinned to one node class.
A process that is not scheduled as a pod (a host-level service) consumes no
`nvidia.com/gpu` device-plugin resource while still holding part of the
card when a model is loaded. Kubernetes therefore hands a GPU step an
"exclusive" card that may in reality have much less free VRAM, and nothing
in the scheduler can see the discrepancy.

The failure that produces is a CUDA OOM inside `gpu_asset`, which burns
its RetryPolicy and fails the nightly phase-2 composite. That is strictly worse
than the scheduling contention the pin was introduced to solve, so the pin is
only safe while peak usage stays well under the headroom. This module makes the
peak an observable number instead of an assumption.

Read it off a run with:
  kubectl -n dagster logs <step-pod> | grep vram
"""

from __future__ import annotations

from typing import Any

# Below this much free VRAM at teardown, the GPU pin is at risk of OOMing
# on a run that overlaps another process holding VRAM. The threshold is a margin
# below which headroom is no longer comfortable, not a sign that anything has
# failed yet.
LOW_HEADROOM_BYTES: int = 1_000_000_000

_GIB = 1024**3


def log_vram_peak(log: Any, label: str) -> dict[str, float] | None:
    """Log peak + currently-free VRAM for a finished GPU step.

    Call AFTER `torch.cuda.synchronize()` and BEFORE `empty_cache()`, so the
    figures describe the step's real high-water mark rather than a
    post-cleanup one. `max_memory_allocated` is a process-lifetime peak that
    `empty_cache()` does not reset, but reading it first keeps the ordering
    obvious to the next reader.

    Never raises: instrumentation must not be able to fail a scoring run.
    Returns the measurements (GiB) for tests, or None when unavailable.
    """
    try:
        import torch
    except ImportError:
        # CI installs a partial dependency set; absence of torch is not a fault.
        return None

    try:
        if not torch.cuda.is_available():
            return None

        peak_alloc = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        free_b, total_b = torch.cuda.mem_get_info()

        stats = {
            "peak_allocated_gib": peak_alloc / _GIB,
            "peak_reserved_gib": peak_reserved / _GIB,
            "free_gib": free_b / _GIB,
            "total_gib": total_b / _GIB,
        }
        log.info(
            f"vram[{label}]: peak_allocated={stats['peak_allocated_gib']:.2f}GiB "
            f"peak_reserved={stats['peak_reserved_gib']:.2f}GiB "
            f"free={stats['free_gib']:.2f}GiB of {stats['total_gib']:.2f}GiB "
            f"device={torch.cuda.get_device_name(0)}"
        )
        if free_b < LOW_HEADROOM_BYTES:
            log.warning(
                f"vram[{label}]: only {stats['free_gib']:.2f}GiB free on "
                f"{torch.cuda.get_device_name(0)}. This is "
                "the Meridian#296 hazard: a process the device plugin cannot "
                "see holds VRAM, so an overlapping run can OOM. Shrink the "
                "GPU phase or the batch."
            )
        return stats
    except Exception as exc:  # pragma: no cover - defensive only
        try:
            log.warning(f"vram[{label}]: measurement failed (non-fatal): {exc!r}")
        except Exception:
            pass
        return None
