"""
burst_backfill.py - CONTROL plane of the elastic burst-compute pattern.

Design: docs/proposals/07-elastic-burst-compute-pipes.md.

This is the parked-session Dagster asset that orchestrates a burst run on an
off-cluster box (a dev machine now; an EC2 spot node later) WITHOUT ever dialing that
box. It:

  1. opens a Dagster Pipes session whose context + messages travel through a
     shared MinIO prefix (PipesS3ContextInjector / PipesS3MessageReader), so the
     burst box only ever talks to MinIO - no inbound port required;
  2. publishes the session's bootstrap env vars to a fixed MinIO key so the
     owner's manual launch script (or a future launcher) can pick them up;
  3. waits for the burst driver's `run_summary.json` to appear in MinIO (the
     transport-simple completion signal), then folds the reported
     materialization back into Dagster lineage.

DELIBERATELY NOT REGISTERED in definitions.py yet. The manual-arm wait loop
below needs to be exercised against dagster 1.13.4's Pipes semantics before it
goes live on the cluster (proposal 07 open question 2). Until then the muscle
plane runs standalone via scripts/burst/pipes_dask_driver.py (which degrades
gracefully with no orchestrator). Register this asset only after that
validation - importing this module has no effect until then.

NOTE: no `from __future__ import annotations` here - it breaks Dagster
AssetExecutionContext validation (a repo convention).
"""

import json
import os
import time
from contextlib import contextmanager

import boto3
from dagster_aws.pipes import PipesS3ContextInjector, PipesS3MessageReader
from pydantic import field_validator

from dagster import AssetExecutionContext, Config, asset, open_pipes_session
from dagster import _check as check


class _ScopedPipesS3MessageReader(PipesS3MessageReader):
    """A Pipes message reader whose chunks live under a caller-chosen prefix.

    `PipesS3MessageReader` has no `key_prefix` parameter (dagster-aws 0.29.x):
    `get_params` mints a bare 30-character random prefix, so message chunks land
    at `s3://<bucket>/<random>/N.json` - the BUCKET ROOT. The external process
    writes those objects.

    The burst box's credential is deliberately scoped to
    `s3://meridian-parquet/burst/*` (plus read-only on the lake), with no access
    to the bucket root. A default reader therefore fails closed at run time: the
    driver's `report_*` calls 403 on write, and the orchestrator sees no messages
    and eventually times out on the completion marker instead - a slow, confusing
    failure a long way from its cause.

    This subclass prepends the configured prefix to the random component, so the
    channel sits inside the scope the key already grants. The external process
    needs no change: `PipesS3MessageWriter` reads `key_prefix` out of the params
    it is handed. The random component is preserved, so two concurrent sessions
    still cannot collide.

    The context injector needs the same treatment but already supports it -
    `PipesS3ContextInjector(key_prefix=...)` prepends to its own random
    component - so the asset passes the same value to both.
    """

    def __init__(self, *, key_prefix: str, **kwargs):
        super().__init__(**kwargs)
        self.key_prefix = check.str_param(key_prefix, "key_prefix")

    @contextmanager
    def get_params(self):
        # The base implementation is a @contextmanager (it does not clean up the
        # chunks, but the contract is a context manager either way), so this must
        # be one too - returning a bare generator silently breaks the session.
        with super().get_params() as params:
            params["key_prefix"] = f"{self.key_prefix}{params['key_prefix']}"
            yield params


class BurstBackfillConfig(Config):
    """Run config for a manual-arm burst backfill."""

    range_start: str  # inclusive, YYYY-MM-DD
    range_end: str  # inclusive, YYYY-MM-DD
    run_id: str
    kernel: str = "verify"
    # Informational: the kernel's device ("cpu"/"gpu"). The kernel itself is
    # authoritative (KERNELS[kernel].device); this is surfaced in the launch
    # instructions and lets a future SSH auto-launcher pick a GPU-capable box.
    device: str = "cpu"
    staging_prefix: str = "burst"
    # Where the Pipes context object and the Pipes message chunks live inside
    # the bucket. MUST stay inside the prefix the burst box's scoped key can
    # read and write, or the channel 403s at both ends - see
    # _ScopedPipesS3MessageReader below. Trailing slash is required: the
    # library concatenates this with a random component, it does not join.
    pipes_prefix: str = "burst/pipes/"
    # How long the parked asset waits for the burst box to finish, and how often
    # it polls the completion marker. Manual-arm runs can be long; size to the job.
    max_wait_seconds: int = 6 * 60 * 60
    poll_seconds: int = 30

    @field_validator("pipes_prefix")
    @classmethod
    def pipes_prefix_must_remain_in_burst_scope(cls, value: str) -> str:
        """Reject a run-config override that would escape the box's S3 grant."""
        if (
            not value.startswith("burst/")
            or not value.endswith("/")
            or value == "burst/"
            or "//" in value
        ):
            raise ValueError(
                "pipes_prefix must be a non-root, slash-delimited prefix under burst/, "
                "for example 'burst/pipes/'"
            )
        return value


def _s3_client():
    endpoint = os.environ.get("MERIDIAN_MINIO_ENDPOINT", "s3.example.com")
    scheme = (
        "https" if os.environ.get("MERIDIAN_MINIO_USE_SSL", "true").lower() == "true" else "http"
    )
    return boto3.client(
        "s3",
        endpoint_url=f"{scheme}://{endpoint}",
        aws_access_key_id=os.environ["MERIDIAN_MINIO_ACCESS_KEY"],
        aws_secret_access_key=os.environ["MERIDIAN_MINIO_SECRET_KEY"],
        region_name=os.environ.get("MERIDIAN_MINIO_REGION", "us-east-1"),
    )


@asset(
    name="burst_backfill",
    description="Manual-arm elastic burst backfill orchestrated via Dagster Pipes over MinIO.",
    compute_kind="pipes",
)
def burst_backfill(context: AssetExecutionContext, config: BurstBackfillConfig):
    bucket = os.environ.get("MERIDIAN_MINIO_BUCKET", "meridian-parquet")
    s3 = _s3_client()
    run_root = f"{config.staging_prefix}/{config.run_id}"

    # Both halves of the channel are pinned inside the burst prefix so the box's
    # scoped key can use them; see _ScopedPipesS3MessageReader.
    context_injector = PipesS3ContextInjector(
        bucket=bucket, client=s3, key_prefix=config.pipes_prefix
    )
    message_reader = _ScopedPipesS3MessageReader(
        bucket=bucket, client=s3, key_prefix=config.pipes_prefix
    )

    with open_pipes_session(
        context=context,
        context_injector=context_injector,
        message_reader=message_reader,
    ) as session:
        # 1. Publish bootstrap env to a fixed key the owner/launcher reads.
        bootstrap = session.get_bootstrap_env_vars()
        s3.put_object(
            Bucket=bucket,
            Key=f"{run_root}/bootstrap.env",
            Body="\n".join(f"{k}={v}" for k, v in bootstrap.items()).encode(),
        )
        context.log.info(
            "Burst session armed. Launch on the burst box:\n"
            f"  aws s3 cp s3://{bucket}/{run_root}/bootstrap.env /tmp/bs.env\n"
            f"  set -a && . /tmp/bs.env && set +a\n"
            f"  python scripts/burst/pipes_dask_driver.py "
            f"--range {config.range_start}:{config.range_end} "
            f"--run-id {config.run_id} --kernel {config.kernel} --parity"
        )

        # 2. Wait for the driver's completion marker (run_summary.json).
        summary_key = f"{run_root}/run_summary.json"
        deadline = config.max_wait_seconds
        waited = 0
        summary = None
        while waited < deadline:
            try:
                body = s3.get_object(Bucket=bucket, Key=summary_key)["Body"].read()
                summary = json.loads(body)
                break
            except s3.exceptions.ClientError:
                time.sleep(config.poll_seconds)
                waited += config.poll_seconds
        if summary is None:
            raise TimeoutError(
                f"burst run {config.run_id} did not report within {deadline}s "
                f"(no {summary_key}); the burst box may be down - resume by relaunching the driver"
            )

        if not summary.get("clean"):
            raise RuntimeError(
                f"burst run {config.run_id} not clean: {summary} - holes or parity mismatch, "
                "NOT promoting to gold"
            )
        context.log.info(
            f"Burst run clean: {summary}. Promotion step (staging -> gold) is separate."
        )

    # Materializations reported by the driver flow back through session results.
    return session.get_results()
