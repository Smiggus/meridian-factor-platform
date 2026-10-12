"""The Pipes channel must stay inside the burst box's scoped credential (#392).

The burst box holds a credential scoped to a single bucket prefix (plus
read-only on the lake), with no access to the bucket root. Dagster's S3 Pipes
transport does not respect that by default:

  * `PipesS3ContextInjector` accepts a `key_prefix` but defaults to None, so the
    context object lands at `s3://<bucket>/<random>/context.json`. The external
    process READS it.
  * `PipesS3MessageReader` (dagster-aws 0.29.x) has no `key_prefix` parameter at
    all. `get_params` mints a bare random prefix, so message chunks land at
    `s3://<bucket>/<random>/N.json`. The external process WRITES them.

Both are at the bucket root, so both 403 for the box. The failure is slow and
far from its cause: the driver's report_* calls fail on write, the orchestrator
never sees a message, and the run dies on the completion-marker timeout with
nothing pointing at permissions. The constraint was confirmed from the box (Pipes
message-bus traffic must live under the scoped prefix or it will 403), which is
what these tests pin.

Source-level and unit assertions rather than an end-to-end run: exercising the
real session needs a live bucket and a real external process.
"""

import pytest

pytest.importorskip("dagster_aws", reason="slim CI does not install dagster-aws")

from meridian_pipeline.burst_backfill import (  # noqa: E402
    BurstBackfillConfig,
    _ScopedPipesS3MessageReader,
)


def _cfg(**kw):
    base = {"range_start": "2020-01-02", "range_end": "2020-01-03", "run_id": "t"}
    base.update(kw)
    return BurstBackfillConfig(**base)


def test_default_pipes_prefix_is_inside_the_scoped_burst_area():
    """The default must be usable on the box without run config."""
    prefix = _cfg().pipes_prefix
    assert prefix.startswith("burst/"), prefix
    assert prefix.endswith("/"), (
        "the library concatenates the prefix with a random component rather than "
        "joining paths, so a missing trailing slash silently produces "
        f"'burst/pipes<random>' at the bucket root: {prefix!r}"
    )


def test_default_pipes_prefix_sits_under_the_staging_prefix():
    cfg = _cfg()
    assert cfg.pipes_prefix.split("/")[0] == cfg.staging_prefix


@pytest.mark.parametrize(
    "pipes_prefix",
    ["", "pipes/", "burst/pipes", "burst//pipes/"],
)
def test_pipes_prefix_rejects_values_outside_the_scoped_burst_area(pipes_prefix):
    with pytest.raises(ValueError, match="pipes_prefix"):
        _cfg(pipes_prefix=pipes_prefix)


def test_message_reader_prefixes_the_random_component():
    reader = _ScopedPipesS3MessageReader(
        bucket="meridian-parquet", client=None, key_prefix="burst/pipes/"
    )
    with reader.get_params() as params:
        key_prefix = params["key_prefix"]

    assert key_prefix.startswith("burst/pipes/"), key_prefix
    assert params["bucket"] == "meridian-parquet"
    # The random component survives, so two concurrent sessions cannot collide.
    assert len(key_prefix) > len("burst/pipes/") + 20


def test_message_reader_random_component_differs_between_sessions():
    reader = _ScopedPipesS3MessageReader(bucket="b", client=None, key_prefix="burst/pipes/")
    with reader.get_params() as a:
        first = a["key_prefix"]
    with reader.get_params() as b:
        second = b["key_prefix"]
    assert first != second


def test_message_reader_get_params_is_a_context_manager():
    """The base class decorates get_params with @contextmanager.

    Overriding it with a bare generator type-checks fine and then breaks the
    session at run time, which is exactly the kind of failure this module exists
    to prevent, so pin the shape.
    """
    reader = _ScopedPipesS3MessageReader(bucket="b", client=None, key_prefix="burst/pipes/")
    obj = reader.get_params()
    assert hasattr(obj, "__enter__") and hasattr(obj, "__exit__")


def test_asset_passes_the_same_prefix_to_both_halves_of_the_channel():
    """Context injection and message return must land in the same scoped area.

    Source-level: constructing the asset's session needs a live S3 client.
    """
    from pathlib import Path

    # burst_backfill is an AssetsDefinition after decoration, so read the module
    # file rather than trying to introspect the object.
    import meridian_pipeline.burst_backfill as mod

    src = Path(mod.__file__).read_text()
    assert "key_prefix=config.pipes_prefix" in src, (
        "both PipesS3ContextInjector and _ScopedPipesS3MessageReader must be given "
        "config.pipes_prefix; a default-constructed injector writes context.json to "
        "the bucket root, which the box cannot read"
    )
    assert src.count("key_prefix=config.pipes_prefix") == 2, (
        "expected exactly two prefixed constructions, the injector and the reader"
    )
    # The unscoped reader is imported and subclassed, never constructed directly.
    assert "= PipesS3MessageReader(" not in src, (
        "the unscoped reader must not be constructed directly; use _ScopedPipesS3MessageReader"
    )
