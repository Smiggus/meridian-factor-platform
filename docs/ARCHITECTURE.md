# Architecture

Two things run from this code. A Dagster code location on k3s computes factor scores and writes them to PostgreSQL, and a small FastAPI service reads them back out. Paths below are relative to the repo root. Some comments and docstrings refer to modules that are not in this copy, `assets.py` for one; the research and the asset bodies stayed in the private repository.

## Writing scores

Everything that lands in `factors.daily_scores` goes through `dagster/meridian_pipeline/db.py`. The daily assets call `upsert_daily_scores`. The burst promote calls `copy_replace_daily_scores`, which does the same job with a bulk `COPY`. A third pair, `stage_daily_scores_rows` and `apply_staged_daily_scores`, splits a write into a stage step and an apply step, and the apply is the part that runs under the lock.

A write is delete and reinsert in one transaction, per `(score_date, factor_name)`. If the universe shrank since the last run, the rows for the dropped tickers are deleted along with the rest instead of lingering. What counts as "the rest" is whatever the caller passes as `owned_factors`; the delete never reaches wider than that, in any group. This makes one mistake expensive. An owned factor that is missing from the payload gets deleted and never replaced, so such a write is refused unless the caller opts out. Batch promoters do opt out, since they hold one scope across many partitions. A write that declares a `replace_date_range` is also compared with the trading calendar, and a missing interior date is an error unless the caller says the gap is intended.

Each write takes a shared advisory lock on the table's range key and then an exclusive lock per score date, in ascending order. The fixed order keeps the lock graph free of cycles. Every advisory lock uses a slot in the shared lock table, which is why the number of dates per transaction is capped, and waits are bounded by `lock_timeout`. `ensure_schema` is idempotent DDL that several run pods may issue at once, so it retries on deadlock with backoff.

Once PostgreSQL has committed, the rows are mirrored to Parquet and Iceberg. That step is best effort. PostgreSQL is the authority and the mirror code swallows its own errors.

`lineage.py` (#294) stamps a `campaign_id` on each row. The migration may not be applied yet, so a capability probe asks whether the column exists, and when it does not the writers emit the SQL they used before lineage. A successful probe is cached. A probe that errors is re-raised, because falling back on an error would let a writer insert unstamped rows with nobody noticing.

## Deciding a date is ready

`cross_section_ready` and `trigger_cross_section_ready` say whether a date may be scored. Both look at the upstream watermark and fall back to a coverage ratio against the trailing median. `assert_cross_section_single_dated` refuses a cross-section that mixes as-of dates, and `assert_universe_coverage` checks the size of the universe.

## Price and return guards

`ohlcv_hygiene.py` puts a floor under bogus and illiquid prints and handles delistings. `adj_close_guard.py` compares `adj_close` with `close` on the same row, which catches a single poisoned value that range checks miss. `splice_guard.py` keeps a reused ticker from blending two companies into one rolling window.

The asset checks are in `checks/`. Two of them block, and both are attached to `momentum_scores`: `check_ohlcv_read_contract` guards the OHLCV read contract, and `check_ret1d_input_quality` (`ret1d_checks.py`) is a tripwire for corrupt upstream `ret_1d`. They run once momentum has computed. Because they block, a failure stops the composite from consuming that day. Each one downgrades violations to WARN for early history (dates before 2015 for the ret_1d check, before 2004 for the contract check), so old vendor quirks do not stall a historical backfill.

The remaining checks do not block. `trading_day_checks.py` sweeps every scored table for non-trading dates; this exists because a writer that takes its partitions from gold reproduces whatever junk dates gold already has. A failing check wrapped with `@alerting_check` sends one ntfy alert per check and partition through `checks/alerting.py` and `notify.py`. In this tree that is `check_trading_day_invariant` only.

## Burst backfills

A backfill too large for the cluster runs on a separate machine. `burst_backfill.py` holds a parked Dagster asset that opens a Pipes session, with context and messages travelling through a MinIO prefix, so the machine needs no inbound port. Until that asset is registered, `pipes_dask_driver.py` runs standalone and behaves the same without an orchestrator. The driver fans partitions out over a local Dask cluster. Workers write score parquet under a staging prefix, and each partition leaves a sentinel so a crashed run resumes where it stopped.

`promote_backfill.py` runs on the cluster and is a series of gates in front of one write. The owned scope in the run manifest is compared with the staged payload in both directions, with the payload read from a sample of dates. A date the kernel undertook that is missing from the staged files stops the promote, and so does a coverage regression against gold, again judged on a sample. A live-only factor, one that cannot be reproduced for a historical date, aborts the promote unless the operator retains its gold rows or allows its erasure by name (#380). The coverage, hole and erasure gates each have their own named override (`--force-coverage`, `--allow-incomplete`, `--allow-erasure`), and one override does not silence another. The write is a delete-replace through `copy_replace_daily_scores`. Afterwards the script rereads a sample of partitions, always including the last, and compares gold with the staged rows to a small tolerance.

Three smaller scripts sit around it. `verify_promoted_range.py` checks a promoted range against the staged payload without reading the promote's own campaign rows. `promote_replay_out.py` upserts replayed output from the separate machine under one named campaign and refuses an incomplete run: the manifest, the directory listing and the per-date row counts have to agree exactly. `ingest_window.py` is a stdlib-only preflight that will not start a campaign while the upstream nightly ingest is running.

## Lakehouse and mirrors

`lakehouse.py` reaches the Iceberg warehouse through pyiceberg with an OIDC bearer token. `parquet_io.py` reads through DuckDB over Parquet or Iceberg, falls back to PostgreSQL, keeps the latest row when keys repeat, and holds the OHLCV read contract.

`checked_mirror.py` and `mirror_digest.py` copy PostgreSQL into Iceberg and compare a per-date content digest on both sides. The digest covers every column and ignores row order, and a mismatch fails the mirror. `lakehouse_parity.py` is a cheaper monitor that compares aggregates. `gold_pg_sync_asset.py` runs the other direction and promotes a finalised Iceberg batch into PostgreSQL within a declared scope.

## Sensors

`sensors.py` drives the daily two-phase run. `watermark_complete_sensor` is the main trigger, `stale_factor_sensor` a backstop, and `request_sensor` picks up dates written to the `factors.compute_requests` table (`db.submit_request` inserts them); nothing in this tree calls it. `core_complete_sensor` joins phase 1 to phase 2 with an age gate and a JSON cursor. It is a plain `@sensor` because `multi_asset_sensor` has a 25-event cap that deadlocks on partial-completion patterns. The rest handle the upstream feed, cache warming, the Iceberg gold sync, backfill completion and alerts.

## API

`api/main.py` mounts four routers. Every route except `/health` needs a token, and the response contract is frozen. The two cache routes depend on `require_admin`, which wants an admin claim or a listed subject once `MERIDIAN_ADMIN_SUBJECTS` is set. Before that any authenticated caller can use them and a warning is logged.

`api/cache.py` is a TTL cache bounded by `API_SCORE_CACHE_MAX_ENTRIES`. It has no per-date eviction. The cache-warm route in `api/routers/admin.py` takes the cache's lock and deletes every key containing the date, which means it reaches into the cache's internals. The cache is per process, so each uvicorn worker has its own copy.
