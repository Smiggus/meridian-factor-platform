# Meridian

Meridian computes cross-sectional factor scores for an equity universe, point in time, and serves them over a read-only REST API. This is a cut-down copy of the private repository with the factor research taken out. What remains is the plumbing: how scores get into PostgreSQL, the guards on bad input, and the machinery for backfilling without putting the live table at risk. Placeholder modules sit where the research plugs in (see Stand-ins below).

## Layout

Scores live in `factors.daily_scores`. The FastAPI app in `api/` reads them and never writes. Auth is JWT, responses are cached in process, and the response contract only grows: fields get added, never renamed or removed.

```
api/                      FastAPI app: auth, cache, scores / universe / risk / admin routers
api_db.py config.py       engine URL, settings
factor_store.py           reads behind the API
risk_metrics.py           risk metrics read path
dagster/meridian_pipeline/
  db.py lineage.py        write path, write campaigns
  lakehouse.py            Iceberg access (pyiceberg, OIDC)
  parquet_io.py           DuckDB / Parquet reads, mirror appends
  checked_mirror*.py      PostgreSQL <-> Iceberg checked mirror
  ohlcv_hygiene.py ...    price guards
  checks/                 asset checks
  sensors.py jobs.py      orchestration and alerting
  k8s_config.py           GPU routing, step-pod resource tiers
dagster/db/migrations/    schema for write campaigns
scripts/burst/            burst driver, staging, promote, verify
scripts/checks/           mirror and replay coverage verifiers
tests/                    about 60 test modules
```

## The write side

Most of the code is here. `dagster/meridian_pipeline/db.py` replaces a date's rows for a declared set of factors in one transaction. It takes per-date advisory locks in a fixed order, so concurrent writers cannot deadlock. A write that would delete a factor without putting it back is refused, and so is a write to a future date. A batch replace over a range is checked against the trading calendar for missing days. Each write belongs to a campaign (`lineage.py`) that records who wrote it, with which code version and from which input data.

Around that sit the guards for bad prices, reused tickers and corrupt returns (`ohlcv_hygiene.py`, `adj_close_guard.py`, `splice_guard.py`, `checks/`), and the burst path in `scripts/burst/`. A separate machine computes a backfill and stages it in object storage; the cluster checks it and promotes it. The machine never writes the real table. There is also Iceberg and Parquet IO, read through DuckDB with a PostgreSQL fallback, and the Dagster sensors and ntfy alerts around the daily job.

`docs/ARCHITECTURE.md` goes into the write path, the promote and the mirrors.

## Stand-ins

`factor_definitions.py` and `partition_scoring.py` hold a small synthetic catalogue, and `score_partition` raises. `definitions.py` registers placeholder assets under the real keys so the jobs and sensors resolve.

## Tests

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r dagster/requirements.txt -r requirements.txt -r requirements-test.txt
pytest -q
ruff check . && ruff format --check .
```

No database or network is needed. Databases are faked, the notification URL points nowhere, and the notifier will not send under pytest. The risk-metrics GPU tests skip without `torch`. The burst driver needs `dask[distributed]` on whichever machine runs it.

## Running the API

```bash
cp .env.example .env     # set JWT_SECRET and the PG* variables
uvicorn api.main:app --port 8100
```

It reads `factors.daily_scores`, `group_scores` and `composite_scores`, so it wants a PostgreSQL that has them. `db.ensure_schema` creates what the write path needs.

## Ticket numbers

Comments cite tickets and merge requests (`#294`, `!474`, `LDP #189`) from the private tracker, and a few point at `docs/` pages that stayed in the private repository. Comments also name modules that are not included here, such as `assets.py`. None of those references resolve from here. They were left in because the number is often the fastest way to find the incident behind a guard, and the comment next to it usually says what went wrong.
