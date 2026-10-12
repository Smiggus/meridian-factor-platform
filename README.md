# Meridian (engineering core)

The data-engineering half of a point-in-time equity factor scoring service:
a read-only FastAPI layer over PostgreSQL, the concurrency-safe score write
path, data-quality and hygiene checks, the burst ship-back / promote / verify
machinery, lakehouse and Parquet IO, alerting sensors, and the container and
CI configuration around them.

This is a **public showcase export** of a private codebase. It is deliberately
small (about 150 files) and shows the *engineering*, not the research. Which
factors exist, how they are built and weighted, the regime and ML models, and
every result are withheld; see [What is not here](#what-is-not-here).

## What is here

| Area | Where | What it shows |
|---|---|---|
| Read API | `api/`, `api_db.py`, `config.py`, `factor_store.py` | Authenticated, read-only FastAPI service over `factors.daily_scores`. Frozen response contract (fields are only ever added), JWT auth, an in-process score cache with a size bound and explicit cache-warm route, NaN-safe serialisation, an as-of date resolver. |
| Score write path | `dagster/meridian_pipeline/db.py` | Delete-replace writes scoped to a declared set of owned factors, per-date PostgreSQL advisory locks taken in a fixed order (no deadlock), lock timeouts, a bulk `COPY` path, a future-date guard, a trading-calendar hole check for batch replaces, and refusal to delete a factor the payload does not carry. |
| Write lineage | `lineage.py`, `db.py`, `dagster/db/migrations/` | Every write belongs to a campaign that records the writer, code version and input vintage, so a bad input can be traced to the exact rows it touched. Additive by design: lineage can never fail a write. |
| Readiness gates | `db.py`, `sensors.py` | A cross-section is only scored when the upstream watermark says the date is complete, and a single-dated check refuses a cross-section that silently mixes as-of dates. |
| Hygiene and checks | `ohlcv_hygiene.py`, `adj_close_guard.py`, `splice_guard.py`, `checks/` | Bogus-print and delisting guards, ticker-reuse splice protection, a blocking OHLCV read-boundary contract, a `ret_1d` corruption tripwire, a trading-day invariant sweep, range verification. |
| Burst compute | `scripts/burst/` | Off-cluster compute over Dagster Pipes and object storage: the box writes a staging prefix and never writes gold; the cluster hole-checks, parity-gates and promotes. Resumable by per-partition sentinels. |
| Verifiers | `scripts/check_trading_day_invariant.py`, `scripts/verify_range.py`, `scripts/checks/`, `scripts/burst/verify_promoted_range.py` | Independent read-only certification of promoted ranges and mirrors. |
| Lakehouse and Parquet IO | `lakehouse.py`, `parquet_io.py`, `checked_mirror.py`, `mirror_digest.py`, `lakehouse_parity.py` | Iceberg warehouse access with OIDC, a DuckDB Parquet read path with a PostgreSQL fallback, an exact per-date content-digest mirror between PostgreSQL and Iceberg, and a parity monitor. |
| Alerting and orchestration | `sensors.py`, `notify.py`, `checks/alerting.py` | Watermark-driven run triggers, a two-phase run bridge, failed-check paging with once-per-key de-duplication, fail-open notifications that refuse to send under pytest. |
| Deployment | `Dockerfile`, `dagster/Dockerfile`, `dagster/meridian_pipeline/k8s_config.py`, `.github/workflows/ci.yml` | Image builds, GPU routing, measured step-pod resource tiers, lint / format / test CI. |

`docs/ARCHITECTURE.md` walks through the write path, the burst promote and the
mirror in more detail.

## What is not here

- **The factor catalogue and construction.** `dagster/meridian_pipeline/factor_definitions.py`
  and `partition_scoring.py` are small stubs that keep the *interfaces* the
  engineering code depends on (a group-to-factor catalogue, the owned-factor
  scope of a promote, `score_partition`) over a synthetic example catalogue.
  `score_partition` raises. `asset_selections.py` and `definitions.py` are
  likewise reduced: the factor assets are registered as stand-ins so the jobs and
  sensors resolve against a complete asset graph.
- **Regime and ML models, weight optimisation, the activation / registry /
  evidence / grading machinery, research scripts and notebooks, and every
  recorded finding.** Large modules that mix engineering with these (`db.py`,
  `sensors.py`, `parquet_io.py`, `k8s_config.py`) are shipped as the real source
  of the engineering subset only.
- **Data.** No vendor data, no trained models, no credentials. Hosts are
  placeholders.

## Layout

```
api/                      FastAPI app: main, auth, cache, scores / universe / risk / admin routers
api_db.py config.py       engine URL, settings
factor_store.py           PostgreSQL / Iceberg-first reads behind the API
risk_metrics.py           risk metrics read path and its on-the-fly fallback
dagster/meridian_pipeline/
  db.py                   the write path (see above)
  lineage.py              write-campaign lineage
  lakehouse.py            Iceberg warehouse access (pyiceberg, OIDC)
  parquet_io.py           DuckDB / Parquet reads, mirror appends, OHLCV contract
  checked_mirror*.py      PostgreSQL <-> Iceberg checked mirror
  ohlcv_hygiene.py ...    hygiene guards
  checks/                 asset checks
  sensors.py jobs.py      alerting and orchestration
  k8s_config.py           GPU routing and step-pod resource tiers
  definitions.py          code-location entry point (factor assets are stand-ins)
dagster/db/migrations/    schema for the write campaigns
scripts/burst/            burst ship-back: driver, staging, promote, verify
scripts/checks/           mirror and replay coverage verifiers
tests/                    about 60 test modules, no database or network needed
```

## Run the tests

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r dagster/requirements.txt -r requirements.txt -r requirements-test.txt
pytest -q
ruff check . && ruff format --check .
```

The suite is hermetic: databases are faked, the notification URL points at a
black hole and the notifier refuses to send under pytest. Tests that need
`torch` (the risk-metrics GPU kernel) skip without it, and the burst driver
needs `dask[distributed]` on the box that runs it.

## Local API

```bash
cp .env.example .env     # set JWT_SECRET and the PG* variables
uvicorn api.main:app --port 8100
```

The API reads `factors.daily_scores`, `group_scores` and `composite_scores`; it
needs a PostgreSQL that has them. The write path creates the schema it needs
(`db.ensure_schema`).
