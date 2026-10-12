-- 2026-08-02_write_campaigns.sql
--
-- Write-campaign lineage (Meridian #294 / #295).
--
-- PROBLEM: on 2026-08-01 we could not answer "what wrote this row, from what
-- input vintage, with what code version". factors.daily_scores carried one
-- `computed_at timestamptz DEFAULT now()` and nothing else, and
-- db.copy_replace_daily_scores OMITTED that column from its COPY list so the
-- table DEFAULT stamped WRITE time rather than compute time. Identifying which
-- historical write campaign produced which rows was done by CLUSTERING
-- computed_at values; that inference is unreliable. The operational cost is the
-- blast radius: with no record of which rows consumed a bad upstream vintage,
-- the only SAFE recompute scope is the full history. This migration is what turns
-- a blanket recompute into a scoped one.
--
-- SHAPE: one `factors.write_campaigns` row per logical write episode (a
-- Dagster asset run, a burst promote, a manual backfill script invocation) and
-- a single `campaign_id bigint` on each row that episode wrote. Normalised
-- rather than four wide text columns per row: on a very large table a bigint is 8
-- bytes against ~60-120 bytes of near-constant text, and the campaign row is
-- the natural place to record the admission gate result, which has nowhere to
-- live in a per-row column at all.
--
-- input_vintage IS THE LOAD-BEARING FIELD, and it is jsonb keyed by source
-- dataset, not a scalar:
--
--   {"ohlcv.prices": {"max_inserted_at": "2026-07-31T23:04:11+00:00",
--                     "reader": "medallion:silver", "n_rows": 4123456},
--    "fundamentals.income_statement": {"max_filing_date": "2026-07-30"},
--    "factors.daily_scores": {"campaigns": [412, 413]}}
--
-- A campaign consumes several upstream datasets whose vintages move
-- independently, so no scalar is a truthful summary of what it read -
-- collapsing them hides which input was stale. The jsonb also
-- records the READER ("medallion:silver" / "flat-parquet" / "postgresql"),
-- because a silent read-path fallback is invisible in the values.
--
-- `input_vintage_max timestamptz` is the denormalised MAX of every parseable
-- timestamp in that jsonb, so the query the incident actually needs -
-- "which scores were computed from inputs read before the fix landed" - is a
-- one-line, B-tree-indexable predicate (`WHERE input_vintage_max <
-- '2026-08-01'`) instead of a jsonb walk. MAX, not MIN: the question is always
-- "could this campaign possibly have seen the fix", and a campaign whose
-- NEWEST input predates the fix definitely could not have. It is written by
-- lineage.vintage_scalar() at insert time so the two representations cannot
-- disagree.
--
-- PRE-EXISTING ROWS get the sentinel campaign_id = 0 (writer='pre_lineage'),
-- NOT a reconstruction. Their real provenance is not recoverable and pretending
-- otherwise is exactly the failure this migration exists to stop. "Unknown
-- provenance" becomes a first-class, queryable state; the sentinel disappears
-- naturally as ranges are re-scored under real campaigns.
--
-- COST ON A VERY LARGE TABLE - MEASURED, NOT ASSUMED. `ADD COLUMN <type>
-- DEFAULT <constant>` has been metadata-only since PostgreSQL 11 (the default
-- is stored in pg_attribute.attmissingval and materialised on read), and
-- `ALTER COLUMN ... DROP DEFAULT` is a catalog-only change that does NOT clear
-- attmissingval. So pre-existing rows keep reading 0 while new rows default to
-- NULL, with no table rewrite. Verified on postgresql:18 (prod is 18.3) against
-- a 2,000,000-row table:
--
--     ALTER TABLE t ADD COLUMN campaign_id bigint DEFAULT 0;   -- 3.179 ms
--     ALTER TABLE t ALTER COLUMN campaign_id DROP DEFAULT;     -- 3.457 ms
--     pg_relation_filenode unchanged (16384 -> 16384), size unchanged (84 MB)
--     pg_attribute.atthasmissing = t, attmissingval = {0}
--     2,000,000 rows read campaign_id = 0; a subsequently inserted row is NULL
--
-- Each statement still takes a brief AccessExclusiveLock, so apply this when
-- no long-running scan holds a conflicting lock on factors.daily_scores; the
-- lock is held for milliseconds, not for a rewrite.
--
-- DELIBERATELY NOT DONE HERE, and why:
--
--   * NO FOREIGN KEY from the score tables to write_campaigns. A validated FK
--     would scan the whole table once, and - worse - every FK adds a per-row
--     referential trigger that fires on INSERT, which is a direct tax on the
--     bulk-COPY promote path that exists precisely because row-at-a-time work
--     was the bottleneck. The reference is logical and documented; the writers
--     only ever insert ids they just created. A `NOT VALID` FK would skip the
--     scan but still add the per-row trigger, so it is not a middle ground.
--   * NO INDEX on factors.daily_scores(campaign_id). Building it over the whole
--     table costs hours and several GB, and while the pre_lineage sentinel
--     dominates the column its selectivity is terrible anyway. Add it later,
--     CONCURRENTLY, and prefer the partial form once real campaigns exist:
--       CREATE INDEX CONCURRENTLY idx_daily_scores_campaign
--         ON factors.daily_scores (campaign_id)
--         WHERE campaign_id IS NOT NULL AND campaign_id <> 0;
--     Blast-radius queries are also normally date-bounded, and the existing
--     (score_date, ...) indexes already carry those.
--
-- Mirrored in db.ensure_write_campaigns_table(), which is the authoritative
-- idempotent DDL (a fresh DB self-bootstraps, and a deployed image will apply
-- the same statements on its first ensure_schema call). This file documents it
-- for a human reading migration history, same convention as every other file
-- in this directory.

-- ── UP ──────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS factors.write_campaigns (
    -- BY DEFAULT (not ALWAYS) so the pre_lineage sentinel below can claim
    -- id 0 explicitly; the identity sequence starts at 1 and cannot collide.
    campaign_id       bigint      GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    -- 'dagster_asset' | 'burst_promote' | 'manual_script' | 'pre_lineage'
    writer            text        NOT NULL,
    code_version      text,
    run_id            text,
    input_vintage     jsonb       NOT NULL DEFAULT '{}'::jsonb,
    input_vintage_max timestamptz,
    started_at        timestamptz NOT NULL DEFAULT now(),
    finished_at       timestamptz,
    gate_result       text,
    detail            jsonb       NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_write_campaigns_writer_started
    ON factors.write_campaigns (writer, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_write_campaigns_vintage_max
    ON factors.write_campaigns (input_vintage_max);
CREATE INDEX IF NOT EXISTS idx_write_campaigns_run
    ON factors.write_campaigns (run_id) WHERE run_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_write_campaigns_vintage
    ON factors.write_campaigns USING gin (input_vintage);

-- The sentinel every pre-lineage row resolves to. gate_result records that
-- nothing gated these rows into gold, which is the honest answer.
INSERT INTO factors.write_campaigns
    (campaign_id, writer, code_version, run_id, input_vintage,
     input_vintage_max, started_at, finished_at, gate_result, detail)
VALUES
    (0, 'pre_lineage', NULL, NULL, '{}'::jsonb, NULL, now(), now(), 'unverified',
     jsonb_build_object(
       'note', 'Rows written before write-campaign lineage existed (Meridian #294). '
               'Real provenance is NOT recoverable and was deliberately not reconstructed; '
               'treat as unknown-provenance. Shrinks as ranges are re-scored.',
       'campaign_origin', 'sentinel'))
ON CONFLICT (campaign_id) DO NOTHING;

-- Score / risk tables get the stamp. Two statements each: the constant DEFAULT
-- makes every pre-existing row read the sentinel with no rewrite, and dropping
-- it makes a NEW unstamped row NULL instead - so an unstamped write after this
-- migration is visibly a bug rather than silently indistinguishable from
-- history. Both statements are metadata-only (see the measurement above).
ALTER TABLE factors.daily_scores      ADD COLUMN IF NOT EXISTS campaign_id bigint DEFAULT 0;
ALTER TABLE factors.daily_scores      ALTER COLUMN campaign_id DROP DEFAULT;
ALTER TABLE factors.group_scores      ADD COLUMN IF NOT EXISTS campaign_id bigint DEFAULT 0;
ALTER TABLE factors.group_scores      ALTER COLUMN campaign_id DROP DEFAULT;
ALTER TABLE factors.composite_scores  ADD COLUMN IF NOT EXISTS campaign_id bigint DEFAULT 0;
ALTER TABLE factors.composite_scores  ALTER COLUMN campaign_id DROP DEFAULT;
ALTER TABLE factors.factor_returns    ADD COLUMN IF NOT EXISTS campaign_id bigint DEFAULT 0;
ALTER TABLE factors.factor_returns    ALTER COLUMN campaign_id DROP DEFAULT;
ALTER TABLE factors.specific_returns  ADD COLUMN IF NOT EXISTS campaign_id bigint DEFAULT 0;
ALTER TABLE factors.specific_returns  ALTER COLUMN campaign_id DROP DEFAULT;
ALTER TABLE factors.specific_risk     ADD COLUMN IF NOT EXISTS campaign_id bigint DEFAULT 0;
ALTER TABLE factors.specific_risk     ALTER COLUMN campaign_id DROP DEFAULT;
ALTER TABLE factors.factor_covariance ADD COLUMN IF NOT EXISTS campaign_id bigint DEFAULT 0;
ALTER TABLE factors.factor_covariance ALTER COLUMN campaign_id DROP DEFAULT;

-- ── DOWN (apply manually if this needs to be rolled back) ─────────────────
-- Safe at any time: no consumer reads campaign_id, and dropping a column is
-- also metadata-only. Drop the columns BEFORE the table so nothing dangles.
--
-- ALTER TABLE factors.daily_scores      DROP COLUMN IF EXISTS campaign_id;
-- ALTER TABLE factors.group_scores      DROP COLUMN IF EXISTS campaign_id;
-- ALTER TABLE factors.composite_scores  DROP COLUMN IF EXISTS campaign_id;
-- ALTER TABLE factors.factor_returns    DROP COLUMN IF EXISTS campaign_id;
-- ALTER TABLE factors.specific_returns  DROP COLUMN IF EXISTS campaign_id;
-- ALTER TABLE factors.specific_risk     DROP COLUMN IF EXISTS campaign_id;
-- ALTER TABLE factors.factor_covariance DROP COLUMN IF EXISTS campaign_id;
-- DROP TABLE IF EXISTS factors.write_campaigns;
