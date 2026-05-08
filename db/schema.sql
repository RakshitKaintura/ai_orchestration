-- ============================================================
-- Mega AI — Master Database Schema
-- PostgreSQL 15
-- Run order: this file is loaded first by docker-entrypoint-initdb.d
-- ============================================================

-- Enable UUID generation
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ─── Jobs ────────────────────────────────────────────────────────────────────
-- Represents a single user query submission and its lifecycle.
CREATE TABLE IF NOT EXISTS jobs (
    id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    query       TEXT        NOT NULL,
    status      TEXT        NOT NULL DEFAULT 'queued'
                            CHECK (status IN ('queued', 'running', 'done', 'failed')),
    error       TEXT,                                       -- populated on failure
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at  TIMESTAMPTZ,
    completed_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at DESC);

-- ─── Execution Traces ────────────────────────────────────────────────────────
-- One row per atomic event in the pipeline.
-- event_type values:
--   orchestrator_plan | agent_start | agent_end | tool_call | tool_result
--   budget_check | budget_violation | compression_triggered | job_complete | job_failed
CREATE TABLE IF NOT EXISTS trace_events (
    id              BIGSERIAL   PRIMARY KEY,
    job_id          UUID        NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    seq             INT         NOT NULL,                   -- ordering within a job (1-indexed)
    agent_id        TEXT,                                   -- NULL for job-level events
    event_type      TEXT        NOT NULL,
    input_hash      TEXT,                                   -- SHA-256[:16] of the input payload
    output_hash     TEXT,                                   -- SHA-256[:16] of the output payload
    payload         JSONB       NOT NULL DEFAULT '{}',      -- full input/output/scores/reasoning
    latency_ms      INT,
    token_count     INT,
    policy_violations JSONB     DEFAULT '[]',               -- list of violation strings
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_trace_events_job_id ON trace_events(job_id);
CREATE INDEX IF NOT EXISTS idx_trace_events_job_seq ON trace_events(job_id, seq);
CREATE INDEX IF NOT EXISTS idx_trace_events_event_type ON trace_events(event_type);
CREATE INDEX IF NOT EXISTS idx_trace_events_agent_id ON trace_events(agent_id);

-- ─── Eval Runs ───────────────────────────────────────────────────────────────
-- Represents a full evaluation run over the 15 test cases.
CREATE TABLE IF NOT EXISTS eval_runs (
    id                  UUID    PRIMARY KEY DEFAULT gen_random_uuid(),
    run_hash            TEXT    UNIQUE NOT NULL,            -- deterministic hash of prompts + case IDs
    triggered_by        TEXT    NOT NULL DEFAULT 'manual'   -- manual | rewrite_approval | auto
                                CHECK (triggered_by IN ('manual', 'rewrite_approval', 'auto')),
    rewrite_id          UUID,                               -- FK set after prompt_rewrites table exists
    scores              JSONB   NOT NULL DEFAULT '{}',      -- raw per-case scores keyed by case_id
    summary             JSONB   NOT NULL DEFAULT '{}',      -- aggregate by category and by dimension
    prompts_snapshot    JSONB   NOT NULL DEFAULT '{}',      -- exact agent prompts used this run
    cases_run           TEXT[]  NOT NULL DEFAULT '{}',      -- list of case IDs included
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_eval_runs_created_at ON eval_runs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_eval_runs_run_hash ON eval_runs(run_hash);

-- ─── Eval Case Results ────────────────────────────────────────────────────────
-- One row per (eval_run, test_case) combination.
CREATE TABLE IF NOT EXISTS eval_case_results (
    id              BIGSERIAL   PRIMARY KEY,
    run_id          UUID        NOT NULL REFERENCES eval_runs(id) ON DELETE CASCADE,
    case_id         TEXT        NOT NULL,
    category        TEXT        NOT NULL
                    CHECK (category IN ('baseline', 'ambiguous', 'adversarial')),
    -- Numeric scores per dimension (0.0–1.0)
    score_correctness       FLOAT,
    score_citations         FLOAT,
    score_contradictions    FLOAT,
    score_tool_efficiency   FLOAT,
    score_budget_compliance FLOAT,
    score_critique_agreement FLOAT,
    -- Textual justifications per dimension
    justification_correctness       TEXT,
    justification_citations         TEXT,
    justification_contradictions    TEXT,
    justification_tool_efficiency   TEXT,
    justification_budget_compliance TEXT,
    justification_critique_agreement TEXT,
    -- Computed aggregate
    overall_score   FLOAT       GENERATED ALWAYS AS (
        (
            COALESCE(score_correctness, 0) +
            COALESCE(score_citations, 0) +
            COALESCE(score_contradictions, 0) +
            COALESCE(score_tool_efficiency, 0) +
            COALESCE(score_budget_compliance, 0) +
            COALESCE(score_critique_agreement, 0)
        ) / 6.0
    ) STORED,
    passed          BOOLEAN,
    job_id          UUID        REFERENCES jobs(id),        -- the pipeline job that produced this result
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_eval_case_results_run_id ON eval_case_results(run_id);
CREATE INDEX IF NOT EXISTS idx_eval_case_results_category ON eval_case_results(category);
CREATE INDEX IF NOT EXISTS idx_eval_case_results_case_id ON eval_case_results(case_id);

-- ─── Prompt Rewrites ─────────────────────────────────────────────────────────
-- Every proposed rewrite from the meta-agent is stored here.
-- Rewrites are NEVER auto-applied. Human approval is required.
CREATE TABLE IF NOT EXISTS prompt_rewrites (
    id                  UUID    PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id              UUID    NOT NULL REFERENCES eval_runs(id),
    agent_id            TEXT    NOT NULL,                   -- which agent's prompt is being rewritten
    dimension           TEXT    NOT NULL,                   -- which scoring dimension triggered this
    old_prompt          TEXT    NOT NULL,
    new_prompt          TEXT    NOT NULL,
    diff                JSONB   NOT NULL DEFAULT '[]',      -- [{type: add|remove|change, line: str, justification: str}]
    justification       TEXT    NOT NULL,
    expected_improvement TEXT   NOT NULL,
    status              TEXT    NOT NULL DEFAULT 'pending'
                        CHECK (status IN ('pending', 'approved', 'rejected')),
    reviewed_by         TEXT,                               -- identifier of the reviewer (future: auth)
    reviewed_at         TIMESTAMPTZ,
    re_eval_run_id      UUID    REFERENCES eval_runs(id),   -- populated after re-eval completes
    delta               JSONB,                              -- {dimension: {before: float, after: float, delta: float}}
    failed_case_ids     TEXT[]  NOT NULL DEFAULT '{}',      -- cases that were failing and triggered this rewrite
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_prompt_rewrites_status ON prompt_rewrites(status);
CREATE INDEX IF NOT EXISTS idx_prompt_rewrites_run_id ON prompt_rewrites(run_id);
CREATE INDEX IF NOT EXISTS idx_prompt_rewrites_agent_id ON prompt_rewrites(agent_id);
CREATE INDEX IF NOT EXISTS idx_prompt_rewrites_created_at ON prompt_rewrites(created_at DESC);

-- Add FK from eval_runs.rewrite_id now that prompt_rewrites exists
ALTER TABLE eval_runs
    ADD CONSTRAINT fk_eval_runs_rewrite_id
    FOREIGN KEY (rewrite_id) REFERENCES prompt_rewrites(id)
    DEFERRABLE INITIALLY DEFERRED;

-- ─── Agent Prompts Registry ───────────────────────────────────────────────────
-- Tracks the active prompt version for each agent.
-- The "active" flag must only be true for one row per agent_id at a time.
CREATE TABLE IF NOT EXISTS agent_prompts (
    id          UUID    PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id    TEXT    NOT NULL,
    version     INT     NOT NULL DEFAULT 1,
    prompt_text TEXT    NOT NULL,
    is_active   BOOLEAN NOT NULL DEFAULT TRUE,
    rewrite_id  UUID    REFERENCES prompt_rewrites(id),     -- NULL for original prompts
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (agent_id, version)
);

CREATE INDEX IF NOT EXISTS idx_agent_prompts_agent_id ON agent_prompts(agent_id);
CREATE INDEX IF NOT EXISTS idx_agent_prompts_active ON agent_prompts(agent_id, is_active);

-- Helper function: deactivate old prompts when a new one is activated
CREATE OR REPLACE FUNCTION deactivate_old_prompts()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.is_active THEN
        UPDATE agent_prompts
        SET is_active = FALSE
        WHERE agent_id = NEW.agent_id
          AND id != NEW.id
          AND is_active = TRUE;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_deactivate_old_prompts
    AFTER INSERT OR UPDATE ON agent_prompts
    FOR EACH ROW EXECUTE FUNCTION deactivate_old_prompts();

-- ─── Sample Product Data (for NL→SQL tool) ────────────────────────────────────
-- A small 50-row catalogue used by the structured data lookup tool.
CREATE TABLE IF NOT EXISTS products (
    id          SERIAL  PRIMARY KEY,
    name        TEXT    NOT NULL,
    category    TEXT    NOT NULL,
    price_usd   NUMERIC(10,2) NOT NULL,
    stock       INT     NOT NULL DEFAULT 0,
    rating      NUMERIC(3,2),
    created_at  DATE    NOT NULL DEFAULT CURRENT_DATE
);

CREATE INDEX IF NOT EXISTS idx_products_category ON products(category);
CREATE INDEX IF NOT EXISTS idx_products_price ON products(price_usd);
