-- ============================================================
-- db/migrations/001_eval_schema_v2.sql
-- Adds columns needed by the eval harness and meta agent
-- Run after schema.sql on existing instances.
-- For fresh containers, schema.sql v2 already includes these.
-- ============================================================

-- eval_runs: relax run_hash (harness doesn't compute one), add status + cases_count
ALTER TABLE eval_runs ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'running'
    CHECK (status IN ('running', 'complete', 'failed'));
ALTER TABLE eval_runs ADD COLUMN IF NOT EXISTS cases_count INT NOT NULL DEFAULT 0;
ALTER TABLE eval_runs ADD COLUMN IF NOT EXISTS completed_at TIMESTAMPTZ;
ALTER TABLE eval_runs ALTER COLUMN run_hash DROP NOT NULL;
ALTER TABLE eval_runs DROP CONSTRAINT IF EXISTS eval_runs_run_hash_key;

-- eval_case_results: add flat columns used by harness INSERT
ALTER TABLE eval_case_results ADD COLUMN IF NOT EXISTS query TEXT;
ALTER TABLE eval_case_results ADD COLUMN IF NOT EXISTS final_answer TEXT;
ALTER TABLE eval_case_results ADD COLUMN IF NOT EXISTS correctness FLOAT;
ALTER TABLE eval_case_results ADD COLUMN IF NOT EXISTS citations FLOAT;
ALTER TABLE eval_case_results ADD COLUMN IF NOT EXISTS contradictions FLOAT;
ALTER TABLE eval_case_results ADD COLUMN IF NOT EXISTS tool_efficiency FLOAT;
ALTER TABLE eval_case_results ADD COLUMN IF NOT EXISTS budget_compliance FLOAT;
ALTER TABLE eval_case_results ADD COLUMN IF NOT EXISTS critique_agreement FLOAT;
ALTER TABLE eval_case_results ADD COLUMN IF NOT EXISTS weighted_total FLOAT;
ALTER TABLE eval_case_results ADD COLUMN IF NOT EXISTS justifications JSONB;
ALTER TABLE eval_case_results ADD COLUMN IF NOT EXISTS latency_ms INT;

-- prompt_rewrites: align column names with meta agent INSERT
ALTER TABLE prompt_rewrites ADD COLUMN IF NOT EXISTS analysis TEXT;
ALTER TABLE prompt_rewrites ADD COLUMN IF NOT EXISTS prompt_before TEXT;
ALTER TABLE prompt_rewrites ADD COLUMN IF NOT EXISTS prompt_after TEXT;
ALTER TABLE prompt_rewrites ADD COLUMN IF NOT EXISTS diff_summary TEXT;
ALTER TABLE prompt_rewrites ADD COLUMN IF NOT EXISTS expected_improvement TEXT;
ALTER TABLE prompt_rewrites ADD COLUMN IF NOT EXISTS confidence FLOAT;
ALTER TABLE prompt_rewrites ALTER COLUMN old_prompt DROP NOT NULL;
ALTER TABLE prompt_rewrites ALTER COLUMN new_prompt DROP NOT NULL;
ALTER TABLE prompt_rewrites ALTER COLUMN justification DROP NOT NULL;
ALTER TABLE prompt_rewrites ALTER COLUMN expected_improvement DROP NOT NULL;

-- agent_prompts: align column names with meta agent fetch
ALTER TABLE agent_prompts ADD COLUMN IF NOT EXISTS system_prompt TEXT;
ALTER TABLE agent_prompts ADD COLUMN IF NOT EXISTS version_label TEXT;
UPDATE agent_prompts SET system_prompt = prompt_text WHERE system_prompt IS NULL;
UPDATE agent_prompts SET version_label = 'v' || version::TEXT WHERE version_label IS NULL;
