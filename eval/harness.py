"""
eval/harness.py

Evaluation Harness — runs the 15 test cases through the full pipeline,
scores all 6 dimensions per case, and stores results in PostgreSQL.

Architecture:
  - Each case runs the full Orchestrator pipeline (same code path as /query)
  - Scoring happens after the pipeline completes
  - Results stored in eval_runs + eval_case_results tables
  - Summary statistics computed after all cases complete

Called by:
  - worker/tasks.py:run_eval_task()      — full 15-case eval
  - worker/tasks.py:run_reeval_task()    — targeted subset re-eval

The harness is intentionally sequential (not parallel) to avoid DB contention
and to produce deterministic trace sequences for debugging.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime
from typing import Any

import anthropic
import instructor

from api.config import get_settings
from api.context_manager import BudgetManager
from api.models.context import SharedContext
from api.orchestrator import Orchestrator
from eval.cases.test_cases import EvalCase, ALL_CASES, get_cases_by_category
from eval.scorers import score_all, SCORER_NAMES

logger = logging.getLogger(__name__)


# ─── DB helpers ───────────────────────────────────────────────────────────────

async def _insert_eval_run(conn, run_id: str, triggered_by: str, cases_count: int) -> None:
    """Insert an eval_runs row."""
    try:
        await conn.execute(
            """
            INSERT INTO eval_runs (id, triggered_by, cases_count, status)
            VALUES ($1, $2, $3, 'running')
            """,
            run_id, triggered_by, cases_count,
        )
    except Exception as e:
        logger.error("eval_run_insert_failed", extra={"error": str(e)})


async def _update_eval_run(conn, run_id: str, summary: dict) -> None:
    """Mark eval_run as complete with summary JSON."""
    try:
        await conn.execute(
            """
            UPDATE eval_runs
            SET status = 'complete',
                summary = $2::jsonb,
                completed_at = NOW()
            WHERE id = $1
            """,
            run_id, json.dumps(summary, default=str),
        )
    except Exception as e:
        logger.error("eval_run_update_failed", extra={"error": str(e)})


async def _insert_case_result(
    conn,
    run_id: str,
    case: EvalCase,
    scores: dict,
    ctx: SharedContext,
) -> None:
    """Insert an eval_case_results row."""
    try:
        await conn.execute(
            """
            INSERT INTO eval_case_results
              (run_id, case_id, category, query, final_answer,
               correctness, citations, contradictions,
               tool_efficiency, budget_compliance, critique_agreement,
               weighted_total, justifications, latency_ms)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
            """,
            run_id,
            case.case_id,
            case.category,
            case.query,
            ctx.final_answer or "",
            scores.get("correctness", {}).get("score", 0.0),
            scores.get("citations", {}).get("score", 0.0),
            scores.get("contradictions", {}).get("score", 0.0),
            scores.get("tool_efficiency", {}).get("score", 0.0),
            scores.get("budget_compliance", {}).get("score", 0.0),
            scores.get("critique_agreement", {}).get("score", 0.0),
            scores.get("weighted_total", 0.0),
            json.dumps({k: scores[k].get("justification") for k in SCORER_NAMES if k in scores}),
            sum(ao.latency_ms for ao in ctx.agent_outputs.values()),
        )
    except Exception as e:
        logger.error("case_result_insert_failed", extra={"error": str(e), "case_id": case.case_id})


# ─── Summary computation ──────────────────────────────────────────────────────

def _compute_summary(all_scores: list[dict], cases: list[EvalCase]) -> dict:
    """
    Compute summary statistics across all scored cases.

    Returns:
        summary_by_category: per-category mean scores for each dimension
        summary_by_dimension: per-dimension stats (mean, worst case, best case)
    """
    by_category: dict[str, list[dict]] = {"baseline": [], "ambiguous": [], "adversarial": []}
    for case, scores in zip(cases, all_scores):
        by_category[case.category].append(scores)

    summary_by_category = {}
    for cat, cat_scores in by_category.items():
        if not cat_scores:
            continue
        summary_by_category[cat] = {
            dim: round(sum(s[dim]["score"] for s in cat_scores) / len(cat_scores), 4)
            for dim in SCORER_NAMES
            if all(dim in s for s in cat_scores)
        }
        summary_by_category[cat]["weighted_total"] = round(
            sum(s["weighted_total"] for s in cat_scores) / len(cat_scores), 4
        )

    summary_by_dimension = {}
    for dim in SCORER_NAMES:
        dim_scores = [(cases[i].case_id, all_scores[i][dim]["score"]) for i in range(len(cases)) if dim in all_scores[i]]
        if not dim_scores:
            continue
        scores_only = [s for _, s in dim_scores]
        worst_case = min(dim_scores, key=lambda x: x[1])
        best_case = max(dim_scores, key=lambda x: x[1])
        summary_by_dimension[dim] = {
            "mean_score": round(sum(scores_only) / len(scores_only), 4),
            "min_score": round(min(scores_only), 4),
            "max_score": round(max(scores_only), 4),
            "worst_case": worst_case[0],
            "best_case": best_case[0],
        }

    overall = [s["weighted_total"] for s in all_scores]
    return {
        "summary_by_category": summary_by_category,
        "summary_by_dimension": summary_by_dimension,
        "overall_mean": round(sum(overall) / len(overall), 4) if overall else 0.0,
        "cases_run": len(cases),
        "cases_failed": sum(1 for s in all_scores if s.get("weighted_total", 0) < 0.4),
    }


# ─── Single case runner ───────────────────────────────────────────────────────

async def run_single_case(
    case: EvalCase,
    llm_client: Any,
    db_conn=None,
    run_id: str | None = None,
) -> tuple[SharedContext, dict]:
    """
    Run a single eval case through the full pipeline and score it.
    Returns (ctx, scores).
    """
    logger.info("eval_case_start", extra={"case_id": case.case_id, "category": case.category})
    t0 = time.perf_counter()

    ctx = SharedContext(query=case.query)
    bm = BudgetManager(ctx)
    orchestrator = Orchestrator(ctx, bm, db_session=db_conn)

    try:
        await orchestrator.run()
    except Exception as e:
        logger.error("eval_pipeline_failed", extra={"case_id": case.case_id, "error": str(e)})
        ctx.final_answer = f"Pipeline failed: {e}"

    latency = int((time.perf_counter() - t0) * 1000)
    logger.info("eval_case_pipeline_done", extra={
        "case_id": case.case_id, "latency_ms": latency,
        "final_answer_length": len(ctx.final_answer or ""),
    })

    # Score all dimensions
    scores = await score_all(case, ctx, llm_client)
    logger.info("eval_case_scored", extra={
        "case_id": case.case_id,
        "weighted_total": scores.get("weighted_total"),
        "correctness": scores.get("correctness", {}).get("score"),
    })

    # Persist to DB
    if db_conn and run_id:
        await _insert_case_result(db_conn, run_id, case, scores, ctx)

    return ctx, scores


# ─── Full eval harness ────────────────────────────────────────────────────────

async def run_eval(
    cases: list[EvalCase] | None = None,
    db_conn=None,
    triggered_by: str = "manual",
) -> dict:
    """
    Run the evaluation harness over a list of cases (defaults to ALL_CASES).

    Returns the full summary dict including per-category and per-dimension stats.
    """
    settings = get_settings()
    if cases is None:
        cases = ALL_CASES

    run_id = str(uuid.uuid4())
    logger.info("eval_run_start", extra={"run_id": run_id, "cases": len(cases), "triggered_by": triggered_by})

    # Create eval_runs row
    if db_conn:
        await _insert_eval_run(db_conn, run_id, triggered_by, len(cases))

    # Build the LLM client for scoring
    raw_client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
    llm_client = instructor.from_anthropic(raw_client)

    # Run cases sequentially
    all_scores = []
    for i, case in enumerate(cases):
        logger.info("eval_progress", extra={
            "run_id": run_id, "case": f"{i+1}/{len(cases)}", "case_id": case.case_id,
        })
        try:
            _, scores = await run_single_case(case, llm_client, db_conn, run_id)
            all_scores.append(scores)
        except Exception as e:
            logger.error("eval_case_fatal", extra={"case_id": case.case_id, "error": str(e)})
            # Push a zero-score result so the run still completes
            all_scores.append({dim: {"score": 0.0, "justification": f"Fatal: {e}"} for dim in SCORER_NAMES} | {"weighted_total": 0.0})

    # Compute summary
    summary = _compute_summary(all_scores, cases)
    summary["run_id"] = run_id
    summary["triggered_by"] = triggered_by

    # Update eval_runs row
    if db_conn:
        await _update_eval_run(db_conn, run_id, summary)

    logger.info("eval_run_complete", extra={
        "run_id": run_id,
        "overall_mean": summary["overall_mean"],
        "cases_run": summary["cases_run"],
    })

    return summary
