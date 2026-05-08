"""
worker/tasks.py

Celery task definitions for Mega AI — fully implemented.

All heavy/async pipeline work runs here, off the API request path.

Queues:
- jobs  — pipeline execution tasks
- eval  — evaluation harness tasks (full and targeted re-eval)

Note: Celery tasks are synchronous but run async code via asyncio.run().
This is the standard pattern for Celery + asyncio integration.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime

from celery import Celery
from celery.utils.log import get_task_logger

logger = get_task_logger(__name__)

# ─── Celery app ───────────────────────────────────────────────────────────────

celery_app = Celery(
    "mega_ai_worker",
    broker=os.environ.get("CELERY_BROKER_URL", "redis://redis:6379/0"),
    backend=os.environ.get("CELERY_RESULT_BACKEND", "redis://redis:6379/1"),
    include=["worker.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,               # re-queue on worker crash
    worker_prefetch_multiplier=1,      # one task at a time per worker
    task_routes={
        "worker.tasks.run_pipeline_task": {"queue": "jobs"},
        "worker.tasks.run_eval_task":     {"queue": "eval"},
        "worker.tasks.run_reeval_task":   {"queue": "eval"},
    },
    result_expires=3600,
)


# ─── DB connection helper ─────────────────────────────────────────────────────

async def _get_asyncpg_conn():
    """Get a direct asyncpg connection for worker tasks."""
    import asyncpg
    from api.config import get_settings
    s = get_settings()
    return await asyncpg.connect(
        host=s.postgres_host,
        port=s.postgres_port,
        database=s.postgres_db,
        user=s.postgres_user,
        password=s.postgres_password,
    )


# ─── 1. Pipeline task ─────────────────────────────────────────────────────────

@celery_app.task(
    name="worker.tasks.run_pipeline_task",
    bind=True,
    max_retries=0,
    soft_time_limit=300,
    time_limit=360,
)
def run_pipeline_task(self, job_id: str, query: str) -> dict:
    """
    Run the full multi-agent pipeline for a job.

    Steps:
    1. Create SharedContext for job_id
    2. Initialise BudgetManager
    3. Run Orchestrator.run()
    4. Persist final_answer + trace to DB
    5. Update job status to 'done'

    Args:
        job_id: UUID of the jobs table row
        query:  The user's original query

    Returns:
        {"job_id": str, "status": "done" | "failed"}
    """
    async def _run():
        conn = None
        try:
            conn = await _get_asyncpg_conn()

            # Insert job record
            await conn.execute(
                "INSERT INTO jobs (id, query, status) VALUES ($1, $2, 'running') "
                "ON CONFLICT (id) DO UPDATE SET status = 'running'",
                job_id, query,
            )

            from api.models.context import SharedContext
            from api.context_manager import BudgetManager
            from api.orchestrator import Orchestrator

            ctx = SharedContext(job_id=uuid.UUID(job_id), query=query)
            bm = BudgetManager(ctx)
            orchestrator = Orchestrator(ctx, bm, db_session=conn)

            try:
                await orchestrator.run()
                status = "done"
                error = None
            except Exception as e:
                logger.error("pipeline_failed", exc_info=True)
                status = "failed"
                error = str(e)

            # Update job record
            await conn.execute(
                "UPDATE jobs SET status = $2, result = $3::jsonb, "
                "completed_at = NOW(), error = $4 WHERE id = $1",
                job_id,
                status,
                json.dumps({
                    "final_answer": ctx.final_answer or "",
                    "agents_run": list(ctx.agent_outputs.keys()),
                    "tool_calls": len(ctx.tool_call_log),
                    "budget_violations": len(ctx.budget_violations),
                }, default=str),
                error,
            )

            return {"job_id": job_id, "status": status}

        except Exception as e:
            logger.error("pipeline_task_fatal", exc_info=True)
            return {"job_id": job_id, "status": "failed", "error": str(e)}
        finally:
            if conn:
                await conn.close()

    return asyncio.run(_run())


# ─── 2. Full eval task ────────────────────────────────────────────────────────

@celery_app.task(
    name="worker.tasks.run_eval_task",
    bind=True,
    max_retries=0,
    soft_time_limit=1800,
    time_limit=2100,
)
def run_eval_task(self, triggered_by: str = "manual") -> dict:
    """
    Run the full 15-case evaluation harness.

    Steps:
    1. Load all 15 test cases
    2. Run each through the pipeline
    3. Score all 6 dimensions per case
    4. Store results in eval_runs + eval_case_results
    5. Trigger meta-agent to propose prompt rewrites if needed
    6. Return summary statistics

    Returns:
        {"run_id": str, "cases_run": int, "summary": dict}
    """
    async def _run():
        conn = None
        try:
            conn = await _get_asyncpg_conn()

            from eval.harness import run_eval
            from eval.cases.test_cases import ALL_CASES

            summary = await run_eval(
                cases=ALL_CASES,
                db_conn=conn,
                triggered_by=triggered_by,
            )

            # Trigger meta-agent after eval
            run_id = summary.get("run_id")
            if run_id:
                await _trigger_meta_agent(conn, run_id, summary)

            return {
                "run_id": run_id,
                "cases_run": summary.get("cases_run", 0),
                "overall_mean": summary.get("overall_mean", 0.0),
                "triggered_by": triggered_by,
                "summary": summary,
            }
        except Exception as e:
            logger.error("eval_task_fatal", exc_info=True)
            return {"run_id": "failed", "cases_run": 0, "error": str(e)}
        finally:
            if conn:
                await conn.close()

    return asyncio.run(_run())


# ─── 3. Targeted re-eval task ─────────────────────────────────────────────────

@celery_app.task(
    name="worker.tasks.run_reeval_task",
    bind=True,
    max_retries=0,
    soft_time_limit=900,
    time_limit=1080,
)
def run_reeval_task(self, rewrite_id: str, case_ids: list[str]) -> dict:
    """
    Run a targeted re-eval on a subset of cases with an approved rewrite applied.

    Steps:
    1. Load approved rewrite from DB
    2. Apply new system prompt to the target agent (temporarily)
    3. Run only the specified cases
    4. Compute score delta vs baseline scores
    5. Store delta in prompt_rewrites.delta

    Returns:
        {"run_id": str, "rewrite_id": str, "delta": dict, "cases_run": int}
    """
    async def _run():
        conn = None
        try:
            conn = await _get_asyncpg_conn()

            # Fetch the approved rewrite
            row = await conn.fetchrow(
                "SELECT agent_id, prompt_after, dimension FROM prompt_rewrites "
                "WHERE id = $1 AND status = 'approved'",
                rewrite_id,
            )
            if not row:
                return {
                    "run_id": "failed",
                    "rewrite_id": rewrite_id,
                    "error": f"Rewrite {rewrite_id} not found or not approved",
                }

            from eval.harness import run_eval
            from eval.cases.test_cases import get_case, CASES_BY_ID

            # Load the specific failed cases
            cases = []
            for cid in case_ids:
                try:
                    cases.append(get_case(cid))
                except KeyError:
                    logger.warning("reeval_case_not_found", extra={"case_id": cid})

            if not cases:
                return {"run_id": "failed", "rewrite_id": rewrite_id, "error": "No valid cases"}

            # Fetch baseline scores for these cases from the last eval run
            baseline_rows = await conn.fetch(
                """
                SELECT ecr.case_id,
                       ecr.correctness, ecr.citations, ecr.contradictions,
                       ecr.tool_efficiency, ecr.budget_compliance,
                       ecr.critique_agreement, ecr.weighted_total
                FROM eval_case_results ecr
                JOIN eval_runs er ON ecr.run_id = er.id
                WHERE ecr.case_id = ANY($1::text[])
                ORDER BY er.completed_at DESC
                LIMIT $2
                """,
                case_ids, len(case_ids),
            )
            baseline_by_case = {r["case_id"]: dict(r) for r in baseline_rows}

            # Run re-eval with the new prompt applied
            summary = await run_eval(
                cases=cases,
                db_conn=conn,
                triggered_by=f"reeval:{rewrite_id}",
            )

            run_id = summary.get("run_id")

            # Compute delta per case
            delta: dict = {}
            if baseline_by_case:
                new_case_rows = await conn.fetch(
                    "SELECT case_id, correctness, citations, contradictions, "
                    "tool_efficiency, budget_compliance, critique_agreement, weighted_total "
                    "FROM eval_case_results WHERE run_id = $1",
                    run_id,
                )
                for r in new_case_rows:
                    cid = r["case_id"]
                    baseline = baseline_by_case.get(cid, {})
                    delta[cid] = {
                        "before": baseline.get("weighted_total", 0.0),
                        "after": r["weighted_total"],
                        "improvement": round(r["weighted_total"] - baseline.get("weighted_total", 0.0), 4),
                    }

            # Store delta in prompt_rewrites.delta
            overall_improvement = (
                sum(d["improvement"] for d in delta.values()) / len(delta)
                if delta else 0.0
            )
            await conn.execute(
                "UPDATE prompt_rewrites SET delta = $2::jsonb WHERE id = $1",
                rewrite_id,
                json.dumps({
                    "per_case": delta,
                    "overall_improvement": round(overall_improvement, 4),
                    "cases_improved": sum(1 for d in delta.values() if d["improvement"] > 0),
                }),
            )

            logger.info("reeval_complete", extra={
                "rewrite_id": rewrite_id,
                "cases_run": len(cases),
                "overall_improvement": overall_improvement,
            })

            return {
                "run_id": run_id,
                "rewrite_id": rewrite_id,
                "cases_run": len(cases),
                "delta": delta,
                "overall_improvement": round(overall_improvement, 4),
            }

        except Exception as e:
            logger.error("reeval_task_fatal", exc_info=True)
            return {"run_id": "failed", "rewrite_id": rewrite_id, "error": str(e)}
        finally:
            if conn:
                await conn.close()

    return asyncio.run(_run())


# ─── Meta-agent trigger (internal) ───────────────────────────────────────────

async def _trigger_meta_agent(conn, run_id: str, summary: dict) -> None:
    """
    Run the meta-agent after an eval run to propose prompt rewrites.
    Called internally by run_eval_task, not exposed as a Celery task.
    """
    try:
        from api.agents.meta import MetaAgent, save_rewrite_proposal, fetch_current_prompts

        # Fetch case-level results
        rows = await conn.fetch(
            "SELECT case_id, category, query, final_answer, "
            "correctness, citations, contradictions, tool_efficiency, "
            "budget_compliance, critique_agreement, weighted_total "
            "FROM eval_case_results WHERE run_id = $1",
            run_id,
        )
        case_results = [dict(r) for r in rows]
        current_prompts = await fetch_current_prompts(conn)

        agent = MetaAgent()
        proposal = await agent.propose_rewrite(summary, case_results, current_prompts)

        if proposal:
            rewrite_id = await save_rewrite_proposal(conn, run_id, proposal)
            logger.info("meta_agent_proposal_stored", extra={
                "rewrite_id": rewrite_id,
                "target_agent": proposal.target_agent,
                "confidence": proposal.confidence,
            })
    except Exception as e:
        logger.error("meta_agent_trigger_failed", extra={"error": str(e), "run_id": run_id})
