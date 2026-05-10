"""
worker/tasks/evaluation.py

Full eval Celery task.
"""
from __future__ import annotations

import asyncio
from celery.utils.log import get_task_logger

from worker.app import celery_app
from worker.tasks.utils import _get_asyncpg_conn

logger = get_task_logger(__name__)


@celery_app.task(
    name="worker.tasks.evaluation.run_eval_task",
    bind=True,
    max_retries=0,
    soft_time_limit=1800,
    time_limit=2100,
)
def run_eval_task(self, triggered_by: str = "manual") -> dict:
    async def _run():
        conn = None
        try:
            conn = await _get_asyncpg_conn()

            from eval.harness import run_eval
            from eval.cases import ALL_CASES

            summary = await run_eval(
                cases=ALL_CASES,
                db_conn=conn,
                triggered_by=triggered_by,
            )

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


async def _trigger_meta_agent(conn, run_id: str, summary: dict) -> None:
    try:
        from api.agents.meta import MetaAgent, save_rewrite_proposal, fetch_current_prompts

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
