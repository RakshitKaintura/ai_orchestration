"""
worker/tasks/reeval.py

Targeted re-eval Celery task.
"""
from __future__ import annotations

import asyncio
import json
from celery.utils.log import get_task_logger

from worker.app import celery_app
from worker.tasks.utils import _get_asyncpg_conn

logger = get_task_logger(__name__)


@celery_app.task(
    name="worker.tasks.reeval.run_reeval_task",
    bind=True,
    max_retries=0,
    soft_time_limit=900,
    time_limit=1080,
)
def run_reeval_task(self, rewrite_id: str, case_ids: list[str]) -> dict:
    async def _run():
        conn = None
        try:
            conn = await _get_asyncpg_conn()

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
            from eval.cases import get_case

            cases = []
            for cid in case_ids:
                try:
                    cases.append(get_case(cid))
                except KeyError:
                    logger.warning("reeval_case_not_found", extra={"case_id": cid})

            if not cases:
                return {"run_id": "failed", "rewrite_id": rewrite_id, "error": "No valid cases"}

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

            summary = await run_eval(
                cases=cases,
                db_conn=conn,
                triggered_by=f"reeval:{rewrite_id}",
            )

            run_id = summary.get("run_id")

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
