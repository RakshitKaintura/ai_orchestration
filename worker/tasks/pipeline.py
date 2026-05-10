"""
worker/tasks/pipeline.py

Pipeline execution Celery task.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from celery.utils.log import get_task_logger

from worker.app import celery_app
from worker.tasks.utils import _get_asyncpg_conn

logger = get_task_logger(__name__)


@celery_app.task(
    name="worker.tasks.pipeline.run_pipeline_task",
    bind=True,
    max_retries=0,
    soft_time_limit=300,
    time_limit=360,
)
def run_pipeline_task(self, job_id: str, query: str) -> dict:
    async def _run():
        conn = None
        try:
            conn = await _get_asyncpg_conn()

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
