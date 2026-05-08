"""
worker/tasks.py

Celery task definitions for Mega AI.
All heavy/async pipeline work runs here, off the API request path.

Queues:
- jobs  — pipeline execution tasks
- eval  — evaluation harness tasks

Tasks:
- run_pipeline_task  — runs the full multi-agent pipeline for a job
- run_eval_task      — runs the full 15-case evaluation harness
- run_reeval_task    — runs a targeted re-eval on failed cases (post-approval)

Implementation note: Full task bodies are implemented in Phases 4, 7, and 8.
This module provides the Celery app configuration and task stubs.
"""

from __future__ import annotations

import os

from celery import Celery

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
    task_acks_late=True,                  # re-queue on worker crash
    worker_prefetch_multiplier=1,         # process one task at a time per worker
    task_routes={
        "worker.tasks.run_pipeline_task": {"queue": "jobs"},
        "worker.tasks.run_eval_task":     {"queue": "eval"},
        "worker.tasks.run_reeval_task":   {"queue": "eval"},
    },
    result_expires=3600,                  # results expire after 1 hour
)


# ─── Pipeline task ────────────────────────────────────────────────────────────

@celery_app.task(
    name="worker.tasks.run_pipeline_task",
    bind=True,
    max_retries=0,          # pipeline failures are not retried at the task level
    soft_time_limit=300,    # 5 min soft limit
    time_limit=360,         # 6 min hard limit
)
def run_pipeline_task(self, job_id: str, query: str) -> dict:
    """
    Run the full multi-agent pipeline for a given job.

    Steps (implemented in Phase 4):
    1. Load SharedContext from DB
    2. Initialise BudgetManager
    3. Run Orchestrator.run()
    4. Write results back to DB
    5. Publish final SSE 'done' event

    Args:
        job_id: UUID of the jobs table row
        query:  The user's original query

    Returns:
        {"job_id": str, "status": "done" | "failed"}
    """
    # TODO Phase 4: full pipeline implementation
    return {"job_id": job_id, "status": "stub", "query": query}


# ─── Eval task ────────────────────────────────────────────────────────────────

@celery_app.task(
    name="worker.tasks.run_eval_task",
    bind=True,
    max_retries=0,
    soft_time_limit=1800,   # 30 min for full 15-case eval
    time_limit=2100,
)
def run_eval_task(self, triggered_by: str = "manual") -> dict:
    """
    Run the full evaluation harness over all 15 test cases.

    Steps (implemented in Phase 7):
    1. Load all test cases from eval/cases/
    2. Run each through the pipeline
    3. Score all 6 dimensions per case
    4. Store results in eval_runs + eval_case_results tables
    5. Trigger meta-agent to propose prompt rewrites

    Returns:
        {"run_id": str, "cases_run": int, "summary": dict}
    """
    # TODO Phase 7: full eval implementation
    return {"run_id": "stub", "cases_run": 0, "triggered_by": triggered_by}


# ─── Re-eval task ─────────────────────────────────────────────────────────────

@celery_app.task(
    name="worker.tasks.run_reeval_task",
    bind=True,
    max_retries=0,
    soft_time_limit=900,
    time_limit=1080,
)
def run_reeval_task(self, rewrite_id: str, case_ids: list[str]) -> dict:
    """
    Run a targeted re-evaluation on a specific subset of cases
    using an approved prompt rewrite.

    Steps (implemented in Phase 8):
    1. Load approved rewrite from DB
    2. Temporarily apply new prompt
    3. Run only the specified failed cases
    4. Compute score delta vs baseline
    5. Store delta in prompt_rewrites.delta

    Returns:
        {"run_id": str, "rewrite_id": str, "delta": dict}
    """
    # TODO Phase 8: re-eval implementation
    return {"run_id": "stub", "rewrite_id": rewrite_id, "cases_run": len(case_ids)}
