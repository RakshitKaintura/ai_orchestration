"""
worker/app.py

Celery application configuration for Mega AI worker.

Import `celery_app` from here in task modules to avoid circular imports.
"""

from __future__ import annotations

import os

from celery import Celery

celery_app = Celery(
    "mega_ai_worker",
    broker=os.environ.get("CELERY_BROKER_URL", "redis://redis:6379/0"),
    backend=os.environ.get("CELERY_RESULT_BACKEND", "redis://redis:6379/1"),
    include=[
        "worker.tasks.pipeline",
        "worker.tasks.evaluation",
        "worker.tasks.reeval",
    ],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_routes={
        "worker.tasks.pipeline.run_pipeline_task": {"queue": "jobs"},
        "worker.tasks.evaluation.run_eval_task":   {"queue": "eval"},
        "worker.tasks.reeval.run_reeval_task":     {"queue": "eval"},
    },
    result_expires=3600,
)
