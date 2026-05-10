"""
api/routes/trace.py

GET /trace/{job_id} — retrieve the full execution trace for a completed job.
"""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import text

from api.database import get_db_session
from api.logging_config import get_logger
from api.routes._schemas import ErrorResponse

router = APIRouter(tags=["Observability"])
logger = get_logger(__name__)


@router.get(
    "/trace/{job_id}",
    summary="Retrieve the full execution trace for a completed job",
    description=(
        "Returns an ordered list of all trace events for the given job ID. "
        "Each event includes: agent_id, event_type, input_hash, output_hash, "
        "latency_ms, token_count, policy_violations, and the full payload. "
        "Events are ordered by their sequence number (seq)."
    ),
    responses={
        200: {"description": "Ordered list of trace events"},
        404: {"model": ErrorResponse, "description": "Job not found"},
    },
)
async def get_trace(job_id: str) -> dict[str, Any]:
    """Retrieve the full execution trace for any job by its ID."""
    try:
        uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ErrorResponse(
                error_code="INVALID_JOB_ID",
                message=f"'{job_id}' is not a valid UUID",
                job_id=job_id,
            ).model_dump(),
        )

    try:
        async with get_db_session() as session:
            job_row = await session.execute(
                text("SELECT id, query, status, error, created_at, completed_at FROM jobs WHERE id = :job_id"),
                {"job_id": job_id},
            )
            job = dict(job_row.fetchone() or {}) if hasattr(job_row, "fetchone") else None

            events_row = await session.execute(
                text(
                    "SELECT seq, agent_id, event_type, input_hash, output_hash, "
                    "payload, latency_ms, token_count, policy_violations, created_at "
                    "FROM trace_events WHERE job_id = :job_id ORDER BY seq ASC"
                ),
                {"job_id": job_id},
            )
            events = [dict(r) for r in (events_row.fetchall() if hasattr(events_row, "fetchall") else [])]
    except Exception as e:
        logger.warning("trace_db_query_failed", extra={"error": str(e), "job_id": job_id})
        job = None
        events = []

    if not events and not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ErrorResponse(
                error_code="JOB_NOT_FOUND",
                message=f"No trace found for job '{job_id}'",
                job_id=job_id,
            ).model_dump(),
        )

    return {
        "job_id": job_id,
        "job": job,
        "trace_events": events,
        "total_events": len(events),
    }
