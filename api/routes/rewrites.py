"""
api/routes/rewrites.py

POST /rewrites/{rewrite_id}/review — approve or reject a pending prompt rewrite.
"""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import text

from api.database import get_db_session
from api.logging_config import get_logger
from api.routes._schemas import ErrorResponse, ReviewRequest

router = APIRouter(tags=["Self-Improvement"])
logger = get_logger(__name__)


@router.post(
    "/rewrites/{rewrite_id}/review",
    summary="Approve or reject a pending prompt rewrite",
    description=(
        "Human-in-the-loop endpoint. Accepts 'approved' or 'rejected'. "
        "On approval, immediately enqueues a targeted re-eval on the previously "
        "failed cases using the new prompt. The rewrite is NEVER auto-applied — "
        "this endpoint is the only mechanism to activate it."
    ),
    responses={
        200: {"description": "Review recorded"},
        404: {"model": ErrorResponse, "description": "Rewrite not found"},
        409: {"model": ErrorResponse, "description": "Rewrite already reviewed"},
        422: {"model": ErrorResponse, "description": "Invalid decision value"},
    },
)
async def review_rewrite(rewrite_id: str, request: ReviewRequest) -> dict[str, Any]:
    """Submit a human approval or rejection for a pending prompt rewrite."""
    try:
        uuid.UUID(rewrite_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ErrorResponse(
                error_code="INVALID_REWRITE_ID",
                message=f"'{rewrite_id}' is not a valid UUID",
            ).model_dump(),
        )

    try:
        async with get_db_session() as session:
            row = await session.execute(
                text("SELECT id, run_id, agent_id, dimension, status FROM prompt_rewrites WHERE id = :rewrite_id"),
                {"rewrite_id": rewrite_id},
            )
            rw = row.fetchone() if hasattr(row, "fetchone") else None

            if not rw:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=ErrorResponse(
                        error_code="REWRITE_NOT_FOUND",
                        message=f"Rewrite '{rewrite_id}' not found.",
                        job_id=rewrite_id,
                    ).model_dump(),
                )

            rw_dict = dict(rw)
            if rw_dict["status"] != "pending":
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=ErrorResponse(
                        error_code="ALREADY_REVIEWED",
                        message=f"Rewrite already has status '{rw_dict['status']}'.",
                        job_id=rewrite_id,
                    ).model_dump(),
                )

            new_status = request.decision
            await session.execute(
                text("UPDATE prompt_rewrites SET status = :status WHERE id = :rewrite_id"),
                {"rewrite_id": rewrite_id, "status": new_status},
            )

            re_eval_triggered = False
            re_eval_run_id = None

            if new_status == "approved":
                from api.agents.meta import apply_approved_rewrite
                await apply_approved_rewrite(session, rewrite_id)

                run_id = rw_dict.get("run_id")
                failed_cases_row = await session.execute(
                    text(
                        "SELECT case_id FROM eval_case_results "
                        "WHERE run_id = :run_id AND weighted_total < 0.6 ORDER BY weighted_total ASC"
                    ),
                    {"run_id": run_id},
                )
                failed_cases = [
                    r._mapping["case_id"]
                    for r in (failed_cases_row.fetchall() if hasattr(failed_cases_row, "fetchall") else [])
                ]

                if failed_cases:
                    from worker.tasks import run_reeval_task
                    task = run_reeval_task.apply_async(
                        args=[rewrite_id, failed_cases], queue="eval"
                    )
                    re_eval_triggered = True
                    re_eval_run_id = task.id
                    logger.info("reeval_enqueued", extra={
                        "rewrite_id": rewrite_id,
                        "cases_count": len(failed_cases),
                        "task_id": task.id,
                    })

    except HTTPException:
        raise
    except Exception as e:
        logger.error("review_rewrite_failed", extra={"error": str(e)})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorResponse(
                error_code="REVIEW_FAILED",
                message=f"Failed to process review: {e}",
            ).model_dump(),
        )

    return {
        "rewrite_id": rewrite_id,
        "status": new_status,
        "re_eval_triggered": re_eval_triggered,
        "re_eval_task_id": re_eval_run_id,
        "message": (
            f"Rewrite {new_status}. Re-eval enqueued on failed cases."
            if re_eval_triggered
            else f"Rewrite {new_status}. No re-eval triggered."
        ),
    }
