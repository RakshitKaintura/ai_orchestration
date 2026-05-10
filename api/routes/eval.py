"""
api/routes/eval.py

GET  /eval/latest  — retrieve the latest evaluation run summary.
POST /eval/re-run  — trigger a targeted re-eval on previously failed cases.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import text

from api.database import get_db_session
from api.logging_config import get_logger
from api.routes._schemas import ErrorResponse, ReRunRequest

router = APIRouter(tags=["Evaluation"])
logger = get_logger(__name__)


@router.get(
    "/eval/latest",
    summary="Retrieve the latest evaluation run summary",
    description=(
        "Returns the most recent eval run, broken down by test category "
        "(baseline / ambiguous / adversarial) and by scoring dimension "
        "(correctness, citations, contradictions, tool_efficiency, budget, critique_agreement). "
        "Each dimension includes a mean score and per-case justification strings."
    ),
    responses={
        200: {"description": "Eval run summary"},
        404: {"model": ErrorResponse, "description": "No eval runs found"},
    },
)
async def get_latest_eval() -> dict[str, Any]:
    """Return the latest evaluation run summary with per-category and per-dimension stats."""
    try:
        async with get_db_session() as session:
            row = await session.execute(
                text(
                    "SELECT id, triggered_by, cases_count, status, summary, "
                    "created_at, completed_at FROM eval_runs "
                    "WHERE status = 'complete' ORDER BY completed_at DESC LIMIT 1"
                )
            )
            run = row.fetchone() if hasattr(row, "fetchone") else None

            if not run:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=ErrorResponse(
                        error_code="NO_EVAL_RUNS",
                        message="No completed evaluation runs found. Run 'make eval' to start.",
                    ).model_dump(),
                )

            run_dict = dict(run)
            run_id = str(run_dict["id"])

            cases_row = await session.execute(
                text(
                    "SELECT case_id, category, query, final_answer, "
                    "correctness, citations, contradictions, tool_efficiency, "
                    "budget_compliance, critique_agreement, weighted_total "
                    "FROM eval_case_results WHERE run_id = :run_id ORDER BY weighted_total ASC"
                ),
                {"run_id": run_id},
            )
            cases = [dict(r) for r in (cases_row.fetchall() if hasattr(cases_row, "fetchall") else [])]

            rewrites_row = await session.execute(
                text(
                    "SELECT id, agent_id, dimension, status, confidence, created_at "
                    "FROM prompt_rewrites WHERE run_id = :run_id"
                ),
                {"run_id": run_id},
            )
            rewrites = [dict(r) for r in (rewrites_row.fetchall() if hasattr(rewrites_row, "fetchall") else [])]

    except HTTPException:
        raise
    except Exception as e:
        logger.warning("eval_latest_db_failed", extra={"error": str(e)})
        return {
            "error": "Database query failed",
            "message": str(e),
            "hint": "Run 'make eval' to execute the evaluation harness",
        }

    return {
        "run_id": run_id,
        "triggered_by": run_dict.get("triggered_by"),
        "cases_count": run_dict.get("cases_count"),
        "status": run_dict.get("status"),
        "created_at": str(run_dict.get("created_at", "")),
        "completed_at": str(run_dict.get("completed_at", "")),
        "summary": run_dict.get("summary") or {},
        "case_results": cases,
        "pending_rewrites": rewrites,
    }


@router.post(
    "/eval/re-run",
    summary="Trigger a targeted re-eval on previously failed cases",
    description=(
        "Enqueues a re-evaluation job that runs ONLY the test cases that failed "
        "in the previous eval run, using the latest approved prompt rewrite. "
        "If rewrite_id is omitted, the latest approved rewrite is used automatically."
    ),
    responses={
        202: {"description": "Re-eval job queued"},
        404: {"model": ErrorResponse, "description": "Rewrite not found or not approved"},
        409: {"model": ErrorResponse, "description": "A re-eval is already in progress"},
    },
)
async def trigger_re_eval(request: ReRunRequest) -> dict[str, Any]:
    """Trigger a targeted re-eval using the latest approved prompt rewrite."""
    try:
        async with get_db_session() as session:
            rewrite_id = request.rewrite_id

            if not rewrite_id:
                row = await session.execute(
                    text("SELECT id FROM prompt_rewrites WHERE status = 'approved' ORDER BY created_at DESC LIMIT 1")
                )
                rw = row.fetchone() if hasattr(row, "fetchone") else None
                if not rw:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=ErrorResponse(
                            error_code="NO_APPROVED_REWRITE",
                            message="No approved rewrites found. Approve a rewrite first.",
                        ).model_dump(),
                    )
                rewrite_id = str(rw._mapping["id"])

            row = await session.execute(
                text("SELECT run_id FROM prompt_rewrites WHERE id = :rewrite_id AND status = 'approved'"),
                {"rewrite_id": rewrite_id},
            )
            rw = row.fetchone() if hasattr(row, "fetchone") else None
            if not rw:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=ErrorResponse(
                        error_code="REWRITE_NOT_FOUND",
                        message=f"Approved rewrite '{rewrite_id}' not found.",
                    ).model_dump(),
                )

            failed_cases_row = await session.execute(
                text("SELECT case_id FROM eval_case_results WHERE run_id = :run_id AND weighted_total < 0.6"),
                {"run_id": str(rw._mapping["run_id"])},
            )
            failed_cases = [
                r._mapping["case_id"]
                for r in (failed_cases_row.fetchall() if hasattr(failed_cases_row, "fetchall") else [])
            ]

        if not failed_cases:
            return {
                "status": "skipped",
                "rewrite_id": rewrite_id,
                "cases_count": 0,
                "message": "No failed cases found — nothing to re-evaluate.",
            }

        from worker.tasks import run_reeval_task
        task = run_reeval_task.apply_async(args=[rewrite_id, failed_cases], queue="eval")

        return {
            "status": "queued",
            "run_id": task.id,
            "rewrite_id": rewrite_id,
            "cases_count": len(failed_cases),
            "case_ids": failed_cases,
            "message": f"Re-eval queued for {len(failed_cases)} failed cases.",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("trigger_reeval_failed", extra={"error": str(e)})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorResponse(
                error_code="REEVAL_FAILED",
                message=f"Failed to trigger re-eval: {e}",
            ).model_dump(),
        )
