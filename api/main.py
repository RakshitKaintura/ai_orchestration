"""
api/main.py

FastAPI application entry point.
All five required endpoints — fully wired as of Day 3.

Endpoints:
1. POST /query          — Submit a query, receive SSE stream (LIVE)
2. GET  /trace/{job_id} — Full execution trace for a job (LIVE)
3. GET  /eval/latest    — Latest eval run summary (stub → Phase 7)
4. POST /rewrites/{id}/review — Approve or reject a prompt rewrite (stub → Phase 8)
5. POST /eval/re-run    — Trigger targeted re-eval (stub → Phase 8)
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from api.config import get_settings
from api.database import get_db_session
from api.logging_config import configure_logging, get_logger
from api.streaming import make_streaming_response

settings = get_settings()
configure_logging(settings.log_level)
logger = get_logger(__name__)


# ─── Lifespan ────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown hooks."""
    logger.info("mega_ai_startup", version="0.1.0", env=settings.log_level)
    # Initialise ChromaDB collection and seed corpus
    try:
        from api.agents.rag import _get_chroma_collection
        _get_chroma_collection()
        logger.info("chromadb_initialised")
    except Exception as e:
        logger.warning("chromadb_init_failed", extra={"error": str(e)})
    yield
    logger.info("mega_ai_shutdown")


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Mega AI — Multi-Agent LLM Orchestration",
    description=(
        "Production-grade multi-agent system with dynamic routing, "
        "RAG, critique/synthesis pipeline, self-improving eval loop, "
        "and real-time SSE streaming."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Error schema ─────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    error_code: str = Field(description="Machine-readable error code")
    message: str = Field(description="Human-readable error description")
    job_id: str | None = Field(default=None, description="Relevant job ID if applicable")


def make_error(
    error_code: str,
    message: str,
    status_code: int,
    job_id: str | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(
            error_code=error_code,
            message=message,
            job_id=job_id,
        ).model_dump(),
    )


# ─── Request / Response models ────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(
        min_length=1,
        max_length=4000,
        description="The user's query to process through the multi-agent pipeline",
        examples=["What are the key differences between RAG and fine-tuning?"],
    )


class ReviewRequest(BaseModel):
    decision: str = Field(
        description="Human decision on the prompt rewrite proposal",
        pattern="^(approved|rejected)$",
        examples=["approved"],
    )


class ReRunRequest(BaseModel):
    rewrite_id: str | None = Field(
        default=None,
        description="UUID of an approved prompt rewrite to use. If omitted, uses the latest approved rewrite.",
    )


# ─── Health check (not in the required 5 but essential for Docker healthcheck) ─

@app.get("/health", tags=["Internal"], summary="Health check")
async def health_check() -> dict[str, str]:
    return {"status": "ok", "service": "mega-ai-api"}


# ─── Endpoint 1: Submit query (SSE stream) ────────────────────────────────────

@app.post(
    "/query",
    tags=["Pipeline"],
    summary="Submit a query to the multi-agent pipeline",
    description=(
        "Accepts a user query, creates an async job, and returns a Server-Sent Events "
        "stream. The stream emits: agent_start, token, tool_call_start, tool_call_end, "
        "budget_update, agent_end, and done events in real time."
    ),
    responses={
        200: {"description": "SSE stream (text/event-stream)"},
        422: {"model": ErrorResponse, "description": "Invalid request body"},
        503: {"model": ErrorResponse, "description": "Pipeline unavailable"},
    },
)
async def submit_query(request: QueryRequest):
    """
    Submit a query and stream the multi-agent pipeline response via SSE.

    The response is a Server-Sent Events stream (text/event-stream).
    Each event has a `type` field:
    - `job_started`     — pipeline begins (includes job_id)
    - `orchestrator_plan` — routing decision with selected agents
    - `agent_start`     — an agent begins processing (includes budget)
    - `token`           — a single streamed word from an agent
    - `tool_call_start` — a tool call is initiated
    - `tool_call_end`   — a tool call completes (latency, accepted flag)
    - `budget_update`   — remaining token budget for the current agent
    - `agent_end`       — an agent finishes (tokens_used, latency_ms)
    - `done`            — the full pipeline is complete (job_id)
    - `error`           — pipeline error (error_code, message, job_id)
    """
    job_id = uuid.uuid4()
    logger.info("query_received", job_id=str(job_id), query_length=len(request.query))

    return make_streaming_response(job_id, request.query)


# ─── Endpoint 2: Get execution trace ──────────────────────────────────────────

@app.get(
    "/trace/{job_id}",
    tags=["Observability"],
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
    """
    Retrieve the full execution trace for any job by its ID.

    Returns an ordered list of all trace events, reconstructing the exact
    sequence of agent decisions, tool calls, and handoffs.
    """
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
            # Fetch job status
            job_row = await session.execute(
                "SELECT id, query, status, error, created_at, completed_at "
                "FROM jobs WHERE id = $1",
                job_id,
            )
            job = dict(job_row.fetchone() or {}) if hasattr(job_row, 'fetchone') else None

            # Fetch ordered trace events
            events_row = await session.execute(
                "SELECT seq, agent_id, event_type, input_hash, output_hash, "
                "payload, latency_ms, token_count, policy_violations, created_at "
                "FROM trace_events WHERE job_id = $1 ORDER BY seq ASC",
                job_id,
            )
            events = [dict(r) for r in (events_row.fetchall() if hasattr(events_row, 'fetchall') else [])]
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


# ─── Endpoint 3: Latest eval summary ──────────────────────────────────────────

@app.get(
    "/eval/latest",
    tags=["Evaluation"],
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
    """
    Return the latest evaluation run summary with per-category and per-dimension stats.
    """
    try:
        async with get_db_session() as session:
            row = await session.execute(
                "SELECT id, triggered_by, cases_count, status, summary, "
                "created_at, completed_at FROM eval_runs "
                "WHERE status = 'complete' ORDER BY completed_at DESC LIMIT 1"
            )
            run = row.fetchone() if hasattr(row, 'fetchone') else None

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

            # Fetch per-case results
            cases_row = await session.execute(
                "SELECT case_id, category, query, final_answer, "
                "correctness, citations, contradictions, tool_efficiency, "
                "budget_compliance, critique_agreement, weighted_total "
                "FROM eval_case_results WHERE run_id = $1 ORDER BY weighted_total ASC",
                run_id,
            )
            cases = [dict(r) for r in (cases_row.fetchall() if hasattr(cases_row, 'fetchall') else [])]

            # Fetch pending rewrites from this run
            rewrites_row = await session.execute(
                "SELECT id, agent_id, dimension, status, confidence, created_at "
                "FROM prompt_rewrites WHERE run_id = $1",
                run_id,
            )
            rewrites = [dict(r) for r in (rewrites_row.fetchall() if hasattr(rewrites_row, 'fetchall') else [])]

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


# ─── Endpoint 4: Approve / reject prompt rewrite ──────────────────────────────

@app.post(
    "/rewrites/{rewrite_id}/review",
    tags=["Self-Improvement"],
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
    """
    Submit a human approval or rejection for a pending prompt rewrite.

    On approval:
    - The rewrite status is set to 'approved'
    - The new prompt is written to agent_prompts (trigger deactivates old)
    - A targeted re-eval is immediately enqueued (only previously failed cases)

    On rejection:
    - The rewrite status is set to 'rejected'
    - No further action is taken
    """
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
            # Fetch the rewrite
            row = await session.execute(
                "SELECT id, run_id, agent_id, dimension, status FROM prompt_rewrites WHERE id = $1",
                rewrite_id,
            )
            rw = row.fetchone() if hasattr(row, 'fetchone') else None

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

            # Update status
            new_status = request.decision  # 'approved' or 'rejected'
            await session.execute(
                "UPDATE prompt_rewrites SET status = $2 WHERE id = $1",
                rewrite_id, new_status,
            )

            re_eval_triggered = False
            re_eval_run_id = None

            if new_status == "approved":
                # Apply the new prompt to agent_prompts
                from api.agents.meta import apply_approved_rewrite
                await apply_approved_rewrite(session, rewrite_id)

                # Find previously failed cases from the originating eval run
                run_id = rw_dict.get("run_id")
                failed_cases_row = await session.execute(
                    "SELECT case_id FROM eval_case_results "
                    "WHERE run_id = $1 AND weighted_total < 0.6 "
                    "ORDER BY weighted_total ASC",
                    run_id,
                )
                failed_cases = [
                    r["case_id"]
                    for r in (failed_cases_row.fetchall() if hasattr(failed_cases_row, 'fetchall') else [])
                ]

                if failed_cases:
                    # Enqueue targeted re-eval via Celery
                    from worker.tasks import run_reeval_task
                    task = run_reeval_task.apply_async(
                        args=[rewrite_id, failed_cases],
                        queue="eval",
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


# ─── Endpoint 5: Trigger targeted re-eval ────────────────────────────────────

@app.post(
    "/eval/re-run",
    tags=["Evaluation"],
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
    """
    Trigger a targeted re-eval using the latest approved prompt rewrite.
    """
    try:
        async with get_db_session() as session:
            # Resolve the rewrite_id — use explicit or find latest approved
            rewrite_id = request.rewrite_id

            if not rewrite_id:
                row = await session.execute(
                    "SELECT id FROM prompt_rewrites WHERE status = 'approved' "
                    "ORDER BY created_at DESC LIMIT 1"
                )
                rw = row.fetchone() if hasattr(row, 'fetchone') else None
                if not rw:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=ErrorResponse(
                            error_code="NO_APPROVED_REWRITE",
                            message="No approved rewrites found. Approve a rewrite first.",
                        ).model_dump(),
                    )
                rewrite_id = str(rw["id"])

            # Find the associated run's failed cases
            row = await session.execute(
                "SELECT run_id FROM prompt_rewrites WHERE id = $1 AND status = 'approved'",
                rewrite_id,
            )
            rw = row.fetchone() if hasattr(row, 'fetchone') else None
            if not rw:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=ErrorResponse(
                        error_code="REWRITE_NOT_FOUND",
                        message=f"Approved rewrite '{rewrite_id}' not found.",
                    ).model_dump(),
                )

            failed_cases_row = await session.execute(
                "SELECT case_id FROM eval_case_results "
                "WHERE run_id = $1 AND weighted_total < 0.6",
                str(rw["run_id"]),
            )
            failed_cases = [
                r["case_id"]
                for r in (failed_cases_row.fetchall() if hasattr(failed_cases_row, 'fetchall') else [])
            ]

        if not failed_cases:
            return {
                "status": "skipped",
                "rewrite_id": rewrite_id,
                "cases_count": 0,
                "message": "No failed cases found — nothing to re-evaluate.",
            }

        from worker.tasks import run_reeval_task
        task = run_reeval_task.apply_async(
            args=[rewrite_id, failed_cases],
            queue="eval",
        )

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
