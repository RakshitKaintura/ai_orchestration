"""
api/main.py

FastAPI application entry point.
Defines the five required endpoints (stubs at this stage — full implementation
follows in later phases). Each endpoint is documented via FastAPI's OpenAPI.

The five endpoints:
1. POST /query          — Submit a query, receive SSE stream
2. GET  /trace/{job_id} — Full execution trace for a job
3. GET  /eval/latest    — Latest eval run summary
4. POST /rewrites/{id}/review — Approve or reject a prompt rewrite
5. POST /eval/re-run    — Trigger targeted re-eval on failed cases
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from api.config import get_settings
from api.logging_config import configure_logging, get_logger

settings = get_settings()
configure_logging(settings.log_level)
logger = get_logger(__name__)


# ─── Lifespan ────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown hooks."""
    logger.info("mega_ai_startup", version="0.1.0", env=settings.log_level)
    # TODO (Phase 3): initialise ChromaDB collection
    # TODO (Phase 2): warm up tool connections
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
    Submit a query and stream the multi-agent pipeline response.

    The response is a Server-Sent Events stream. Each event has a `type` field:
    - `agent_start`     — an agent begins processing
    - `token`           — a single streamed token from an agent
    - `tool_call_start` — a tool call is initiated
    - `tool_call_end`   — a tool call completes (with latency and accepted flag)
    - `budget_update`   — remaining token budget for the current agent
    - `agent_end`       — an agent finishes
    - `done`            — the full pipeline is complete

    **Implementation note**: Full SSE streaming implemented in Phase 5.
    This stub returns a 202 with the job ID.
    """
    job_id = str(uuid.uuid4())
    logger.info("query_received", job_id=job_id, query_length=len(request.query))

    # TODO (Phase 4 + 5): enqueue Celery task, return SSE stream
    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={
            "job_id": job_id,
            "status": "queued",
            "message": "Job queued. SSE streaming implemented in Phase 5.",
            "stream_url": f"/query/stream/{job_id}",
        },
    )


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

    This endpoint reconstructs the exact sequence of agent decisions,
    tool calls, and handoffs in chronological order.

    **Implementation note**: DB query implemented in Phase 6.
    """
    # Validate UUID format
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

    # TODO (Phase 6): query trace_events table, return ordered events
    return {
        "job_id": job_id,
        "status": "stub",
        "trace_events": [],
        "message": "Full trace query implemented in Phase 6.",
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
    Return the latest evaluation run summary.

    **Implementation note**: DB query implemented in Phase 7.
    """
    # TODO (Phase 7): query eval_runs table, return latest
    return {
        "status": "stub",
        "message": "Eval run query implemented in Phase 7.",
        "latest_run": None,
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
    - A targeted re-eval is immediately enqueued (only failed cases)
    - The performance delta is stored once the re-eval completes

    On rejection:
    - The rewrite status is set to 'rejected'
    - No further action is taken

    **Implementation note**: Implemented in Phase 8.
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

    # TODO (Phase 8): update prompt_rewrites table, trigger re-eval if approved
    return {
        "rewrite_id": rewrite_id,
        "status": "stub",
        "decision": request.decision,
        "re_eval_triggered": False,
        "message": "Approval flow implemented in Phase 8.",
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
    Trigger a targeted re-evaluation using approved prompt rewrites.

    **Implementation note**: Implemented in Phase 8.
    """
    # TODO (Phase 8): enqueue Celery eval task with filtered case list
    return {
        "status": "stub",
        "rewrite_id": request.rewrite_id,
        "cases_count": 0,
        "message": "Re-eval trigger implemented in Phase 8.",
    }
