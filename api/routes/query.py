"""
api/routes/query.py

POST /query — submit a user query; returns SSE stream.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter
from pydantic import BaseModel, Field

from api.logging_config import get_logger
from api.streaming import make_streaming_response

router = APIRouter(tags=["Pipeline"])
logger = get_logger(__name__)


class QueryRequest(BaseModel):
    query: str = Field(
        min_length=1,
        max_length=4000,
        description="The user's query to process through the multi-agent pipeline",
        examples=["What are the key differences between RAG and fine-tuning?"],
    )


@router.post(
    "/query",
    summary="Submit a query to the multi-agent pipeline",
    description=(
        "Accepts a user query, creates an async job, and returns a Server-Sent Events "
        "stream. The stream emits: agent_start, token, tool_call_start, tool_call_end, "
        "budget_update, agent_end, and done events in real time."
    ),
    responses={
        200: {"description": "SSE stream (text/event-stream)"},
        422: {"description": "Invalid request body"},
        503: {"description": "Pipeline unavailable"},
    },
)
async def submit_query(request: QueryRequest):
    """
    Submit a query and stream the multi-agent pipeline response via SSE.

    SSE event types:
    - job_started       — pipeline begins (includes job_id)
    - orchestrator_plan — routing decision with selected agents
    - agent_start       — an agent begins processing (includes budget)
    - token             — a single streamed word from an agent
    - tool_call_start   — a tool call is initiated
    - tool_call_end     — a tool call completes (latency, accepted flag)
    - budget_update     — remaining token budget for the current agent
    - agent_end         — an agent finishes (tokens_used, latency_ms)
    - done              — the full pipeline is complete (job_id)
    - error             — pipeline error (error_code, message, job_id)
    """
    job_id = uuid.uuid4()
    logger.info("query_received", job_id=str(job_id), query_length=len(request.query))
    return make_streaming_response(job_id, request.query)
