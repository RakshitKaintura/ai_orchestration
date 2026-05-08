"""
api/streaming.py

Server-Sent Events streaming layer for Mega AI.

Event types emitted over the SSE stream:
  orchestrator_plan  — routing decision with agent list and reasoning
  agent_start        — agent begins (includes budget)
  token              — single streamed word from an agent's output
  tool_call_start    — a tool call is initiated
  tool_call_end      — a tool call completes (latency, accepted flag)
  budget_update      — remaining tokens for current agent
  agent_end          — agent finished (tokens_used, latency_ms)
  done               — full pipeline complete (job_id)
  error              — pipeline error (error_code, message, job_id)

The client can reconstruct the full execution state from these events alone.

Implementation:
  - FastAPI endpoint returns EventSourceResponse (sse-starlette)
  - Orchestrator pushes events onto an asyncio.Queue
  - stream_job() reads from the queue and yields formatted SSE events
  - Pipeline runs concurrently via asyncio.create_task()
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncIterator
from uuid import UUID

from sse_starlette.sse import EventSourceResponse

logger = logging.getLogger(__name__)


# ─── SSE event formatter ──────────────────────────────────────────────────────

def _format_sse_event(data: dict, event: str | None = None) -> dict:
    """Format a dict as an SSE event dict for sse-starlette."""
    return {
        "event": event or data.get("type", "message"),
        "data": json.dumps(data, default=str),
    }


# ─── Stream generator ─────────────────────────────────────────────────────────

async def stream_pipeline(
    job_id: UUID,
    query: str,
    db_session=None,
) -> AsyncIterator[dict]:
    """
    Async generator that:
    1. Creates SharedContext + BudgetManager for the job
    2. Creates an asyncio.Queue for SSE events
    3. Runs the Orchestrator in a background task
    4. Yields SSE events as they arrive from the queue
    5. Terminates when the orchestrator emits 'done' or 'error'

    Yields dicts compatible with sse-starlette EventSourceResponse.
    """
    from api.models.context import SharedContext
    from api.context_manager import BudgetManager
    from api.orchestrator import Orchestrator

    sse_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)

    # Build context for this job
    ctx = SharedContext(job_id=job_id, query=query)
    bm = BudgetManager(ctx)

    # Yield immediate acknowledgment
    yield _format_sse_event({
        "type": "job_started",
        "job_id": str(job_id),
        "query": query,
    })

    # Start the orchestrator pipeline as a background task
    orchestrator = Orchestrator(ctx, bm, db_session=db_session, sse_queue=sse_queue)
    pipeline_task = asyncio.create_task(orchestrator.run())

    # Stream events until 'done' or 'error' arrives
    try:
        while True:
            try:
                event = await asyncio.wait_for(sse_queue.get(), timeout=120.0)
            except asyncio.TimeoutError:
                # Heartbeat to keep connection alive
                yield _format_sse_event({"type": "heartbeat"})
                continue

            yield _format_sse_event(event)

            if event.get("type") in ("done", "error"):
                break

    except asyncio.CancelledError:
        logger.warning("sse_stream_cancelled", extra={"job_id": str(job_id)})
        pipeline_task.cancel()
        raise

    except Exception as e:
        logger.error("sse_stream_error", extra={"error": str(e), "job_id": str(job_id)})
        yield _format_sse_event({
            "type": "error",
            "error_code": "PIPELINE_ERROR",
            "message": str(e),
            "job_id": str(job_id),
        })

    finally:
        # Ensure the pipeline task completes cleanly
        if not pipeline_task.done():
            try:
                await asyncio.wait_for(pipeline_task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pipeline_task.cancel()


def make_streaming_response(
    job_id: UUID,
    query: str,
    db_session=None,
) -> EventSourceResponse:
    """
    Create an EventSourceResponse for a pipeline job.
    Used by the POST /query endpoint.
    """
    return EventSourceResponse(
        stream_pipeline(job_id, query, db_session),
        headers={
            "Cache-Control": "no-cache",
            "X-Job-ID": str(job_id),
        },
    )
