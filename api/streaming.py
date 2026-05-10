import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import AsyncGenerator
from uuid import UUID

from fastapi.responses import StreamingResponse
from sqlalchemy import text

from api.database import get_db_session
from api.orchestrator import Orchestrator
from api.context_manager import BudgetManager
from api.models.context import SharedContext

logger = logging.getLogger(__name__)

async def stream_pipeline(job_id: UUID, query: str) -> AsyncGenerator[str, None]:
    """
    Execute the multi-agent pipeline and yield status events as they occur.
    Yields formatted SSE strings.
    """
    # 1. Initialise job in database
    try:
        async with get_db_session() as db:
            await db.execute(text(
                "INSERT INTO jobs (id, query, status, created_at, started_at) "
                "VALUES (:id, :query, 'running', :now, :now)"
            ), {"id": str(job_id), "query": query, "now": datetime.now(timezone.utc)})
    except Exception as db_err:
        logger.warning("job_db_insert_failed", extra={"error": str(db_err), "job_id": str(job_id)})

    # 2. Setup orchestrator
    ctx = SharedContext(job_id=job_id, query=query)
    bm = BudgetManager(ctx)
    sse_queue = asyncio.Queue()
    orchestrator = Orchestrator(ctx, bm, sse_queue=sse_queue)

    # 3. Start pipeline in background
    pipeline_task = asyncio.create_task(orchestrator.run())

    def _format_sse_event(data: dict) -> str:
        return f"data: {json.dumps(data)}\n\n"

    # Emit initial job_started event
    yield _format_sse_event({"type": "job_started", "job_id": str(job_id)})

    try:
        # 4. Stream events to the caller
        while True:
            if pipeline_task.done() and sse_queue.empty():
                break

            try:
                event = await asyncio.wait_for(sse_queue.get(), timeout=1.0)
                yield _format_sse_event(event)
            except asyncio.TimeoutError:
                if pipeline_task.done() and sse_queue.empty():
                    break
                continue

    except Exception as e:
        logger.error("sse_relay_error", extra={"error": str(e), "job_id": str(job_id)})
        pass

    finally:
        logger.info("sse_relay_finished", extra={"job_id": str(job_id), "task_done": pipeline_task.done()})


def make_streaming_response(job_id: UUID, query: str) -> StreamingResponse:
    """
    Convenience wrapper to return a FastAPI StreamingResponse for the pipeline.
    """
    return StreamingResponse(
        stream_pipeline(job_id, query),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Job-ID": str(job_id),
            "Access-Control-Expose-Headers": "X-Job-ID"
        },
    )
