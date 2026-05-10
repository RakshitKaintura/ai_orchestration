"""
api/orchestrator.py

Master Orchestrator
-------------------
The ONLY component that directly invokes agents. Agents never call each other.
The orchestrator mediates all handoffs via SharedContext.

Responsibilities:
  1. Plan:    Call LLM to produce a RoutingPlan (which agents, what order, what budget)
  2. Execute: For each agent in order:
               a. Declare budget
               b. Check if compression needed (is_near_limit)
               c. Invoke agent
               d. Immediately invoke CritiqueAgent on the result
               e. Log trace event
  3. Synthesise: Invoke SynthesisAgent last
  4. Finalise:   Mark job done, return SharedContext

Tool retry logic (in code, not prompts):
  Each failure mode routes to a different input mutation:
    timeout  → simplify_input
    empty    → broaden_input
    malformed → fix_format
  Up to MAX_TOOL_RETRIES (2) retries, each logged separately.

All state is written to SharedContext and to trace_events (PostgreSQL).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, AsyncIterator
from uuid import UUID

from pydantic import BaseModel, Field
from sqlalchemy import text

from api.agents.compression import compress_context_async
from api.agents.critique import CritiqueAgent
from api.agents.decomposition import DecompositionAgent
from api.agents.llm_retry import call_with_retry
from api.agents.rag import RAGAgent
from api.agents.synthesis import SynthesisAgent
from api.config import get_settings
from api.context_manager import BudgetManager
from api.database import get_db_session
from api.models.context import AgentOutput, RoutingPlan, SharedContext

logger = logging.getLogger(__name__)

# SSE event queue type
SSEQueue = asyncio.Queue


# ─── Routing plan (Instructor) ────────────────────────────────────────────────

class AgentSelection(BaseModel):
    agents_to_invoke: list[str] = Field(
        description=(
            "Ordered list of agent IDs to invoke. "
            "Always start with 'decomposition'. Always end with 'synthesis'. "
            "Include 'rag' for factual/retrieval questions. "
            "Valid agents: decomposition, rag, synthesis."
        )
    )
    per_agent_budget: dict[str, int] = Field(
        description="Token budget for each agent. Must match agents_to_invoke list."
    )
    reasoning: str = Field(
        description="Explanation of why these agents were selected in this order"
    )
    query_complexity: str = Field(
        description="One of: simple | moderate | complex"
    )


# ─── Trace event helpers ──────────────────────────────────────────────────────

def _short_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


async def _write_trace_event(
    db_session,
    job_id: UUID,
    seq: int,
    event_type: str,
    payload: dict,
    agent_id: str | None = None,
    latency_ms: int | None = None,
    token_count: int | None = None,
    input_hash: str | None = None,
    output_hash: str | None = None,
    policy_violations: list[str] | None = None,
) -> None:
    """Write a single trace event row to the database."""
    sql = text("""
        INSERT INTO trace_events
          (job_id, seq, agent_id, event_type, input_hash, output_hash,
           payload, latency_ms, token_count, policy_violations)
        VALUES (:job_id, :seq, :agent_id, :event_type, :input_hash, :output_hash,
                :payload, :latency_ms, :token_count, :policy_violations)
    """)
    params = {
        "job_id": str(job_id),
        "seq": seq,
        "agent_id": agent_id,
        "event_type": event_type,
        "input_hash": input_hash,
        "output_hash": output_hash,
        "payload": json.dumps(payload, default=str),
        "latency_ms": latency_ms,
        "token_count": token_count,
        "policy_violations": json.dumps(policy_violations or []),
    }

    if db_session is not None:
        try:
            await db_session.execute(sql, params)
        except Exception as e:
            logger.error("trace_write_failed", extra={"error": str(e), "event_type": event_type})
    else:
        # Auto-acquire session if not provided
        try:
            async with get_db_session() as db:
                await db.execute(sql, params)
        except Exception as e:
            # Silently fail if DB is unavailable during trace
            logger.error("trace_write_failed_auto_session", extra={"error": str(e), "event_type": event_type})


# ─── Orchestrator ─────────────────────────────────────────────────────────────

class Orchestrator:
    """
    Master orchestrator with Dual-Key Rotation and Pacing.
    """

    def __init__(
        self,
        ctx: SharedContext,
        bm: BudgetManager,
        db_session=None,
        sse_queue: SSEQueue | None = None,
    ) -> None:
        self.ctx = ctx
        self.bm = bm
        self.db = db_session
        self.sse = sse_queue
        self._seq = 0

    def _api_keys(self) -> list[str]:
        """Return all configured API keys (deduplicated, non-empty)."""
        settings = get_settings()
        seen, keys = set(), []
        for k in [settings.google_api_key, settings.gemini_api_key]:
            if k and k not in seen:
                seen.add(k)
                keys.append(k)
        return keys

    # ── SSE emit helpers ──────────────────────────────────────────────────────

    async def _emit(self, event: dict) -> None:
        if self.sse is not None:
            await self.sse.put(event)

    async def _emit_token(self, agent_id: str, text: str) -> None:
        await self._emit({"type": "token", "agent": agent_id, "text": text})

    async def _emit_agent_start(self, agent_id: str, budget: int) -> None:
        await self._emit({"type": "agent_start", "agent": agent_id, "budget": budget})

    async def _emit_agent_end(self, agent_id: str, tokens_used: int, latency_ms: int) -> None:
        await self._emit({
            "type": "agent_end", "agent": agent_id,
            "tokens_used": tokens_used, "latency_ms": latency_ms,
        })

    async def _emit_tool_start(self, agent_id: str, tool: str) -> None:
        await self._emit({"type": "tool_call_start", "agent": agent_id, "tool": tool})

    async def _emit_tool_end(self, agent_id: str, tool: str, latency_ms: int, accepted: bool) -> None:
        await self._emit({
            "type": "tool_call_end", "agent": agent_id, "tool": tool,
            "latency_ms": latency_ms, "accepted": accepted,
        })

    async def _emit_budget_update(self, agent_id: str) -> None:
        try:
            remaining = self.bm.check_remaining(agent_id)
            await self._emit({
                "type": "budget_update", "agent": agent_id, "remaining": remaining,
            })
        except KeyError:
            pass

    # ── Trace logging ─────────────────────────────────────────────────────────

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    async def _trace(self, event_type: str, payload: dict, **kwargs) -> None:
        await _write_trace_event(
            self.db, self.ctx.job_id, self._next_seq(),
            event_type, payload, **kwargs,
        )

    # ── Routing plan ──────────────────────────────────────────────────────────

    async def plan(self) -> RoutingPlan:
        settings = get_settings()
        t0 = time.perf_counter()
        api_keys = self._api_keys()

        prompt = (
            f"User query: {self.ctx.query}\n\n"
            "Decide which agents to invoke and in what order. "
            "decomposition is always first. synthesis is always last. "
            "Include rag for factual questions. "
            "Valid agents: decomposition, rag, synthesis."
        )

        def _plan(client):
            return client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                response_model=AgentSelection,
            )

        try:
            selection: AgentSelection = await call_with_retry(
                _plan, api_keys=api_keys, model_name=settings.primary_model
            )
        except Exception as e:
            logger.warning("orchestrator_plan_failed_using_default", extra={"error": str(e)})
            selection = AgentSelection(
                agents_to_invoke=["decomposition", "rag", "synthesis"],
                per_agent_budget={
                    "decomposition": settings.decomposition_budget,
                    "rag": settings.rag_budget,
                    "synthesis": settings.synthesis_budget,
                },
                reasoning=f"Default plan (routing failed: {type(e).__name__})",
                query_complexity="moderate",
            )

        latency = int((time.perf_counter() - t0) * 1000)
        routing_plan = RoutingPlan(
            agents_selected=selection.agents_to_invoke,
            per_agent_budget=selection.per_agent_budget,
            reasoning=selection.reasoning,
        )
        self.ctx.routing_plan = routing_plan
        await self._trace("orchestrator_plan", payload={"agents": routing_plan.agents_selected, "reasoning": routing_plan.reasoning}, latency_ms=latency)
        await self._emit({"type": "orchestrator_plan", "agents": routing_plan.agents_selected, "reasoning": routing_plan.reasoning})
        return routing_plan

    # ── Agent invocation ──────────────────────────────────────────────────────

    async def _invoke_agent(self, agent_id: str, budget: int) -> AgentOutput:
        await self._emit_agent_start(agent_id, budget)
        await self._trace("agent_start", {"agent_id": agent_id}, agent_id=agent_id)

        t0 = time.perf_counter()
        output = None

        try:
            if agent_id == "decomposition":
                agent = DecompositionAgent(self.ctx, self.bm)
            elif agent_id == "rag":
                agent = RAGAgent(self.ctx, self.bm)
            elif agent_id == "synthesis":
                agent = SynthesisAgent(self.ctx, self.bm)
            else:
                raise ValueError(f"Unknown agent: {agent_id}")

            # Agents handle their own retries internally via call_with_retry
            output = await agent.run()

        except Exception as e:
            latency = int((time.perf_counter() - t0) * 1000)
            logger.error("agent_invocation_failed", extra={"agent": agent_id, "error": str(e)})
            await self._trace("agent_error", {"error": str(e)}, agent_id=agent_id, latency_ms=latency)
            output = AgentOutput(
                agent_id=agent_id,
                output=f"[{agent_id}] failed after all retries: {type(e).__name__}",
                token_count=0,
                latency_ms=latency,
            )

        self.ctx.agent_outputs[agent_id] = output
        for word in output.output.split():
            await self._emit_token(agent_id, word + " ")
            await asyncio.sleep(0.01)

        await self._emit_agent_end(agent_id, output.token_count, output.latency_ms)
        await self._trace("agent_end", {"token_count": output.token_count}, agent_id=agent_id, latency_ms=output.latency_ms)
        return output

    async def _invoke_critique(self, target_agent_id: str) -> AgentOutput:
        """Critique is best-effort — failure here never blocks the main pipeline."""
        scoped_id = f"critique_{target_agent_id}"
        await self._emit({"type": "agent_start", "agent": scoped_id, "budget": 4000})
        t0 = time.perf_counter()
        try:
            agent = CritiqueAgent(self.ctx, self.bm, target_agent_id)
            critique_output = await agent.run()  # CritiqueAgent handles its own retries
        except Exception as e:
            logger.warning("critique_skipped", extra={"target": target_agent_id, "error": str(e)})
            critique_output = AgentOutput(
                agent_id=scoped_id,
                output=f"Critique skipped ({type(e).__name__})",
                token_count=0,
            )
        self.ctx.agent_outputs[scoped_id] = critique_output
        await self._emit_agent_end(scoped_id, critique_output.token_count, int((time.perf_counter()-t0)*1000))
        return critique_output

    # ── Main run loop ─────────────────────────────────────────────────────────

    async def run(self) -> SharedContext:
        t0 = time.perf_counter()
        settings = get_settings()

        try:
            plan = await self.plan()
            agents = plan.agents_selected

            for i, agent_id in enumerate(agents):
                if agent_id == "synthesis": continue
                
                # HEAVY PACING for free tier
                if i > 0: await asyncio.sleep(5.0) 

                tool_count_before = len(self.ctx.tool_call_log)
                await self._invoke_agent(agent_id, 4000)
                await self._log_tool_calls(agent_id, tool_count_before)

                await asyncio.sleep(3.0)
                await self._invoke_critique(agent_id)

            if "synthesis" in agents:
                await asyncio.sleep(8.0) # Maximum silence before synthesis
                await self._invoke_agent("synthesis", 5000)

            self.ctx.completed_at = datetime.now(timezone.utc)
            await self._emit({"type": "done", "job_id": str(self.ctx.job_id)})

            async with get_db_session() as db:
                await db.execute(text("UPDATE jobs SET status='done', final_answer=:answer, completed_at=:now WHERE id=:id"), 
                                 {"answer": self.ctx.final_answer, "now": datetime.now(timezone.utc), "id": str(self.ctx.job_id)})

            return self.ctx

        except Exception as e:
            logger.error("pipeline_failed", extra={"error": str(e)})
            async with get_db_session() as db:
                await db.execute(text("UPDATE jobs SET status='failed', error=:error, completed_at=:now WHERE id=:id"), 
                                 {"error": str(e), "now": datetime.now(timezone.utc), "id": str(self.ctx.job_id)})
            await self._emit({"type": "error", "message": str(e), "job_id": str(self.ctx.job_id)})
            raise

    async def _log_tool_calls(self, agent_id: str, before_count: int) -> None:
        new_calls = self.ctx.tool_call_log[before_count:]
        for tc in new_calls:
            await self._emit_tool_start(agent_id, tc.tool)
            await self._emit_tool_end(agent_id, tc.tool, tc.latency_ms, tc.accepted)
