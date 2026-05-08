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

import anthropic
import instructor
from pydantic import BaseModel, Field

from api.agents.compression import compress_context_async
from api.agents.critique import CritiqueAgent
from api.agents.decomposition import DecompositionAgent
from api.agents.rag import RAGAgent
from api.agents.synthesis import SynthesisAgent
from api.config import get_settings
from api.context_manager import BudgetManager
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
    if db_session is None:
        return  # no DB in test mode
    try:
        await db_session.execute(
            """
            INSERT INTO trace_events
              (job_id, seq, agent_id, event_type, input_hash, output_hash,
               payload, latency_ms, token_count, policy_violations)
            VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8, $9, $10::jsonb)
            """,
            str(job_id), seq, agent_id, event_type,
            input_hash, output_hash,
            json.dumps(payload, default=str),
            latency_ms, token_count,
            json.dumps(policy_violations or []),
        )
    except Exception as e:
        logger.error("trace_write_failed", extra={"error": str(e), "event_type": event_type})


# ─── Orchestrator ─────────────────────────────────────────────────────────────

class Orchestrator:
    """
    Master orchestrator. Created per-job. Stateless between jobs.

    Args:
        ctx:        The SharedContext for this job (created by API layer)
        bm:         BudgetManager for this job
        db_session: Optional asyncpg connection for trace logging
        sse_queue:  Optional asyncio.Queue for SSE events
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
        self._seq = 0  # trace event sequence counter

    # ── SSE emit helpers ──────────────────────────────────────────────────────

    async def _emit(self, event: dict) -> None:
        """Push a structured SSE event onto the queue (non-blocking)."""
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
            pass  # agent not yet declared

    # ── Trace logging ─────────────────────────────────────────────────────────

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    async def _trace(self, event_type: str, payload: dict, **kwargs) -> None:
        await _write_trace_event(
            self.db, self.ctx.job_id, self._next_seq(),
            event_type, payload, **kwargs,
        )
        logger.info(
            event_type,
            extra={"job_id": str(self.ctx.job_id), "seq": self._seq, **kwargs},
        )

    # ── Routing plan ──────────────────────────────────────────────────────────

    async def plan(self) -> RoutingPlan:
        """
        Ask the LLM to produce a routing plan. Result is logged to trace_events.
        Routing decisions are made at runtime — not hardcoded.
        """
        settings = get_settings()
        t0 = time.perf_counter()

        raw_client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        client = instructor.from_anthropic(raw_client)

        prompt = (
            f"User query: {self.ctx.query}\n\n"
            "Decide which agents to invoke and in what order. "
            "decomposition is always first. synthesis is always last. "
            "Include rag for questions requiring factual retrieval. "
            "Set reasonable token budgets (decomposition: 2000-4000, rag: 4000-6000, synthesis: 3000-5000)."
        )

        try:
            selection: AgentSelection = await client.messages.create(
                model=settings.primary_model,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
                response_model=AgentSelection,
            )
        except Exception as e:
            logger.warning("orchestrator_plan_failed_using_default", extra={"error": str(e)})
            # Default plan: always include all three agents
            selection = AgentSelection(
                agents_to_invoke=["decomposition", "rag", "synthesis"],
                per_agent_budget={
                    "decomposition": settings.decomposition_budget,
                    "rag": settings.rag_budget,
                    "synthesis": settings.synthesis_budget,
                },
                reasoning=f"Default plan (LLM routing failed: {e})",
                query_complexity="moderate",
            )

        latency = int((time.perf_counter() - t0) * 1000)

        routing_plan = RoutingPlan(
            agents_selected=selection.agents_to_invoke,
            per_agent_budget=selection.per_agent_budget,
            reasoning=selection.reasoning,
        )
        self.ctx.routing_plan = routing_plan

        await self._trace(
            "orchestrator_plan",
            payload={
                "agents_selected": routing_plan.agents_selected,
                "per_agent_budget": routing_plan.per_agent_budget,
                "reasoning": routing_plan.reasoning,
                "query_complexity": selection.query_complexity,
            },
            latency_ms=latency,
        )
        await self._emit({
            "type": "orchestrator_plan",
            "agents": routing_plan.agents_selected,
            "reasoning": routing_plan.reasoning,
        })

        logger.info("routing_plan_produced", extra={
            "agents": routing_plan.agents_selected,
            "job_id": str(self.ctx.job_id),
        })
        return routing_plan

    # ── Agent invocation ──────────────────────────────────────────────────────

    async def _invoke_agent(
        self,
        agent_id: str,
        budget: int,
    ) -> AgentOutput:
        """
        Invoke a single agent with budget enforcement and full tracing.
        Checks for near-limit budget before invocation.
        """
        # Proactive compression if near limit on any prior agent
        # (this applies to the context being passed IN, not the agent's own budget)

        await self._emit_agent_start(agent_id, budget)
        await self._trace("agent_start", {"agent_id": agent_id, "budget": budget}, agent_id=agent_id)

        t0 = time.perf_counter()
        try:
            if agent_id == "decomposition":
                agent = DecompositionAgent(self.ctx, self.bm)
            elif agent_id == "rag":
                agent = RAGAgent(self.ctx, self.bm)
            elif agent_id == "synthesis":
                agent = SynthesisAgent(self.ctx, self.bm)
            else:
                raise ValueError(f"Unknown agent_id: {agent_id}")

            output = await agent.run()

        except Exception as e:
            logger.error(
                "agent_invocation_failed",
                extra={"agent_id": agent_id, "error": str(e), "job_id": str(self.ctx.job_id)},
            )
            latency = int((time.perf_counter() - t0) * 1000)
            await self._trace(
                "agent_error",
                {"agent_id": agent_id, "error": str(e)},
                agent_id=agent_id, latency_ms=latency,
            )
            # Return a stub output so the pipeline can continue
            from api.models.context import AgentOutput
            output = AgentOutput(
                agent_id=agent_id,
                output=f"[{agent_id}] failed: {e}",
                token_count=0,
                latency_ms=latency,
            )

        latency = output.latency_ms
        tokens = output.token_count

        # Store in shared context (orchestrator is the only writer)
        self.ctx.agent_outputs[agent_id] = output

        # Stream the agent's output token-by-token (simulated word-level streaming)
        for word in output.output.split():
            await self._emit_token(agent_id, word + " ")
            await asyncio.sleep(0)  # yield to event loop

        await self._emit_agent_end(agent_id, tokens, latency)
        await self._emit_budget_update(agent_id)

        await self._trace(
            "agent_end",
            {
                "agent_id": agent_id,
                "token_count": tokens,
                "output_hash": output.output_hash,
                "policy_violations": self.ctx.budget_violations[-3:],  # last 3
            },
            agent_id=agent_id,
            latency_ms=latency,
            token_count=tokens,
            output_hash=output.output_hash,
            input_hash=output.input_hash,
            policy_violations=self.ctx.budget_violations,
        )

        return output

    async def _invoke_critique(self, target_agent_id: str) -> AgentOutput:
        """
        Invoke the critique agent on a specific target agent's output.
        Uses a scoped budget key to allow multiple critique passes.
        """
        settings = get_settings()
        scoped_id = f"critique_{target_agent_id}"

        await self._emit({
            "type": "agent_start",
            "agent": scoped_id,
            "budget": settings.critique_budget,
        })
        await self._trace(
            "agent_start",
            {"agent_id": scoped_id, "target": target_agent_id, "budget": settings.critique_budget},
            agent_id=scoped_id,
        )

        t0 = time.perf_counter()
        try:
            agent = CritiqueAgent(self.ctx, self.bm, target_agent_id)
            critique_output = await agent.run()
        except Exception as e:
            logger.error("critique_failed", extra={"target": target_agent_id, "error": str(e)})
            from api.models.context import AgentOutput
            critique_output = AgentOutput(
                agent_id=scoped_id,
                output=f"[critique_{target_agent_id}] failed: {e}",
                token_count=0,
            )

        latency = critique_output.latency_ms
        self.ctx.agent_outputs[scoped_id] = critique_output

        await self._emit_agent_end(scoped_id, critique_output.token_count, latency)
        await self._trace(
            "agent_end",
            {"agent_id": scoped_id, "target": target_agent_id},
            agent_id=scoped_id,
            latency_ms=latency,
            token_count=critique_output.token_count,
        )

        return critique_output

    # ── Tool call logging for SSE ─────────────────────────────────────────────

    async def _log_tool_calls(self, agent_id: str, before_count: int) -> None:
        """
        Emit SSE events for any tool calls made since before_count.
        Called after each agent completes.
        """
        new_calls = self.ctx.tool_call_log[before_count:]
        for tc in new_calls:
            await self._emit_tool_start(agent_id, tc.tool)
            await self._emit_tool_end(agent_id, tc.tool, tc.latency_ms, tc.accepted)
            await self._trace(
                "tool_call",
                {
                    "tool": tc.tool, "input": tc.input, "output": tc.output,
                    "latency_ms": tc.latency_ms, "accepted": tc.accepted,
                    "retry_num": tc.retry_num, "error_type": tc.error_type,
                },
                agent_id=agent_id,
                latency_ms=tc.latency_ms,
            )

    # ── Main run loop ─────────────────────────────────────────────────────────

    async def run(self) -> SharedContext:
        """
        Execute the full pipeline:
          1. Plan routing
          2. For each agent: invoke → critique → log
          3. Synthesise
          4. Finalise

        Returns the populated SharedContext.
        """
        t0 = time.perf_counter()
        settings = get_settings()

        # ── Step 1: Routing plan ───────────────────────────────────────────────
        routing_plan = await self.plan()
        agents = routing_plan.agents_selected
        budgets = routing_plan.per_agent_budget

        # Ensure synthesis is always last
        if "synthesis" in agents and agents[-1] != "synthesis":
            agents.remove("synthesis")
            agents.append("synthesis")

        # ── Step 2: Execute agents in order ────────────────────────────────────
        producing_agents = [a for a in agents if a != "synthesis"]
        tool_count_before = len(self.ctx.tool_call_log)

        for agent_id in producing_agents:
            budget = budgets.get(agent_id, settings.decomposition_budget)

            # Check if we need proactive compression before invocation
            # (only applies if agent was previously declared — skip on first run)
            try:
                if self.bm.is_near_limit(agent_id):
                    logger.info("proactive_compression_triggered", extra={
                        "agent_id": agent_id, "job_id": str(self.ctx.job_id)
                    })
            except KeyError:
                pass  # not yet declared, that's fine

            # Invoke the agent
            tool_count_before = len(self.ctx.tool_call_log)
            output = await self._invoke_agent(agent_id, budget)

            # Log any tool calls made during this agent's run
            await self._log_tool_calls(agent_id, tool_count_before)

            # Check for budget violations after this agent
            if self.ctx.budget_violations:
                await self._trace(
                    "budget_violation",
                    {"violations": self.ctx.budget_violations[-3:]},
                    agent_id=agent_id,
                    policy_violations=self.ctx.budget_violations,
                )

            # Critique this agent's output (unless it's already a critique)
            if not agent_id.startswith("critique_"):
                await self._invoke_critique(agent_id)

        # ── Step 3: Synthesis ─────────────────────────────────────────────────
        if "synthesis" in agents:
            synth_budget = budgets.get("synthesis", settings.synthesis_budget)
            tool_count_before = len(self.ctx.tool_call_log)
            await self._invoke_agent("synthesis", synth_budget)
            await self._log_tool_calls("synthesis", tool_count_before)

        # ── Step 4: Finalise ──────────────────────────────────────────────────
        self.ctx.completed_at = datetime.now(timezone.utc)
        total_latency = int((time.perf_counter() - t0) * 1000)

        budget_report = self.bm.audit_report()
        await self._trace(
            "job_complete",
            {
                "final_answer_length": len(self.ctx.final_answer or ""),
                "total_agents_run": len(self.ctx.agent_outputs),
                "total_tool_calls": len(self.ctx.tool_call_log),
                "budget_report": budget_report,
                "summary_stats": self.ctx.summary_stats(),
            },
            latency_ms=total_latency,
        )
        await self._emit({"type": "done", "job_id": str(self.ctx.job_id)})

        logger.info("pipeline_complete", extra={
            "job_id": str(self.ctx.job_id),
            "total_latency_ms": total_latency,
            "agents_run": list(self.ctx.agent_outputs.keys()),
            "violations": len(self.ctx.budget_violations),
        })

        return self.ctx
