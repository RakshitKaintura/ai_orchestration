"""
api/tools/self_reflection.py

Self-Reflection Tool — Tool #4
Reads the calling agent's own previous outputs from SharedContext and uses
an LLM to identify contradictions between them and the current context.

This tool is called by agents that want to check their own consistency
before producing a final output. It reads from SharedContext (not the DB),
which means it operates on the live pipeline state.

Failure contract (enforced in code, not prompts):
  - malformed → no prior agent outputs exist in SharedContext (nothing to reflect on)
  - empty     → no contradictions found (this is a SUCCESS, not a failure)
  - timeout   → LLM call exceeds REFLECTION_TIMEOUT_SECONDS
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import anthropic
import instructor
from pydantic import BaseModel, Field

from api.config import get_settings
from api.models.context import SharedContext
from api.models.tools import ToolResult, SelfReflectionContradiction

REFLECTION_TIMEOUT_SECONDS = 15.0
TOOL_NAME = "self_reflection"


# ─── Instructor structured output ────────────────────────────────────────────

class ReflectionOutput(BaseModel):
    contradictions: list[SelfReflectionContradiction] = Field(
        default_factory=list,
        description="List of contradictions found. Empty list = no contradictions.",
    )
    consistency_score: float = Field(
        ge=0.0, le=1.0,
        description="0.0 = highly inconsistent, 1.0 = fully consistent",
    )
    summary: str = Field(
        description="One-paragraph summary of consistency analysis",
    )


async def _analyze_consistency(
    agent_outputs_text: str,
    current_query: str,
) -> ReflectionOutput:
    """LLM call to identify contradictions across agent outputs."""
    settings = get_settings()
    raw_client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
    client = instructor.from_anthropic(raw_client)

    return await client.messages.create(
        model=settings.primary_model,
        max_tokens=1000,
        messages=[{
            "role": "user",
            "content": (
                "Review the following outputs from multiple AI agents working on the same query. "
                "Identify any factual contradictions between them — places where one agent says "
                "something that directly conflicts with what another agent says.\n\n"
                "Be precise: quote the exact conflicting spans.\n\n"
                f"Original query: {current_query}\n\n"
                f"Agent outputs:\n{agent_outputs_text}\n\n"
                "Return a list of contradictions found (empty list if none), "
                "a consistency score (0.0-1.0), and a summary."
            ),
        }],
        response_model=ReflectionOutput,
    )


def _build_outputs_text(ctx: SharedContext, requesting_agent: str) -> str:
    """Format all agent outputs into a single text for the reflection LLM."""
    lines = []
    for agent_id, output in ctx.agent_outputs.items():
        lines.append(f"=== {agent_id.upper()} ===")
        lines.append(output.output[:2000])  # cap at 2000 chars per agent
        lines.append("")
    return "\n".join(lines)


# ─── Public tool function ─────────────────────────────────────────────────────

async def self_reflection(input_data: dict, ctx: SharedContext) -> ToolResult:
    """
    Tool #4 — Self-Reflection

    Input schema:
        requesting_agent (str, required) — ID of the agent requesting reflection
        focus_on         (str, optional) — specific topic to focus the contradiction check on

    Requires: SharedContext passed as second argument (special signature vs other tools)

    Failure contract:
        malformed  → requesting_agent is empty, or no prior agent outputs exist in context
        timeout    → LLM call exceeds REFLECTION_TIMEOUT_SECONDS
        empty      → (NOT used — empty contradictions list = success with empty data)
    """
    t0 = time.perf_counter()

    # ── Malformed check ───────────────────────────────────────────────────────
    requesting_agent = input_data.get("requesting_agent")
    if not isinstance(requesting_agent, str) or not requesting_agent.strip():
        return ToolResult.malformed(
            source=TOOL_NAME,
            message="'requesting_agent' must be a non-empty string",
        )

    if not ctx.agent_outputs:
        return ToolResult.malformed(
            source=TOOL_NAME,
            message="No prior agent outputs exist in context. Nothing to reflect on.",
        )

    # ── Build context text ────────────────────────────────────────────────────
    outputs_text = _build_outputs_text(ctx, requesting_agent)
    if not outputs_text.strip():
        return ToolResult.malformed(
            source=TOOL_NAME,
            message="All agent outputs are empty. Cannot perform reflection.",
        )

    # ── LLM reflection call ───────────────────────────────────────────────────
    try:
        reflection = await asyncio.wait_for(
            _analyze_consistency(outputs_text, ctx.query),
            timeout=REFLECTION_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        latency = (time.perf_counter() - t0) * 1000
        return ToolResult.timeout(
            source=TOOL_NAME,
            latency_ms=latency,
            message=f"Self-reflection LLM call timed out after {REFLECTION_TIMEOUT_SECONDS}s",
        )
    except Exception as e:
        return ToolResult.malformed(
            source=TOOL_NAME,
            message=f"Self-reflection failed: {e}",
        )

    latency = (time.perf_counter() - t0) * 1000

    # Note: empty contradictions list is a success (not an empty failure)
    return ToolResult.ok(
        data={
            "contradictions": [c.model_dump() for c in reflection.contradictions],
            "consistency_score": reflection.consistency_score,
            "summary": reflection.summary,
            "agents_reviewed": list(ctx.agent_outputs.keys()),
        },
        source=TOOL_NAME,
        latency_ms=latency,
    )


# ─── Adapter for call_tool_with_retry ─────────────────────────────────────────
# self_reflection has a non-standard signature (requires ctx).
# This factory creates a standard (input_data) → ToolResult coroutine
# that can be passed to call_tool_with_retry.

def make_self_reflection_fn(ctx: SharedContext):
    """Return a standard tool function with ctx pre-bound."""
    async def _bound(input_data: dict) -> ToolResult:
        return await self_reflection(input_data, ctx)
    return _bound
