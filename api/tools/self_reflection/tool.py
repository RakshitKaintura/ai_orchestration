"""
api/tools/self_reflection/tool.py

Self-Reflection Tool — Tool #4
"""

from __future__ import annotations

import asyncio
import time

import google.generativeai as genai
import instructor
from pydantic import BaseModel, Field

from api.config import get_settings
from api.models.context import SharedContext
from api.models.tools import SelfReflectionContradiction, ToolResult

REFLECTION_TIMEOUT_SECONDS = 15.0
TOOL_NAME = "self_reflection"


class ReflectionOutput(BaseModel):
    contradictions: list[SelfReflectionContradiction] = Field(
        default_factory=list,
        description="List of contradictions found. Empty list = no contradictions.",
    )
    consistency_score: float = Field(
        ge=0.0, le=1.0,
        description="0.0 = highly inconsistent, 1.0 = fully consistent",
    )
    summary: str = Field(description="One-paragraph summary of consistency analysis")


async def _analyze_consistency(agent_outputs_text: str, current_query: str) -> ReflectionOutput:
    settings = get_settings()
    genai.configure(api_key=settings.google_api_key or settings.gemini_api_key)
    client = instructor.from_gemini(
        client=genai.GenerativeModel(model_name=settings.primary_model),
        mode=instructor.Mode.GEMINI_JSON,
    )
    return await asyncio.to_thread(
        client.chat.completions.create,
        messages=[{"role": "user", "content": (
            "Review the following outputs from multiple AI agents working on the same query. "
            "Identify any factual contradictions between them — places where one agent says "
            "something that directly conflicts with what another agent says.\n\n"
            "Be precise: quote the exact conflicting spans.\n\n"
            f"Original query: {current_query}\n\n"
            f"Agent outputs:\n{agent_outputs_text}\n\n"
            "Return a list of contradictions found (empty list if none), "
            "a consistency score (0.0-1.0), and a summary."
        )}],
        response_model=ReflectionOutput,
    )


def _build_outputs_text(ctx: SharedContext, requesting_agent: str) -> str:
    lines = []
    for agent_id, output in ctx.agent_outputs.items():
        lines.append(f"=== {agent_id.upper()} ===")
        lines.append(output.output[:2000])
        lines.append("")
    return "\n".join(lines)


async def self_reflection(input_data: dict, ctx: SharedContext) -> ToolResult:
    """
    Tool #4 — Self-Reflection

    Input schema:
        requesting_agent (str, required) — ID of the agent requesting reflection
        focus_on         (str, optional) — specific topic to focus on
    """
    t0 = time.perf_counter()

    requesting_agent = input_data.get("requesting_agent")
    if not isinstance(requesting_agent, str) or not requesting_agent.strip():
        return ToolResult.malformed(source=TOOL_NAME,
                                    message="'requesting_agent' must be a non-empty string")

    if not ctx.agent_outputs:
        return ToolResult.malformed(source=TOOL_NAME,
                                    message="No prior agent outputs exist in context. Nothing to reflect on.")

    outputs_text = _build_outputs_text(ctx, requesting_agent)
    if not outputs_text.strip():
        return ToolResult.malformed(source=TOOL_NAME,
                                    message="All agent outputs are empty. Cannot perform reflection.")

    try:
        reflection = await asyncio.wait_for(
            _analyze_consistency(outputs_text, ctx.query),
            timeout=REFLECTION_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        latency = (time.perf_counter() - t0) * 1000
        return ToolResult.timeout(source=TOOL_NAME, latency_ms=latency,
                                  message=f"Self-reflection LLM call timed out after {REFLECTION_TIMEOUT_SECONDS}s")
    except Exception as e:
        return ToolResult.malformed(source=TOOL_NAME, message=f"Self-reflection failed: {e}")

    latency = (time.perf_counter() - t0) * 1000
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


def make_self_reflection_fn(ctx: SharedContext):
    """Return a standard tool callable with ctx pre-bound."""
    async def _bound(input_data: dict) -> ToolResult:
        return await self_reflection(input_data, ctx)
    return _bound
