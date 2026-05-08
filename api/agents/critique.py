"""
api/agents/critique.py

Critique Agent
--------------
Reviews every other agent's output and assigns SPAN-LEVEL confidence scores.
The critique agent NEVER flags a whole output as wrong — it must identify
specific verbatim text spans that are uncertain or incorrect.

Called by the orchestrator AFTER every producing agent completes.
Results are written back into:
  SharedContext.agent_outputs[target_agent_id].claim_scores

Budget: 4000 tokens (CRITIQUE_BUDGET env var)

Structured output:
  - claim_scores:       list[ClaimScore]  — per-span confidence + flag
  - overall_confidence: float
  - summary:            str
"""

from __future__ import annotations

import logging
import time

import anthropic
import instructor
from pydantic import BaseModel, Field

from api.agents.base import BaseAgent
from api.agents.compression import compress_context_async
from api.config import get_settings
from api.context_manager import BudgetManager
from api.models.context import AgentOutput, ClaimScore, SharedContext

logger = logging.getLogger(__name__)


# ─── Instructor structured outputs ────────────────────────────────────────────

class SpanAssessment(BaseModel):
    span: str = Field(
        description="Exact verbatim text span from the agent's output being assessed. "
                    "Must be a substring of the output text."
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="0.0 = almost certainly wrong, 1.0 = almost certainly correct",
    )
    flagged: bool = Field(
        description="True if you disagree with or are uncertain about this claim"
    )
    reason: str = Field(
        default="",
        description="If flagged=True, explain specifically why you disagree or are uncertain",
    )
    source_chunk_id: str = Field(
        default="",
        description="If you verified this span against a retrieved chunk, the chunk ID",
    )


class CritiqueResult(BaseModel):
    claim_scores: list[SpanAssessment] = Field(
        description="Per-span assessments. Must cover key claims. Minimum 1 entry.",
        min_length=1,
    )
    overall_confidence: float = Field(
        ge=0.0, le=1.0,
        description="Overall confidence in the target agent's output as a whole",
    )
    summary: str = Field(
        description="One-paragraph summary of the critique findings",
    )
    has_critical_errors: bool = Field(
        description="True if any span is flagged AND confidence < 0.4 (critical disagreement)",
    )


# ─── System prompt ────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a critique agent in a multi-agent AI system.

Your job: review another agent's output and assess each key claim at the SPAN level.

Rules:
1. You MUST identify specific verbatim text spans, not the whole output.
2. Quote the exact text from the output — do not paraphrase.
3. Assign confidence 0.0-1.0 per span (1.0 = definitely correct, 0.0 = definitely wrong).
4. Flag a span (flagged=True) if confidence < 0.7 or if you actively disagree.
5. For flagged spans, always provide a specific reason.
6. You MUST produce at least one SpanAssessment, even for high-quality outputs.
7. For outputs with no errors, still assess the key claims with high confidence scores.
8. Do NOT assess the entire output at once — break it into its constituent claims."""


# ─── Agent ────────────────────────────────────────────────────────────────────

class CritiqueAgent(BaseAgent):
    agent_id = "critique"
    default_budget = 4000

    def __init__(
        self,
        ctx: SharedContext,
        bm: BudgetManager,
        target_agent_id: str,
    ) -> None:
        super().__init__(ctx, bm)
        self.target_agent_id = target_agent_id

    async def run(self) -> AgentOutput:
        settings = get_settings()
        t0 = time.perf_counter()

        # Critique agent is declared fresh for each target agent
        # The orchestrator uses a scoped agent_id to avoid double-declare
        scoped_id = f"critique_{self.target_agent_id}"
        self.agent_id = scoped_id
        self._declare_budget(settings.critique_budget)

        # Fetch the target agent's output
        target_output = self.ctx.agent_outputs.get(self.target_agent_id)
        if not target_output:
            logger.warning("critique_target_not_found", extra={
                "target": self.target_agent_id, "job_id": str(self.ctx.job_id)
            })
            return self._make_output(
                output=f"No output found for agent '{self.target_agent_id}'",
                structured={"error": "target_not_found"},
            )

        # Build critique prompt
        target_text = target_output.output[:3000]  # cap for budget
        prompt = (
            f"Agent being reviewed: {self.target_agent_id}\n\n"
            f"Agent output:\n{target_text}\n\n"
            f"Original query that produced this output: {self.ctx.query}\n\n"
            "Assess each key claim at the span level. Identify any errors, unsupported claims, "
            "or uncertain statements with specific quoted spans."
        )
        input_hash = self._hash(prompt)

        if not self._check_and_add(prompt):
            prompt = await self.bm.compress_and_record(scoped_id, prompt)
            self.bm.force_add(scoped_id, prompt)

        # LLM critique call
        raw_client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        client = instructor.from_anthropic(raw_client)

        try:
            critique: CritiqueResult = await client.messages.create(
                model=settings.primary_model,
                max_tokens=1200,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
                response_model=CritiqueResult,
            )
        except Exception as e:
            logger.error("critique_llm_failed", extra={"error": str(e)})
            # Fallback: high-confidence pass-through
            critique = CritiqueResult(
                claim_scores=[SpanAssessment(
                    span=target_text[:100],
                    confidence=0.7,
                    flagged=False,
                    reason="",
                )],
                overall_confidence=0.7,
                summary=f"Critique LLM failed: {e}. Cannot assess output.",
                has_critical_errors=False,
            )

        # Convert to ClaimScore objects and write back to the target's output
        claim_scores = []
        for sa in critique.claim_scores:
            cs = ClaimScore(
                span=sa.span,
                confidence=sa.confidence,
                flagged=sa.flagged,
                reason=sa.reason if sa.reason else None,
                source_chunk_id=sa.source_chunk_id if sa.source_chunk_id else None,
            )
            claim_scores.append(cs)

        # Write results back into the target agent's output
        if self.target_agent_id in self.ctx.agent_outputs:
            self.ctx.agent_outputs[self.target_agent_id].claim_scores = claim_scores

        # Build output text
        flagged_count = sum(1 for sa in critique.claim_scores if sa.flagged)
        output_lines = [
            f"CRITIQUE of [{self.target_agent_id}] — overall confidence: {critique.overall_confidence:.2f}",
            f"Critical errors: {'YES ⚠' if critique.has_critical_errors else 'none'}",
            f"Spans assessed: {len(critique.claim_scores)} | Flagged: {flagged_count}",
            f"\nSummary: {critique.summary}",
            "\nFlagged spans:",
        ]
        for sa in critique.claim_scores:
            if sa.flagged:
                output_lines.append(f"  ⚠ [{sa.confidence:.2f}] \"{sa.span[:80]}…\" — {sa.reason}")

        output_text = "\n".join(output_lines)
        self.bm.force_add(scoped_id, output_text)
        latency = int((time.perf_counter() - t0) * 1000)

        return self._make_output(
            output=output_text,
            structured={
                "target_agent_id": self.target_agent_id,
                "claim_scores": [cs.model_dump() for cs in claim_scores],
                "overall_confidence": critique.overall_confidence,
                "summary": critique.summary,
                "has_critical_errors": critique.has_critical_errors,
                "flagged_count": flagged_count,
            },
            token_count=self.bm.get_all_states()[scoped_id].consumed,
            latency_ms=latency,
            input_hash=input_hash,
        )
