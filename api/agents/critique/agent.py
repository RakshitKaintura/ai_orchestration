"""
api/agents/critique/agent.py

Critique Agent — span-level confidence scoring of any agent's output.
"""

from __future__ import annotations

import asyncio
import logging
import time

import google.generativeai as genai
import instructor

from api.agents.base import BaseAgent
from api.agents.critique.schemas import CritiqueResult, SpanAssessment
from api.config import get_settings
from api.context_manager import BudgetManager
from api.models.context import AgentOutput, ClaimScore, SharedContext

logger = logging.getLogger(__name__)

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


class CritiqueAgent(BaseAgent):
    agent_id = "critique"
    default_budget = 4000

    def __init__(self, ctx: SharedContext, bm: BudgetManager, target_agent_id: str) -> None:
        super().__init__(ctx, bm)
        self.target_agent_id = target_agent_id

    async def run(self) -> AgentOutput:
        settings = get_settings()
        t0 = time.perf_counter()

        scoped_id = f"critique_{self.target_agent_id}"
        self.agent_id = scoped_id
        self._declare_budget(settings.critique_budget)

        target_output = self.ctx.agent_outputs.get(self.target_agent_id)
        if not target_output:
            logger.warning("critique_target_not_found", extra={
                "target": self.target_agent_id, "job_id": str(self.ctx.job_id)
            })
            return self._make_output(
                output=f"No output found for agent '{self.target_agent_id}'",
                structured={"error": "target_not_found"},
            )

        target_text = target_output.output[:3000]
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

        genai.configure(api_key=settings.google_api_key or settings.gemini_api_key)
        client = instructor.from_gemini(
            client=genai.GenerativeModel(model_name=settings.primary_model),
            mode=instructor.Mode.GEMINI_JSON,
        )

        try:
            critique: CritiqueResult = await asyncio.to_thread(
                client.chat.completions.create,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                response_model=CritiqueResult,
            )
        except Exception as e:
            logger.error("critique_llm_failed", extra={"error": str(e)})
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

        claim_scores = [
            ClaimScore(
                span=sa.span,
                confidence=sa.confidence,
                flagged=sa.flagged,
                reason=sa.reason if sa.reason else None,
                source_chunk_id=sa.source_chunk_id if sa.source_chunk_id else None,
            )
            for sa in critique.claim_scores
        ]

        if self.target_agent_id in self.ctx.agent_outputs:
            self.ctx.agent_outputs[self.target_agent_id].claim_scores = claim_scores

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
                output_lines.append(f'  ⚠ [{sa.confidence:.2f}] "{sa.span[:80]}…" — {sa.reason}')

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
