"""
api/agents/synthesis.py

Synthesis Agent
---------------
Merges outputs from all preceding agents into a single coherent final answer.

Responsibilities:
  1. Read all AgentOutput objects from SharedContext
  2. Read all ClaimScore lists (from critique agent) on each output
  3. Resolve contradictions flagged by the critique agent — NOT surface them to user
  4. Produce a final answer where every sentence has a provenance entry
  5. Write final_answer and provenance_map to SharedContext

Budget: 5000 tokens (SYNTHESIS_BUDGET env var)

Structured output:
  - final_answer:           str
  - provenance_map:         list[ProvenanceEntry]
  - contradictions_resolved: list[str]  — description of each resolved contradiction
  - unresolvable_issues:    list[str]   — issues that could not be resolved (noted internally)
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
from api.models.context import AgentOutput, ProvenanceEntry, SharedContext

logger = logging.getLogger(__name__)


# ─── Instructor structured outputs ────────────────────────────────────────────

class ProvenanceItem(BaseModel):
    sentence: str = Field(description="Exact sentence from the final answer")
    source_agent: str = Field(description="Which agent produced the underlying information")
    source_chunk_id: str = Field(
        default="",
        description="Chunk ID if the sentence is grounded in a retrieved chunk",
    )
    source_chunk_excerpt: str = Field(
        default="",
        description="Brief excerpt from the source chunk (≤150 chars)",
    )


class ContradictionResolution(BaseModel):
    description: str = Field(description="What the contradiction was")
    resolution: str = Field(description="How it was resolved in the final answer")
    agents_involved: list[str] = Field(description="Which agents produced conflicting claims")


class SynthesisResult(BaseModel):
    final_answer: str = Field(
        description="The complete, contradiction-free final answer to present to the user"
    )
    provenance_map: list[ProvenanceItem] = Field(
        description="Per-sentence provenance. Every sentence in final_answer must appear here.",
        min_length=1,
    )
    contradictions_resolved: list[ContradictionResolution] = Field(
        default_factory=list,
        description="Each contradiction that was detected and resolved",
    )
    unresolvable_issues: list[str] = Field(
        default_factory=list,
        description="Issues that could not be resolved (stored internally, NOT in final_answer)",
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Overall confidence in the final synthesised answer",
    )


# ─── System prompt ────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are the synthesis agent in a multi-agent AI system.

Your job: merge all agent outputs into a single, coherent final answer.

Rules:
1. Read all agent outputs and their critique results carefully.
2. RESOLVE contradictions — do NOT surface them to the user. If two agents disagree,
   reason about which is more likely correct based on the evidence, and use that.
3. Write a final_answer where EVERY sentence can be traced to a source.
4. For each sentence in the final_answer, create a provenance_map entry linking it to
   the source agent and, where possible, the specific chunk ID.
5. List every contradiction you resolved with a description and resolution.
6. Sentences that are genuinely unresolvable should be omitted from the final_answer
   and noted in unresolvable_issues.
7. The final answer must be clear, accurate, and directly address the original query."""


# ─── Agent ────────────────────────────────────────────────────────────────────

class SynthesisAgent(BaseAgent):
    agent_id = "synthesis"
    default_budget = 5000

    async def run(self) -> AgentOutput:
        settings = get_settings()
        t0 = time.perf_counter()

        self._declare_budget(settings.synthesis_budget)

        # ── Assemble all agent outputs and critique results ────────────────────
        agent_summaries = []
        for agent_id, ao in self.ctx.agent_outputs.items():
            flagged_spans = [
                cs for cs in ao.claim_scores if cs.flagged
            ]
            summary = (
                f"=== {agent_id.upper()} ===\n"
                f"{ao.output[:2000]}\n"
            )
            if flagged_spans:
                summary += f"\nCritique flagged {len(flagged_spans)} span(s):\n"
                for cs in flagged_spans[:5]:  # cap at 5 to save tokens
                    summary += f"  ⚠ \"{cs.span[:80]}\" — {cs.reason}\n"
            agent_summaries.append(summary)

        all_outputs_text = "\n\n".join(agent_summaries)

        # ── Build synthesis prompt ─────────────────────────────────────────────
        prompt = (
            f"Original query: {self.ctx.query}\n\n"
            f"All agent outputs and critique results:\n\n"
            f"{all_outputs_text}\n\n"
            "Synthesise a single, coherent final answer. "
            "Resolve all contradictions. "
            "Map every sentence to its source agent and chunk."
        )
        input_hash = self._hash(prompt)

        # Budget check and compress if needed
        if not self._check_and_add(prompt):
            prompt = await self.bm.compress_and_record(self.agent_id, prompt)
            self.bm.force_add(self.agent_id, prompt)

        # ── LLM synthesis call ────────────────────────────────────────────────
        raw_client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        client = instructor.from_anthropic(raw_client)

        try:
            synthesis: SynthesisResult = await client.messages.create(
                model=settings.primary_model,
                max_tokens=2000,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
                response_model=SynthesisResult,
            )
        except Exception as e:
            logger.error("synthesis_llm_failed", extra={"error": str(e)})
            # Graceful fallback: take the RAG agent's answer if available
            rag_output = self.ctx.agent_outputs.get("rag")
            fallback_answer = (
                rag_output.structured_output.get("answer", "Unable to synthesise an answer.")
                if rag_output else "Unable to synthesise an answer."
            )
            synthesis = SynthesisResult(
                final_answer=fallback_answer,
                provenance_map=[ProvenanceItem(
                    sentence=fallback_answer[:200],
                    source_agent="rag" if rag_output else "fallback",
                    source_chunk_id="",
                )],
                contradictions_resolved=[],
                unresolvable_issues=[f"Synthesis LLM failed: {e}"],
                confidence=0.3,
            )

        # ── Write to SharedContext ────────────────────────────────────────────
        self.ctx.final_answer = synthesis.final_answer

        provenance_entries = []
        for pi in synthesis.provenance_map:
            entry = ProvenanceEntry(
                sentence=pi.sentence,
                source_agent=pi.source_agent,
                source_chunk_id=pi.source_chunk_id if pi.source_chunk_id else None,
                source_chunk_excerpt=pi.source_chunk_excerpt if pi.source_chunk_excerpt else None,
            )
            provenance_entries.append(entry)
        self.ctx.provenance_map = provenance_entries

        self.ctx.contradictions_resolved = [
            f"{cr.description} → {cr.resolution}"
            for cr in synthesis.contradictions_resolved
        ]

        # ── Build output text ─────────────────────────────────────────────────
        output_lines = [
            f"SYNTHESIS (confidence={synthesis.confidence:.2f})",
            f"\n{synthesis.final_answer}",
            f"\n\nProvenance map ({len(provenance_entries)} entries):",
        ]
        for pe in provenance_entries[:5]:  # show first 5 in output text
            cid = f" [{pe.source_chunk_id}]" if pe.source_chunk_id else ""
            output_lines.append(f"  • {pe.sentence[:80]}… → {pe.source_agent}{cid}")
        if len(provenance_entries) > 5:
            output_lines.append(f"  … and {len(provenance_entries) - 5} more")

        if synthesis.contradictions_resolved:
            output_lines.append(f"\nContradictions resolved: {len(synthesis.contradictions_resolved)}")
            for cr in synthesis.contradictions_resolved:
                output_lines.append(f"  ✓ {cr.description[:100]}")

        if synthesis.unresolvable_issues:
            output_lines.append(f"\n[Internal] Unresolvable issues: {len(synthesis.unresolvable_issues)}")

        output_text = "\n".join(output_lines)
        self.bm.force_add(self.agent_id, output_text)
        latency = int((time.perf_counter() - t0) * 1000)

        return self._make_output(
            output=output_text,
            structured={
                "final_answer": synthesis.final_answer,
                "provenance_map": [pe.model_dump() for pe in provenance_entries],
                "contradictions_resolved": [
                    cr.model_dump() for cr in synthesis.contradictions_resolved
                ],
                "unresolvable_issues": synthesis.unresolvable_issues,
                "confidence": synthesis.confidence,
            },
            token_count=self.bm.get_all_states()[self.agent_id].consumed,
            latency_ms=latency,
            input_hash=input_hash,
        )
