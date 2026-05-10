"""
api/agents/synthesis/agent.py

Synthesis Agent — merges all agent outputs into one final answer with provenance.
"""

from __future__ import annotations

import asyncio
import logging
import time

import google.generativeai as genai
import instructor

from api.agents.base import BaseAgent
from api.agents.llm_retry import call_with_retry
from api.agents.synthesis.schemas import ContradictionResolution, ProvenanceItem, SynthesisResult
from api.config import get_settings
from api.models.context import AgentOutput, ProvenanceEntry

logger = logging.getLogger(__name__)

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


class SynthesisAgent(BaseAgent):
    agent_id = "synthesis"
    default_budget = 5000

    async def run(self) -> AgentOutput:
        settings = get_settings()
        t0 = time.perf_counter()

        self._declare_budget(settings.synthesis_budget)

        agent_summaries = []
        for agent_id, ao in self.ctx.agent_outputs.items():
            flagged_spans = [cs for cs in ao.claim_scores if cs.flagged]
            summary = f"=== {agent_id.upper()} ===\n{ao.output[:2000]}\n"
            if flagged_spans:
                summary += f"\nCritique flagged {len(flagged_spans)} span(s):\n"
                for cs in flagged_spans[:5]:
                    summary += f'  ⚠ "{cs.span[:80]}" — {cs.reason}\n'
            agent_summaries.append(summary)

        all_outputs_text = "\n\n".join(agent_summaries)

        prompt = (
            f"Original query: {self.ctx.query}\n\n"
            f"All agent outputs and critique results:\n\n"
            f"{all_outputs_text}\n\n"
            "Synthesise a single, coherent final answer. "
            "Resolve all contradictions. "
            "Map every sentence to its source agent and chunk."
        )
        input_hash = self._hash(prompt)

        if not self._check_and_add(prompt):
            prompt = await self.bm.compress_and_record(self.agent_id, prompt)
            self.bm.force_add(self.agent_id, prompt)

        api_keys = [k for k in [settings.google_api_key, settings.gemini_api_key] if k]
        system_content = _SYSTEM_PROMPT
        user_content = prompt

        def _synth(client):
            return client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content},
                ],
                response_model=SynthesisResult,
            )

        try:
            synthesis: SynthesisResult = await call_with_retry(
                _synth, api_keys=api_keys, model_name=settings.primary_model
            )
        except Exception as e:
            logger.error("synthesis_llm_failed", extra={"error": str(e)})
            # Build a clean fallback from whatever agents DID produce
            rag_out = self.ctx.agent_outputs.get("rag")
            decomp_out = self.ctx.agent_outputs.get("decomposition")
            if rag_out and rag_out.structured_output and not str(rag_out.structured_output.get("answer","")).startswith("I was unable"):
                fallback_text = rag_out.structured_output.get("answer", "")
            elif decomp_out:
                fallback_text = decomp_out.output[:1200]
            else:
                fallback_text = f"Unable to generate a complete answer for: {self.ctx.query}"
            synthesis = SynthesisResult(
                final_answer=fallback_text,
                provenance_map=[ProvenanceItem(
                    sentence=fallback_text[:200],
                    source_agent="rag" if rag_out else "decomposition",
                )],
                unresolvable_issues=[],
                confidence=0.3,
            )

        self.ctx.final_answer = synthesis.final_answer

        provenance_entries = [
            ProvenanceEntry(
                sentence=pi.sentence,
                source_agent=pi.source_agent,
                source_chunk_id=pi.source_chunk_id if pi.source_chunk_id else None,
                source_chunk_excerpt=pi.source_chunk_excerpt if pi.source_chunk_excerpt else None,
            )
            for pi in synthesis.provenance_map
        ]
        self.ctx.provenance_map = provenance_entries

        self.ctx.contradictions_resolved = [
            f"{cr.description} → {cr.resolution}"
            for cr in synthesis.contradictions_resolved
        ]

        output_lines = [
            f"SYNTHESIS (confidence={synthesis.confidence:.2f})",
            f"\n{synthesis.final_answer}",
            f"\n\nProvenance map ({len(provenance_entries)} entries):",
        ]
        for pe in provenance_entries[:5]:
            cid = f" [{pe.source_chunk_id}]" if pe.source_chunk_id else ""
            output_lines.append(f"  • {pe.sentence[:80]}… → {pe.source_agent}{cid}")
        if len(provenance_entries) > 5:
            output_lines.append(f"  … and {len(provenance_entries) - 5} more")

        if synthesis.contradictions_resolved:
            output_lines.append(f"\nContradictions resolved: {len(synthesis.contradictions_resolved)}")
            for cr in synthesis.contradictions_resolved:
                output_lines.append(f"  ✓ {cr.description[:100]}")

        output_text = "\n".join(output_lines)
        self.bm.force_add(self.agent_id, output_text)
        latency = int((time.perf_counter() - t0) * 1000)

        return self._make_output(
            output=output_text,
            structured={
                "final_answer": synthesis.final_answer,
                "provenance_map": [pe.model_dump() for pe in provenance_entries],
                "contradictions_resolved": [cr.model_dump() for cr in synthesis.contradictions_resolved],
                "unresolvable_issues": synthesis.unresolvable_issues,
                "confidence": synthesis.confidence,
            },
            token_count=self.bm.get_all_states()[self.agent_id].consumed,
            latency_ms=latency,
            input_hash=input_hash,
        )
