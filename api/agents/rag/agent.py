"""
api/agents/rag/agent.py

RAG Agent — mandatory two-hop retrieval + per-claim citations.
"""

from __future__ import annotations

import asyncio
import logging
import time

import google.generativeai as genai
import instructor

from api.agents.base import BaseAgent
from api.agents.llm_retry import call_with_retry
from api.agents.rag.retriever import retrieve, _get_chroma_collection
from api.agents.rag.schemas import ChunkCitation, FollowUpQuery, RAGAnswer
from api.config import get_settings
from api.models.context import AgentOutput

logger = logging.getLogger(__name__)


class RAGAgent(BaseAgent):
    agent_id = "rag"
    default_budget = 6000

    async def run(self) -> AgentOutput:
        settings = get_settings()
        t0 = time.perf_counter()

        self._declare_budget(settings.rag_budget)
        query = self.ctx.query
        input_hash = self._hash(query)

        api_keys = [k for k in [settings.google_api_key, settings.gemini_api_key] if k]

        # ── HOP 1 ─────────────────────────────────────────────────────────────
        hop1_chunks: list[dict] = []
        try:
            hop1_chunks = await retrieve(query, n=3)
            logger.info("rag_hop1_complete", extra={
                "chunks": len(hop1_chunks), "job_id": str(self.ctx.job_id)
            })
        except Exception as e:
            logger.error("rag_hop1_failed", extra={"error": str(e)})

        hop1_text = "\n\n".join(
            f"[{c['id']}] {c['text']}" for c in hop1_chunks
        ) or "No chunks retrieved in hop 1."

        if not self._check_and_add(hop1_text):
            hop1_text = await self.bm.compress_and_record(self.agent_id, hop1_text)
            self.bm.force_add(self.agent_id, hop1_text)

        # ── GENERATE FOLLOW-UP QUERY ───────────────────────────────────────────
        follow_up_query = query
        try:
            followup_content = (
                f"Original question: {query}\n\n"
                f"After reading these retrieved passages:\n{hop1_text}\n\n"
                "What additional information would you need to give a complete answer? "
                "Write a specific follow-up retrieval query to find it."
            )
            def _followup(client):
                return client.chat.completions.create(
                    messages=[{"role": "user", "content": followup_content}],
                    response_model=FollowUpQuery,
                )
            fq_result = await call_with_retry(_followup, api_keys=api_keys, model_name=settings.primary_model)
            follow_up_query = fq_result.query
            logger.info("rag_followup_query", extra={"query": follow_up_query[:100]})
        except Exception as e:
            logger.warning("rag_followup_failed", extra={"error": str(e)})

        # ── HOP 2 ─────────────────────────────────────────────────────────────
        hop1_ids = {c["id"] for c in hop1_chunks}
        hop2_chunks: list[dict] = []
        try:
            hop2_chunks = await retrieve(follow_up_query, n=3, exclude_ids=hop1_ids)
            logger.info("rag_hop2_complete", extra={
                "chunks": len(hop2_chunks), "job_id": str(self.ctx.job_id)
            })
        except Exception as e:
            logger.error("rag_hop2_failed", extra={"error": str(e)})

        hop2_text = "\n\n".join(
            f"[{c['id']}] {c['text']}" for c in hop2_chunks
        ) or "No additional chunks retrieved in hop 2."

        if not self._check_and_add(hop2_text):
            hop2_text = await self.bm.compress_and_record(self.agent_id, hop2_text)
            self.bm.force_add(self.agent_id, hop2_text)

        # ── FINAL SYNTHESIS ────────────────────────────────────────────────────
        all_chunks = hop1_chunks + hop2_chunks
        all_chunks_text = "\n\n".join(
            f"[{c['id']}] (hop {'1' if c['id'] in hop1_ids else '2'}) {c['text']}"
            for c in all_chunks
        )

        synthesis_prompt = (
            f"Question: {query}\n\n"
            f"Retrieved chunks (cite by ID):\n{all_chunks_text}\n\n"
            f"Follow-up query used for second retrieval: {follow_up_query}\n\n"
            "Synthesise a complete answer. For every claim, cite which chunk ID supports it. "
            "If the chunks are insufficient to fully answer the question, say so."
        )

        if not self._check_and_add(synthesis_prompt):
            synthesis_prompt = await self.bm.compress_and_record(self.agent_id, synthesis_prompt)
            self.bm.force_add(self.agent_id, synthesis_prompt)

        try:
            def _rag_synth(client):
                return client.chat.completions.create(
                    messages=[{"role": "user", "content": synthesis_prompt}],
                    response_model=RAGAnswer,
                )
            rag_answer: RAGAnswer = await call_with_retry(
                _rag_synth, api_keys=api_keys, model_name=settings.primary_model
            )
        except Exception as e:
            logger.error("rag_synthesis_failed", extra={"error": str(e)})
            # Build a clean fallback answer from the raw retrieved chunks
            if all_chunks:
                chunk_texts = " ".join(c["text"][:300] for c in all_chunks[:3])
                fallback_ans = f"Based on retrieved information: {chunk_texts[:800]}"
            else:
                fallback_ans = f"I was unable to retrieve relevant information to answer: {query}"
            rag_answer = RAGAnswer(
                answer=fallback_ans,
                citations=[ChunkCitation(
                    chunk_id=all_chunks[0]["id"] if all_chunks else "none",
                    claim="fallback from retrieved chunks",
                    contribution="LLM synthesis failed, using raw chunks",
                )],
                follow_up_query_used=follow_up_query,
                confidence=0.1,
                retrieval_sufficient=bool(all_chunks),
                gaps="LLM synthesis unavailable",
            )

        output_lines = [
            f"RAG ANSWER (confidence={rag_answer.confidence:.2f}):",
            rag_answer.answer,
            f"\nFollow-up query used: {rag_answer.follow_up_query_used}",
            f"\nChunks retrieved: hop1={len(hop1_chunks)}, hop2={len(hop2_chunks)}",
            "\nCitations:",
        ]
        for cit in rag_answer.citations:
            output_lines.append(f'  [{cit.chunk_id}] → "{cit.claim}" — {cit.contribution}')

        if not rag_answer.retrieval_sufficient:
            output_lines.append(f"\n⚠ Retrieval gaps: {rag_answer.gaps}")

        output_text = "\n".join(output_lines)
        self.bm.force_add(self.agent_id, output_text)
        latency = int((time.perf_counter() - t0) * 1000)

        return self._make_output(
            output=output_text,
            structured={
                "answer": rag_answer.answer,
                "citations": [c.model_dump() for c in rag_answer.citations],
                "follow_up_query_used": rag_answer.follow_up_query_used,
                "confidence": rag_answer.confidence,
                "retrieval_sufficient": rag_answer.retrieval_sufficient,
                "gaps": rag_answer.gaps,
                "hop1_chunk_ids": [c["id"] for c in hop1_chunks],
                "hop2_chunk_ids": [c["id"] for c in hop2_chunks],
            },
            token_count=self.bm.get_all_states()[self.agent_id].consumed,
            latency_ms=latency,
            input_hash=input_hash,
        )
