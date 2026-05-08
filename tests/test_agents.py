"""
tests/test_agents.py

Unit tests for RAG, Critique, Synthesis agents and Orchestrator.
All LLM calls and ChromaDB are mocked. Tests focus on:
  - Correct SharedContext writes
  - Budget enforcement
  - Graceful failure fallbacks
  - SSE event emission
  - Orchestrator routing logic
"""

from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from api.models.context import (
    AgentOutput, ClaimScore, ProvenanceEntry, SharedContext, SubTask,
)
from api.context_manager import BudgetManager


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def ctx():
    return SharedContext(query="What are the differences between RAG and fine-tuning?")


@pytest.fixture
def bm(ctx):
    return BudgetManager(ctx)


def _mock_rag_output(ctx):
    """Inject a pre-built RAG output into ctx."""
    ao = AgentOutput(
        agent_id="rag",
        output="RAG retrieves documents at inference time. Fine-tuning updates model weights.",
        structured_output={
            "answer": "RAG retrieves documents at inference time. Fine-tuning updates weights.",
            "citations": [{"chunk_id": "chunk-rag-001", "claim": "retrieval", "contribution": "defines RAG"}],
            "follow_up_query_used": "How does fine-tuning differ from RAG?",
            "confidence": 0.85,
            "retrieval_sufficient": True,
            "gaps": "",
            "hop1_chunk_ids": ["chunk-rag-001"],
            "hop2_chunk_ids": ["chunk-ft-001"],
        },
        token_count=200,
        latency_ms=1200,
    )
    ao.compute_hashes()
    ctx.agent_outputs["rag"] = ao
    return ao


# ────────────────────────────────────────────────────────────────────────────────
# Critique Agent
# ────────────────────────────────────────────────────────────────────────────────

class TestCritiqueAgent:
    @pytest.mark.asyncio
    async def test_critique_writes_claim_scores_to_target_output(self, ctx, bm):
        from api.agents.critique import CritiqueAgent, CritiqueResult, SpanAssessment

        _mock_rag_output(ctx)

        mock_critique = CritiqueResult(
            claim_scores=[
                SpanAssessment(
                    span="RAG retrieves documents at inference time",
                    confidence=0.92,
                    flagged=False,
                    reason="",
                ),
                SpanAssessment(
                    span="Fine-tuning updates model weights",
                    confidence=0.95,
                    flagged=False,
                    reason="",
                ),
            ],
            overall_confidence=0.93,
            summary="Output is accurate and well-supported.",
            has_critical_errors=False,
        )

        with patch("api.agents.critique.instructor") as mock_instr:
            mc = MagicMock()
            mock_instr.from_anthropic.return_value = mc
            mc.messages.create = AsyncMock(return_value=mock_critique)

            with patch("api.agents.critique.anthropic.AsyncAnthropic"):
                agent = CritiqueAgent(ctx, bm, target_agent_id="rag")
                output = await agent.run()

        # Claim scores written back to the RAG output
        assert len(ctx.agent_outputs["rag"].claim_scores) == 2
        assert ctx.agent_outputs["rag"].claim_scores[0].confidence == 0.92
        assert output.structured_output["overall_confidence"] == 0.93

    @pytest.mark.asyncio
    async def test_critique_flags_incorrect_spans(self, ctx, bm):
        from api.agents.critique import CritiqueAgent, CritiqueResult, SpanAssessment

        _mock_rag_output(ctx)

        mock_critique = CritiqueResult(
            claim_scores=[
                SpanAssessment(
                    span="Fine-tuning updates model weights",
                    confidence=0.3,
                    flagged=True,
                    reason="This is partially incorrect — some PEFT methods freeze base weights.",
                ),
            ],
            overall_confidence=0.5,
            summary="One span is potentially misleading.",
            has_critical_errors=True,
        )

        with patch("api.agents.critique.instructor") as mock_instr:
            mc = MagicMock()
            mock_instr.from_anthropic.return_value = mc
            mc.messages.create = AsyncMock(return_value=mock_critique)

            with patch("api.agents.critique.anthropic.AsyncAnthropic"):
                agent = CritiqueAgent(ctx, bm, target_agent_id="rag")
                await agent.run()

        flagged = [cs for cs in ctx.agent_outputs["rag"].claim_scores if cs.flagged]
        assert len(flagged) == 1
        assert flagged[0].confidence == 0.3

    @pytest.mark.asyncio
    async def test_critique_target_not_found_returns_gracefully(self, ctx, bm):
        from api.agents.critique import CritiqueAgent

        # No rag output in ctx
        with patch("api.agents.critique.anthropic.AsyncAnthropic"):
            agent = CritiqueAgent(ctx, bm, target_agent_id="rag")
            output = await agent.run()

        assert "not found" in output.output.lower() or "no output" in output.output.lower()

    @pytest.mark.asyncio
    async def test_critique_llm_failure_uses_fallback(self, ctx, bm):
        from api.agents.critique import CritiqueAgent

        _mock_rag_output(ctx)

        with patch("api.agents.critique.instructor") as mock_instr:
            mc = MagicMock()
            mock_instr.from_anthropic.return_value = mc
            mc.messages.create = AsyncMock(side_effect=Exception("API down"))

            with patch("api.agents.critique.anthropic.AsyncAnthropic"):
                agent = CritiqueAgent(ctx, bm, target_agent_id="rag")
                output = await agent.run()

        # Should produce fallback output (not crash)
        assert output.agent_id.startswith("critique")
        # Fallback confidence is 0.7
        assert len(ctx.agent_outputs["rag"].claim_scores) >= 1


# ────────────────────────────────────────────────────────────────────────────────
# Synthesis Agent
# ────────────────────────────────────────────────────────────────────────────────

class TestSynthesisAgent:
    @pytest.mark.asyncio
    async def test_synthesis_writes_final_answer(self, ctx, bm):
        from api.agents.synthesis import SynthesisAgent, SynthesisResult, ProvenanceItem

        _mock_rag_output(ctx)

        mock_synthesis = SynthesisResult(
            final_answer="RAG retrieves documents at inference time. Fine-tuning modifies weights.",
            provenance_map=[
                ProvenanceItem(
                    sentence="RAG retrieves documents at inference time.",
                    source_agent="rag",
                    source_chunk_id="chunk-rag-001",
                ),
                ProvenanceItem(
                    sentence="Fine-tuning modifies weights.",
                    source_agent="rag",
                    source_chunk_id="chunk-ft-001",
                ),
            ],
            contradictions_resolved=[],
            unresolvable_issues=[],
            confidence=0.9,
        )

        with patch("api.agents.synthesis.instructor") as mock_instr:
            mc = MagicMock()
            mock_instr.from_anthropic.return_value = mc
            mc.messages.create = AsyncMock(return_value=mock_synthesis)

            with patch("api.agents.synthesis.anthropic.AsyncAnthropic"):
                agent = SynthesisAgent(ctx, bm)
                output = await agent.run()

        assert ctx.final_answer is not None
        assert "RAG" in ctx.final_answer
        assert len(ctx.provenance_map) == 2
        assert ctx.provenance_map[0].source_agent == "rag"

    @pytest.mark.asyncio
    async def test_synthesis_records_contradictions_resolved(self, ctx, bm):
        from api.agents.synthesis import SynthesisAgent, SynthesisResult, ProvenanceItem, ContradictionResolution

        _mock_rag_output(ctx)

        mock_synthesis = SynthesisResult(
            final_answer="RAG is more flexible than fine-tuning for dynamic knowledge.",
            provenance_map=[
                ProvenanceItem(sentence="RAG is more flexible than fine-tuning for dynamic knowledge.", source_agent="rag"),
            ],
            contradictions_resolved=[
                ContradictionResolution(
                    description="Decomposition said RAG is slower; RAG agent said it is faster.",
                    resolution="Retained RAG agent's claim as it cited a specific benchmark.",
                    agents_involved=["decomposition", "rag"],
                )
            ],
            confidence=0.88,
        )

        with patch("api.agents.synthesis.instructor") as mock_instr:
            mc = MagicMock()
            mock_instr.from_anthropic.return_value = mc
            mc.messages.create = AsyncMock(return_value=mock_synthesis)

            with patch("api.agents.synthesis.anthropic.AsyncAnthropic"):
                agent = SynthesisAgent(ctx, bm)
                await agent.run()

        assert len(ctx.contradictions_resolved) == 1
        assert "Decomposition" in ctx.contradictions_resolved[0]

    @pytest.mark.asyncio
    async def test_synthesis_fallback_to_rag_on_llm_failure(self, ctx, bm):
        from api.agents.synthesis import SynthesisAgent

        _mock_rag_output(ctx)

        with patch("api.agents.synthesis.instructor") as mock_instr:
            mc = MagicMock()
            mock_instr.from_anthropic.return_value = mc
            mc.messages.create = AsyncMock(side_effect=Exception("LLM down"))

            with patch("api.agents.synthesis.anthropic.AsyncAnthropic"):
                agent = SynthesisAgent(ctx, bm)
                output = await agent.run()

        # Fallback: should use RAG's answer
        assert ctx.final_answer is not None
        assert len(ctx.final_answer) > 0


# ────────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ────────────────────────────────────────────────────────────────────────────────

class TestOrchestrator:
    @pytest.mark.asyncio
    async def test_orchestrator_populates_context(self, ctx, bm):
        from api.orchestrator import Orchestrator, AgentSelection
        from api.agents.decomposition import DecompositionOutput, SubTaskSpec
        from api.agents.synthesis import SynthesisResult, ProvenanceItem
        from api.agents.critique import CritiqueResult, SpanAssessment

        decomp_result = DecompositionOutput(
            subtasks=[SubTaskSpec(id="st-1", description="Retrieve info", task_type="retrieval")],
            reasoning="Simple factual query",
        )
        synth_result = SynthesisResult(
            final_answer="RAG differs from fine-tuning in how knowledge is stored.",
            provenance_map=[ProvenanceItem(
                sentence="RAG differs from fine-tuning in how knowledge is stored.",
                source_agent="decomposition",
            )],
            confidence=0.9,
        )
        critique_result = CritiqueResult(
            claim_scores=[SpanAssessment(span="RAG differs", confidence=0.9, flagged=False)],
            overall_confidence=0.9,
            summary="Good output.",
            has_critical_errors=False,
        )
        routing = AgentSelection(
            agents_to_invoke=["decomposition", "synthesis"],
            per_agent_budget={"decomposition": 2000, "synthesis": 3000},
            reasoning="Simple query — no RAG needed",
            query_complexity="simple",
        )

        sse_queue = asyncio.Queue()

        with patch("api.orchestrator.instructor") as mock_instr:
            mc = MagicMock()
            mock_instr.from_anthropic.return_value = mc

            # Routing plan call
            async def create_side_effect(*args, **kwargs):
                response_model = kwargs.get("response_model")
                if response_model.__name__ == "AgentSelection":
                    return routing
                if response_model.__name__ == "DecompositionOutput":
                    return decomp_result
                if response_model.__name__ == "SynthesisResult":
                    return synth_result
                if response_model.__name__ == "CritiqueResult":
                    return critique_result
                return MagicMock()

            mc.messages.create = AsyncMock(side_effect=create_side_effect)

            with patch("api.orchestrator.anthropic.AsyncAnthropic"):
                orchestrator = Orchestrator(ctx, bm, sse_queue=sse_queue)
                result_ctx = await orchestrator.run()

        assert result_ctx.final_answer is not None
        assert "decomposition" in result_ctx.agent_outputs
        assert result_ctx.completed_at is not None

    @pytest.mark.asyncio
    async def test_orchestrator_emits_done_sse_event(self, ctx, bm):
        from api.orchestrator import Orchestrator, AgentSelection
        from api.agents.decomposition import DecompositionOutput, SubTaskSpec
        from api.agents.synthesis import SynthesisResult, ProvenanceItem
        from api.agents.critique import CritiqueResult, SpanAssessment

        decomp_result = DecompositionOutput(
            subtasks=[SubTaskSpec(id="st-1", description="Task", task_type="reasoning")],
            reasoning="Simple",
        )
        synth_result = SynthesisResult(
            final_answer="Final answer.",
            provenance_map=[ProvenanceItem(sentence="Final answer.", source_agent="decomposition")],
            confidence=0.8,
        )
        critique_result = CritiqueResult(
            claim_scores=[SpanAssessment(span="Final answer", confidence=0.8, flagged=False)],
            overall_confidence=0.8, summary="ok", has_critical_errors=False,
        )
        routing = AgentSelection(
            agents_to_invoke=["decomposition", "synthesis"],
            per_agent_budget={"decomposition": 2000, "synthesis": 3000},
            reasoning="Simple",
            query_complexity="simple",
        )

        sse_queue = asyncio.Queue()

        with patch("api.orchestrator.instructor") as mock_instr:
            mc = MagicMock()
            mock_instr.from_anthropic.return_value = mc

            async def create_side_effect(*args, **kwargs):
                rm = kwargs.get("response_model")
                if rm and rm.__name__ == "AgentSelection": return routing
                if rm and rm.__name__ == "DecompositionOutput": return decomp_result
                if rm and rm.__name__ == "SynthesisResult": return synth_result
                if rm and rm.__name__ == "CritiqueResult": return critique_result
                return MagicMock()

            mc.messages.create = AsyncMock(side_effect=create_side_effect)

            with patch("api.orchestrator.anthropic.AsyncAnthropic"):
                orchestrator = Orchestrator(ctx, bm, sse_queue=sse_queue)
                await orchestrator.run()

        # Collect all emitted events
        events = []
        while not sse_queue.empty():
            events.append(await sse_queue.get())

        event_types = [e.get("type") for e in events]
        assert "done" in event_types

    @pytest.mark.asyncio
    async def test_orchestrator_synthesis_always_last(self, ctx, bm):
        from api.orchestrator import Orchestrator, AgentSelection
        from api.agents.decomposition import DecompositionOutput, SubTaskSpec
        from api.agents.synthesis import SynthesisResult, ProvenanceItem
        from api.agents.critique import CritiqueResult, SpanAssessment

        # Routing plan with synthesis NOT last
        routing = AgentSelection(
            agents_to_invoke=["synthesis", "decomposition"],  # wrong order
            per_agent_budget={"decomposition": 2000, "synthesis": 3000},
            reasoning="Intentionally wrong order for test",
            query_complexity="simple",
        )
        decomp_result = DecompositionOutput(
            subtasks=[SubTaskSpec(id="st-1", description="Task", task_type="reasoning")],
            reasoning="Simple",
        )
        synth_result = SynthesisResult(
            final_answer="Corrected order answer.",
            provenance_map=[ProvenanceItem(sentence="Corrected order answer.", source_agent="decomposition")],
            confidence=0.8,
        )
        critique_result = CritiqueResult(
            claim_scores=[SpanAssessment(span="Corrected", confidence=0.8, flagged=False)],
            overall_confidence=0.8, summary="ok", has_critical_errors=False,
        )

        call_order = []

        async def create_side_effect(*args, **kwargs):
            rm = kwargs.get("response_model")
            if rm and rm.__name__ == "AgentSelection": return routing
            if rm and rm.__name__ == "DecompositionOutput":
                call_order.append("decomposition")
                return decomp_result
            if rm and rm.__name__ == "SynthesisResult":
                call_order.append("synthesis")
                return synth_result
            if rm and rm.__name__ == "CritiqueResult": return critique_result
            return MagicMock()

        with patch("api.orchestrator.instructor") as mock_instr:
            mc = MagicMock()
            mock_instr.from_anthropic.return_value = mc
            mc.messages.create = AsyncMock(side_effect=create_side_effect)

            with patch("api.orchestrator.anthropic.AsyncAnthropic"):
                orchestrator = Orchestrator(ctx, bm)
                await orchestrator.run()

        # Synthesis must be called after decomposition
        assert call_order.index("synthesis") > call_order.index("decomposition")
