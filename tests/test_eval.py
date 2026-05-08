"""
tests/test_eval.py

Unit tests for the evaluation harness:
  - Test case structure validation
  - Individual scorer logic (no LLM for most)
  - Summary computation
  - Meta agent dimension-to-agent mapping
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from eval.cases.test_cases import (
    ALL_CASES, BASELINE_CASES, AMBIGUOUS_CASES, ADVERSARIAL_CASES,
    get_case, get_cases_by_category, EvalCase,
)
from eval.scorers import (
    score_citations, score_contradictions, score_tool_efficiency,
    score_budget_compliance, score_critique_agreement,
    SCORER_NAMES, SCORER_WEIGHTS,
)
from api.models.context import SharedContext, AgentOutput, ProvenanceEntry, ClaimScore
from api.context_manager import BudgetManager


# ────────────────────────────────────────────────────────────────────────────────
# Test Case Structure
# ────────────────────────────────────────────────────────────────────────────────

class TestCaseStructure:
    def test_total_case_count(self):
        assert len(ALL_CASES) == 15

    def test_category_counts(self):
        assert len(BASELINE_CASES) == 5
        assert len(AMBIGUOUS_CASES) == 5
        assert len(ADVERSARIAL_CASES) == 5

    def test_all_case_ids_unique(self):
        ids = [c.case_id for c in ALL_CASES]
        assert len(ids) == len(set(ids))

    def test_all_cases_have_expected_answer(self):
        for case in ALL_CASES:
            assert case.expected_answer.strip(), f"Case {case.case_id} has empty expected_answer"

    def test_all_cases_have_query(self):
        for case in ALL_CASES:
            assert case.query.strip(), f"Case {case.case_id} has empty query"

    def test_adversarial_cases_have_forbidden_claims(self):
        for case in ADVERSARIAL_CASES:
            assert len(case.forbidden_claims) > 0, f"Adversarial case {case.case_id} has no forbidden_claims"

    def test_get_case_by_id(self):
        case = get_case("base-001")
        assert case.case_id == "base-001"
        assert case.category == "baseline"

    def test_get_case_invalid_id_raises(self):
        with pytest.raises(KeyError):
            get_case("nonexistent-999")

    def test_get_cases_by_category(self):
        baseline = get_cases_by_category("baseline")
        assert len(baseline) == 5
        assert all(c.category == "baseline" for c in baseline)

    def test_case_ids_follow_naming_convention(self):
        """All case IDs follow 'prefix-NNN' format."""
        import re
        pattern = re.compile(r"^(base|amb|adv)-\d{3}$")
        for case in ALL_CASES:
            assert pattern.match(case.case_id), f"Bad case_id format: {case.case_id}"


# ────────────────────────────────────────────────────────────────────────────────
# Citations Scorer
# ────────────────────────────────────────────────────────────────────────────────

class TestCitationsScorer:
    @pytest.fixture
    def ctx_with_provenance(self):
        ctx = SharedContext(query="test")
        ctx.provenance_map = [
            ProvenanceEntry(sentence="RAG retrieves docs.", source_agent="rag", source_chunk_id="chunk-rag-001"),
            ProvenanceEntry(sentence="Fine-tuning updates weights.", source_agent="rag", source_chunk_id="chunk-ft-001"),
        ]
        return ctx

    @pytest.mark.asyncio
    async def test_no_provenance_scores_zero(self):
        case = get_case("base-001")
        ctx = SharedContext(query="test")
        score, justification = await score_citations(case, ctx)
        assert score == 0.0
        assert "No provenance_map" in justification

    @pytest.mark.asyncio
    async def test_good_citations_scores_well(self, ctx_with_provenance):
        case = get_case("base-001")
        with patch("eval.scorers._get_chroma_collection", return_value=None):
            score, justification = await score_citations(case, ctx_with_provenance)
        # Should score based on expected chunk coverage
        assert score >= 0.0
        assert "Cited" in justification

    @pytest.mark.asyncio
    async def test_no_expected_chunks_scores_high(self):
        case = EvalCase(
            case_id="test-000",
            category="baseline",
            query="Simple question",
            expected_answer="Answer",
            expected_chunk_ids=[],  # No chunks required
        )
        ctx = SharedContext(query="test")
        ctx.provenance_map = []  # No citations either
        score, justification = await score_citations(case, ctx)
        assert score == 1.0


# ────────────────────────────────────────────────────────────────────────────────
# Contradictions Scorer
# ────────────────────────────────────────────────────────────────────────────────

class TestContradictionsScorer:
    @pytest.mark.asyncio
    async def test_surfaced_contradiction_penalised(self):
        case = get_case("adv-005")
        ctx = SharedContext(query="test")
        ctx.final_answer = "One source says RAG is faster, another source says fine-tuning is faster."
        score, justification = await score_contradictions(case, ctx)
        assert score < 0.5
        assert "surfaced" in justification.lower() or "one source says" in justification.lower()

    @pytest.mark.asyncio
    async def test_clean_answer_scores_high(self):
        case = get_case("base-001")
        ctx = SharedContext(query="test")
        ctx.final_answer = "RAG retrieves documents; fine-tuning updates weights."
        ctx.contradictions_resolved = []
        score, justification = await score_contradictions(case, ctx)
        assert score >= 0.8

    @pytest.mark.asyncio
    async def test_resolved_contradictions_score_perfect(self):
        case = get_case("base-001")
        ctx = SharedContext(query="test")
        ctx.final_answer = "RAG is better for dynamic knowledge."
        ctx.contradictions_resolved = ["Agent A said faster; Agent B said slower → resolved: slower is correct"]
        score, justification = await score_contradictions(case, ctx)
        assert score == 1.0
        assert "resolved" in justification.lower()


# ────────────────────────────────────────────────────────────────────────────────
# Tool Efficiency Scorer
# ────────────────────────────────────────────────────────────────────────────────

class TestToolEfficiencyScorer:
    @pytest.mark.asyncio
    async def test_no_tools_required_no_tools_used_scores_perfect(self):
        case = get_case("base-001")  # no requires_tools
        ctx = SharedContext(query="test")
        score, justification = await score_tool_efficiency(case, ctx)
        assert score == 1.0

    @pytest.mark.asyncio
    async def test_required_tool_not_used_penalised(self):
        case = get_case("base-005")  # requires code_sandbox
        ctx = SharedContext(query="test")
        score, justification = await score_tool_efficiency(case, ctx)
        assert score < 1.0
        assert "code_sandbox" in justification

    @pytest.mark.asyncio
    async def test_extra_tool_penalised(self):
        from api.models.context import ToolCall
        case = get_case("base-001")  # requires no tools
        ctx = SharedContext(query="test")
        # Simulate unnecessary tool call
        ctx.tool_call_log = [
            ToolCall(tool="web_search", input={}, output={}, latency_ms=100, success=True, accepted=True, retry_num=0, agent_id="rag"),
            ToolCall(tool="code_sandbox", input={}, output={}, latency_ms=200, success=True, accepted=True, retry_num=0, agent_id="rag"),
        ]
        score, justification = await score_tool_efficiency(case, ctx)
        assert score < 1.0
        assert "Extra" in justification or "extra" in justification.lower()

    @pytest.mark.asyncio
    async def test_retry_penalised(self):
        from api.models.context import ToolCall
        case = get_case("base-001")
        ctx = SharedContext(query="test")
        ctx.tool_call_log = [
            ToolCall(tool="web_search", input={}, output={}, latency_ms=100, success=False, error_type="timeout", accepted=False, retry_num=0, agent_id="rag"),
            ToolCall(tool="web_search", input={}, output={}, latency_ms=100, success=True, accepted=True, retry_num=1, agent_id="rag"),
        ]
        score, justification = await score_tool_efficiency(case, ctx)
        assert score < 1.0
        assert "Retries" in justification or "retry" in justification.lower()


# ────────────────────────────────────────────────────────────────────────────────
# Budget Compliance Scorer
# ────────────────────────────────────────────────────────────────────────────────

class TestBudgetComplianceScorer:
    @pytest.mark.asyncio
    async def test_no_violations_scores_perfect(self):
        case = get_case("base-001")
        ctx = SharedContext(query="test")
        score, justification = await score_budget_compliance(case, ctx)
        assert score == 1.0
        assert "No budget violations" in justification

    @pytest.mark.asyncio
    async def test_one_violation_reduces_score(self):
        case = get_case("base-001")
        ctx = SharedContext(query="test")
        ctx.budget_violations = ["BUDGET_VIOLATION | agent=rag | attempted=7000 | remaining=100"]
        score, justification = await score_budget_compliance(case, ctx)
        assert score == pytest.approx(0.85)

    @pytest.mark.asyncio
    async def test_many_violations_floors_at_zero(self):
        case = get_case("base-001")
        ctx = SharedContext(query="test")
        ctx.budget_violations = [f"violation_{i}" for i in range(10)]
        score, justification = await score_budget_compliance(case, ctx)
        assert score == 0.0


# ────────────────────────────────────────────────────────────────────────────────
# Critique Agreement Scorer
# ────────────────────────────────────────────────────────────────────────────────

class TestCritiqueAgreementScorer:
    @pytest.mark.asyncio
    async def test_no_critique_output_defaults_to_half(self):
        case = get_case("base-001")
        ctx = SharedContext(query="test")
        score, justification = await score_critique_agreement(case, ctx)
        assert score == 0.5
        assert "No critique output" in justification

    @pytest.mark.asyncio
    async def test_high_confidence_critique_scores_high(self):
        case = get_case("base-001")
        ctx = SharedContext(query="test")
        ao = AgentOutput(
            agent_id="critique_synthesis",
            output="Good output.",
            structured_output={"overall_confidence": 0.92, "has_critical_errors": False},
        )
        ctx.agent_outputs["critique_synthesis"] = ao
        score, justification = await score_critique_agreement(case, ctx)
        assert score == pytest.approx(0.92)

    @pytest.mark.asyncio
    async def test_critical_errors_cap_score(self):
        case = get_case("base-001")
        ctx = SharedContext(query="test")
        ao = AgentOutput(
            agent_id="critique_synthesis",
            output="Critical errors found.",
            structured_output={"overall_confidence": 0.9, "has_critical_errors": True},
        )
        ctx.agent_outputs["critique_synthesis"] = ao
        score, justification = await score_critique_agreement(case, ctx)
        assert score <= 0.5
        assert "CRITICAL ERRORS" in justification


# ────────────────────────────────────────────────────────────────────────────────
# Scorer weights
# ────────────────────────────────────────────────────────────────────────────────

class TestScorerWeights:
    def test_weights_sum_to_one(self):
        total = sum(SCORER_WEIGHTS.values())
        assert abs(total - 1.0) < 1e-6, f"Weights sum to {total}, expected 1.0"

    def test_all_dimensions_have_weights(self):
        for dim in SCORER_NAMES:
            assert dim in SCORER_WEIGHTS, f"Dimension '{dim}' has no weight"


# ────────────────────────────────────────────────────────────────────────────────
# Meta Agent
# ────────────────────────────────────────────────────────────────────────────────

class TestMetaAgent:
    def test_dimension_agent_map_covers_all_dimensions(self):
        from api.agents.meta import DIMENSION_AGENT_MAP
        for dim in SCORER_NAMES:
            assert dim in DIMENSION_AGENT_MAP, f"Dimension '{dim}' not in DIMENSION_AGENT_MAP"

    @pytest.mark.asyncio
    async def test_no_rewrite_when_all_scores_above_threshold(self):
        from api.agents.meta import MetaAgent

        summary = {
            "summary_by_dimension": {
                "correctness": {"mean_score": 0.92},
                "citations": {"mean_score": 0.88},
                "contradictions": {"mean_score": 0.90},
                "tool_efficiency": {"mean_score": 0.95},
                "budget_compliance": {"mean_score": 0.87},
                "critique_agreement": {"mean_score": 0.91},
            },
            "overall_mean": 0.905,
        }

        agent = MetaAgent()
        proposal = await agent.propose_rewrite(summary, [], {})
        assert proposal is None  # All above 0.85

    @pytest.mark.asyncio
    async def test_proposes_rewrite_for_worst_dimension(self):
        from api.agents.meta import MetaAgent
        from pydantic import BaseModel, Field

        summary = {
            "summary_by_dimension": {
                "correctness": {"mean_score": 0.90},
                "citations": {"mean_score": 0.45},  # worst
                "contradictions": {"mean_score": 0.88},
                "tool_efficiency": {"mean_score": 0.92},
                "budget_compliance": {"mean_score": 0.87},
                "critique_agreement": {"mean_score": 0.91},
            },
            "overall_mean": 0.82,
        }

        class MockProposal(BaseModel):
            target_agent: str = "rag"
            target_dimension: str = "citations"
            analysis: str = "Citations are poor"
            rewritten_prompt: str = "Improved RAG prompt..."
            diff_summary: str = "Added citation requirements"
            expected_improvement: str = "Better chunk citations"
            confidence: float = 0.75

        with patch("api.agents.meta.instructor") as mock_instr:
            mc = MagicMock()
            mock_instr.from_anthropic.return_value = mc
            mc.messages.create = AsyncMock(return_value=MockProposal())

            with patch("api.agents.meta.anthropic.AsyncAnthropic"):
                agent = MetaAgent()
                proposal = await agent.propose_rewrite(summary, [], {"rag": "old prompt"})

        assert proposal is not None
        assert proposal.target_dimension == "citations"
        assert proposal.target_agent == "rag"
