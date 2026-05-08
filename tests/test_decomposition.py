"""
tests/test_decomposition.py

Unit tests for the Decomposition Agent's DAG validation logic.
The LLM call is mocked — we test structure, validation, and error handling.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from api.models.context import SharedContext, SubTask
from api.context_manager import BudgetManager
from api.agents.decomposition import DecompositionOutput, SubTaskSpec


# ─── DAG Validation tests (no LLM required) ──────────────────────────────────

class TestDecompositionOutputValidation:
    def test_single_task_valid(self):
        out = DecompositionOutput(
            subtasks=[SubTaskSpec(id="st-1", description="do it", task_type="reasoning")],
            reasoning="Simple task",
        )
        assert len(out.subtasks) == 1

    def test_linear_chain_valid(self):
        out = DecompositionOutput(
            subtasks=[
                SubTaskSpec(id="st-1", description="first", task_type="retrieval"),
                SubTaskSpec(id="st-2", description="second", task_type="reasoning", depends_on=["st-1"]),
                SubTaskSpec(id="st-3", description="third", task_type="synthesis", depends_on=["st-2"]),
            ],
            reasoning="Linear chain",
        )
        assert len(out.subtasks) == 3

    def test_diamond_dag_valid(self):
        """A→B, A→C, B→D, C→D is a valid DAG."""
        out = DecompositionOutput(
            subtasks=[
                SubTaskSpec(id="a", description="a", task_type="retrieval"),
                SubTaskSpec(id="b", description="b", task_type="reasoning", depends_on=["a"]),
                SubTaskSpec(id="c", description="c", task_type="computation", depends_on=["a"]),
                SubTaskSpec(id="d", description="d", task_type="synthesis", depends_on=["b", "c"]),
            ],
            reasoning="Diamond",
        )
        assert len(out.subtasks) == 4

    def test_cycle_detected(self):
        """A→B→C→A should fail."""
        with pytest.raises(Exception, match="cycle"):
            DecompositionOutput(
                subtasks=[
                    SubTaskSpec(id="a", description="a", task_type="reasoning", depends_on=["c"]),
                    SubTaskSpec(id="b", description="b", task_type="reasoning", depends_on=["a"]),
                    SubTaskSpec(id="c", description="c", task_type="reasoning", depends_on=["b"]),
                ],
                reasoning="Cyclic",
            )

    def test_missing_dependency_id(self):
        """Depends on an ID that doesn't exist."""
        with pytest.raises(Exception, match="does not exist"):
            DecompositionOutput(
                subtasks=[
                    SubTaskSpec(id="st-1", description="task", task_type="reasoning", depends_on=["nonexistent"]),
                ],
                reasoning="Bad dep",
            )

    def test_self_dependency_is_cycle(self):
        """A task depending on itself is a cycle."""
        with pytest.raises(Exception):
            DecompositionOutput(
                subtasks=[
                    SubTaskSpec(id="a", description="self-referential", task_type="reasoning", depends_on=["a"]),
                ],
                reasoning="Self-dep",
            )

    def test_parallel_tasks_valid(self):
        """Multiple tasks with no dependencies can run in parallel."""
        out = DecompositionOutput(
            subtasks=[
                SubTaskSpec(id="a", description="parallel a", task_type="retrieval"),
                SubTaskSpec(id="b", description="parallel b", task_type="retrieval"),
                SubTaskSpec(id="c", description="parallel c", task_type="computation"),
                SubTaskSpec(id="merge", description="merge", task_type="synthesis", depends_on=["a", "b", "c"]),
            ],
            reasoning="Fan-in",
        )
        assert len(out.subtasks) == 4

    def test_ambiguous_flag(self):
        out = DecompositionOutput(
            subtasks=[SubTaskSpec(id="st-1", description="clarify", task_type="reasoning")],
            reasoning="Unclear",
            is_ambiguous=True,
            ambiguity_notes="No context provided",
        )
        assert out.is_ambiguous is True
        assert "context" in out.ambiguity_notes


# ─── Agent integration (mocked LLM) ──────────────────────────────────────────

class TestDecompositionAgent:
    @pytest.fixture
    def ctx_and_bm(self):
        ctx = SharedContext(query="Compare RAG and fine-tuning for production LLM systems")
        bm = BudgetManager(ctx)
        return ctx, bm

    @pytest.mark.asyncio
    async def test_run_writes_subtasks_to_context(self, ctx_and_bm):
        ctx, bm = ctx_and_bm
        from api.agents.decomposition import DecompositionAgent

        mock_result = DecompositionOutput(
            subtasks=[
                SubTaskSpec(id="st-1", description="Retrieve RAG info", task_type="retrieval"),
                SubTaskSpec(id="st-2", description="Retrieve fine-tuning info", task_type="retrieval"),
                SubTaskSpec(id="st-3", description="Compare", task_type="reasoning", depends_on=["st-1", "st-2"]),
            ],
            reasoning="Need retrieval before comparison",
        )

        with patch("api.agents.decomposition.instructor") as mock_instructor:
            mock_client = MagicMock()
            mock_instructor.from_anthropic.return_value = mock_client
            mock_client.messages.create = AsyncMock(return_value=mock_result)

            with patch("api.agents.decomposition.anthropic.AsyncAnthropic"):
                agent = DecompositionAgent(ctx, bm)
                output = await agent.run()

        assert len(ctx.subtasks) == 3
        assert ctx.subtasks[0].id == "st-1"
        assert ctx.subtasks[2].depends_on == ["st-1", "st-2"]
        assert output.agent_id == "decomposition"

    @pytest.mark.asyncio
    async def test_run_produces_output_hash(self, ctx_and_bm):
        ctx, bm = ctx_and_bm
        from api.agents.decomposition import DecompositionAgent

        mock_result = DecompositionOutput(
            subtasks=[SubTaskSpec(id="st-1", description="do something", task_type="reasoning")],
            reasoning="Simple",
        )

        with patch("api.agents.decomposition.instructor") as mock_instructor:
            mock_client = MagicMock()
            mock_instructor.from_anthropic.return_value = mock_client
            mock_client.messages.create = AsyncMock(return_value=mock_result)

            with patch("api.agents.decomposition.anthropic.AsyncAnthropic"):
                agent = DecompositionAgent(ctx, bm)
                output = await agent.run()

        assert output.output_hash is not None
        assert len(output.output_hash) == 16

    @pytest.mark.asyncio
    async def test_llm_failure_produces_fallback(self, ctx_and_bm):
        ctx, bm = ctx_and_bm
        from api.agents.decomposition import DecompositionAgent

        with patch("api.agents.decomposition.instructor") as mock_instructor:
            mock_client = MagicMock()
            mock_instructor.from_anthropic.return_value = mock_client
            mock_client.messages.create = AsyncMock(side_effect=Exception("API Error"))

            with patch("api.agents.decomposition.anthropic.AsyncAnthropic"):
                agent = DecompositionAgent(ctx, bm)
                output = await agent.run()

        # Should produce a single fallback subtask
        assert len(ctx.subtasks) == 1
        assert ctx.subtasks[0].id == "st-1"
        assert "Fallback" in output.structured["reasoning"]
