"""
tests/test_context.py

Unit tests for api/models/context.py — SharedContext and related models.
Run with: pytest tests/test_context.py -v
"""

from __future__ import annotations

import pytest
from uuid import UUID

from api.models.context import (
    AgentOutput,
    ClaimScore,
    ProvenanceEntry,
    RoutingPlan,
    SharedContext,
    SubTask,
    ToolCall,
)


# ─── SubTask tests ────────────────────────────────────────────────────────────

class TestSubTask:
    def test_default_status_is_pending(self):
        st = SubTask(description="do something")
        assert st.status == "pending"

    def test_default_depends_on_is_empty(self):
        st = SubTask(description="do something")
        assert st.depends_on == []

    def test_id_is_generated(self):
        st = SubTask(description="do something")
        assert st.id is not None
        assert len(st.id) > 0

    def test_explicit_id(self):
        st = SubTask(id="abc123", description="test")
        assert st.id == "abc123"

    def test_can_set_depends_on(self):
        st = SubTask(description="child", depends_on=["parent-1", "parent-2"])
        assert len(st.depends_on) == 2

    def test_task_type_validation(self):
        st = SubTask(description="compute", task_type="computation")
        assert st.task_type == "computation"

    def test_invalid_task_type_raises(self):
        with pytest.raises(Exception):
            SubTask(description="x", task_type="invalid_type")  # type: ignore[arg-type]


# ─── SharedContext tests ──────────────────────────────────────────────────────

class TestSharedContext:
    def test_job_id_is_uuid(self):
        ctx = SharedContext(query="test query")
        assert isinstance(ctx.job_id, UUID)

    def test_query_stored(self):
        ctx = SharedContext(query="what is the capital of France?")
        assert ctx.query == "what is the capital of France?"

    def test_initial_state_empty(self):
        ctx = SharedContext(query="test")
        assert ctx.subtasks == []
        assert ctx.agent_outputs == {}
        assert ctx.tool_call_log == []
        assert ctx.final_answer is None
        assert ctx.budget_violations == []
        assert ctx.provenance_map == []

    def test_context_hash_is_deterministic(self):
        ctx = SharedContext(query="test", job_id=UUID("12345678-1234-5678-1234-567812345678"))
        h1 = ctx.context_hash()
        h2 = ctx.context_hash()
        assert h1 == h2

    def test_context_hash_changes_on_mutation(self):
        ctx = SharedContext(query="test")
        h1 = ctx.context_hash()
        ctx.budget_violations.append("violation!")
        h2 = ctx.context_hash()
        assert h1 != h2

    def test_get_pending_subtasks_no_deps(self):
        ctx = SharedContext(query="test")
        st1 = SubTask(id="a", description="first")
        st2 = SubTask(id="b", description="second")
        ctx.subtasks = [st1, st2]
        pending = ctx.get_pending_subtasks()
        assert len(pending) == 2

    def test_get_pending_subtasks_respects_deps(self):
        ctx = SharedContext(query="test")
        parent = SubTask(id="p", description="parent", status="pending")
        child = SubTask(id="c", description="child", depends_on=["p"])
        ctx.subtasks = [parent, child]
        pending = ctx.get_pending_subtasks()
        # child should not be pending because parent is not done
        assert len(pending) == 1
        assert pending[0].id == "p"

    def test_get_pending_subtasks_releases_when_dep_done(self):
        ctx = SharedContext(query="test")
        parent = SubTask(id="p", description="parent", status="done")
        child = SubTask(id="c", description="child", depends_on=["p"])
        ctx.subtasks = [parent, child]
        pending = ctx.get_pending_subtasks()
        assert len(pending) == 1
        assert pending[0].id == "c"

    def test_all_subtasks_done_true(self):
        ctx = SharedContext(query="test")
        ctx.subtasks = [
            SubTask(id="a", description="a", status="done"),
            SubTask(id="b", description="b", status="done"),
        ]
        assert ctx.all_subtasks_done() is True

    def test_all_subtasks_done_false(self):
        ctx = SharedContext(query="test")
        ctx.subtasks = [
            SubTask(id="a", description="a", status="done"),
            SubTask(id="b", description="b", status="pending"),
        ]
        assert ctx.all_subtasks_done() is False

    def test_has_budget_violations_false(self):
        ctx = SharedContext(query="test")
        assert ctx.has_budget_violations() is False

    def test_has_budget_violations_true(self):
        ctx = SharedContext(query="test")
        ctx.budget_violations.append("some violation")
        assert ctx.has_budget_violations() is True

    def test_get_subtask_by_id_found(self):
        ctx = SharedContext(query="test")
        st = SubTask(id="xyz", description="test subtask")
        ctx.subtasks = [st]
        found = ctx.get_subtask_by_id("xyz")
        assert found is not None
        assert found.id == "xyz"

    def test_get_subtask_by_id_not_found(self):
        ctx = SharedContext(query="test")
        found = ctx.get_subtask_by_id("nonexistent")
        assert found is None

    def test_summary_stats_structure(self):
        ctx = SharedContext(query="test")
        stats = ctx.summary_stats()
        assert "total_tokens_consumed" in stats
        assert "agents_completed" in stats
        assert "total_tool_calls" in stats
        assert "budget_violations" in stats

    def test_summary_stats_counts_agents(self):
        ctx = SharedContext(query="test")
        ctx.agent_outputs["rag"] = AgentOutput(
            agent_id="rag", output="some output", token_count=500
        )
        stats = ctx.summary_stats()
        assert stats["agents_completed"] == 1
        assert stats["total_tokens_consumed"] == 500
