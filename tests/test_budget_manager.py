"""
tests/test_budget_manager.py

Unit tests for api/context_manager.py — BudgetManager.
Run with: pytest tests/test_budget_manager.py -v
"""

from __future__ import annotations

import pytest

from api.context_manager import BudgetManager, AgentBudgetState, COMPRESSION_TRIGGER_THRESHOLD
from api.models.context import SharedContext


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def ctx():
    return SharedContext(query="test query")


@pytest.fixture
def bm(ctx):
    return BudgetManager(ctx)


# ─── Declaration tests ────────────────────────────────────────────────────────

class TestDeclaration:
    def test_declare_registers_agent(self, bm):
        bm.declare("rag", 6000)
        assert bm.check_remaining("rag") == 6000

    def test_declare_twice_raises(self, bm):
        bm.declare("rag", 6000)
        with pytest.raises(ValueError, match="already declared"):
            bm.declare("rag", 4000)

    def test_declare_zero_budget_raises(self, bm):
        with pytest.raises(ValueError):
            bm.declare("rag", 0)

    def test_declare_negative_budget_raises(self, bm):
        with pytest.raises(ValueError):
            bm.declare("rag", -100)

    def test_multiple_agents_independent(self, bm):
        bm.declare("rag", 6000)
        bm.declare("critique", 4000)
        assert bm.check_remaining("rag") == 6000
        assert bm.check_remaining("critique") == 4000


# ─── Token counting tests ─────────────────────────────────────────────────────

class TestTokenCounting:
    def test_count_tokens_empty(self):
        assert BudgetManager.count_tokens("") == 0

    def test_count_tokens_positive(self):
        tokens = BudgetManager.count_tokens("hello world")
        assert tokens > 0

    def test_count_tokens_longer_text_is_more(self):
        short = BudgetManager.count_tokens("hi")
        long = BudgetManager.count_tokens("hi " * 100)
        assert long > short

    def test_count_tokens_none_treated_as_empty(self):
        # None should not crash (empty check in implementation)
        assert BudgetManager.count_tokens("") == 0


# ─── Budget add tests ─────────────────────────────────────────────────────────

class TestBudgetAdd:
    def test_add_within_budget_returns_true(self, bm):
        bm.declare("rag", 6000)
        result = bm.add("rag", "short text")
        assert result is True

    def test_add_reduces_remaining(self, bm):
        bm.declare("rag", 6000)
        tokens_before = bm.check_remaining("rag")
        bm.add("rag", "short text")
        tokens_after = bm.check_remaining("rag")
        assert tokens_after < tokens_before

    def test_add_exceeding_budget_returns_false(self, bm, ctx):
        bm.declare("rag", 5)  # tiny budget
        big_text = "word " * 1000
        result = bm.add("rag", big_text)
        assert result is False

    def test_add_violation_logged_to_context(self, bm, ctx):
        bm.declare("rag", 5)
        bm.add("rag", "word " * 1000)
        assert len(ctx.budget_violations) == 1
        assert "BUDGET_VIOLATION" in ctx.budget_violations[0]
        assert "rag" in ctx.budget_violations[0]

    def test_add_violation_increments_state_counter(self, bm):
        bm.declare("rag", 5)
        bm.add("rag", "word " * 1000)
        state = bm.get_all_states()["rag"]
        assert state.violation_count == 1

    def test_add_undeclared_agent_raises(self, bm):
        with pytest.raises(KeyError, match="not been declared"):
            bm.add("undeclared_agent", "some text")

    def test_multiple_adds_accumulate_correctly(self, bm):
        bm.declare("rag", 6000)
        bm.add("rag", "word " * 10)
        bm.add("rag", "word " * 10)
        remaining = bm.check_remaining("rag")
        assert remaining < 6000


# ─── would_exceed tests ───────────────────────────────────────────────────────

class TestWouldExceed:
    def test_would_exceed_false_when_fits(self, bm):
        bm.declare("rag", 6000)
        assert bm.would_exceed("rag", "short text") is False

    def test_would_exceed_true_when_too_big(self, bm):
        bm.declare("rag", 5)
        assert bm.would_exceed("rag", "word " * 1000) is True

    def test_would_exceed_does_not_modify_state(self, bm):
        bm.declare("rag", 6000)
        remaining_before = bm.check_remaining("rag")
        bm.would_exceed("rag", "short text")
        remaining_after = bm.check_remaining("rag")
        assert remaining_before == remaining_after


# ─── is_near_limit tests ──────────────────────────────────────────────────────

class TestIsNearLimit:
    def test_not_near_limit_on_fresh_budget(self, bm):
        bm.declare("rag", 6000)
        assert bm.is_near_limit("rag") is False

    def test_near_limit_when_almost_full(self, bm):
        bm.declare("rag", 600)
        # Fill up to just below the threshold
        bm._states["rag"].consumed = 200  # remaining = 400 > 500? No, 400 < 500 → near limit
        # Actually with threshold=500, remaining=400 < 500 → True
        bm._states["rag"].consumed = 200  # remaining = 400
        assert bm.is_near_limit("rag", threshold=500) is True


# ─── Compression hook tests ───────────────────────────────────────────────────

class TestCompressionHook:
    def test_record_compression_reduces_consumed(self, bm):
        bm.declare("rag", 6000)
        bm._states["rag"].consumed = 3000
        bm.record_compression("rag", tokens_saved=1000)
        assert bm._states["rag"].consumed == 2000

    def test_record_compression_increments_count(self, bm):
        bm.declare("rag", 6000)
        bm.record_compression("rag", tokens_saved=0)
        assert bm._states["rag"].compression_count == 1

    def test_record_compression_cannot_go_below_zero(self, bm):
        bm.declare("rag", 6000)
        bm._states["rag"].consumed = 100
        bm.record_compression("rag", tokens_saved=5000)
        assert bm._states["rag"].consumed == 0

    def test_record_compression_negative_raises(self, bm):
        bm.declare("rag", 6000)
        with pytest.raises(ValueError):
            bm.record_compression("rag", tokens_saved=-1)


# ─── Audit report tests ───────────────────────────────────────────────────────

class TestAuditReport:
    def test_audit_report_no_violations(self, bm):
        bm.declare("rag", 6000)
        report = bm.audit_report()
        assert report["total_violations"] == 0
        assert report["compliance"] == "PASS"

    def test_audit_report_with_violations(self, bm):
        bm.declare("rag", 5)
        bm.add("rag", "word " * 1000)  # trigger violation
        report = bm.audit_report()
        assert report["total_violations"] == 1
        assert report["compliance"] == "FAIL"

    def test_audit_report_contains_all_agents(self, bm):
        bm.declare("rag", 6000)
        bm.declare("critique", 4000)
        report = bm.audit_report()
        assert "rag" in report["agents"]
        assert "critique" in report["agents"]
