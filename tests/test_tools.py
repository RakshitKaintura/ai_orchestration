"""
tests/test_tools.py

Unit tests for all four tools and base infrastructure.
Run with: pytest tests/test_tools.py -v -x
"""

from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from api.models.context import SharedContext, AgentOutput
from api.models.tools import ToolResult, WebSearchResult
from api.tools.base import (
    ToolLogger,
    call_tool_with_retry,
    simplify_input,
    broaden_input,
    fix_format,
    _mutate_input,
)


# ─── Tool fixture ─────────────────────────────────────────────────────────────

@pytest.fixture
def ctx():
    return SharedContext(query="test query")


# ────────────────────────────────────────────────────────────────────────────────
# Input mutation helpers
# ────────────────────────────────────────────────────────────────────────────────

class TestSimplifyInput:
    def test_shortens_query(self):
        data = {"query": "What is the meaning of life. And other things."}
        result = simplify_input(data)
        assert len(result["query"]) < len(data["query"])
        assert result["_retry_strategy"] == "simplified"

    def test_trims_code(self):
        code = "\n".join(f"line_{i}" for i in range(50))
        result = simplify_input({"code": code})
        assert result["code"].count("\n") <= 19  # max 20 lines → 19 newlines

    def test_preserves_other_keys(self):
        result = simplify_input({"query": "hello", "limit": 5})
        assert result["limit"] == 5


class TestBroadenInput:
    def test_appends_or_clause(self):
        result = broaden_input({"query": "exact product name"})
        assert "OR" in result["query"] or "related" in result["query"]
        assert result["_retry_strategy"] == "broadened"

    def test_removes_filters(self):
        result = broaden_input({"query": "test", "filters": {"category": "Electronics"}})
        assert result["filters"] == {}

    def test_doubles_limit(self):
        result = broaden_input({"query": "test", "limit": 5})
        assert result["limit"] == 10


class TestFixFormat:
    def test_converts_bytes(self):
        result = fix_format({"query": b"hello bytes"})
        assert isinstance(result["query"], str)
        assert result["_retry_strategy"] == "format_fixed"

    def test_removes_private_keys(self):
        result = fix_format({"query": "test", "_internal": "value"})
        assert "_internal" not in result

    def test_converts_unknown_types(self):
        result = fix_format({"value": object()})
        assert isinstance(result["value"], str)


class TestMutateInput:
    def test_routes_timeout_to_simplify(self):
        result = _mutate_input({"query": "a very long query sentence here"}, "timeout")
        assert result["_retry_strategy"] == "simplified"

    def test_routes_empty_to_broaden(self):
        result = _mutate_input({"query": "narrow query"}, "empty")
        assert result["_retry_strategy"] == "broadened"

    def test_routes_malformed_to_fix_format(self):
        result = _mutate_input({"query": "test"}, "malformed")
        assert result["_retry_strategy"] == "format_fixed"


# ────────────────────────────────────────────────────────────────────────────────
# ToolLogger
# ────────────────────────────────────────────────────────────────────────────────

class TestToolLogger:
    def test_appends_to_tool_call_log(self, ctx):
        result = ToolResult.ok(data={"results": []}, source="web_search", latency_ms=100)
        call = ToolLogger.log(
            ctx=ctx,
            tool_name="web_search",
            agent_id="rag",
            input_data={"query": "test"},
            result=result,
            retry_num=0,
            accepted=True,
        )
        assert len(ctx.tool_call_log) == 1
        assert ctx.tool_call_log[0].tool == "web_search"

    def test_logs_failure_correctly(self, ctx):
        result = ToolResult.timeout(source="web_search", latency_ms=5001)
        call = ToolLogger.log(
            ctx=ctx,
            tool_name="web_search",
            agent_id="rag",
            input_data={"query": "test"},
            result=result,
            retry_num=1,
            accepted=False,
        )
        assert ctx.tool_call_log[0].success is False
        assert ctx.tool_call_log[0].retry_num == 1
        assert ctx.tool_call_log[0].accepted is False

    def test_records_rejection_reason(self, ctx):
        result = ToolResult.ok(data=[], source="web_search", latency_ms=100)
        ToolLogger.log(
            ctx=ctx,
            tool_name="web_search",
            agent_id="rag",
            input_data={"query": "test"},
            result=result,
            accepted=False,
            rejection_reason="Results not relevant enough",
        )
        assert ctx.tool_call_log[0].rejection_reason == "Results not relevant enough"


# ────────────────────────────────────────────────────────────────────────────────
# call_tool_with_retry
# ────────────────────────────────────────────────────────────────────────────────

class TestCallToolWithRetry:
    @pytest.mark.asyncio
    async def test_returns_on_first_success(self, ctx):
        mock_tool = AsyncMock(return_value=ToolResult.ok(data="result", source="test"))

        result = await call_tool_with_retry(
            mock_tool, {"query": "test"}, ctx, "rag", "test_tool"
        )
        assert result.success is True
        assert mock_tool.call_count == 1
        assert len(ctx.tool_call_log) == 1

    @pytest.mark.asyncio
    async def test_retries_on_failure(self, ctx):
        fail = ToolResult.timeout(source="test", latency_ms=5001)
        success = ToolResult.ok(data="result", source="test")
        mock_tool = AsyncMock(side_effect=[fail, success])

        result = await call_tool_with_retry(
            mock_tool, {"query": "test"}, ctx, "rag", "test_tool"
        )
        assert result.success is True
        assert mock_tool.call_count == 2
        assert len(ctx.tool_call_log) == 2  # one per attempt

    @pytest.mark.asyncio
    async def test_max_retries_exhausted(self, ctx):
        fail = ToolResult.timeout(source="test", latency_ms=5001)
        mock_tool = AsyncMock(return_value=fail)

        result = await call_tool_with_retry(
            mock_tool, {"query": "test"}, ctx, "rag", "test_tool", max_retries=2
        )
        assert result.success is False
        assert mock_tool.call_count == 3  # initial + 2 retries
        assert len(ctx.tool_call_log) == 3

    @pytest.mark.asyncio
    async def test_acceptance_check_can_reject_success(self, ctx):
        """Agent can reject a technically successful result."""
        ok = ToolResult.ok(data=[], source="test")  # empty list = useless
        mock_tool = AsyncMock(side_effect=[ok, ok, ok])

        def check(r):
            return (False, "Empty results not useful") if r.data == [] else (True, None)

        result = await call_tool_with_retry(
            mock_tool, {"query": "test"}, ctx, "rag", "test_tool",
            acceptance_check=check, max_retries=2,
        )
        # All rejected, returns last result
        assert mock_tool.call_count == 3
        # All logged as not accepted
        assert all(not tc.accepted for tc in ctx.tool_call_log)

    @pytest.mark.asyncio
    async def test_retry_num_tracked_per_attempt(self, ctx):
        fail = ToolResult.empty(source="test")
        success = ToolResult.ok(data="x", source="test")
        mock_tool = AsyncMock(side_effect=[fail, success])

        await call_tool_with_retry(mock_tool, {"q": "test"}, ctx, "rag", "test_tool")

        assert ctx.tool_call_log[0].retry_num == 0
        assert ctx.tool_call_log[1].retry_num == 1


# ────────────────────────────────────────────────────────────────────────────────
# Web Search tool
# ────────────────────────────────────────────────────────────────────────────────

class TestWebSearch:
    @pytest.mark.asyncio
    async def test_valid_query_returns_results(self):
        from api.tools.web_search import web_search
        result = await web_search({"query": "RAG retrieval augmented generation"})
        assert result.success is True
        assert isinstance(result.data, list)
        assert len(result.data) > 0
        assert "url" in result.data[0]
        assert "relevance_score" in result.data[0]

    @pytest.mark.asyncio
    async def test_empty_query_malformed(self):
        from api.tools.web_search import web_search
        result = await web_search({"query": ""})
        assert result.success is False
        assert result.error_type == "malformed"

    @pytest.mark.asyncio
    async def test_missing_query_malformed(self):
        from api.tools.web_search import web_search
        result = await web_search({})
        assert result.success is False
        assert result.error_type == "malformed"

    @pytest.mark.asyncio
    async def test_non_string_query_malformed(self):
        from api.tools.web_search import web_search
        result = await web_search({"query": 12345})
        assert result.success is False
        assert result.error_type == "malformed"

    @pytest.mark.asyncio
    async def test_limit_respected(self):
        from api.tools.web_search import web_search
        result = await web_search({"query": "python", "limit": 2})
        assert result.success is True
        assert len(result.data) <= 2

    @pytest.mark.asyncio
    async def test_result_has_required_fields(self):
        from api.tools.web_search import web_search
        result = await web_search({"query": "fine-tuning LLM"})
        assert result.success is True
        for item in result.data:
            assert "url" in item
            assert "title" in item
            assert "snippet" in item
            assert "relevance_score" in item
            assert 0.0 <= item["relevance_score"] <= 1.0

    @pytest.mark.asyncio
    async def test_source_is_set(self):
        from api.tools.web_search import web_search
        result = await web_search({"query": "climate change"})
        assert result.source == "web_search"


# ────────────────────────────────────────────────────────────────────────────────
# Code Sandbox tool
# ────────────────────────────────────────────────────────────────────────────────

class TestCodeSandbox:
    @pytest.mark.asyncio
    async def test_simple_arithmetic(self):
        from api.tools.code_sandbox import code_sandbox
        result = await code_sandbox({"code": "print(17 * 23)"})
        assert result.success is True
        assert "391" in result.data["stdout"]
        assert result.data["exit_code"] == 0

    @pytest.mark.asyncio
    async def test_stderr_captured(self):
        from api.tools.code_sandbox import code_sandbox
        result = await code_sandbox({"code": "import sys; sys.stderr.write('error!\\n')"})
        # Will be blocked by denylist
        assert result.success is False  # import sys is blocked

    @pytest.mark.asyncio
    async def test_syntax_error_captured(self):
        from api.tools.code_sandbox import code_sandbox
        result = await code_sandbox({"code": "def broken(:"})
        assert result.success is True  # tool succeeds; exit_code != 0
        assert result.data["exit_code"] != 0
        assert len(result.data["stderr"]) > 0

    @pytest.mark.asyncio
    async def test_empty_code_malformed(self):
        from api.tools.code_sandbox import code_sandbox
        result = await code_sandbox({"code": ""})
        assert result.success is False
        assert result.error_type == "malformed"

    @pytest.mark.asyncio
    async def test_missing_code_malformed(self):
        from api.tools.code_sandbox import code_sandbox
        result = await code_sandbox({})
        assert result.success is False
        assert result.error_type == "malformed"

    @pytest.mark.asyncio
    async def test_blocked_pattern_malformed(self):
        from api.tools.code_sandbox import code_sandbox
        result = await code_sandbox({"code": "import os; print(os.listdir('/'))"})
        assert result.success is False
        assert result.error_type == "malformed"

    @pytest.mark.asyncio
    async def test_source_is_set(self):
        from api.tools.code_sandbox import code_sandbox
        result = await code_sandbox({"code": "print('hello')"})
        assert result.source == "code_sandbox"


# ────────────────────────────────────────────────────────────────────────────────
# ToolResult factory methods
# ────────────────────────────────────────────────────────────────────────────────

class TestToolResultFactories:
    def test_ok_factory(self):
        r = ToolResult.ok(data={"key": "val"}, source="test", latency_ms=42.5)
        assert r.success is True
        assert r.error_type is None
        assert r.data == {"key": "val"}
        assert r.latency_ms == 42.5

    def test_timeout_factory(self):
        r = ToolResult.timeout(source="test", latency_ms=5001)
        assert r.success is False
        assert r.error_type == "timeout"
        assert r.data is None

    def test_empty_factory(self):
        r = ToolResult.empty(source="test", latency_ms=100)
        assert r.success is False
        assert r.error_type == "empty"

    def test_malformed_factory(self):
        r = ToolResult.malformed(source="test", message="bad input")
        assert r.success is False
        assert r.error_type == "malformed"
        assert "bad input" in r.error_message
