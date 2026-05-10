"""eval/scorers/tool_efficiency.py — Tool efficiency scorer."""

from __future__ import annotations

import logging

from eval.cases import EvalCase
from eval.scorers.base import ScoreResult
from api.models.context import SharedContext

logger = logging.getLogger(__name__)


async def score_tool_efficiency(case: EvalCase, ctx: SharedContext, **_) -> ScoreResult:
    """
    Tool efficiency scorer.

    Score starts at 1.0, decreases for:
    - Missing required tools (-20% recall penalty per missing tool)
    - Extra unnecessary tool calls (-0.1 per extra unique tool)
    - Retries (-0.05 per retry)
    """
    tool_calls = ctx.tool_call_log
    required = set(case.requires_tools)

    invoked_tools = {tc.tool for tc in tool_calls if tc.retry_num == 0}

    if required:
        missing = required - invoked_tools
        recall = len(required & invoked_tools) / len(required)
        if missing:
            return (
                recall * 0.8,
                f"Required tools not invoked: {missing}. Recall: {recall:.0%}.",
            )
    else:
        recall = 1.0

    extra_tools = invoked_tools - required
    extra_penalty = 0.1 * len(extra_tools)

    retries = sum(1 for tc in tool_calls if tc.retry_num > 0)
    retry_penalty = 0.05 * retries

    score = max(0.0, 1.0 - extra_penalty - retry_penalty)
    justification = (
        f"Required: {required or 'none'} | Invoked: {invoked_tools}. "
        f"Extra tools: {extra_tools} (penalty: -{extra_penalty:.2f}). "
        f"Retries: {retries} (penalty: -{retry_penalty:.2f}). "
        f"Score: {score:.2f}."
    )
    return (score, justification)
