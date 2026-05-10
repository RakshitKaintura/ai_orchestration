"""eval/scorers/critique_agreement.py — Critique agreement scorer."""

from __future__ import annotations

import logging

from eval.cases import EvalCase
from eval.scorers.base import ScoreResult
from api.models.context import SharedContext

logger = logging.getLogger(__name__)


async def score_critique_agreement(case: EvalCase, ctx: SharedContext, **_) -> ScoreResult:
    """
    Critique agreement scorer.

    Uses the critique agent's overall_confidence for the synthesis output.
    Falls back to RAG critique, then decomposition critique.
    Returns 0.5 if no critique output is found.
    """
    critique_output = (
        ctx.agent_outputs.get("critique_synthesis")
        or ctx.agent_outputs.get("critique_rag")
        or ctx.agent_outputs.get("critique_decomposition")
    )

    if critique_output and critique_output.structured_output:
        conf = critique_output.structured_output.get("overall_confidence", 0.5)
        has_critical = critique_output.structured_output.get("has_critical_errors", False)

        if has_critical:
            score = min(conf, 0.5)
            justification = (
                f"Critique flagged CRITICAL ERRORS. "
                f"Overall confidence: {conf:.2f}. Capped score: {score:.2f}."
            )
        else:
            score = conf
            justification = (
                f"Critique overall confidence: {conf:.2f}. "
                f"No critical errors flagged."
            )
        return (score, justification)

    return (0.5, "No critique output found. Cannot assess critique agreement. Defaulting to 0.5.")
