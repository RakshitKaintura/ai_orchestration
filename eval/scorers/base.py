"""
eval/scorers/base.py

Shared types, scorer registry, and the score_all() aggregator.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from eval.cases import EvalCase
from api.models.context import SharedContext

logger = logging.getLogger(__name__)

ScoreResult = tuple[float, str]  # (score, justification)

SCORER_NAMES = [
    "correctness",
    "citations",
    "contradictions",
    "tool_efficiency",
    "budget_compliance",
    "critique_agreement",
]

SCORER_WEIGHTS = {
    "correctness": 0.35,
    "citations": 0.15,
    "contradictions": 0.15,
    "tool_efficiency": 0.10,
    "budget_compliance": 0.10,
    "critique_agreement": 0.15,
}


async def score_all(case: EvalCase, ctx: SharedContext, llm_client: Any) -> dict[str, dict]:
    """
    Run all 6 scorers and return per-dimension results + weighted aggregate.

    Returns:
        {
            "correctness": {"score": float, "justification": str},
            "citations": {...},
            ...
            "weighted_total": float,
        }
    """
    from eval.scorers.correctness import score_correctness
    from eval.scorers.citations import score_citations
    from eval.scorers.contradictions import score_contradictions
    from eval.scorers.tool_efficiency import score_tool_efficiency
    from eval.scorers.budget_compliance import score_budget_compliance
    from eval.scorers.critique_agreement import score_critique_agreement

    results: dict[str, Any] = {}

    scores_raw = await asyncio.gather(
        score_correctness(case, ctx, llm_client),
        score_citations(case, ctx),
        score_contradictions(case, ctx),
        score_tool_efficiency(case, ctx),
        score_budget_compliance(case, ctx),
        score_critique_agreement(case, ctx),
        return_exceptions=True,
    )

    for name, raw in zip(SCORER_NAMES, scores_raw):
        if isinstance(raw, Exception):
            logger.error(f"scorer_{name}_failed", extra={"error": str(raw)})
            results[name] = {"score": 0.5, "justification": f"Scorer error: {raw}"}
        else:
            score_val, justification = raw  # type: ignore[misc]
            results[name] = {"score": round(float(score_val), 4), "justification": justification}

    weighted_total = sum(
        results[name]["score"] * SCORER_WEIGHTS[name]
        for name in SCORER_NAMES
    )
    results["weighted_total"] = round(weighted_total, 4)

    return results
