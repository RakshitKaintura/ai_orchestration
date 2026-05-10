"""eval/scorers/budget_compliance.py — Budget compliance scorer."""

from __future__ import annotations

import logging

from eval.cases import EvalCase
from eval.scorers.base import ScoreResult
from api.models.context import SharedContext

logger = logging.getLogger(__name__)


async def score_budget_compliance(case: EvalCase, ctx: SharedContext, **_) -> ScoreResult:
    """
    Budget compliance scorer.

    1.0 if no budget violations occurred.
    Reduced by 0.15 per violation, minimum 0.0.
    """
    violations = ctx.budget_violations
    if not violations:
        return (1.0, "No budget violations. All agents stayed within token budget.")

    score = max(0.0, 1.0 - 0.15 * len(violations))
    justification = (
        f"{len(violations)} budget violation(s) recorded. "
        f"Score: {score:.2f}. "
        f"First violation: {violations[0][:100]}"
    )
    return (score, justification)
