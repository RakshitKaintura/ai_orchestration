"""
eval/scorers/__init__.py

Public exports for the eval scorers package.
Import score_all() from here for the evaluation harness.
"""

from eval.scorers.base import SCORER_NAMES, SCORER_WEIGHTS, ScoreResult, score_all
from eval.scorers.correctness import score_correctness
from eval.scorers.citations import score_citations
from eval.scorers.contradictions import score_contradictions
from eval.scorers.tool_efficiency import score_tool_efficiency
from eval.scorers.budget_compliance import score_budget_compliance
from eval.scorers.critique_agreement import score_critique_agreement

__all__ = [
    "score_all",
    "score_correctness",
    "score_citations",
    "score_contradictions",
    "score_tool_efficiency",
    "score_budget_compliance",
    "score_critique_agreement",
    "SCORER_NAMES",
    "SCORER_WEIGHTS",
    "ScoreResult",
]
