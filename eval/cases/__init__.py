"""
eval/cases/__init__.py

Public exports for eval test cases.
"""

from eval.cases.schemas import EvalCase
from eval.cases.baseline import BASELINE_CASES
from eval.cases.ambiguous import AMBIGUOUS_CASES
from eval.cases.adversarial import ADVERSARIAL_CASES

ALL_CASES: list[EvalCase] = BASELINE_CASES + AMBIGUOUS_CASES + ADVERSARIAL_CASES
CASES_BY_ID: dict[str, EvalCase] = {c.case_id: c for c in ALL_CASES}

def get_case(case_id: str) -> EvalCase:
    """Get a single eval case by ID. Raises KeyError if not found."""
    if case_id not in CASES_BY_ID:
        raise KeyError(f"Eval case '{case_id}' not found. Available: {list(CASES_BY_ID.keys())}")
    return CASES_BY_ID[case_id]

def get_cases_by_category(category: str) -> list[EvalCase]:
    """Get all cases for a category: baseline | ambiguous | adversarial."""
    return [c for c in ALL_CASES if c.category == category]

__all__ = [
    "EvalCase",
    "ALL_CASES",
    "CASES_BY_ID",
    "get_case",
    "get_cases_by_category",
    "BASELINE_CASES",
    "AMBIGUOUS_CASES",
    "ADVERSARIAL_CASES",
]
