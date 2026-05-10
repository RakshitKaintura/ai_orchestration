"""eval/cases/schemas.py — EvalCase dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class EvalCase:
    case_id: str
    category: Literal["baseline", "ambiguous", "adversarial"]
    query: str
    expected_answer: str
    expected_chunk_ids: list[str] = field(default_factory=list)
    requires_tools: list[str] = field(default_factory=list)
    forbidden_claims: list[str] = field(default_factory=list)
    notes: str = ""
