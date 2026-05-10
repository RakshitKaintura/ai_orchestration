"""
api/agents/critique/schemas.py

Instructor-structured output models for the Critique Agent.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SpanAssessment(BaseModel):
    span: str = Field(
        description=(
            "Exact verbatim text span from the agent's output being assessed. "
            "Must be a substring of the output text."
        )
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="0.0 = almost certainly wrong, 1.0 = almost certainly correct",
    )
    flagged: bool = Field(
        description="True if you disagree with or are uncertain about this claim"
    )
    reason: str = Field(
        default="",
        description="If flagged=True, explain specifically why you disagree or are uncertain",
    )
    source_chunk_id: str = Field(
        default="",
        description="If you verified this span against a retrieved chunk, the chunk ID",
    )


class CritiqueResult(BaseModel):
    claim_scores: list[SpanAssessment] = Field(
        description="Per-span assessments. Must cover key claims. Minimum 1 entry.",
        min_length=1,
    )
    overall_confidence: float = Field(
        ge=0.0, le=1.0,
        description="Overall confidence in the target agent's output as a whole",
    )
    summary: str = Field(
        description="One-paragraph summary of the critique findings",
    )
    has_critical_errors: bool = Field(
        description="True if any span is flagged AND confidence < 0.4 (critical disagreement)",
    )
