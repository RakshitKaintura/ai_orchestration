"""
api/agents/meta/schemas.py

Instructor-structured output models for the Meta Agent.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class PromptRewriteProposal(BaseModel):
    target_agent: str = Field(
        description=(
            "Agent ID whose prompt should be rewritten "
            "(decomposition/rag/critique/synthesis/orchestrator)"
        )
    )
    target_dimension: str = Field(
        description="The eval dimension this rewrite aims to improve"
    )
    analysis: str = Field(
        description=(
            "3-5 sentence analysis of WHY this agent/dimension is underperforming "
            "based on the failed cases and scores provided"
        )
    )
    rewritten_prompt: str = Field(
        description="The complete new system prompt for the target agent"
    )
    diff_summary: str = Field(
        description="Concise description of what changed between the current and new prompt, and why"
    )
    expected_improvement: str = Field(
        description="Specific, testable prediction of how this rewrite will improve the failing cases"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="How confident you are this rewrite will improve the target dimension",
    )
