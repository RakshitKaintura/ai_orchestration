"""
api/agents/synthesis/schemas.py

Instructor-structured output models for the Synthesis Agent.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ProvenanceItem(BaseModel):
    sentence: str = Field(description="Exact sentence from the final answer")
    source_agent: str = Field(description="Which agent produced the underlying information")
    source_chunk_id: str = Field(
        default="",
        description="Chunk ID if the sentence is grounded in a retrieved chunk",
    )
    source_chunk_excerpt: str = Field(
        default="",
        description="Brief excerpt from the source chunk (≤150 chars)",
    )


class ContradictionResolution(BaseModel):
    description: str = Field(description="What the contradiction was")
    resolution: str = Field(description="How it was resolved in the final answer")
    agents_involved: list[str] = Field(description="Which agents produced conflicting claims")


class SynthesisResult(BaseModel):
    final_answer: str = Field(
        description="The complete, contradiction-free final answer to present to the user"
    )
    provenance_map: list[ProvenanceItem] = Field(
        description="Per-sentence provenance. Every sentence in final_answer must appear here.",
        min_length=1,
    )
    contradictions_resolved: list[ContradictionResolution] = Field(
        default_factory=list,
        description="Each contradiction that was detected and resolved",
    )
    unresolvable_issues: list[str] = Field(
        default_factory=list,
        description="Issues that could not be resolved (stored internally, NOT in final_answer)",
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Overall confidence in the final synthesised answer",
    )
