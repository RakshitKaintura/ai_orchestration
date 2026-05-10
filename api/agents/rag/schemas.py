"""
api/agents/rag/schemas.py

Instructor-structured output models for the RAG Agent.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChunkCitation(BaseModel):
    chunk_id: str = Field(description="ID of the chunk that supports this claim")
    claim: str = Field(description="The specific claim in the answer supported by this chunk")
    contribution: str = Field(description="How this chunk contributed to this part of the answer")


class FollowUpQuery(BaseModel):
    query: str = Field(description="The follow-up retrieval query for the second hop")
    reasoning: str = Field(description="Why this follow-up query is needed")


class RAGAnswer(BaseModel):
    answer: str = Field(description="The complete answer synthesised from retrieved chunks")
    citations: list[ChunkCitation] = Field(
        description="Per-claim citations linking answer sentences to source chunks",
        min_length=1,
    )
    follow_up_query_used: str = Field(description="The hop-2 query that was used")
    confidence: float = Field(ge=0.0, le=1.0, description="Overall confidence in the answer")
    retrieval_sufficient: bool = Field(
        description="True if retrieved chunks contained enough information to answer"
    )
    gaps: str = Field(
        default="",
        description="If retrieval_sufficient=False, what information was missing",
    )
