"""api/agents/rag/__init__.py — public exports for rag subpackage."""

from api.agents.rag.agent import RAGAgent
from api.agents.rag.retriever import _get_chroma_collection, retrieve
from api.agents.rag.schemas import ChunkCitation, FollowUpQuery, RAGAnswer

__all__ = [
    "RAGAgent",
    "_get_chroma_collection",
    "retrieve",
    "embed",
    "ChunkCitation",
    "FollowUpQuery",
    "RAGAnswer",
]
