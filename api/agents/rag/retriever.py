"""
api/agents/rag/retriever.py

ChromaDB retrieval layer for the RAG Agent.

Responsibilities:
  - Lazy singleton ChromaDB client + collection
  - Corpus seeding from eval/corpus/*.json files (with inline fallback)
  - Embedding via OpenAI text-embedding-3-small
  - Two-hop retrieval with deduplication

This module is imported by the RAGAgent and by the citations scorer.
It is kept separate so ChromaDB access can be mocked in tests without
instantiating the full agent.
"""

from __future__ import annotations

import json
import logging
import pathlib

import google.generativeai as genai

from api.config import get_settings

logger = logging.getLogger(__name__)

# ─── ChromaDB lazy singleton ─────────────────────────────────────────────────

_chroma_client = None
_chroma_collection = None


def _get_chroma_collection():
    global _chroma_client, _chroma_collection
    if _chroma_collection is not None:
        return _chroma_collection

    try:
        import chromadb
        settings = get_settings()
        _chroma_client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        _chroma_collection = _chroma_client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        if _chroma_collection.count() == 0:
            _seed_corpus(_chroma_collection)
        return _chroma_collection
    except Exception as e:
        logger.error("chroma_init_failed", extra={"error": str(e)})
        return None


# ─── Corpus seeding ───────────────────────────────────────────────────────────

def _seed_corpus(collection) -> None:
    """Seed ChromaDB with evaluation corpus. Falls back to inline corpus."""
    corpus_dir = pathlib.Path("/app/eval/corpus")
    if not corpus_dir.exists():
        corpus_dir = pathlib.Path("eval/corpus")
    if not corpus_dir.exists():
        logger.warning("corpus_dir_not_found — using inline fallback corpus")
        _seed_inline_corpus(collection)
        return

    docs, ids, metas = [], [], []
    for f in sorted(corpus_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text())
            for chunk in data.get("chunks", []):
                docs.append(chunk["text"])
                ids.append(chunk["id"])
                metas.append({
                    "source": chunk.get("source", f.stem),
                    "title": chunk.get("title", ""),
                })
        except Exception as e:
            logger.warning("corpus_file_load_failed", extra={"file": str(f), "error": str(e)})

    if docs:
        collection.add(documents=docs, ids=ids, metadatas=metas)
        logger.info("corpus_seeded", extra={"chunks": len(docs)})
    else:
        _seed_inline_corpus(collection)


def _seed_inline_corpus(collection) -> None:
    """Inline fallback: 30 curated chunks covering all eval case topics."""
    chunks = [
        # RAG
        ("chunk-rag-001", "RAG (Retrieval-Augmented Generation) combines a neural retriever with a sequence-to-sequence generator. The retriever fetches relevant documents from a knowledge base, and the generator conditions on both the query and retrieved documents to produce an answer.", "rag_paper"),
        ("chunk-rag-002", "Unlike purely parametric models that store knowledge in weights, RAG can access non-parametric memory at inference time. This allows updating the knowledge base without retraining the model.", "rag_paper"),
        ("chunk-rag-003", "Multi-hop RAG involves iterative retrieval where intermediate answers guide subsequent retrieval steps. This is essential for complex questions requiring synthesis across multiple documents.", "rag_multihop"),
        ("chunk-rag-004", "RAG models struggle with retrieval recall when the relevant document is not in the top-k results. Dense retrieval methods like DPR improve recall over sparse BM25 but require careful index management.", "rag_challenges"),
        # Fine-tuning
        ("chunk-ft-001", "Fine-tuning adapts a pre-trained model's weights to a specific domain by training on labelled examples. Unlike RAG, fine-tuned knowledge is encoded in the model weights and cannot be updated without retraining.", "finetuning_guide"),
        ("chunk-ft-002", "LoRA (Low-Rank Adaptation) reduces fine-tuning cost by injecting trainable rank-decomposition matrices into transformer layers while keeping base weights frozen. Typical LoRA rank values range from 4 to 64.", "lora_paper"),
        ("chunk-ft-003", "Full fine-tuning on large models requires significant GPU memory (often 4-8x model size). Parameter-efficient methods like LoRA, QLoRA, and IA3 reduce this requirement by 10-100x.", "peft_survey"),
        # LLM internals
        ("chunk-llm-001", "Transformer attention computes Q, K, V projections and scales dot products by 1/sqrt(d_k) to prevent gradient vanishing in softmax. Multi-head attention allows attending to information from different subspaces.", "transformer_paper"),
        ("chunk-llm-002", "In-context learning (ICL) allows LLMs to perform new tasks by conditioning on examples in the prompt, without weight updates. ICL performance is sensitive to example order and formatting.", "icl_survey"),
        ("chunk-llm-003", "Chain-of-thought (CoT) prompting improves reasoning by asking models to produce intermediate steps. CoT works best in models with >100B parameters and on tasks requiring multi-step arithmetic or logical reasoning.", "cot_paper"),
        ("chunk-llm-004", "Constitutional AI (CAI) from Anthropic trains models using AI feedback. A 'critic' model evaluates outputs against a set of principles, and the resulting preference data is used for RLHF training.", "cai_paper"),
        # Climate
        ("chunk-climate-001", "According to IPCC AR6, global surface temperature increased by approximately 1.1°C above pre-industrial levels (1850-1900) between 2011-2020. The rate of warming has accelerated since the 1970s.", "ipcc_ar6"),
        ("chunk-climate-002", "The 1.5°C warming threshold is significant because many non-linear tipping points are estimated to activate between 1.5-2°C of warming, including permafrost collapse, coral bleaching, and ice sheet destabilisation.", "ipcc_ar6"),
        ("chunk-climate-003", "Carbon dioxide (CO2) is the primary greenhouse gas driving anthropogenic climate change, followed by methane (CH4) and nitrous oxide (N2O). CO2 persists in the atmosphere for centuries.", "climate_basics"),
        # History / Facts
        ("chunk-hist-001", "Albert Einstein did NOT fail mathematics as a child — this is a popular myth. He excelled at mathematics and physics, mastering differential calculus by age 15. He failed the entrance exam to ETH Zurich on his first attempt in 1895, but this was because he was two years younger than required.", "einstein_bio"),
        ("chunk-hist-002", "The Great Wall of China is NOT visible from space with the naked eye. Multiple astronauts have confirmed this, including Chinese astronaut Yang Liwei. NASA has also stated the wall is too narrow (4.5-9 metres wide) to be resolved from low Earth orbit.", "great_wall_myth"),
        ("chunk-hist-003", "The capital of France is Paris. Paris has been the capital since 987 AD and is home to approximately 2.1 million people in the city proper, and 12 million in the greater metropolitan area.", "france_facts"),
        ("chunk-hist-004", "William Shakespeare wrote Hamlet approximately between 1599 and 1601. The play is a tragedy in five acts. It was first performed at the Globe Theatre in London.", "shakespeare_facts"),
        ("chunk-hist-005", "Water boils at 100°C (212°F) at standard atmospheric pressure (101.325 kPa). At higher altitudes where pressure is lower, water boils at a lower temperature.", "chemistry_basics"),
        # Planets
        ("chunk-astro-001", "The eight planets of the Solar System in order from the Sun: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune. Pluto was reclassified as a dwarf planet in 2006 by the IAU.", "solar_system"),
        ("chunk-astro-002", "Jupiter is the largest planet in the Solar System, with a mass 2.5 times that of all other planets combined. Its Great Red Spot is a storm that has persisted for at least 350 years.", "solar_system"),
        # Math
        ("chunk-math-001", "17 multiplied by 23 equals 391. This can be computed as (17 × 20) + (17 × 3) = 340 + 51 = 391.", "arithmetic"),
        ("chunk-math-002", "Prime factorisation of 391: 391 = 17 × 23. Both 17 and 23 are prime numbers, so 391 is a semiprime (a product of exactly two prime numbers).", "arithmetic"),
        # Software engineering
        ("chunk-se-001", "Docker containers provide OS-level virtualisation. Unlike virtual machines, containers share the host kernel and are therefore more lightweight. Docker Compose orchestrates multi-container applications.", "docker_docs"),
        ("chunk-se-002", "PostgreSQL is a relational database known for ACID compliance, extensibility, and support for advanced data types including JSONB. JSONB stores JSON as binary for faster querying.", "postgres_docs"),
        ("chunk-se-003", "FastAPI is an async Python web framework built on Starlette and Pydantic. It auto-generates OpenAPI documentation and achieves throughput comparable to Node.js via asyncio.", "fastapi_docs"),
        ("chunk-se-004", "Celery is a distributed task queue for Python. Workers consume tasks from a broker (typically Redis or RabbitMQ). Task routing allows directing tasks to specific worker queues.", "celery_docs"),
        # Adversarial
        ("chunk-adv-001", "Prompt injection is an attack where malicious instructions embedded in user input attempt to override LLM system instructions. Defence strategies include instruction hierarchies, input sanitisation, and output validation.", "security_llm"),
        ("chunk-adv-002", "Hallucination in LLMs refers to generating plausible-sounding but factually incorrect content. RAG reduces hallucination by grounding outputs in retrieved documents, but does not eliminate it.", "llm_failures"),
        ("chunk-adv-003", "Context window poisoning occurs when retrieved chunks contain adversarial content designed to manipulate the LLM's output. This is a vector-specific attack on RAG systems.", "security_rag"),
    ]

    docs = [c[1] for c in chunks]
    ids = [c[0] for c in chunks]
    metas = [{"source": c[2], "title": c[0]} for c in chunks]
    collection.add(documents=docs, ids=ids, metadatas=metas)
    logger.info("inline_corpus_seeded", extra={"chunks": len(chunks)})


# ─── Embedding ────────────────────────────────────────────────────────────────

async def retrieve(
    query: str,
    n: int = 3,
    exclude_ids: set[str] | None = None,
) -> list[dict]:
    """
    Embed query, query ChromaDB, return list of chunk dicts.
    Excludes chunk IDs in exclude_ids (for hop-2 deduplication).
    """
    collection = _get_chroma_collection()
    if collection is None:
        return []

    fetch_n = n + (len(exclude_ids) if exclude_ids else 0) + 2
    results = collection.query(
        query_texts=[query],
        n_results=min(fetch_n, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for i, doc_id in enumerate(results["ids"][0]):
        if exclude_ids and doc_id in exclude_ids:
            continue
        chunks.append({
            "id": doc_id,
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i].get("source", "unknown"),
            "distance": results["distances"][0][i],
        })
        if len(chunks) >= n:
            break

    return chunks
