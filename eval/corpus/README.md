# eval/corpus/

This directory contains JSON files for the RAG corpus.
Each file has the following schema:

```json
{
  "source": "source_name",
  "chunks": [
    {
      "id": "unique-chunk-id",
      "text": "chunk content",
      "source": "source_name",
      "title": "optional title"
    }
  ]
}
```

The RAG agent seeds ChromaDB from these files on startup.
If this directory is empty, the agent falls back to the inline corpus
defined in `api/agents/rag.py` (_seed_inline_corpus).

The inline corpus contains 30 chunks covering:
- RAG and retrieval fundamentals
- Fine-tuning and LoRA
- LLM internals (transformers, CoT, CAI)
- Climate science (IPCC AR6)
- Historical facts (Einstein, Great Wall, etc.)
- Solar system and astronomy
- Mathematics (arithmetic, primes)
- Software engineering (Docker, PostgreSQL, FastAPI, Celery)
- Adversarial/security content (prompt injection, hallucination)
