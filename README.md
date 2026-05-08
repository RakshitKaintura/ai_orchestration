# Mega AI — Real-Time Multi-Agent LLM Orchestration System

> Production-grade multi-agent pipeline with dynamic routing, multi-hop RAG,
> critique/synthesis agents, self-improving eval loop, and real-time SSE streaming.

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green.svg)](https://fastapi.tiangolo.com)
[![PostgreSQL 15](https://img.shields.io/badge/PostgreSQL-15-blue.svg)](https://postgresql.org)
[![Docker](https://img.shields.io/badge/Docker-Compose-blue.svg)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Quick Start (< 5 minutes)

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/mega-ai.git
cd mega-ai

# 2. Configure (add your API keys)
cp .env.example .env
# Edit .env: set ANTHROPIC_API_KEY and OPENAI_API_KEY

# 3. Start all services
docker compose up -d

# 4. Verify
curl http://localhost:8000/health
# → {"status": "ok", "service": "mega-ai-api"}

# 5. View API docs
open http://localhost:8000/docs

# 6. View log UI
open http://localhost:8080
```

**No other steps required.** All services start via Docker Compose. No manual DB migrations — the schema is auto-applied on first run.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLIENT (SSE Consumer)                     │
└───────────────────────────────┬─────────────────────────────────┘
                                │ POST /query (SSE stream)
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI API Server (:8000)                    │
│  • 5 endpoints (query, trace, eval, rewrite review, re-eval)    │
│  • SSE streaming via sse-starlette                              │
│  • Structured logging (structlog → JSONL)                       │
└───────────────────────────────┬─────────────────────────────────┘
                                │ enqueue
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Celery Worker (2 queues)                       │
│  jobs queue:  run_pipeline_task                                  │
│  eval queue:  run_eval_task, run_reeval_task                     │
└────────┬────────────────────────────────┬────────────────────────┘
         │                                │
         ▼                                ▼
┌─────────────────┐              ┌─────────────────┐
│   Orchestrator  │              │  Eval Harness   │
│  (dynamic LLM   │              │  (15 test cases,│
│   routing plan) │              │   6 scorers)    │
└────────┬────────┘              └────────┬────────┘
         │ mediates all handoffs          │
         │ (agents never talk directly)   │
         ▼                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                       SharedContext (Pydantic)                   │
│  job_id · query · subtasks · agent_outputs · tool_call_log      │
│  final_answer · provenance_map · budget_violations              │
└─────────────────────────────────────────────────────────────────┘
         │
         ├─── Decomposition Agent (budget: 4000 tok)
         │      └─ breaks query into typed sub-tasks with DAG deps
         │
         ├─── RAG Agent (budget: 6000 tok)
         │      └─ multi-hop retrieval (ChromaDB) + chunk citations
         │
         ├─── Critique Agent (budget: 4000 tok)
         │      └─ span-level confidence scoring of every output
         │
         ├─── Synthesis Agent (budget: 5000 tok)
         │      └─ merges outputs, resolves contradictions, provenance map
         │
         ├─── Compression Agent (budget: 2000 tok)
         │      └─ triggered by BudgetManager on near-limit detection
         │
         └─── Meta Agent (post-eval)
                └─ proposes prompt rewrites for worst-scoring dimension

┌─────────────────────────────────────────────────────────────────┐
│                         Infrastructure                           │
│  PostgreSQL 15  │  Redis 7  │  ChromaDB  │  Log UI (:8080)     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Services

| Service   | Port | Description |
|-----------|------|-------------|
| `api`     | 8000 | FastAPI application server |
| `worker`  | —    | Celery background worker (jobs + eval queues) |
| `db`      | 5432 | PostgreSQL 15 |
| `redis`   | 6379 | Celery broker + result backend |
| `log-ui`  | 8080 | Flask log query UI |

---

## Agent Descriptions

### Orchestrator
- **Role**: The sole mediator. Decides which agents to invoke, in what order, and with what token budget. Uses a structured LLM call (Instructor) to produce a `RoutingPlan` with explicit reasoning.
- **Decision boundary**: Only invokes agents; never produces content itself.
- **Budget**: 3000 tokens

### Decomposition Agent
- **Role**: Breaks ambiguous queries into typed sub-tasks (`retrieval`, `computation`, `reasoning`, `synthesis`) with explicit dependency graphs.
- **Decision boundary**: Validates that the dependency graph is a DAG. Raises if cycles detected.
- **Budget**: 4000 tokens

### RAG Agent
- **Role**: Retrieves document chunks from ChromaDB, performs two hops of retrieval, and produces an answer with per-sentence chunk citations.
- **Decision boundary**: Requires exactly two retrieval hops. Will not answer from a single hop.
- **Budget**: 6000 tokens

### Critique Agent
- **Role**: Reviews every other agent's output and assigns span-level confidence scores. Flags specific text spans it disagrees with.
- **Decision boundary**: Flags spans, not whole outputs. Must produce at least one `ClaimScore` per reviewed output.
- **Budget**: 4000 tokens

### Synthesis Agent
- **Role**: Merges all agent outputs, resolves contradictions flagged by the critique agent, and produces the final answer with a complete provenance map.
- **Decision boundary**: Contradictions are resolved internally, not surfaced to the user.
- **Budget**: 5000 tokens

### Compression Agent
- **Role**: Triggered automatically by the BudgetManager when an agent is near its token limit. Compresses natural language prose while preserving structured data verbatim.
- **Decision boundary**: Lossless for JSON, scores, citations; lossy only for filler prose.
- **Budget**: 2000 tokens

### Meta Agent
- **Role**: Post-eval agent that reads failure cases, identifies the worst-performing prompt dimension, and proposes a rewritten prompt with a structured diff. Rewrites are never auto-applied.
- **Decision boundary**: Proposes one rewrite per eval run. Stores the proposal; requires human approval.

---

## API Reference

### 1. `POST /query` — Submit a query
```json
// Request
{"query": "What are the key differences between RAG and fine-tuning?"}

// Response: SSE stream
data: {"type": "agent_start", "agent": "decomposition", "budget": 4000}
data: {"type": "token", "agent": "decomposition", "text": "The "}
data: {"type": "tool_call_start", "agent": "rag", "tool": "web_search"}
data: {"type": "tool_call_end", "agent": "rag", "tool": "web_search", "latency_ms": 234, "accepted": true}
data: {"type": "budget_update", "agent": "rag", "remaining": 4821}
data: {"type": "agent_end", "agent": "rag", "tokens_used": 1179}
data: {"type": "done", "job_id": "uuid"}
```

### 2. `GET /trace/{job_id}` — Get execution trace
```json
{
  "job_id": "uuid",
  "trace_events": [
    {"seq": 1, "agent_id": "orchestrator", "event_type": "orchestrator_plan",
     "payload": {"agents_selected": ["decomposition", "rag", "critique", "synthesis"]},
     "latency_ms": 342, "token_count": 89}
  ]
}
```

### 3. `GET /eval/latest` — Latest eval summary
```json
{
  "run_id": "uuid",
  "created_at": "2024-01-01T00:00:00Z",
  "summary_by_category": {
    "baseline": {"mean_score": 0.92},
    "ambiguous": {"mean_score": 0.71},
    "adversarial": {"mean_score": 0.64}
  },
  "summary_by_dimension": {
    "correctness": {"mean_score": 0.88, "worst_case": "adv-003"},
    "citations": {"mean_score": 0.79}
  }
}
```

### 4. `POST /rewrites/{rewrite_id}/review` — Approve/reject rewrite
```json
// Request
{"decision": "approved"}

// Response
{"rewrite_id": "uuid", "status": "approved", "re_eval_triggered": true}
```

### 5. `POST /eval/re-run` — Trigger targeted re-eval
```json
// Request (rewrite_id optional)
{"rewrite_id": "uuid"}

// Response
{"run_id": "uuid", "cases_count": 3, "status": "queued"}
```

---

## Running the Evaluation

```bash
# Full 15-case eval
make eval

# By category
make eval-baseline
make eval-adversarial

# View results
curl http://localhost:8000/eval/latest | python -m json.tool
```

### Scoring Dimensions

| Dimension | Description |
|-----------|-------------|
| `correctness` | Answer vs expected answer (LLM-as-judge, 0–1) |
| `citations` | Provenance map citations reference real chunks |
| `contradictions` | Contradictions resolved in synthesis, not surfaced |
| `tool_efficiency` | Penalise unnecessary tool calls (-0.1 per excess call) |
| `budget_compliance` | 1.0 if no violations, reduced per violation |
| `critique_agreement` | Critique agent agrees with final answer |

---

## Self-Improving Loop

1. After each eval, the meta-agent reads the lowest-scoring dimension
2. It proposes a rewritten prompt for the responsible agent
3. The proposal is stored with status `pending` — **never auto-applied**
4. A human reviews via `POST /rewrites/{id}/review {"decision": "approved"}`
5. On approval, a targeted re-eval runs on only the previously failed cases
6. Score delta (before vs after) is stored in `prompt_rewrites.delta`
7. Every step is timestamped and queryable

---

## Known Limitations

See [LIMITATIONS.md](LIMITATIONS.md) for a full honest assessment.

Key limitations:
- **RAG**: Second retrieval hop is LLM-generated; no guaranteed optimality
- **Sandbox**: subprocess-based, not Firecracker-isolated
- **Adversarial**: 5 hand-crafted cases, not a comprehensive red-team set
- **Compression**: Generic prompt, no domain-aware verification
- **Self-improvement**: LLM-proposed rewrites are inconsistent; needs A/B infra

---

## What's Next

- Replace subprocess sandbox with Firecracker or gVisor
- Add A/B testing infrastructure for prompt rewrite evaluation
- Implement iterative chain-of-thought retrieval in the RAG agent
- Add authentication to the rewrite approval endpoint
- Build a proper eval dashboard (React + recharts)
- Add support for tool parallelism (run independent tools concurrently)
- Implement cross-run regression detection with alerts

---

## Development

```bash
# Run tests
make test

# Lint
make lint

# Format
make format

# Open psql
make shell-db
```

---

## AI Collaboration Disclosure

This project was built with significant AI assistance from **Antigravity (Google Deepmind)**.

**Human-designed** (not AI): all architectural contracts (SharedContext protocol, failure modes, budget constraints, evaluation dimensions, adversarial guards, safety rules — specifically the never-auto-apply constraint for prompt rewrites).

**AI-generated**: implementations of the specified contracts (agent code, tool code, DB queries, test scaffolding, Docker configuration).

All AI-generated code was reviewed line-by-line and tested before commit. See **[REPORT.md §10](REPORT.md#10-ai-tool-usage--attestation)** for the full file-level attestation table and workflow description.

