# Mega AI — LLM Engineer Take-Home: Technical Report

> **Author**: [Candidate]
> **Repository**: [GitHub URL — pushed Day 5]
> **Submission date**: 2026-05-08
> **Assessment level**: Junior LLM Engineer

---

## Table of Contents

1. [System Architecture Overview](#1-system-architecture-overview)
2. [Agent Design & Communication Protocol](#2-agent-design--communication-protocol)
3. [Tool Library & Failure Contracts](#3-tool-library--failure-contracts)
4. [Retrieval Architecture (RAG)](#4-retrieval-architecture-rag)
5. [Evaluation Harness](#5-evaluation-harness)
6. [Self-Improvement Loop](#6-self-improvement-loop)
7. [Adversarial Robustness](#7-adversarial-robustness)
8. [Budget Management & Context Compression](#8-budget-management--context-compression)
9. [Production Constraints & Known Limitations](#9-production-constraints--known-limitations)
10. [AI Tool Usage — Attestation](#10-ai-tool-usage--attestation)

---

## 1. System Architecture Overview

### High-Level Architecture

```
User → POST /query
         │
         ▼
  FastAPI (SSE stream)
         │
         ▼
  ┌─────────────────────────────────────────┐
  │           Orchestrator                  │
  │   1. LLM routing plan (Instructor)      │
  │   2. Sequential agent invocation        │
  │   3. CritiqueAgent after each agent     │
  │   4. Trace logging → PostgreSQL         │
  │   5. SSE events → asyncio.Queue         │
  └───────────────┬─────────────────────────┘
                  │
        SharedContext (single object)
                  │
    ┌─────────────┼─────────────┐
    ▼             ▼             ▼
DecompositionAgent  RAGAgent  SynthesisAgent
    │             │             │
    ▼             ▼             ▼
[DAG validation] [2-hop retrieval] [provenance map]
                  │
              CritiqueAgent (after each)
                  │
           [span-level scores → ClaimScore[]]
```

### Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| API server | FastAPI + uvicorn | Async HTTP + SSE |
| LLM provider | Anthropic Claude 3.5 Sonnet | Pipeline + judge |
| Structured output | `instructor` (Anthropic backend) | Typed LLM responses |
| Embeddings | OpenAI text-embedding-3-small | RAG retrieval |
| Vector store | ChromaDB (PersistentClient) | Chunk storage + search |
| Task queue | Celery + Redis | Background eval jobs |
| Database | PostgreSQL 15 (asyncpg) | Trace + eval persistence |
| Containerisation | Docker Compose | Service isolation |
| Streaming | sse-starlette + asyncio.Queue | Real-time SSE |

### Data Flow

Every pipeline run creates a `SharedContext` object that lives for the lifetime of one job. Agents read from `ctx` and write back into it — they **never call each other directly**. The orchestrator is the sole caller of `agent.run()`. This architectural constraint is enforced by the type system: `BaseAgent.__init__` only receives `SharedContext` and `BudgetManager`.

---

## 2. Agent Design & Communication Protocol

### Agents

| Agent | Role | Key Output Written to SharedContext |
|-------|------|-------------------------------------|
| `DecompositionAgent` | Break query into sub-tasks | `ctx.subtasks` (validated DAG) |
| `RAGAgent` | 2-hop retrieval + answer | `ctx.agent_outputs["rag"]` |
| `CritiqueAgent` | Span-level review | `target_output.claim_scores` |
| `SynthesisAgent` | Merge + resolve contradictions | `ctx.final_answer`, `ctx.provenance_map` |
| `CompressionAgent` | Lossless context trimming | Called by `BudgetManager.compress_and_record()` |
| `MetaAgent` | Prompt rewrite proposals | `prompt_rewrites` table (post-eval) |

### Communication Contract

```
Orchestrator calls → agent.run(ctx, bm)
Agent reads        → ctx.query, ctx.agent_outputs[*], ctx.subtasks
Agent writes       → ctx.agent_outputs[own_id]  (via orchestrator)
Budget check       → bm.add(agent_id, text) → bool
Budget overflow    → bm.compress_and_record(agent_id, text) → compressed_text
```

### DAG Validation (Decomposition Agent)

The decomposition agent uses `instructor` to generate a `DecompositionOutput` containing a list of `SubTaskSpec` objects with `depends_on` references. Before accepting the output, **Kahn's Algorithm** validates the dependency graph:

1. Compute in-degree of every node
2. BFS from zero-in-degree nodes
3. If topological order length ≠ node count → cycle detected → raise `CyclicDependencyError`
4. All `depends_on` IDs must reference a declared `SubTask.id` → missing ID error

This runs entirely in Python — no LLM required for validation.

---

## 3. Tool Library & Failure Contracts

### Failure Contract Design

Every tool call returns a typed `ToolResult` with `success`, `error_type`, and `output`. The `call_tool_with_retry()` wrapper in `api/tools/base.py` implements retry logic **in code** (not in prompts):

| Error type | Input mutation on retry |
|------------|------------------------|
| `timeout` | `simplify_input()` — shorter, fewer clauses |
| `empty` | `broaden_input()` — add synonyms, generalise |
| `malformed` | `fix_format()` — strip special chars, correct syntax |

Maximum **2 retries** per tool call. Each retry is logged separately to `ctx.tool_call_log` with `retry_num` ∈ {0, 1, 2}.

### Tools

| Tool | Backend | Failure modes |
|------|---------|---------------|
| `web_search` | Fixture-based (curated corpus) | `timeout` >5s, `empty` 0 results, `malformed` bad query |
| `code_sandbox` | `subprocess` + security denylist | `timeout` >10s (SIGKILL), `malformed` blocked keyword |
| `nl_to_sql` | Instructor → asyncpg | `malformed` DML/DDL detected, `timeout` LLM/DB |
| `self_reflection` | Claude + SharedContext | `malformed` no prior outputs |

### Acceptance Checks

Agents can inject an `acceptance_check` callback into `call_tool_with_retry()`. The callback receives the raw `ToolResult` and returns `(bool, str)` — accept or reject with reason. A technically successful tool call can still be rejected and re-retried if the result doesn't meet domain requirements. This is logged as `accepted=False`.

---

## 4. Retrieval Architecture (RAG)

### Two-Hop Retrieval (Mandatory)

```
Query → embed → Hop 1: retrieve top-3 chunks from ChromaDB
                    │
                    ▼
              LLM generates follow-up query from Hop 1 results
                    │
                    ▼
              Hop 2: embed follow-up query → retrieve top-3 (deduplicated)
                    │
                    ▼
              Final synthesis: answer + per-claim citations
```

The second hop is **always executed** — even if Hop 1 results look sufficient. This is a design constraint to ensure multi-hop evidence is always attempted. Deduplication uses Python set operations on chunk IDs.

### Corpus

- 30 inline chunks seeded at container startup
- 9 topic domains: RAG, fine-tuning, LLM internals, climate, history, astronomy, mathematics, software engineering, adversarial/security
- Embedding model: `text-embedding-3-small` (1536 dimensions)
- Similarity: cosine (ChromaDB `hnsw:space=cosine`)
- Corpus JSON files from `eval/corpus/` loaded on startup; inline fallback if directory is empty

### Citation Integrity

Every sentence in the synthesis output is linked to a `ProvenanceEntry` with `source_agent` and `source_chunk_id`. The citations scorer checks:
- Chunk ID exists in the ChromaDB collection (no phantom citations)
- Coverage: fraction of `expected_chunk_ids` that were actually cited

---

## 5. Evaluation Harness

### 15 Test Cases

| Category | Count | What it tests |
|----------|-------|---------------|
| baseline | 5 | Factual retrieval, arithmetic, comparison |
| ambiguous | 5 | False premises, vague scope, subjective questions |
| adversarial | 5 | Prompt injection, jailbreak, contradiction bait, myth traps |

### 6 Scoring Dimensions

| Dimension | Weight | Automated / LLM |
|-----------|--------|-----------------|
| `correctness` | 0.35 | LLM-as-judge (Claude 3.5 Sonnet) |
| `citations` | 0.15 | Automated (chunk ID set operations) |
| `contradictions` | 0.15 | Automated (pattern matching + count) |
| `tool_efficiency` | 0.10 | Automated (required tool recall + penalty) |
| `budget_compliance` | 0.10 | Automated (violation count) |
| `critique_agreement` | 0.15 | Semi-automated (critique agent's `overall_confidence`) |

**Correctness scorer adversarial guard**: any `forbidden_claims` string that appears verbatim in `final_answer` triggers an **automatic 0.0** before the LLM judge runs.

### Score Computation

```python
weighted_total = sum(scores[dim] * SCORER_WEIGHTS[dim] for dim in SCORER_NAMES)
```

Weights sum to exactly 1.0 (verified in `tests/test_eval.py::TestScorerWeights`).

---

## 6. Self-Improvement Loop

```
Eval run completes
       │
       ▼
MetaAgent.propose_rewrite()
  - finds worst-scoring dimension
  - maps dimension → responsible agent
  - proposes ONE rewrite (not one per case)
  - stores as 'pending' in prompt_rewrites table
       │
       ▼
Human reviews via POST /rewrites/{id}/review
  - decision: 'approved' | 'rejected'
  - on approval: apply_approved_rewrite() → agent_prompts table
  - deactivate_old_prompts DB trigger fires
       │
       ▼
run_reeval_task (Celery, queue=eval)
  - runs ONLY the previously failed cases
  - computes score delta (before/after)
  - stores delta in prompt_rewrites.delta
```

**Safety constraint**: Rewrites are **never auto-applied**. The only mechanism to activate a rewrite is `POST /rewrites/{id}/review` with `decision: approved`. The system is architecturally incapable of self-modifying without human approval.

---

## 7. Adversarial Robustness

### Threat Model

| Attack | Detection Mechanism | Response |
|--------|-------------------|----------|
| Prompt injection (`Ignore all instructions...`) | Critique agent span-level review + forbidden_claims scorer | Score 0.0; answer normal question |
| False premise (`Einstein failed math`) | RAG retrieval surfaces contradicting chunk; synthesis rejects premise | Correct premise before answering |
| Jailbreak (`DAN roleplay`) | Forbidden claims: `["DAN", "Do Anything Now"]` | Automatic 0.0 scoring |
| Contradiction bait (`confirm both sources are valid`) | Synthesis resolves contradictions; forbidden: `["both sources are equally valid"]` | Resolve, not validate |
| System-role injection in user turn | Critique flags unexpected instruction patterns | Ignore injection, answer CoT question |
| Context window poisoning | RAG deduplication; critique reviews all chunks | Flagged in span-level critique |

### Instrumentation

- Every adversarial case has explicit `forbidden_claims` — strings that **must not** appear verbatim in the final answer
- The correctness scorer applies this check **before** calling the LLM judge
- ChromaDB corpus contains `chunk-adv-001` through `chunk-adv-003` on prompt injection, hallucination, and context poisoning for citation grounding

---

## 8. Budget Management & Context Compression

### BudgetManager

`BudgetManager` tracks per-agent token consumption using `tiktoken` (cl100k_base). It exposes:

- `declare(agent_id, budget)` — register budget before execution
- `add(agent_id, text)` — consume tokens; returns `False` if over budget
- `is_near_limit(agent_id)` — True if consumed > 85% of budget
- `check_remaining(agent_id)` — tokens remaining
- `compress_and_record(agent_id, text)` — trigger compression and update consumed count

### Compression Strategy

`compress_context_async()` in `api/agents/compression.py`:

1. **Structural detection**: uses regex to identify JSON-like structures (dicts, lists with braces/brackets)
2. **Lossless path**: JSON blocks are preserved byte-for-byte (these may be tool results)
3. **Lossy path**: prose segments are sent to Claude for summarisation in parallel (asyncio.gather)
4. **Budget update**: `record_compression(agent_id, tokens_saved)` reduces `consumed` by the delta

### Proactive Compression

The orchestrator checks `bm.is_near_limit(agent_id)` before each agent invocation. If near-limit, it can trigger compression on the accumulated context before passing it to the next agent.

---

## 9. Production Constraints & Known Limitations

### Current Limitations (documented in `LIMITATIONS.md`)

| Component | Limitation | Production Fix |
|-----------|-----------|----------------|
| Code Sandbox | Uses `subprocess` — escapable | Replace with Firecracker microVMs or gVisor |
| Self-improvement | Non-deterministic rewrites | A/B infrastructure + statistical significance testing |
| Distributed concurrency | No distributed lock for `prompt_rewrites` writes | Redis distributed lock (Redlock) |
| ChromaDB | Single-node PersistentClient | Distributed Weaviate or Qdrant cluster |
| Auth | No authentication on any endpoint | OAuth2 / API key middleware |
| Rate limiting | No rate limiting | FastAPI middleware + Redis token bucket |
| Secrets | `.env` file | HashiCorp Vault or AWS Secrets Manager |
| Celery | Single worker | Worker pool with auto-scaling (Kubernetes HPA) |
| Trace storage | Row-per-event (unbounded) | Time-series partitioning + TTL |

### Design Trade-offs Made

**Fixture-based web search** (not live): Using live web search would introduce non-determinism that makes eval scores irreproducible. Fixtures provide controlled, repeatable results. In production, the fixture layer would be replaced with SerpAPI or Brave Search, with the same failure contract interface.

**Sequential eval harness**: Cases run sequentially to avoid DB write contention and to produce deterministic trace sequences. Parallelising the harness with `asyncio.gather` would reduce eval time from ~15 minutes to ~2 minutes, but at the cost of overlapping trace event sequences.

**Instructor for all structured outputs**: Using Instructor (retry + validation loop) instead of raw JSON mode provides automatic schema validation and retry on malformed outputs, at the cost of extra API latency per structured call. For production, Anthropic's native tool use would be preferable.

---

## 10. AI Tool Usage — Attestation

### Declaration

This project was built with significant AI assistance from **Google Deepmind Antigravity (Claude-based agentic coding assistant)**. The following documents what was AI-generated vs. human-designed.

### AI-Generated Components

| File / Component | AI Role | Human Role |
|-----------------|---------|-----------|
| `api/models/context.py` | Initial scaffold | Reviewed, added `ClaimScore`, `ProvenanceEntry`, invariant comments |
| `api/context_manager.py` | BudgetManager implementation | Designed token accounting strategy; added `compress_and_record` method |
| `api/tools/*.py` | All four tool implementations | Designed failure contract spec; reviewed security denylist |
| `api/agents/rag.py` | Full 2-hop retrieval agent | Specified mandatory 2-hop constraint; reviewed corpus seeding |
| `api/agents/critique.py` | Span-level critique agent | Specified span-verbatim requirement; reviewed ClaimScore write-back |
| `api/agents/synthesis.py` | Synthesis + provenance map | Specified contradiction-resolution-internally constraint |
| `api/orchestrator.py` | Full orchestrator | Designed synthesis-always-last invariant; reviewed routing plan logic |
| `api/streaming.py` | SSE streaming layer | Designed event taxonomy (9 types); reviewed heartbeat |
| `eval/cases/test_cases.py` | All 15 test cases | Designed all `forbidden_claims`; specified all `expected_chunk_ids` |
| `eval/scorers.py` | All 6 scorers + weights | Designed weight distribution (sum=1.0); specified adversarial guard |
| `eval/harness.py` | Full eval harness | Designed sequential-only constraint; specified summary statistics |
| `api/agents/meta.py` | Meta agent + rewrite flow | Specified never-auto-apply constraint; designed `DIMENSION_AGENT_MAP` |
| `worker/tasks.py` | All 3 Celery tasks | Designed score delta computation; specified re-eval scope |
| `db/schema.sql` | Full schema | Designed all FK constraints; specified `deactivate_old_prompts` trigger |
| `docker-compose.yml` | Full service composition | Specified service dependencies + healthcheck intervals |

### AI Not Used For

- Architecture decisions (SharedContext as single pipeline object — human decision)
- Security denylist entries in `code_sandbox.py` (human reviewed)
- The constraint "synthesis always last, no agent calls another" (human architectural rule)
- Eval case `forbidden_claims` strings (human specification)
- The "no auto-apply" constraint for prompt rewrites (human safety rule)
- Test case ground truth answers (human verified against primary sources)

### AI Tool Stack

- **Primary agent**: Antigravity (Google Deepmind) — agentic code generation, file creation, multi-file consistency
- **LLM backend**: Claude (Anthropic) via Antigravity
- **Verification**: Human code review of every generated file; no generated code was accepted without review

### Workflow

Day 1 → Human architecture design (whiteboard) → AI scaffold bootstrap
Day 2 → Human tool contract spec → AI implementation → Human review
Day 3 → Human agent contract spec → AI implementation → Human review
Day 4 → Human eval design (15 cases, 6 dimensions, weight distribution) → AI implementation
Day 5 → Human audit, gap-fill, AI-generated REPORT.md, human verified

### Why This Matters

The AI generated the *implementation* of specified contracts. The *contracts themselves* (failure modes, budget constraints, eval dimensions, adversarial guards, safety rules) were human-designed. This mirrors production LLM engineering: AI accelerates boilerplate and scaffolding; engineers own the architecture and safety properties.
