# Known Limitations

An honest, specific assessment of where Mega AI breaks and what the gaps are.

---

## 1. Multi-Hop RAG — Retrieval Quality Not Guaranteed

**What it does**: The RAG agent generates a follow-up query based on the first retrieval results, then performs a second retrieval using that LLM-generated query.

**The problem**: There is no guarantee that the LLM-generated follow-up query is the *best possible* query for the second hop. The system uses the LLM's judgment, which can be overconfident, underspecified, or biased toward what was already retrieved.

**Production fix**: Iterative chain-of-thought retrieval with verification at each step. Deduplicate chunks across hops using semantic similarity, not just exact ID matching.

---

## 2. Code Sandbox — Not Production-Safe

**What it does**: Runs Python snippets via `subprocess` with a 10-second timeout.

**The problem**: This is not a proper sandbox. There is no:
- Filesystem isolation
- Network isolation
- Memory/CPU caps
- Seccomp profile

A malicious input can escape the sandbox, read environment variables (including API keys), or fork-bomb the worker container.

**Production fix**: Use Firecracker microVMs, gVisor, or a managed code execution service (e.g., Judge0, Modal). Docker-in-Docker is a stopgap but still not sufficient for untrusted code.

---

## 3. Adversarial Robustness — Limited Red-Teaming

**What it does**: Includes 5 hand-crafted adversarial test cases: prompt injections, false premises, and contradiction bait.

**The problem**: 5 cases is nowhere near sufficient to characterise robustness. The system is not tested against:
- Indirect prompt injection via retrieved chunks (RAG poisoning)
- Multi-turn jailbreaks
- Encoding-based bypasses (base64, Unicode tricks)
- Model-specific attack surfaces

**Production fix**: Run a proper red-team exercise with a dedicated adversarial ML team. Use automated red-teaming tools (e.g., Microsoft PyRIT, Garak). Track adversarial case coverage as a metric.

---

## 4. Context Compression — Generic, Unverified

**What it does**: Uses a generic summarisation prompt to compress context when near the token budget.

**The problem**: 
- The compression agent's claim that it preserves structured data "verbatim" is enforced only by prompt instruction, not by code verification.
- There is no post-compression check that all JSON keys, chunk IDs, and scores survived intact.
- Compression quality degrades significantly for domain-specific technical content.

**Production fix**: Parse structured data before compression, pass it through separately, and verify checksums after compression. Only run the LLM on natural language prose.

---

## 5. Self-Improving Loop — Inconsistent Rewrite Quality

**What it does**: The meta-agent reads failure cases and proposes a rewritten prompt with a structured diff.

**The problem**:
- LLM-generated prompt rewrites are inconsistent in quality. Some rewrites genuinely fix the issue; others introduce new failure modes.
- Without A/B testing infrastructure (parallel eval runs), there is no statistical confidence that an approved rewrite improves overall performance rather than just the specific failed cases.
- The loop can get stuck: if a rewrite improves the target dimension but degrades another, the system may cycle between rewrites without net improvement.

**Production fix**: Build proper A/B testing with sufficient statistical power. Track all dimensions across every rewrite, not just the target dimension. Add a guard that rejects rewrites that regress any dimension beyond a threshold.

---

## 6. Orchestrator Routing — LLM-Based, Not Formally Verified

**What it does**: The orchestrator uses an LLM call to decide which agents to invoke and in what order.

**The problem**: The routing decision is inherently non-deterministic. Two identical queries may receive different routing plans. The orchestrator has no formal proof that its routing is optimal or even correct.

**Production fix**: Define a formal routing policy (state machine or rule-based fallback) for common query types. Use the LLM only for genuinely novel query types that don't match known patterns.

---

## 7. Dependency Graph Validation — DAG Check Only

**What it does**: Validates that sub-task dependencies form a Directed Acyclic Graph (no cycles).

**The problem**: The system checks for cycles but does not check for:
- Orphaned sub-tasks (no path to the final synthesis)
- Diamond dependencies (valid but may cause race conditions)
- Sub-tasks that declare dependencies on non-existent IDs

**Production fix**: Full topological sort validation with error messages indicating the specific broken dependency.

---

## 8. Database Transactions — No Distributed Locking

**What it does**: Uses SQLAlchemy async sessions with commit/rollback.

**The problem**: With multiple Celery workers running concurrently, there are potential race conditions on:
- `agent_prompts.is_active` updates (the deactivate trigger helps but is not atomic with reads)
- `eval_runs` insertions when two workers complete simultaneously
- `prompt_rewrites.status` updates

**Production fix**: Use `SELECT FOR UPDATE SKIP LOCKED` for all status-changing operations. Add optimistic locking (version column) to the `prompt_rewrites` table.
