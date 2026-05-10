"""eval/cases/ambiguous.py — Ambiguous evaluation test cases."""

from eval.cases.schemas import EvalCase

AMBIGUOUS_CASES: list[EvalCase] = [
    EvalCase(
        case_id="amb-001",
        category="ambiguous",
        query="Which is better: RAG or fine-tuning?",
        expected_answer=(
            "Neither is universally better — it depends on the use case. RAG is preferable when "
            "the knowledge base changes frequently, when you need source attribution, or when you "
            "cannot afford to retrain. Fine-tuning is preferable for consistent style, format "
            "adaptation, and task-specific behaviour. Many production systems use both together."
        ),
        expected_chunk_ids=["chunk-rag-001", "chunk-ft-001"],
        notes="Ambiguous 'which is better' question — correct answer acknowledges tradeoffs",
    ),
    EvalCase(
        case_id="amb-002",
        category="ambiguous",
        query="Tell me everything about Einstein.",
        expected_answer=(
            "Albert Einstein (1879-1955) was a theoretical physicist who developed the theory of "
            "relativity. In his 'miraculous year' (1905) he published four landmark papers. "
            "Contrary to popular myth, he did not fail mathematics — he mastered calculus by age 15. "
            "He failed the ETH Zurich entrance exam on his first attempt at age 16 because he was "
            "two years younger than required, not due to academic failure."
        ),
        expected_chunk_ids=["chunk-hist-001"],
        forbidden_claims=["failed mathematics", "failed math class"],
        notes="Broad scope query — tests summarisation and myth-busting",
    ),
    EvalCase(
        case_id="amb-003",
        category="ambiguous",
        query="How does attention work?",
        expected_answer=(
            "In transformer models, attention computes query (Q), key (K), and value (V) projections "
            "and computes scaled dot-product attention: softmax(QK^T / sqrt(d_k)) * V. "
            "The scaling by 1/sqrt(d_k) prevents vanishing gradients in softmax. "
            "Multi-head attention allows the model to attend to information from different representation subspaces."
        ),
        expected_chunk_ids=["chunk-llm-001"],
        notes="Ambiguous: 'attention' could mean many things — agent should assume transformers",
    ),
    EvalCase(
        case_id="amb-004",
        category="ambiguous",
        query="What are the best products under $100?",
        expected_answer=(
            "Based on the product database, the best-rated products under $100 include items from "
            "Electronics, Books, and Accessories categories. The specific top products by rating "
            "can be retrieved from the database."
        ),
        requires_tools=["nl_to_sql"],
        notes="Requires NL→SQL tool to query the products table; vague 'best' criterion",
    ),
    EvalCase(
        case_id="amb-005",
        category="ambiguous",
        query="Is Python good?",
        expected_answer=(
            "Python is widely considered excellent for data science, machine learning, and scripting "
            "due to its readable syntax, rich ecosystem (NumPy, Pandas, PyTorch), and async support "
            "via asyncio. It is less suited for CPU-intensive tasks where compiled languages like "
            "C++ or Rust perform better."
        ),
        expected_chunk_ids=["chunk-se-003"],
        notes="Subjective question requiring balanced nuanced answer",
    ),
]
