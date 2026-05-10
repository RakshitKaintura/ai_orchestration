"""eval/cases/baseline.py — Baseline evaluation test cases."""

from eval.cases.schemas import EvalCase

BASELINE_CASES: list[EvalCase] = [
    EvalCase(
        case_id="base-001",
        category="baseline",
        query="What is the difference between RAG and fine-tuning for production LLM systems?",
        expected_answer=(
            "RAG (Retrieval-Augmented Generation) retrieves relevant documents at inference time "
            "from a knowledge base and uses them to ground the LLM's response, without modifying "
            "model weights. Fine-tuning adapts model weights on domain-specific data, encoding "
            "knowledge directly in parameters. RAG is better for dynamic, frequently-updated "
            "knowledge; fine-tuning is better for style, format, and task-specific behaviour "
            "that must generalise without retrieval."
        ),
        expected_chunk_ids=["chunk-rag-001", "chunk-rag-002", "chunk-ft-001", "chunk-ft-002"],
        notes="Core comparison question — tests both RAG retrieval and synthesis quality",
    ),
    EvalCase(
        case_id="base-002",
        category="baseline",
        query="How many planets are in the solar system, and what is the largest?",
        expected_answer=(
            "There are 8 planets in the Solar System: Mercury, Venus, Earth, Mars, Jupiter, "
            "Saturn, Uranus, and Neptune. Pluto was reclassified as a dwarf planet in 2006. "
            "Jupiter is the largest planet — its mass is greater than all other planets combined."
        ),
        expected_chunk_ids=["chunk-astro-001", "chunk-astro-002"],
        notes="Simple factual question with corpus coverage",
    ),
    EvalCase(
        case_id="base-003",
        category="baseline",
        query="What is LoRA and how does it reduce fine-tuning compute cost?",
        expected_answer=(
            "LoRA (Low-Rank Adaptation) reduces fine-tuning cost by injecting trainable low-rank "
            "decomposition matrices into transformer layers while keeping the original model weights "
            "frozen. Only the adapter matrices are updated during training, reducing trainable "
            "parameters by 10-100x compared to full fine-tuning, with minimal performance loss."
        ),
        expected_chunk_ids=["chunk-ft-002", "chunk-ft-003"],
        notes="Tests LoRA-specific retrieval and technical accuracy",
    ),
    EvalCase(
        case_id="base-004",
        category="baseline",
        query="What does the IPCC AR6 report say about global temperature rise?",
        expected_answer=(
            "According to IPCC AR6, global surface temperature increased by approximately 1.1°C "
            "above pre-industrial (1850-1900) levels between 2011-2020. The 1.5°C threshold is "
            "significant because many non-linear climate tipping points are estimated to activate "
            "in the 1.5-2°C range."
        ),
        expected_chunk_ids=["chunk-climate-001", "chunk-climate-002"],
        notes="Tests precise retrieval of scientific report data",
    ),
    EvalCase(
        case_id="base-005",
        category="baseline",
        query="What is 17 multiplied by 23?",
        expected_answer="17 multiplied by 23 equals 391. This can be computed as (17 × 20) + (17 × 3) = 340 + 51 = 391.",
        expected_chunk_ids=["chunk-math-001"],
        requires_tools=["code_sandbox"],
        notes="Simple arithmetic — tests code sandbox tool invocation",
    ),
]
