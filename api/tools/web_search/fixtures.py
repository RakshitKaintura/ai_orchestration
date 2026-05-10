"""
api/tools/web_search/fixtures.py

Static fixture data for the web search tool.
Keyed by topic cluster, matched via keyword presence in the query.
In production, replace _execute_search() in tool.py with a real API call.
"""

from __future__ import annotations

# Each result: (url, title, snippet, relevance_score)
FIXTURES: dict[str, list[tuple[str, str, str, float]]] = {
    "rag": [
        ("https://arxiv.org/abs/2005.11401", "RAG: Retrieval-Augmented Generation for NLP",
         "Combines parametric and non-parametric memory for open-domain QA. The retriever fetches relevant passages, the generator conditions on them.", 0.97),
        ("https://www.pinecone.io/learn/retrieval-augmented-generation/", "Retrieval-Augmented Generation — Pinecone",
         "RAG allows LLMs to reference authoritative knowledge bases outside their training data before generating a response.", 0.95),
        ("https://python.langchain.com/docs/concepts/rag/", "RAG — LangChain Docs",
         "LangChain's RAG pipeline: document loaders, text splitters, embeddings, vector stores, and retrievers.", 0.91),
        ("https://www.databricks.com/glossary/retrieval-augmented-generation", "What is RAG? — Databricks",
         "RAG is an AI framework for retrieving facts from an external knowledge base to ground LLMs on the most accurate, up-to-date information.", 0.88),
        ("https://huggingface.co/docs/transformers/model_doc/rag", "RAG — HuggingFace Transformers",
         "RAG models use a retriever and a generator. The retriever encodes questions and documents into a shared embedding space.", 0.85),
    ],
    "fine-tuning": [
        ("https://platform.openai.com/docs/guides/fine-tuning", "Fine-tuning — OpenAI Docs",
         "Fine-tuning adapts a pre-trained model by updating its weights on a curated dataset. Improves few-shot performance on specific tasks.", 0.96),
        ("https://arxiv.org/abs/2106.09685", "LoRA: Low-Rank Adaptation of Large Language Models",
         "LoRA injects trainable low-rank matrices into transformer layers, dramatically reducing trainable parameters while maintaining model quality.", 0.94),
        ("https://www.anyscale.com/blog/fine-tuning-llms-lora-or-full-parameter", "Fine-tuning LLMs: LoRA vs Full Parameter",
         "Full fine-tuning updates all weights; LoRA freezes base weights and trains adapters. LoRA uses 10-100x fewer parameters.", 0.90),
    ],
    "llm": [
        ("https://arxiv.org/abs/2303.08774", "GPT-4 Technical Report — OpenAI",
         "GPT-4 is a large multimodal model trained with RLHF. It achieves human-level performance on various professional and academic benchmarks.", 0.96),
        ("https://www.anthropic.com/research/claude-3", "Claude 3 Model Card — Anthropic",
         "Claude 3 exhibits near-human comprehension on complex tasks. Uses Constitutional AI and RLHF for alignment.", 0.95),
        ("https://arxiv.org/abs/2201.11903", "Chain-of-Thought Prompting Elicits Reasoning",
         "CoT prompting allows language models to decompose complex problems into intermediate steps, significantly improving accuracy on reasoning tasks.", 0.93),
        ("https://arxiv.org/abs/2310.06825", "Self-RAG: Learning to Retrieve, Generate, and Critique",
         "Self-RAG trains an LLM to decide when to retrieve, critique its own outputs, and generate with reflection tokens.", 0.89),
    ],
    "climate": [
        ("https://www.ipcc.ch/report/ar6/wg1/", "IPCC AR6 Working Group I — The Physical Science Basis",
         "Global surface temperature increased 1.1°C above 1850-1900 levels between 2011-2020. Human influence has warmed the atmosphere, ocean and land.", 0.97),
        ("https://climate.nasa.gov/evidence/", "Evidence — NASA Global Climate Change",
         "Global temperature rise, warming oceans, shrinking ice sheets, glacial retreat, sea level rise, extreme weather events.", 0.95),
        ("https://www.unep.org/explore-topics/climate-action", "Climate Action — UNEP",
         "Without deep emissions reductions, global warming will exceed 1.5°C by the early 2030s.", 0.91),
    ],
    "python": [
        ("https://docs.python.org/3/", "Python 3 Documentation — python.org",
         "Comprehensive reference for Python 3.11+. Includes language reference, library reference, and tutorials.", 0.97),
        ("https://realpython.com/python-async-await/", "Async IO in Python — Real Python",
         "asyncio provides infrastructure for writing single-threaded concurrent code using coroutines, multiplexing I/O access over sockets and other resources.", 0.93),
        ("https://peps.python.org/pep-0634/", "PEP 634 — Structural Pattern Matching",
         "Python 3.10 introduces structural pattern matching: match statements and case clauses.", 0.87),
    ],
    "einstein": [
        ("https://www.britannica.com/biography/Albert-Einstein", "Albert Einstein — Britannica",
         "Albert Einstein excelled in mathematics and physics from a young age. By age 12 he had mastered algebra and Euclidean geometry over a single summer.", 0.98),
        ("https://www.aps.org/publications/apsnews/200512/history.cfm", "Einstein's Miraculous Year — APS",
         "In 1905, Einstein published four landmark papers: on the photoelectric effect, Brownian motion, special relativity, and mass-energy equivalence.", 0.96),
        ("https://history.aip.org/exhibits/einstein/early1.htm", "Einstein's Early Years — AIP",
         "Contrary to popular myth, Einstein did not fail mathematics. He showed exceptional mathematical ability throughout his schooling in Germany and Switzerland.", 0.99),
    ],
    "great wall": [
        ("https://www.smithsonianmag.com/history/the-myth-of-the-great-wall-30402043/", "The Myth of the Great Wall — Smithsonian",
         "The Great Wall of China is NOT visible from space with the naked eye. This is a widely-repeated myth debunked by multiple astronauts including Chinese astronaut Yang Liwei.", 0.99),
        ("https://www.nasa.gov/vision/space/workinginspace/great_wall.html", "Great Wall — NASA",
         "NASA confirms the Great Wall is not visible from low Earth orbit without optical aids. It's too narrow (15-30 feet wide) to resolve from orbit.", 0.98),
        ("https://www.history.com/topics/ancient-china/great-wall-of-china", "Great Wall of China — History.com",
         "Construction of the Great Wall spanned many centuries, from the 7th century BC to the 17th century AD. Its primary purpose was military defense.", 0.90),
    ],
    "default": [
        ("https://en.wikipedia.org/wiki/Artificial_intelligence", "Artificial intelligence — Wikipedia",
         "AI is intelligence demonstrated by machines. Major subfields include machine learning, natural language processing, computer vision, and robotics.", 0.75),
        ("https://arxiv.org/abs/2304.01852", "Sparks of Artificial General Intelligence — Microsoft Research",
         "GPT-4 exhibits sparks of AGI across diverse tasks. Discusses reasoning, common sense, theory of mind, and emergent capabilities.", 0.72),
        ("https://www.deeplearning.ai/", "DeepLearning.AI — Andrew Ng",
         "Courses and resources for machine learning, deep learning, and AI. MLOps, LLMOps, and generative AI specializations.", 0.70),
    ],
}

# Keywords → fixture cluster mapping
KEYWORD_MAP: list[tuple[list[str], str]] = [
    (["rag", "retrieval", "augmented", "retrieval-augmented"], "rag"),
    (["fine-tuning", "fine tuning", "finetune", "lora", "adapter"], "fine-tuning"),
    (["llm", "language model", "gpt", "claude", "transformer", "prompt"], "llm"),
    (["climate", "global warming", "carbon", "emission", "ipcc"], "climate"),
    (["python", "asyncio", "pep", "pandas", "numpy"], "python"),
    (["einstein", "relativity", "physics"], "einstein"),
    (["great wall", "china", "visible from space"], "great wall"),
]


def select_fixture(query: str) -> list[tuple[str, str, str, float]]:
    """Select the best-matching fixture cluster for a given query."""
    q = query.lower()
    for keywords, cluster in KEYWORD_MAP:
        if any(kw in q for kw in keywords):
            return FIXTURES[cluster]
    return FIXTURES["default"]
