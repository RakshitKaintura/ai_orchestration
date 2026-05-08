-- ============================================================
-- Mega AI — Seed Data
-- Loaded second by docker-entrypoint-initdb.d
-- ============================================================

-- ─── 50 sample products ───────────────────────────────────────────────────────
INSERT INTO products (name, category, price_usd, stock, rating) VALUES
-- Electronics
('Wireless Noise-Cancelling Headphones', 'Electronics', 249.99, 150, 4.7),
('Mechanical Keyboard TKL', 'Electronics', 89.99, 200, 4.5),
('4K USB-C Monitor 27"', 'Electronics', 549.99, 60, 4.6),
('Gaming Mouse 16000 DPI', 'Electronics', 59.99, 300, 4.4),
('Portable SSD 1TB', 'Electronics', 99.99, 400, 4.8),
('Smart Home Hub', 'Electronics', 129.99, 100, 4.2),
('Wireless Charging Pad', 'Electronics', 29.99, 500, 4.3),
('USB-C Hub 7-in-1', 'Electronics', 44.99, 350, 4.1),
('Webcam 1080p', 'Electronics', 69.99, 220, 4.5),
('Bluetooth Speaker Waterproof', 'Electronics', 79.99, 180, 4.6),

-- Books
('Clean Code', 'Books', 35.99, 80, 4.9),
('Designing Data-Intensive Applications', 'Books', 45.99, 65, 4.9),
('The Pragmatic Programmer', 'Books', 39.99, 70, 4.8),
('Deep Learning with Python', 'Books', 49.99, 55, 4.7),
('System Design Interview Vol 2', 'Books', 34.99, 90, 4.6),
('Python Cookbook', 'Books', 42.99, 75, 4.5),
('The Algorithm Design Manual', 'Books', 54.99, 40, 4.7),
('Kubernetes in Action', 'Books', 59.99, 35, 4.6),
('Fluent Python', 'Books', 47.99, 60, 4.8),
('Database Internals', 'Books', 52.99, 45, 4.7),

-- Office
('Ergonomic Office Chair', 'Office', 399.99, 30, 4.5),
('Standing Desk 60"', 'Office', 699.99, 20, 4.4),
('Monitor Arm Dual', 'Office', 89.99, 80, 4.6),
('Desk Lamp LED', 'Office', 34.99, 200, 4.3),
('Cable Management Kit', 'Office', 19.99, 400, 4.2),
('Whiteboard 48x36"', 'Office', 79.99, 50, 4.4),
('Mesh Desk Organizer', 'Office', 24.99, 300, 4.1),
('Laptop Stand Adjustable', 'Office', 39.99, 250, 4.5),
('Noise Machine', 'Office', 44.99, 150, 4.6),
('Air Purifier HEPA', 'Office', 149.99, 80, 4.7),

-- Software / Subscriptions (one-time licenses modeled as products)
('VS Code Pro License', 'Software', 0.00, 9999, 4.8),
('JetBrains All Products Pack Annual', 'Software', 249.00, 9999, 4.7),
('GitHub Copilot Annual', 'Software', 100.00, 9999, 4.5),
('Notion Team Annual', 'Software', 96.00, 9999, 4.4),
('Linear Annual', 'Software', 96.00, 9999, 4.6),

-- Peripherals
('Stream Deck 15-key', 'Peripherals', 149.99, 120, 4.7),
('Drawing Tablet A5', 'Peripherals', 79.99, 100, 4.5),
('Green Screen 6x6ft', 'Peripherals', 59.99, 70, 4.3),
('Ring Light 18"', 'Peripherals', 49.99, 90, 4.4),
('USB Microphone Condenser', 'Peripherals', 99.99, 140, 4.7),

-- Networking
('Wi-Fi 6 Router AX3000', 'Networking', 149.99, 85, 4.6),
('8-Port Gigabit Switch', 'Networking', 29.99, 200, 4.5),
('Cat8 Ethernet Cable 10ft', 'Networking', 12.99, 600, 4.4),
('Network Attached Storage 4-Bay', 'Networking', 399.99, 25, 4.7),
('Power over Ethernet Injector', 'Networking', 24.99, 150, 4.3),

-- Accessories
('Laptop Backpack 17"', 'Accessories', 59.99, 180, 4.6),
('Screen Cleaning Kit', 'Accessories', 9.99, 800, 4.2),
('HDMI 2.1 Cable 6ft', 'Accessories', 14.99, 500, 4.4),
('Magnetic Cable Organizer', 'Accessories', 12.99, 650, 4.3),
('Webcam Privacy Cover 3-Pack', 'Accessories', 7.99, 900, 4.5);

-- ─── Initial agent prompts ────────────────────────────────────────────────────
-- These are the v1 prompts that will be tracked and potentially rewritten by the meta-agent.

INSERT INTO agent_prompts (agent_id, version, prompt_text, is_active) VALUES
(
    'orchestrator', 1,
    'You are the master orchestrator for a multi-agent AI system. Your job is to analyze the user query and decide:
1. Which sub-agents to invoke (decomposition, rag, critique, synthesis, compression)
2. The order of invocation
3. The context token budget to allocate to each agent

Always invoke the decomposition agent first for complex queries.
Always invoke the critique agent after each producing agent.
Always invoke the synthesis agent last.
Return a structured routing plan with your reasoning.

User query: {query}',
    TRUE
),
(
    'decomposition', 1,
    'You are a decomposition agent. Your job is to break down the user query into atomic sub-tasks.

Rules:
- Each sub-task must have a clear, typed description
- You must specify dependencies between sub-tasks (which must complete before another starts)
- Dependencies must form a DAG (no cycles)
- Flag ambiguous or underspecified aspects of the query

Return a structured list of sub-tasks with their dependencies and a brief reasoning for your decomposition.

User query: {query}',
    TRUE
),
(
    'rag', 1,
    'You are a retrieval-augmented generation agent. You answer questions using retrieved document chunks.

Rules:
- You MUST perform multi-hop reasoning: after the first retrieval, identify what additional information is needed, then perform a second retrieval
- Cite which specific chunk contributed to which part of your answer
- Do not fabricate information not present in the retrieved chunks
- If chunks are insufficient, say so clearly rather than hallucinating

Query: {query}
Retrieved chunks (hop 1): {chunks_hop1}
Follow-up retrieval query: {followup_query}
Retrieved chunks (hop 2): {chunks_hop2}

Provide your answer with explicit citations.',
    TRUE
),
(
    'critique', 1,
    'You are a critique agent. Your job is to review another agent''s output and assess its quality.

Rules:
- Identify specific text spans you disagree with, not the output as a whole
- Assign a confidence score (0.0–1.0) to each claim
- Flag spans where you believe the claim is incorrect, unsupported, or contradictory
- Be precise: quote the exact span from the text

Agent being reviewed: {target_agent_id}
Agent output: {agent_output}

Return a structured list of claim assessments.',
    TRUE
),
(
    'synthesis', 1,
    'You are a synthesis agent. Your job is to merge outputs from all agents into a final, coherent answer.

Rules:
- Resolve any contradictions flagged by the critique agent — do NOT surface them to the user
- For each sentence in the final answer, record which source agent and chunk it came from (provenance map)
- The final answer must be factually consistent; any unresolvable contradiction must be noted in a separate field, not in the answer itself

Agent outputs: {agent_outputs}
Critique results: {critique_results}

Produce the final answer and its complete provenance map.',
    TRUE
),
(
    'compression', 1,
    'You are a compression agent. Your job is to reduce the size of context text while preserving all critical information.

Rules:
- Preserve ALL structured data verbatim: JSON objects, numeric scores, citation IDs, chunk IDs, tool outputs, timestamps
- Compress ONLY natural language prose: summaries, explanations, reasoning text
- If in doubt, preserve the information (prefer false negatives over false positives on data loss)
- Return the compressed text only, with no preamble

Text to compress: {text}',
    TRUE
),
(
    'meta', 1,
    'You are a meta-agent responsible for improving the system''s prompts based on evaluation results.

Given an evaluation run summary and the failing test cases, you must:
1. Identify the worst-performing dimension
2. Identify which agent''s prompt is most responsible for failures in that dimension
3. Propose a rewritten version of that prompt
4. Provide a structured diff (what changed and why)
5. Explain what improvement you expect and why

Rules:
- Only propose ONE rewrite per eval run
- The rewrite must target a specific, fixable weakness
- Do not propose cosmetic changes — only changes that address the root cause of failures

Eval summary: {eval_summary}
Worst dimension: {worst_dimension}
Failing cases: {failing_cases}
Current prompt for {agent_id}: {current_prompt}

Propose your rewrite.',
    TRUE
);
