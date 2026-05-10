import os
import psycopg2
import json
from dotenv import load_dotenv

load_dotenv()

db_url = os.environ.get("DATABASE_URL")
conn = psycopg2.connect(db_url)
cur = conn.cursor()

print("Starting backfill...")

# 1. Fetch synthesis outputs from trace_events
# The payload of agent_end contains the data
cur.execute("""
    SELECT job_id, payload 
    FROM trace_events 
    WHERE agent_id = 'synthesis' AND event_type = 'agent_end'
""")
rows = cur.fetchall()
print(f"Found {len(rows)} potential answers in trace history.")

updated = 0
for job_id, payload in rows:
    try:
        # Payload is stored as JSON in DB (or string depending on driver)
        if isinstance(payload, str):
            data = json.loads(payload)
        else:
            data = payload
        
        # In agent_end, the output isn't directly in the payload, 
        # it's usually the event before it or we need to find the specific 'synthesis' result.
        # Wait, let's check the trace_events for a synthesis agent.
        pass
    except:
        continue

# Actually, it's easier to just look for the last 'agent_end' or 'done' event's preceding agent output.
# Or better: Search for ANY event where agent_id='synthesis' and payload has text.
cur.execute("""
    SELECT job_id, payload->>'output' as output
    FROM trace_events
    WHERE agent_id = 'synthesis' AND event_type = 'agent_end'
""")
# Wait, the current payload for agent_end doesn't have 'output'.
# Let's check what the payload actually contains.

cur.execute("SELECT payload FROM trace_events WHERE agent_id = 'synthesis' LIMIT 1")
sample = cur.fetchone()
print(f"Sample synthesis payload: {sample}")

# Re-evaluating: The SynthesisAgent.run() returns an AgentOutput.
# The Orchestrator._invoke_agent calls agent.run() and THEN traces 'agent_end'.
# The trace 'agent_end' payload has:
# payload={ "agent_id": agent_id, "token_count": tokens, "output_hash": output.output_hash, ... }
# IT DOES NOT HAVE THE OUTPUT TEXT.

# WAIT! Where is the output text stored?
# It's NOT stored in trace_events currently! Only hashes and metadata are stored in trace_events.
# The ONLY place the final answer was stored was in the SharedContext (in memory).

# Oh no! That means for the OLD jobs, the final answer was never persisted to the DB,
# only the traces of the process were.
# This is exactly why we added the 'final_answer' column!
