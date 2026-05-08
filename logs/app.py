"""
logs/app.py

Lightweight Flask log query UI.
Provides two pages:
  /           — list recent jobs with status, timing, and token totals
  /trace/<id> — render the full ordered execution trace for a job

Reads directly from PostgreSQL via psycopg2 (sync, simple).
"""

from __future__ import annotations

import os
from datetime import datetime

import psycopg2
import psycopg2.extras
from flask import Flask, abort, render_template_string

app = Flask(__name__)


# ─── DB connection ────────────────────────────────────────────────────────────

def get_conn():
    return psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST", "db"),
        port=int(os.environ.get("POSTGRES_PORT", "5432")),
        dbname=os.environ.get("POSTGRES_DB", "mega_ai"),
        user=os.environ.get("POSTGRES_USER", "mega_ai_user"),
        password=os.environ.get("POSTGRES_PASSWORD", ""),
        cursor_factory=psycopg2.extras.RealDictCursor,
    )


# ─── HTML templates ───────────────────────────────────────────────────────────

BASE_STYLE = """
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', sans-serif; background: #0f1117; color: #e0e0e0; padding: 2rem; }
  h1 { color: #7c6af7; margin-bottom: 1.5rem; font-size: 1.8rem; }
  h2 { color: #a89bff; margin: 1.5rem 0 0.5rem; }
  a { color: #7c6af7; text-decoration: none; }
  a:hover { text-decoration: underline; }
  table { width: 100%; border-collapse: collapse; margin-top: 1rem; }
  th { background: #1e2030; color: #a89bff; padding: 0.6rem 1rem; text-align: left; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; }
  td { padding: 0.6rem 1rem; border-bottom: 1px solid #1e2030; font-size: 0.9rem; }
  tr:hover td { background: #1a1d2e; }
  .badge { display: inline-block; padding: 0.2em 0.6em; border-radius: 4px; font-size: 0.8rem; font-weight: 600; }
  .badge-queued   { background: #2d3748; color: #a0aec0; }
  .badge-running  { background: #2a4365; color: #63b3ed; }
  .badge-done     { background: #1c4532; color: #68d391; }
  .badge-failed   { background: #4a1c1c; color: #fc8181; }
  .event-card { background: #1e2030; border-left: 3px solid #7c6af7; padding: 0.8rem 1rem; margin-bottom: 0.5rem; border-radius: 4px; }
  .event-type { font-weight: 700; color: #7c6af7; font-size: 0.85rem; }
  .event-agent { color: #a89bff; font-size: 0.85rem; margin-left: 0.5rem; }
  .event-time { color: #718096; font-size: 0.78rem; float: right; }
  .event-payload { margin-top: 0.4rem; font-size: 0.82rem; color: #a0aec0; white-space: pre-wrap; word-break: break-all; }
  .stat-bar { display: flex; gap: 2rem; background: #1e2030; padding: 1rem 1.5rem; border-radius: 8px; margin-bottom: 1.5rem; }
  .stat { text-align: center; }
  .stat-val { font-size: 1.6rem; font-weight: 700; color: #7c6af7; }
  .stat-label { font-size: 0.8rem; color: #718096; margin-top: 0.2rem; }
  .back-link { margin-bottom: 1rem; display: block; font-size: 0.9rem; }
  .violation { color: #fc8181; font-weight: 600; }
</style>
"""

INDEX_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Mega AI — Log Query UI</title>
  {{ style|safe }}
</head>
<body>
  <h1>⚡ Mega AI — Recent Jobs</h1>

  <div class="stat-bar">
    <div class="stat"><div class="stat-val">{{ total }}</div><div class="stat-label">Total Jobs</div></div>
    <div class="stat"><div class="stat-val">{{ done }}</div><div class="stat-label">Completed</div></div>
    <div class="stat"><div class="stat-val">{{ running }}</div><div class="stat-label">Running</div></div>
    <div class="stat"><div class="stat-val">{{ failed }}</div><div class="stat-label">Failed</div></div>
  </div>

  <table>
    <thead>
      <tr>
        <th>Job ID</th>
        <th>Status</th>
        <th>Query (truncated)</th>
        <th>Created</th>
        <th>Duration</th>
        <th>Trace</th>
      </tr>
    </thead>
    <tbody>
    {% for job in jobs %}
      <tr>
        <td><code style="font-size:0.8rem">{{ job.id[:8] }}…</code></td>
        <td><span class="badge badge-{{ job.status }}">{{ job.status }}</span></td>
        <td>{{ job.query[:80] }}{% if job.query|length > 80 %}…{% endif %}</td>
        <td>{{ job.created_at.strftime('%Y-%m-%d %H:%M:%S') if job.created_at else '—' }}</td>
        <td>
          {% if job.completed_at and job.started_at %}
            {{ ((job.completed_at - job.started_at).total_seconds() | round(1)) }}s
          {% else %}—{% endif %}
        </td>
        <td><a href="/trace/{{ job.id }}">View →</a></td>
      </tr>
    {% else %}
      <tr><td colspan="6" style="text-align:center;color:#718096;padding:2rem">No jobs yet. Submit a query via POST /query</td></tr>
    {% endfor %}
    </tbody>
  </table>
</body>
</html>
"""

TRACE_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Mega AI — Trace {{ job_id[:8] }}</title>
  {{ style|safe }}
</head>
<body>
  <a class="back-link" href="/">← Back to jobs</a>
  <h1>Execution Trace</h1>
  <p style="color:#718096;margin-bottom:1rem">
    Job: <code>{{ job_id }}</code> &nbsp;·&nbsp;
    Status: <span class="badge badge-{{ job.status }}">{{ job.status }}</span>
    {% if job.error %}&nbsp;·&nbsp;<span class="violation">{{ job.error }}</span>{% endif %}
  </p>

  <div class="stat-bar">
    <div class="stat"><div class="stat-val">{{ events|length }}</div><div class="stat-label">Events</div></div>
    <div class="stat"><div class="stat-val">{{ total_tokens }}</div><div class="stat-label">Total Tokens</div></div>
    <div class="stat"><div class="stat-val">{{ violations }}</div><div class="stat-label">Violations</div></div>
    <div class="stat"><div class="stat-val">{{ tool_calls }}</div><div class="stat-label">Tool Calls</div></div>
  </div>

  <h2>Events (chronological)</h2>
  {% for ev in events %}
    <div class="event-card">
      <span class="event-type">#{{ ev.seq }} {{ ev.event_type }}</span>
      {% if ev.agent_id %}<span class="event-agent">· {{ ev.agent_id }}</span>{% endif %}
      <span class="event-time">
        {{ ev.created_at.strftime('%H:%M:%S.%f')[:-3] if ev.created_at else '' }}
        {% if ev.latency_ms %}&nbsp;({{ ev.latency_ms }}ms){% endif %}
        {% if ev.token_count %}&nbsp;· {{ ev.token_count }} tok{% endif %}
      </span>
      {% if ev.policy_violations %}
        <div class="violation">⚠ {{ ev.policy_violations }}</div>
      {% endif %}
      {% if ev.payload %}
        <div class="event-payload">{{ ev.payload | tojson(indent=2) if ev.payload else '' }}</div>
      {% endif %}
    </div>
  {% else %}
    <p style="color:#718096">No trace events recorded yet.</p>
  {% endfor %}
</body>
</html>
"""


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT id, query, status, error, created_at, started_at, completed_at
            FROM jobs
            ORDER BY created_at DESC
            LIMIT 100
        """)
        jobs = cur.fetchall()

        cur.execute("SELECT COUNT(*) AS n FROM jobs")
        total = cur.fetchone()["n"]
        cur.execute("SELECT COUNT(*) AS n FROM jobs WHERE status='done'")
        done = cur.fetchone()["n"]
        cur.execute("SELECT COUNT(*) AS n FROM jobs WHERE status='running'")
        running = cur.fetchone()["n"]
        cur.execute("SELECT COUNT(*) AS n FROM jobs WHERE status='failed'")
        failed = cur.fetchone()["n"]

    return render_template_string(
        INDEX_TEMPLATE,
        jobs=jobs, total=total, done=done, running=running, failed=failed,
        style=BASE_STYLE,
    )


@app.route("/trace/<job_id>")
def trace(job_id: str):
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, query, status, error, created_at, started_at, completed_at FROM jobs WHERE id = %s",
            (job_id,),
        )
        job = cur.fetchone()
        if not job:
            abort(404, description=f"Job '{job_id}' not found")

        cur.execute("""
            SELECT seq, agent_id, event_type, input_hash, output_hash,
                   payload, latency_ms, token_count, policy_violations, created_at
            FROM trace_events
            WHERE job_id = %s
            ORDER BY seq ASC
        """, (job_id,))
        events = cur.fetchall()

    total_tokens = sum(e["token_count"] or 0 for e in events)
    violations = sum(
        len(e["policy_violations"]) if e["policy_violations"] else 0
        for e in events
    )
    tool_calls = sum(1 for e in events if e["event_type"] == "tool_call")

    return render_template_string(
        TRACE_TEMPLATE,
        job_id=job_id, job=job, events=events,
        total_tokens=total_tokens, violations=violations, tool_calls=tool_calls,
        style=BASE_STYLE,
    )


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    host = os.environ.get("LOG_UI_HOST", "0.0.0.0")
    port = int(os.environ.get("LOG_UI_PORT", "8080"))
    app.run(host=host, port=port, debug=False)
