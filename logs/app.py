import os
import psycopg2
import psycopg2.extras
from flask import Flask, render_template_string, abort, request
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

def get_conn():
    return psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST", "db"),
        port=os.environ.get("POSTGRES_PORT", "5432"),
        dbname=os.environ.get("POSTGRES_DB", "postgres"),
        user=os.environ.get("POSTGRES_USER", "postgres.hikcnmqvxnoqchbqyhrj"),
        password=os.environ.get("POSTGRES_PASSWORD", ""),
        cursor_factory=psycopg2.extras.RealDictCursor,
    )

# ─── HTML templates ───────────────────────────────────────────────────────────

BASE_STYLE = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

  :root {
    --bg: #0F172A;
    --sidebar: #111827;
    --card-bg: #1E293B;
    --primary: #10A37F;
    --primary-hover: #0c8a6a;
    --text: #F8FAFC;
    --text-dim: #94A3B8;
    --border: #334155;
    --success: #10A37F;
    --error: #f85149;
    --running: #58a6ff;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }
  
  body { 
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; 
    background: var(--bg); 
    color: var(--text); 
    line-height: 1.5;
    display: flex;
    height: 100vh;
    overflow: hidden;
  }

  /* ─── SIDEBAR ─── */
  aside.sidebar {
    width: 260px;
    background: var(--sidebar);
    border-right: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    padding: 1rem 0.75rem;
    flex-shrink: 0;
  }

  .new-chat-btn {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: transparent;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.75rem;
    color: var(--text);
    cursor: pointer;
    text-decoration: none;
    font-weight: 500;
    transition: background 0.2s;
    margin-bottom: 1.5rem;
  }
  .new-chat-btn:hover {
    background: var(--card-bg);
  }

  .history-header {
    font-size: 0.75rem;
    color: var(--text-dim);
    font-weight: 600;
    padding: 0 0.5rem 0.5rem 0.5rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .history-list {
    flex: 1;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
  }

  .history-item {
    display: block;
    padding: 0.75rem;
    border-radius: 8px;
    color: var(--text);
    text-decoration: none;
    font-size: 0.875rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    transition: background 0.2s;
    position: relative;
  }
  .history-item:hover {
    background: var(--card-bg);
  }
  .history-item.active {
    background: var(--card-bg);
    font-weight: 500;
  }
  .status-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 0.5rem;
  }
  .status-done { background: var(--success); }
  .status-failed { background: var(--error); }
  .status-running { background: var(--running); }

  /* ─── MAIN CONTENT ─── */
  main.content {
    flex: 1;
    display: flex;
    flex-direction: column;
    position: relative;
    background: var(--bg);
  }

  header.top-nav {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1rem 2rem;
    border-bottom: 1px solid var(--border);
    background: var(--bg);
    z-index: 10;
  }
  
  .top-nav h1 {
    font-size: 1.1rem;
    font-weight: 600;
  }

  .chat-container {
    flex: 1;
    overflow-y: auto;
    padding: 2rem;
    padding-bottom: 150px; /* space for input */
    display: flex;
    flex-direction: column;
    align-items: center;
  }
  
  .chat-content {
    width: 100%;
    max-width: 800px;
    display: flex;
    flex-direction: column;
    gap: 2rem;
  }

  /* ─── CHAT BUBBLES ─── */
  .message {
    display: flex;
    gap: 1.5rem;
    width: 100%;
  }
  .message.user {
    justify-content: flex-end;
  }
  .message.assistant {
    justify-content: flex-start;
  }

  .avatar {
    width: 30px;
    height: 30px;
    border-radius: 4px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: bold;
    font-size: 0.85rem;
    flex-shrink: 0;
  }
  .avatar.user-av { background: #5536DA; color: white; }
  .avatar.bot-av { background: var(--primary); color: white; }

  .bubble {
    padding: 1rem 1.25rem;
    border-radius: 12px;
    max-width: 85%;
    line-height: 1.6;
    font-size: 0.95rem;
  }
  .message.user .bubble {
    background: var(--card-bg);
    border: 1px solid var(--border);
    color: var(--text);
    border-bottom-right-radius: 2px;
  }
  .message.assistant .bubble {
    background: transparent;
    color: var(--text);
    padding: 0;
  }

  /* ─── TRACE EVENTS ─── */
  .event-card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.25rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
  }
  .event-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.75rem;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.5rem;
  }
  .event-type {
    font-weight: 600;
    color: var(--primary);
    font-size: 0.85rem;
    letter-spacing: 0.5px;
  }
  .event-meta {
    font-size: 0.8rem;
    color: var(--text-dim);
    font-family: monospace;
  }
  .event-payload {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 1rem;
    font-family: 'Fira Code', Consolas, Monaco, monospace;
    font-size: 0.85rem;
    color: #a5d6ff;
    overflow-x: auto;
    white-space: pre-wrap;
  }

  /* ─── INPUT BAR ─── */
  .input-wrapper {
    position: absolute;
    bottom: 0;
    left: 0;
    width: 100%;
    padding: 2rem;
    background: linear-gradient(180deg, transparent, var(--bg) 20%);
    display: flex;
    justify-content: center;
    align-items: flex-end;
    flex-direction: column;
  }
  
  .input-box {
    width: 100%;
    max-width: 800px;
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 0.5rem 0.5rem 0.5rem 1.25rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
    box-shadow: 0 0 15px rgba(0,0,0,0.5);
    margin: 0 auto;
    transition: border-color 0.2s;
  }
  .input-box:focus-within {
    border-color: var(--primary);
  }

  input[type="text"] {
    flex: 1;
    background: transparent;
    border: none;
    color: var(--text);
    font-size: 1rem;
    padding: 0.5rem 0;
  }
  input[type="text"]:focus {
    outline: none;
  }

  .send-btn {
    background: var(--primary);
    color: white;
    border: none;
    border-radius: 8px;
    width: 36px;
    height: 36px;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    transition: background 0.2s;
  }
  .send-btn:hover {
    background: var(--primary-hover);
  }
  .send-btn:disabled {
    background: var(--border);
    cursor: not-allowed;
  }

  #statusMsg {
    width: 100%;
    max-width: 800px;
    margin: 0.5rem auto 0 auto;
    font-size: 0.8rem;
    color: var(--text-dim);
    text-align: center;
    display: none;
  }

  /* Empty State */
  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100%;
    color: var(--text-dim);
  }
  .empty-state h2 {
    font-size: 2rem;
    color: var(--text);
    margin-bottom: 1rem;
    font-weight: 600;
  }
  
  /* Scrollbars */
  ::-webkit-scrollbar { width: 8px; height: 8px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
  ::-webkit-scrollbar-thumb:hover { background: #475569; }

  /* Syntax Highlighting */
  .string { color: #a5d6ff; }
  .number { color: #79c0ff; }
  .boolean { color: var(--primary); }
  .null { color: var(--error); }
  .key { color: #7ee787; font-weight: 600; }
</style>

<script>
async function submitQuery() {
    const input = document.getElementById('queryInput');
    const btn = document.getElementById('submitBtn');
    const status = document.getElementById('statusMsg');
    const query = input.value.trim();
    if (!query) return;
    btn.disabled = true;
    status.style.display = 'block';
    status.innerText = 'Orchestrating agents...';
    try {
        const response = await fetch('http://localhost:8000/query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: query })
        });
        if (response.ok) {
            const jobId = response.headers.get('X-Job-ID');
            status.innerText = 'Success! Opening trace...';
            setTimeout(() => {
                if (jobId) window.location.href = `/trace/${jobId}`;
                else window.location.reload();
            }, 800);
        } else {
            status.innerText = 'API Error: ' + response.status;
            btn.disabled = false;
        }
    } catch (e) {
        status.innerText = 'Connection failed.';
        btn.disabled = false;
    }
}
</script>
"""

SIDEBAR_HTML = """
  <aside class="sidebar">
    <a href="/" class="new-chat-btn">
      <span>New pipeline</span>
      <svg stroke="currentColor" fill="none" stroke-width="2" viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round" height="16" width="16" xmlns="http://www.w3.org/2000/svg"><line x1="12" y1="5" x2="12" y2="19"></line><line x1="5" y1="12" x2="19" y2="12"></line></svg>
    </a>
    <div class="history-header">Recent History</div>
    <div class="history-list">
      {% for j in jobs %}
      <a href="/trace/{{ j.id }}" class="history-item {% if current_job_id == j.id %}active{% endif %}">
        <span class="status-dot status-{{ j.status }}"></span>
        {{ j.query }}
      </a>
      {% endfor %}
    </div>
  </aside>
"""

INDEX_TEMPLATE = BASE_STYLE + """
""" + SIDEBAR_HTML + """
  <main class="content">
    <header class="top-nav">
      <h1>Mega AI</h1>
      <div style="font-size: 0.85rem; color: var(--text-dim);">
        Jobs: <span style="color:var(--text);">{{ stats.total }}</span> | 
        Completed: <span style="color:var(--success);">{{ stats.done }}</span> | 
        Running: <span style="color:var(--running);">{{ stats.running }}</span>
      </div>
    </header>

    <div class="chat-container">
      <div class="empty-state">
        <div style="width: 60px; height: 60px; background: var(--primary); border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-bottom: 1.5rem;">
          <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2"><path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>
        </div>
        <h2>How can I help you today?</h2>
        <p>Ask a complex query. The multi-agent pipeline will take care of the rest.</p>
      </div>
    </div>

    <div class="input-wrapper">
      <div class="input-box">
        <input type="text" id="queryInput" placeholder="Message Mega AI..." onkeydown="if(event.key==='Enter') submitQuery()" />
        <button id="submitBtn" class="send-btn" onclick="submitQuery()">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>
        </button>
      </div>
      <div id="statusMsg"></div>
    </div>
  </main>
"""

TRACE_TEMPLATE = BASE_STYLE + """
""" + SIDEBAR_HTML + """
  <main class="content">
    <header class="top-nav">
      <h1>Trace <span style="font-weight:400; color:var(--text-dim); font-size:0.9rem; margin-left:0.5rem;">{{ job.id[:8] }}...</span></h1>
      <div style="font-size: 0.85rem; color: var(--text-dim); text-transform: uppercase; font-weight: 600;">
        <span class="status-dot status-{{ job.status }}" style="margin-right:0.25rem;"></span>{{ job.status }}
      </div>
    </header>

    <div class="chat-container" id="chatContainer">
      <div class="chat-content">
        
        <!-- User Query Bubble -->
        <div class="message user">
          <div class="bubble">{{ job.query }}</div>
          <div class="avatar user-av">U</div>
        </div>

        <!-- System / Agent Output Bubble -->
        <div class="message assistant">
          <div class="avatar bot-av">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>
          </div>
          <div class="bubble" style="width: 100%;">
            
            {% if job.status == 'failed' and job.error %}
            <div class="event-card" style="border-color: var(--error);">
              <div class="event-header">
                <span class="event-type" style="color: var(--error);">Pipeline Error</span>
              </div>
              <div class="event-payload" style="color: #ff7b72;">{{ job.error }}</div>
            </div>
            {% endif %}

            {% for ev in events %}
            <div class="event-card">
              <div class="event-header">
                <div>
                  <span class="event-type">#{{ ev.seq }} {{ ev.event_type }}</span>
                  {% if ev.agent_id %}
                  <span class="event-meta" style="margin-left: 0.75rem; color: var(--primary); background: rgba(16, 163, 127, 0.1); padding: 0.1rem 0.4rem; border-radius: 4px;">
                    @{{ ev.agent_id }}
                  </span>
                  {% endif %}
                </div>
                <div class="event-meta">{{ ev.created_at.strftime('%H:%M:%S.%f')[:-3] if ev.created_at else '' }}</div>
              </div>
              <pre class="event-payload"><code class="json">{{ ev.payload | tojson(indent=2) | safe }}</code></pre>
            </div>
            {% endfor %}
            
            {% if job.final_answer %}
            <div class="event-card" style="border-color: var(--primary);">
              <div class="event-header">
                <span class="event-type">Final Synthesized Answer</span>
              </div>
              <div class="event-payload" style="color: var(--text); font-family: 'Inter', sans-serif; white-space: pre-wrap;">{{ job.final_answer }}</div>
            </div>
            {% endif %}

          </div>
        </div>

      </div>
    </div>

    <div class="input-wrapper">
      <div class="input-box">
        <input type="text" id="queryInput" placeholder="Message Mega AI..." onkeydown="if(event.key==='Enter') submitQuery()" />
        <button id="submitBtn" class="send-btn" onclick="submitQuery()">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>
        </button>
      </div>
      <div id="statusMsg"></div>
    </div>
  </main>

  <script>
    {% if job.status == 'running' %}
    setTimeout(() => window.location.reload(), 3000);
    {% endif %}
    
    // Auto-scroll to bottom of chat
    const container = document.getElementById('chatContainer');
    container.scrollTop = container.scrollHeight;

    // Syntax Highlighting
    document.querySelectorAll('code.json').forEach((block) => {
      let json = block.innerHTML;
      json = json.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
      block.innerHTML = json.replace(/("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\\s*:)?|\\b(true|false|null)\\b|-?\\d+(?:\\.\\d*)?(?:[eE][+\\-]?\\d+)?)/g, function (match) {
          let cls = 'number';
          if (/^"/.test(match)) {
              if (/:$/.test(match)) { cls = 'key'; } 
              else { cls = 'string'; }
          } else if (/true|false/.test(match)) { cls = 'boolean'; } 
          else if (/null/.test(match)) { cls = 'null'; }
          return '<span class="' + cls + '">' + match + '</span>';
      });
    });
  </script>
"""

def fetch_sidebar_jobs(cur):
    cur.execute("SELECT id, query, status, created_at FROM jobs ORDER BY created_at DESC LIMIT 50")
    return cur.fetchall()

@app.route("/")
def index():
    with get_conn() as conn:
        cur = conn.cursor()
        jobs = fetch_sidebar_jobs(cur)
        cur.execute("""
            SELECT
                count(*) as total,
                count(*) filter (where status = 'done') as done,
                count(*) filter (where status = 'running') as running,
                count(*) filter (where status = 'failed') as failed
            FROM jobs
        """)
        stats = cur.fetchone()
    return render_template_string(INDEX_TEMPLATE, jobs=jobs, stats=stats, current_job_id=None)

@app.route("/trace/<job_id>")
def trace(job_id):
    with get_conn() as conn:
        cur = conn.cursor()
        jobs = fetch_sidebar_jobs(cur)
        cur.execute("SELECT * FROM jobs WHERE id = %s", (job_id,))
        job = cur.fetchone()
        if not job: abort(404)
        cur.execute("SELECT * FROM trace_events WHERE job_id = %s ORDER BY seq ASC", (job_id,))
        events = cur.fetchall()
    return render_template_string(TRACE_TEMPLATE, job=job, events=events, jobs=jobs, current_job_id=job_id)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
