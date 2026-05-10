import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

# Use the full DATABASE_URL for robustness
db_url = os.environ.get("DATABASE_URL")

conn = psycopg2.connect(db_url)
cur = conn.cursor()

# 1. Check if final_answer exists
cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'jobs'")
cols = [c[0] for c in cur.fetchall()]
print(f"Columns in 'jobs': {cols}")

if 'final_answer' not in cols:
    print("Adding 'final_answer' column...")
    cur.execute("ALTER TABLE jobs ADD COLUMN final_answer TEXT")
    conn.commit()
    print("Done.")
else:
    print("'final_answer' already exists.")

cur.close()
conn.close()
