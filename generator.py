import os, json, requests
from datetime import date, datetime

API_KEY = os.getenv("GEMINI_API_KEY")
assert API_KEY, "GEMINI_API_KEY not found"

TODAY = str(date.today())
MONTH = TODAY[:7]
LOG_DIR = "data/logs"
MONTHLY_DIR = "reports/monthly"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MONTHLY_DIR, exist_ok=True)

# Load history
history = {}
if os.path.exists("output/stats.json"):
    with open("output/stats.json") as f:
        history = json.load(f)

LAST_TASKS = list(history.values())[-15:]

PROMPT = f"""
You are an AI mentor for Data Science, Machine Learning and AI.

These are my last tasks:
{LAST_TASKS}

Generate ONE new LEAN but REAL-WORLD task focused on:
- Python data analysis
- Feature engineering
- Model training / evaluation
- SQL for analytics
- Pandas / NumPy
- ML algorithms
- AI experimentation

Rules:
- Do NOT repeat previous tasks
- Keep it small but meaningful
- Avoid theory-only tasks
- Prefer practical coding
- Make it doable in ~45 minutes

Respond ONLY in JSON:
{{ "task":"...", "focus":"...", "dataset":"...", "hint":"..." }}
"""

URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={API_KEY}"

res = requests.post(URL, json={"contents":[{"parts":[{"text":PROMPT}]}]})
res.raise_for_status()

raw = res.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
task = json.loads(raw)

DAILY = {
    **task,
    "date": TODAY,
    "timestamp": datetime.utcnow().isoformat() + "Z"
}

# Save daily logs
with open(f"{LOG_DIR}/{TODAY}.json", "w") as f:
    json.dump(DAILY, f, indent=2)

# Today output
with open("output/today.json", "w") as f:
    json.dump(DAILY, f, indent=2)

history[TODAY] = task["task"]
with open("output/stats.json", "w") as f:
    json.dump(history, f, indent=2)

# Streak
with open("output/streak.json", "w") as f:
    json.dump({"total_days": len(history), "last_updated": TODAY}, f, indent=2)

# Markdown output
with open("output/task.md", "w") as f:
    f.write(
        f"# 🤖 AI Daily Lab — {TODAY}\n\n"
        f"### ✅ Task\n{task['task']}\n\n"
        f"### 🧭 Focus\n{task['focus']}\n\n"
        f"### 📂 Dataset\n{task['dataset']}\n\n"
        f"### 💡 Hint\n{task['hint']}\n"
    )

# Monthly report
MONTH_FILE = f"{MONTHLY_DIR}/{MONTH}.json"
month = {}
if os.path.exists(MONTH_FILE):
    with open(MONTH_FILE) as f:
        month = json.load(f)

month[TODAY] = task["task"]
with open(MONTH_FILE, "w") as f:
    json.dump(month, f, indent=2)

print("✅ AI task generated successfully")
