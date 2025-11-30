import os, json
from datetime import date, datetime
from google import genai

# API KEY from GitHub secret
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
assert os.getenv("GEMINI_API_KEY"), "GEMINI_API_KEY not set"

TODAY = str(date.today())
MONTH = TODAY[:7]

LOG_DIR = "data/logs"
MONTHLY_DIR = "reports/monthly"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MONTHLY_DIR, exist_ok=True)

# Load history
stats_file = "output/stats.json"
history = {}

if os.path.exists(stats_file):
    with open(stats_file) as f:
        history = json.load(f)

LAST_TASKS = list(history.values())[-15:]

PROMPT = f"""
You are an advanced AI mentor for Data Science, Machine Learning, and AI.

Recent tasks:
{LAST_TASKS}

Generate ONE lean, practical task for:
- pandas / numpy
- ML pipelines
- feature engineering
- model evaluation
- SQL analytics
- data visualization
- basic AI experimentation

Rules:
- Do not repeat past tasks
- Make it coding-oriented
- Should be doable in 45 minutes
- Avoid theory-only answers

Respond STRICTLY in JSON format:

{{
  "task": "...",
  "focus": "...",
  "dataset": "...",
  "hint": "..."
}}
"""

# AI REQUEST
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=PROMPT
)

raw = response.text.strip()

# Remove Markdown code fences if present
if raw.startswith("```"):
    raw = raw.split("```")[1]

raw = raw.replace("json", "").strip()

# Parse JSON safely
try:
    task = json.loads(raw)
except Exception:
    raise Exception("Gemini returned invalid JSON:\n" + raw)


DAILY = {**task, "date": TODAY, "timestamp": datetime.utcnow().isoformat() + "Z"}

# Write logs
with open(f"{LOG_DIR}/{TODAY}.json", "w") as f:
    json.dump(DAILY, f, indent=2)

with open("output/today.json", "w") as f:
    json.dump(DAILY, f, indent=2)

history[TODAY] = DAILY["task"]

with open("output/stats.json", "w") as f:
    json.dump(history, f, indent=2)

with open("output/streak.json", "w") as f:
    json.dump({"total_days": len(history), "last_updated": TODAY}, f, indent=2)

# Write Markdown
with open("output/task.md", "w") as f:
    f.write(
        f"# AI Daily Lab — {TODAY}\n\n"
        f"## Task\n{DAILY['task']}\n\n"
        f"## Focus\n{DAILY['focus']}\n\n"
        f"## Dataset\n{DAILY['dataset']}\n\n"
        f"## Hint\n{DAILY['hint']}\n"
    )

# Monthly Report
MONTH_FILE = f"{MONTHLY_DIR}/{MONTH}.json"
month = {}

if os.path.exists(MONTH_FILE):
    with open(MONTH_FILE) as f:
        month = json.load(f)

month[TODAY] = DAILY["task"]

with open(MONTH_FILE, "w") as f:
    json.dump(month, f, indent=2)

print("Gemini AI task generated successfully")
