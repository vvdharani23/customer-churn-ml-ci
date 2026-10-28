
import json
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

METRICS_FILE = BASE_DIR / "retraining_metrics.json"
CANDIDATE_MODEL = BASE_DIR / "customer_churn_model_retrained.pkl"
APPROVED_MODEL = BASE_DIR / "customer_churn_model.pkl"

MIN_ACCURACY = 0.60

if not METRICS_FILE.exists():
    raise FileNotFoundError("retraining_metrics.json not found.")

if not CANDIDATE_MODEL.exists():
    raise FileNotFoundError("Retrained candidate model not found.")

with open(METRICS_FILE, "r", encoding="utf-8") as file:
    metrics = json.load(file)

accuracy = float(metrics["accuracy"])

print("Candidate accuracy:", round(accuracy, 4))
print("Required accuracy:", MIN_ACCURACY)

if accuracy < MIN_ACCURACY:
    print("QUALITY GATE: FAIL")
    print("Existing approved model remains unchanged.")
    raise SystemExit(1)

shutil.copy2(CANDIDATE_MODEL, APPROVED_MODEL)

print("QUALITY GATE: PASS")
print("Candidate promoted to:", APPROVED_MODEL.name)
