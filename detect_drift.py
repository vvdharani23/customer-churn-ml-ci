
import pandas as pd
import json

REFERENCE_FILE = "customer_churn_raw_50_samples (1).csv"
CURRENT_FILE = "current_churn_data.csv"
# Load datasets
reference = pd.read_csv(REFERENCE_FILE)
current = pd.read_csv(CURRENT_FILE)

numeric_columns = [
    "Age",
    "TenureMonths",
    "MonthlyCharges",
    "TotalCharges",
    "SupportCalls",
    "SatisfactionScore"
]

results = {}

# Detect drift
for column in numeric_columns:
    reference_mean = reference[column].mean()
    current_mean = current[column].mean()

    if reference_mean == 0:
        drift_percent = 0 if current_mean == 0 else 100
    else:
        drift_percent = abs(
            (current_mean - reference_mean) / reference_mean
        ) * 100

    results[column] = {
        "reference_mean": round(float(reference_mean), 3),
        "current_mean": round(float(current_mean), 3),
        "drift_percent": round(float(drift_percent), 2),
        "drift_detected": bool(drift_percent > 10)
    }

# Create report
report = {
    "reference_records": int(len(reference)),
    "current_records": int(len(current)),
    "drift_threshold_percent": 10,
    "results": results,
    "drift_detected": any(
        item["drift_detected"] for item in results.values()
    )
}

# Save report
with open("drift_report.json", "w") as file:
    json.dump(report, file, indent=4)

# Display report
print(json.dumps(report, indent=4))
print("\nDrift report saved to drift_report.json")
