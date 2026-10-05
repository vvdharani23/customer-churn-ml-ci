import json

with open("metrics.json", "r") as file:
    metrics = json.load(file)

accuracy = metrics["accuracy"]

if accuracy >= 0.60:
    print("Quality Gate PASSED")
else:
    print("Quality Gate FAILED")
    exit(1)
