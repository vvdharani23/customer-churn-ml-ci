import json
import joblib
import pandas as pd


def test_model_file():
    model = joblib.load("customer_churn_model.pkl")
    assert model is not None


def test_metrics_file():
    with open("metrics.json", "r") as file:
        metrics = json.load(file)

    assert "accuracy" in metrics
    assert metrics["accuracy"] >= 0.9

def test_dataset():
    data = pd.read_csv("customer_churn_raw_50_samples (1).csv")

    assert len(data) > 0
    assert "Churn" in data.columns
