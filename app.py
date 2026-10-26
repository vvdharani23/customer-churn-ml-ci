
import logging
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)

# Configure application logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

MODEL_PATH = Path("customer_churn_model.pkl")

FEATURES = [
    "Age",
    "TenureMonths",
    "MonthlyCharges",
    "TotalCharges",
    "Contract",
    "InternetService",
    "PaymentMethod",
    "SupportCalls",
    "SatisfactionScore"
]


@app.before_request
def log_request():
    logger.info(
        "Request received: method=%s path=%s",
        request.method,
        request.path
    )


@app.after_request
def log_response(response):
    logger.info(
        "Request completed: method=%s path=%s status=%s",
        request.method,
        request.path,
        response.status_code
    )
    return response


@app.errorhandler(Exception)
def handle_unexpected_error(error):
    logger.exception("Unexpected application error")
    return jsonify({
        "error": "An internal server error occurred"
    }), 500


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "customer_churn_model.pkl was not found. "
            "Run the training pipeline first."
        )

    return joblib.load(MODEL_PATH)


@app.get("/")
def health_check():
    return jsonify({
        "status": "ok",
        "service": "customer-churn-prediction"
    })


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)

    if not data:
        logger.warning("Prediction request has a missing or invalid JSON body")
        return jsonify({
            "error": "JSON request body is required"
        }), 400

    missing_fields = [
        feature for feature in FEATURES
        if feature not in data
    ]

    if missing_fields:
        logger.warning(
            "Prediction request missing required fields: %s",
            missing_fields
        )
        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing_fields
        }), 400

    try:
        sample = pd.DataFrame([{
            feature: data[feature]
            for feature in FEATURES
        }])

        model = load_model()
        prediction_code = int(model.predict(sample)[0])
        prediction = "Yes" if prediction_code == 1 else "No"

        logger.info("Customer churn prediction completed successfully")

        return jsonify({
            "prediction": prediction,
            "prediction_code": prediction_code
        })

    except Exception:
        logger.exception("Customer churn prediction failed")
        return jsonify({
            "error": "Prediction could not be completed"
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
