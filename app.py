
import logging
import time
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request, Response
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

app = Flask(__name__)

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

# Prometheus metrics
REQUEST_COUNT = Counter(
    "churn_api_requests_total",
    "Total API requests",
    ["method", "endpoint", "status"]
)

REQUEST_LATENCY = Histogram(
    "churn_api_request_duration_seconds",
    "API request duration in seconds",
    ["endpoint"]
)

PREDICTION_COUNT = Counter(
    "churn_predictions_total",
    "Total successful churn predictions",
    ["prediction"]
)

ERROR_COUNT = Counter(
    "churn_api_errors_total",
    "Total API errors",
    ["endpoint", "status"]
)

MODEL_STATUS = Gauge(
    "churn_model_status",
    "Model availability: 1 available, 0 unavailable"
)

request_start_times = {}


@app.before_request
def before_request():
    request_start_times[id(request._get_current_object())] = time.time()
    logger.info(
        "Request received: method=%s path=%s",
        request.method,
        request.path
    )


@app.after_request
def after_request(response):
    endpoint = request.path
    status = str(response.status_code)

    REQUEST_COUNT.labels(
        request.method, endpoint, status
    ).inc()

    start = request_start_times.pop(
        id(request._get_current_object()), time.time()
    )
    REQUEST_LATENCY.labels(endpoint).observe(time.time() - start)

    if response.status_code >= 400:
        ERROR_COUNT.labels(endpoint, status).inc()

    logger.info(
        "Request completed: method=%s path=%s status=%s",
        request.method, endpoint, status
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
        MODEL_STATUS.set(0)
        raise FileNotFoundError(
            "customer_churn_model.pkl was not found. "
            "Run the training pipeline first."
        )

    model = joblib.load(MODEL_PATH)
    MODEL_STATUS.set(1)
    return model


@app.get("/")
def health_check():
    return jsonify({
        "status": "ok",
        "service": "customer-churn-prediction"
    })


@app.get("/metrics")
def metrics():
    return Response(
        generate_latest(),
        mimetype=CONTENT_TYPE_LATEST
    )


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

        PREDICTION_COUNT.labels(prediction).inc()
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
