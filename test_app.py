import unittest

from app import app


class TestCustomerChurnApplication(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()

    def test_health_endpoint(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.get_json()["status"],
            "ok"
        )

    def test_low_risk_customer_prediction(self):
        response = self.client.post(
            "/predict",
            json={
                "Age": 30,
                "TenureMonths": 36,
                "MonthlyCharges": 50,
                "TotalCharges": 1800,
                "Contract": "One year",
                "InternetService": "DSL",
                "PaymentMethod": "Bank transfer",
                "SupportCalls": 1,
                "SatisfactionScore": 5
            }
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
         response.get_json()["prediction"],
         "INVALID"
     )

    def test_high_risk_customer_prediction(self):
        response = self.client.post(
            "/predict",
            json={
                "Age": 65,
                "TenureMonths": 2,
                "MonthlyCharges": 95,
                "TotalCharges": 190,
                "Contract": "Month-to-month",
                "InternetService": "Fiber optic",
                "PaymentMethod": "Electronic check",
                "SupportCalls": 6,
                "SatisfactionScore": 1
            }
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn(
            response.get_json()["prediction"],
            ["Yes", "No"]
        )

    def test_missing_field_validation(self):
        response = self.client.post(
            "/predict",
            json={
                "Age": 30,
                "TenureMonths": 36
            }
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn(
            "missing_fields",
            response.get_json()
        )


if __name__ == "__main__":
    unittest.main()
