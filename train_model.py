import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix


# Load dataset
data = pd.read_csv("customer_churn_raw_50_samples (1).csv")

# Remove duplicate rows
data = data.drop_duplicates()

# Remove unnecessary columns
if "CustomerID" in data.columns:
    data = data.drop("CustomerID", axis=1)

# Separate input and target
X = data.drop("Churn", axis=1)
y = data["Churn"].map({"No": 0, "Yes": 1})

# Numerical and categorical columns
numeric_columns = [
    "Age",
    "TenureMonths",
    "MonthlyCharges",
    "TotalCharges",
    "SupportCalls",
    "SatisfactionScore"
]

categorical_columns = [
    "Contract",
    "InternetService",
    "PaymentMethod"
]

# Preprocessing
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_columns),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns)
])

# SVM pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", SVC(kernel="linear"))
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Train
print("Training customer churn model...")
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, predictions)
matrix = confusion_matrix(y_test, predictions)

print("Training completed")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(matrix)

# Save model
joblib.dump(model, "customer_churn_model.pkl")

# Save metrics
metrics = {
    "accuracy": float(accuracy),
    "test_records": int(len(X_test))
}

with open("metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

print("Model saved successfully")
print("Metrics saved successfully")
