
import pandas as pd
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

DATA_FILE = "current_churn_data.csv"

data = pd.read_csv(DATA_FILE)

# Select features and target
X = data.drop(columns=["Churn", "CustomerID"])
y = data["Churn"].map({"No": 0, "Yes": 1})

numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()
categorical_columns = X.select_dtypes(exclude=["number"]).columns.tolist()

# Preprocess numeric and categorical data
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_transformer, numeric_columns),
    ("categorical", categorical_transformer, categorical_columns)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

# Train and evaluate
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

# Save the newly trained model separately
joblib.dump(model, "customer_churn_model_retrained.pkl")

metrics = {
    "accuracy": float(accuracy),
    "training_records": int(len(X_train)),
    "testing_records": int(len(X_test)),
    "model_file": "customer_churn_model_retrained.pkl"
}

with open("retraining_metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

print("Model retraining completed successfully!")
print(json.dumps(metrics, indent=4))
