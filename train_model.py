
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# Load datasets
train = pd.read_csv("dataset/train.csv")
validation = pd.read_csv("dataset/validation.csv")
test = pd.read_csv("dataset/test.csv")

# Separate features and target
X_train = train.drop("PlacementStatus", axis=1)
y_train = train["PlacementStatus"]

X_val = validation.drop("PlacementStatus", axis=1)
y_val = validation["PlacementStatus"]

X_test = test.drop("PlacementStatus", axis=1)
y_test = test["PlacementStatus"]

# Identify categorical and numerical columns
categorical = X_train.select_dtypes(include=["object"]).columns
numerical = X_train.select_dtypes(exclude=["object"]).columns

# Preprocess categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", "passthrough", numerical)
    ]
)

# Create Random Forest model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ))
])

# Train model
print("Training model...")
model.fit(X_train, y_train)

# Validation evaluation
val_predictions = model.predict(X_val)

print("\nVALIDATION RESULTS")
print("Accuracy:", accuracy_score(y_val, val_predictions))
print(classification_report(y_val, val_predictions, zero_division=0))

# Final test evaluation
test_predictions = model.predict(X_test)

print("\nFINAL TEST RESULTS")
print("Accuracy:", accuracy_score(y_test, test_predictions))
print(classification_report(y_test, test_predictions, zero_division=0))

# Save trained model
joblib.dump(model, "placement_model.pkl")

print("\nModel trained and saved successfully!")
