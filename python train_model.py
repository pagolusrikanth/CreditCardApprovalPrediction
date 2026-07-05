import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# -------------------------------
# Load Dataset
# -------------------------------

df = pd.read_csv("dataset/clean_dataset.csv")

print("Dataset Loaded Successfully")
print(df.head())

# -------------------------------
# Encode Categorical Features
# -------------------------------

categorical_columns = [
    "Industry",
    "Ethnicity",
    "Citizen"
]

encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# -------------------------------
# Features & Target
# -------------------------------

X = df.drop("Approved", axis=1)
y = df["Approved"]          # Already 0 and 1

# -------------------------------
# Train Test Split
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# -------------------------------
# Random Forest Model
# -------------------------------

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# -------------------------------
# Evaluation
# -------------------------------

y_pred = model.predict(X_test)

print("\nAccuracy")
print(accuracy_score(y_test, y_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))

# -------------------------------
# Save Model
# -------------------------------

joblib.dump(model, "models/credit_card_model.pkl")
joblib.dump(encoders, "models/encoders.pkl")

print("\nModel Saved Successfully")
print("Encoders Saved Successfully")

# -------------------------------
# Feature Importance
# -------------------------------

importance = pd.DataFrame(
    {
        "Feature": X.columns,
        "Importance": model.feature_importances_,
    }
)

importance = importance.sort_values(
    by="Importance",
    ascending=False,
)

print("\nFeature Importance")
print(importance)