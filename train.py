
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# Load dataset
df = pd.read_csv("data/tourism.csv")

# Basic preprocessing
df = df.fillna(method="ffill")

X = df.drop(["CustomerID", "ProdTaken"], axis=1)
y = df["ProdTaken"]

# Simple encoding for safety
X = pd.get_dummies(X)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
model.fit(X_train, y_train)

# Save model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/model.pkl")

print("Training complete. Model saved.")
