import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib
import os

# Load dataset
df = pd.read_csv("data/tourism.csv")

print("Columns in dataset:", df.columns.tolist())

# Forward fill missing values
df = df.ffill()

# Automatically detect target column
if "ProdTaken" in df.columns:
    target_col = "ProdTaken"
elif "prod_taken" in df.columns:
    target_col = "prod_taken"
else:
    raise Exception("Target column not found in dataset")

# Remove ID column safely if present
if "CustomerID" in df.columns:
    df = df.drop("CustomerID", axis=1)
if "customer_id" in df.columns:
    df = df.drop("customer_id", axis=1)

X = df.drop(target_col, axis=1)
y = df[target_col]

# Encode categorical variables
X = pd.get_dummies(X)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = XGBClassifier(eval_metric="logloss")
model.fit(X_train, y_train)

# Save model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/model.pkl")

print("Training complete. Model saved.")
