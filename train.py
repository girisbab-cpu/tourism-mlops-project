import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier


# Load dataset
df = pd.read_csv("data/tourism.csv")

# Fill missing values
df = df.ffill().bfill()

# 🔥 Use LAST column as target (most ML datasets follow this)
target_col = df.columns[-1]

y = df[target_col]
X = df.drop(columns=[target_col])

# Remove obvious ID columns if present
for col in X.columns:
    if "id" in col.lower():
        X = X.drop(columns=[col])

# One-hot encode categorical variables
X = pd.get_dummies(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    random_state=42,
    eval_metric="logloss",
    use_label_encoder=False
)

model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
print("Accuracy:", acc)

# Save model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/model.pkl")

print("Training complete. Model saved.")
