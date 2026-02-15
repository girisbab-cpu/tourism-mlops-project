import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


def pick_target_column(df: pd.DataFrame) -> str:
    # 1) Prefer known target names (case-insensitive)
    known = {"prodtaken", "prod_taken", "target", "label", "y", "class"}
    lower_map = {c.lower(): c for c in df.columns}
    for k in known:
        if k in lower_map:
            return lower_map[k]

    # 2) Heuristic: pick a low-cardinality column (likely classification label)
    n = len(df)
    candidates = []
    for c in df.columns:
        nunique = df[c].nunique(dropna=True)
        if 1 < nunique <= min(20, max(2, n // 10)):
            candidates.append((nunique, c))

    # Prefer binary if present
    binary = [c for (u, c) in candidates if u == 2]
    if binary:
        return binary[0]

    if candidates:
        candidates.sort(key=lambda x: x[0])  # smallest nunique first
        return candidates[0][1]

    # 3) Fallback: last column
    return df.columns[-1]


def drop_id_like_columns(X: pd.DataFrame) -> pd.DataFrame:
    n = len(X)
    drop_cols = []
    for c in X.columns:
        cl = c.lower()
        nunique = X[c].nunique(dropna=True)
        # obvious id column name OR almost-unique column
        if "id" in cl or nunique >= 0.98 * n:
            drop_cols.append(c)
    return X.drop(columns=drop_cols, errors="ignore")


# ---------------- MAIN ----------------

df = pd.read_csv("data/tourism.csv")

# Drop common junk columns
df = df.drop(columns=[c for c in df.columns if c.lower().startswith("unnamed")], errors="ignore")

# Fill missing values
df = df.ffill().bfill()

# Pick target safely
target_col = pick_target_column(df)

y = df[target_col].copy()
X = df.drop(columns=[target_col], errors="ignore").copy()

# Remove ID-like columns
X = drop_id_like_columns(X)

# If any remaining missing (just in case)
X = X.ffill().bfill()

# Encode features
X = pd.get_dummies(X, drop_first=False)
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

# Encode target if needed (handles Yes/No, True/False, strings)
y = y.fillna(y.mode().iloc[0] if not y.mode().empty else y)
le = LabelEncoder()
if y.dtype == "object" or str(y.dtype).startswith("bool") or str(y.dtype).startswith("category"):
    y_enc = le.fit_transform(y.astype(str))
else:
    # still safe: if numeric but not int, cast
    y_enc = y.astype(int) if pd.api.types.is_integer_dtype(y) else y.astype(float)

# Ensure at least 2 classes
unique_classes = np.unique(y_enc)
if len(unique_classes) < 2:
    raise ValueError(f"Target '{target_col}' has <2 classes. Detected classes: {unique_classes}")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc if len(unique_classes) > 1 else None
)

# Train model (CPU-safe)
model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    eval_metric="logloss",
    tree_method="hist"
)

model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
print("Target column used:", target_col)
print("Accuracy:", acc)

# Save model + label encoder + columns (important for inference)
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/model.pkl")
joblib.dump(le, "model/label_encoder.pkl")
joblib.dump(list(X.columns), "model/feature_columns.pkl")

print("Training complete. Model saved.")
