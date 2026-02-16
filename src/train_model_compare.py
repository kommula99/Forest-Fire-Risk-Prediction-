"""
train_model_compare.py
- Trains 3 models: Logistic Regression, Random Forest, Gradient Boosting
- Uses FUTURE_LABEL dataset (no leakage)
- Encodes categorical columns
- Samples data for memory-safe training
- Compares ROC-AUC
- Saves comparison metrics
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import json

# ---------------- PATHS ---------------- #

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed_with_future_label.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRICS_FILE = MODEL_DIR / "model_comparison_metrics.json"

# ---------------- SETTINGS ---------------- #

SAMPLE_ROWS = 150000   # keep moderate to avoid laptop overheating
RANDOM_STATE = 42

# ---------------- LOAD DATA ---------------- #

def load_data():
    print(f"Loading processed data from {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    print(f"Data shape: {df.shape}")
    return df

# ---------------- PREPARE FEATURES ---------------- #

def prepare_features(df):

    if "future_label" not in df.columns:
        raise KeyError("future_label column not found. Check dataset.")

    # Sample to reduce memory
    if SAMPLE_ROWS is not None and len(df) > SAMPLE_ROWS:
        print(f"Sampling {SAMPLE_ROWS} rows from {len(df)} for training...")
        df = df.sample(n=SAMPLE_ROWS, random_state=RANDOM_STATE).reset_index(drop=True)

    # Separate target FIRST (important)
    y = df["future_label"].astype(int)

    # Drop columns that cause leakage or not useful
    drop_cols = [
        "future_label",
        "frp",
        "brightness",
        "frp_high",
        "bright_high",
        "date"
    ]

    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Encode categorical columns
    obj_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if obj_cols:
        print("Encoding categorical columns:", obj_cols)
        X = pd.get_dummies(X, columns=obj_cols, drop_first=True)

    X = X.fillna(0)

    return X, y

# ---------------- TRAIN & COMPARE ---------------- #

def train_and_compare():

    df = load_data()
    X, y = prepare_features(df)

    print("Feature matrix shape:", X.shape)

    stratify_param = y if len(np.unique(y)) > 1 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=stratify_param
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=500, solver="liblinear"),
        "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE)
    }

    results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)

        probs = model.predict_proba(X_test)[:, 1]
        roc = float(roc_auc_score(y_test, probs))

        results[name] = roc
        print(f"{name} ROC-AUC: {roc:.4f}")

    print("\n" + "="*40)
    print("FINAL ROC-AUC COMPARISON")
    print("="*40)

    for name, score in results.items():
        print(f"{name:<20} : {score:.4f}")

    best_model = max(results, key=results.get)
    print("\nBest Model:", best_model)

    # Save results
    with open(METRICS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved comparison metrics to", METRICS_FILE)

# ---------------- MAIN ---------------- #

if __name__ == "__main__":
    train_and_compare()
