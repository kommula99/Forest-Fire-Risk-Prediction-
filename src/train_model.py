"""
train_model.py (robust version)
- Encodes any object/categorical columns with get_dummies
- Samples data for memory-safe training
- Saves model + metrics + feature importance
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import json
import matplotlib.pyplot as plt

# Paths
DATA_FILE = Path("../data/processed_viirs.csv").resolve()
MODEL_DIR = Path("../models").resolve()
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_FILE = MODEL_DIR / "rf_viirs.pkl"
FEATURE_IMP_FILE = MODEL_DIR / "feature_importance.png"
METRICS_FILE = MODEL_DIR / "metrics.json"

# Training settings (reduce for quick runs)
SAMPLE_ROWS = 500000        # number of rows to sample for training; set None to use all (not recommended on laptop)
N_ESTIMATORS = 100         # number of trees (reduce to speed up)
RANDOM_STATE = 42

def load_data():
    print(f"Loading processed data from {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    print(f"Data shape: {df.shape}")
    return df

def prepare_features(df):
    # Ensure label exists
    if "label" not in df.columns:
        raise KeyError("Label column not found in processed data. Run preprocessing to create labels.")

    # Optionally sample rows to avoid OOM on laptop
    if SAMPLE_ROWS is not None and len(df) > SAMPLE_ROWS:
        print(f"Sampling {SAMPLE_ROWS} rows from {len(df)} for training...")
        df = df.sample(n=SAMPLE_ROWS, random_state=RANDOM_STATE).reset_index(drop=True)
    else:
        print("Using full dataset for training (may be slow).")

    # Drop columns we do not want as features if present
    drop_cols = ["date", "time", "time_str", "acq_date", "acq_time"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Separate target
    y = df["label"].astype(int)
    X = df.drop(columns=["label"])

    # Identify non-numeric/object columns and encode them
    obj_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if obj_cols:
        print("Categorical/object columns detected and will be one-hot encoded:", obj_cols)
        X = pd.get_dummies(X, columns=obj_cols, drop_first=True)
    else:
        print("No object columns detected.")

    # Ensure numeric matrix and fill NaNs
    X = X.fillna(0)

    # Final sanity: convert any remaining non-numeric columns to numeric if possible
    for col in X.columns:
        if X[col].dtype == "object":
            try:
                X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)
            except Exception:
                X[col] = 0

    return X, y

def train_and_evaluate():
    df = load_data()
    X, y = prepare_features(df)
    print("Feature matrix shape:", X.shape)

    # Use stratify if both classes exist
    stratify_param = y if len(np.unique(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=stratify_param
    )

    clf = RandomForestClassifier(n_estimators=N_ESTIMATORS, n_jobs=-1, random_state=RANDOM_STATE)
    print("Training RandomForest (this may take some time)...")
    clf.fit(X_train, y_train)

    print("Evaluating...")
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None
    report = classification_report(y_test, preds, output_dict=True)
    cm = confusion_matrix(y_test, preds).tolist()
    roc = float(roc_auc_score(y_test, probs)) if probs is not None else None

    metrics = {
        "classification_report": report,
        "confusion_matrix": cm,
        "roc_auc": roc,
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1])
    }

    print("Classification report:\n", classification_report(y_test, preds))
    print("Confusion matrix:\n", cm)
    print("ROC AUC:", roc)

    # Save metrics
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved metrics to", METRICS_FILE)

    # Save model
    joblib.dump(clf, MODEL_FILE)
    print("Saved model to", MODEL_FILE)

    # Feature importance plot (top 30)
    try:
        fi = clf.feature_importances_
        feat_names = X.columns
        idx = np.argsort(fi)[-30:]
        plt.figure(figsize=(8,10))
        plt.barh(feat_names[idx], fi[idx])
        plt.title("Top feature importances")
        plt.tight_layout()
        plt.savefig(FEATURE_IMP_FILE)
        print("Saved feature importance plot to", FEATURE_IMP_FILE)
    except Exception as e:
        print("Could not save feature importance plot:", e)

if __name__ == "__main__":
    train_and_evaluate()
