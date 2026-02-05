print("SCRIPT STARTED")

import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import joblib, json

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed_with_future_label.csv"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

MODEL = MODELS / "rf_quick.pkl"
METRICS = MODELS / "metrics_quick.json"

print("Loading data...")
df = pd.read_csv(DATA, parse_dates=["date"])
df["year"] = df["date"].dt.year

train = df[df.year == 2023].sample(150_000, random_state=42)
test = df[df.year == 2024].sample(50_000, random_state=42)

drop = ["frp", "brightness", "frp_high", "bright_high", "date"]
train = train.drop(columns=[c for c in drop if c in train.columns])
test = test.drop(columns=[c for c in drop if c in test.columns])

y_train = train.future_label
X_train = pd.get_dummies(train.drop(columns="future_label")).fillna(0)

y_test = test.future_label
X_test = pd.get_dummies(test.drop(columns="future_label")).fillna(0)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

print("Training model...")
model = RandomForestClassifier(n_estimators=50, n_jobs=2, random_state=42)
model.fit(X_train, y_train)

probs = model.predict_proba(X_test)[:, 1]
roc = roc_auc_score(y_test, probs)

joblib.dump(model, MODEL)
json.dump({"roc_auc": roc}, open(METRICS, "w"), indent=2)

print("DONE ✅ ROC AUC:", roc)
