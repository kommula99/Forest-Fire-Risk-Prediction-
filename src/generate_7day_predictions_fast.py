import pandas as pd
import joblib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed_with_future_label.csv"
MODEL = ROOT / "models" / "rf_quick.pkl"
OUT = ROOT / "data" / "risk_7days.csv"

print("Loading model...")
model = joblib.load(MODEL)

print("Loading data...")
df = pd.read_csv(DATA)

# SMALL SAMPLE for speed (important)
df = df.sample(5000, random_state=42).reset_index(drop=True)

drop_cols = ["frp", "brightness", "frp_high", "bright_high", "date", "future_label"]
X = pd.get_dummies(df.drop(columns=[c for c in drop_cols if c in df.columns]))
X = X.reindex(columns=model.feature_names_in_, fill_value=0)

print("Predicting risk...")
probs = model.predict_proba(X)[:, 1]

# Fake 7-day evolution (acceptable for demo)
for d in range(1, 8):
    df[f"day_{d}"] = (probs * (0.9 ** (d-1))).clip(0, 1)

df[["lat", "lon"] + [f"day_{d}" for d in range(1,8)]].to_csv(OUT, index=False)
print("Saved:", OUT)