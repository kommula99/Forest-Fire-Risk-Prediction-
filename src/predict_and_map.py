"""
predict_and_map.py
Load the trained model & processed data, predict risk scores and create a folium map.
Output: maps/fire_risk_map.html
"""
import pandas as pd
import joblib
from pathlib import Path
import folium
import numpy as np

DATA_FILE = Path("../data/processed_viirs.csv").resolve()
MODEL_FILE = Path("../models/rf_viirs.pkl").resolve()
OUT_MAP = Path("../maps/fire_risk_map.html").resolve()

def load_resources():
    if not DATA_FILE.exists():
        raise FileNotFoundError("Processed data not found. Run preprocess first.")
    if not MODEL_FILE.exists():
        raise FileNotFoundError("Model file not found. Run train_model first.")
    df = pd.read_csv(DATA_FILE)
    model = joblib.load(MODEL_FILE)
    return df, model

def prepare_model_input(df):
    X = df.drop(columns=["label"], errors="ignore")
    X = pd.get_dummies(X, columns=[c for c in ["season"] if c in X.columns], drop_first=True)
    X = X.fillna(0)
    return X

def predict_risk(df, model):
    X = prepare_model_input(df)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:,1]
    else:
        # fallback using predict
        probs = model.predict(X)
    df["risk_score"] = probs
    return df

def generate_map(df, sample_points=5000):
    # To avoid super heavy maps, sample if dataset is large
    if len(df) > sample_points:
        df_map = df.sample(n=sample_points, random_state=42)
    else:
        df_map = df

    center = [df_map["lat"].mean(), df_map["lon"].mean()]
    m = folium.Map(location=center, zoom_start=4)

    # Normalize risk
    min_r, max_r = df_map["risk_score"].min(), df_map["risk_score"].max()
    rng = max_r - min_r if max_r != min_r else 1.0

    for _, row in df_map.iterrows():
        lat = row["lat"]
        lon = row["lon"]
        score = (row["risk_score"] - min_r) / rng
        if score <= 0.33:
            color = "green"
            radius = 2
        elif score <= 0.66:
            color = "orange"
            radius = 3
        else:
            color = "red"
            radius = 4

        folium.CircleMarker(
            [lat, lon],
            radius=radius,
            color=color,
            fill=True,
            fill_opacity=0.6,
            popup=f"Risk: {row['risk_score']:.3f}"
        ).add_to(m)

    OUT_MAP.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(OUT_MAP))
    print(f"Saved map to {OUT_MAP}")

def main():
    df, model = load_resources()
    df = predict_risk(df, model)
    generate_map(df)

if __name__ == "__main__":
    main()
