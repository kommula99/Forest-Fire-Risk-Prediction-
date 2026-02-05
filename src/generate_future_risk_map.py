import pandas as pd
import numpy as np
from pathlib import Path
import folium
import joblib

print("MAP SCRIPT STARTED")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed_with_future_label.csv"
MODEL = ROOT / "models" / "rf_quick.pkl"
MAPS = ROOT / "maps"
MAPS.mkdir(exist_ok=True)

OUT_MAP = MAPS / "future_risk_map.html"

print("Loading model...")
model = joblib.load(MODEL)

print("Loading data...")
df = pd.read_csv(DATA, parse_dates=["date"])

# sample for visualization
df = df.sample(10_000, random_state=42).reset_index(drop=True)

# same preprocessing as training
drop = ["frp", "brightness", "frp_high", "bright_high"]
df = df.drop(columns=[c for c in drop if c in df.columns], errors="ignore")

X = pd.get_dummies(df.drop(columns=["future_label", "date"])).fillna(0)
X = X.reindex(columns=model.feature_names_in_, fill_value=0)

print("Predicting future fire risk...")
df["risk_prob"] = model.predict_proba(X)[:, 1]

# --------------------------------------------------
# MAP
# --------------------------------------------------

m = folium.Map(location=[20, 78], zoom_start=4, tiles=None)

folium.TileLayer(
    tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
    attr="© OpenTopoMap contributors",
    name="Terrain",
    control=False
).add_to(m)


def risk_color(p):
    if p >= 0.66:
        return "red"
    elif p >= 0.33:
        return "orange"
    else:
        return "green"


print("Adding points to map...")
for _, row in df.iterrows():
    # STEP A IMPROVEMENTS 👇
    radius = 2 + row["risk_prob"] * 6      # size by risk
    opacity = 0.3 + row["risk_prob"] * 0.6 # intensity by risk

    folium.CircleMarker(
        location=[row["lat"], row["lon"]],
        radius=radius,
        color=risk_color(row["risk_prob"]),
        fill=True,
        fill_opacity=opacity,
        popup=f"""
        <b>Fire Risk Probability</b>: {row['risk_prob']:.2f}<br>
        Latitude: {row['lat']}<br>
        Longitude: {row['lon']}
        """
    ).add_to(m)

m.save(OUT_MAP)
print("MAP GENERATED:", OUT_MAP)
