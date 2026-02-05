"""
preprocess.py
Load merged_viirs.csv -> clean -> feature-engineer -> save processed_viirs.csv
Adjust column names mapping if your CSV uses different names.
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("../data").resolve()
INFILE = DATA_DIR / "merged_viirs.csv"
OUTFILE = DATA_DIR / "processed_viirs.csv"

# Common VIIRS column names expected: latitude, longitude, acq_date, acq_time, brightness, frp, confidence, daynight
# If your CSV uses different names, update the 'COLS' mapping below.
COLS = {
    "lat": ["latitude", "LAT", "LATITUDE"],
    "lon": ["longitude", "LON", "LONGITUDE"],
    "date": ["acq_date", "acqdate", "acqDate"],
    "time": ["acq_time", "acqtime", "acqTime"],
    "brightness": ["brightness", "bright_ti4", "bright_ti5"],
    "frp": ["frp", "FRP"],
    "confidence": ["confidence", "CONFIDENCE"],
    "daynight": ["daynight", "day_night"]
}

def find_col(df, names):
    for n in names:
        if n in df.columns:
            return n
    return None

def standardize_columns(df: pd.DataFrame):
    col_map = {}
    for key, candidates in COLS.items():
        found = find_col(df, candidates)
        if found:
            col_map[found] = key
    df = df.rename(columns=col_map)
    return df

def load_and_standardize(path=INFILE):
    print(f"Loading {path} ...")
    df = pd.read_csv(path)
    df = standardize_columns(df)
    print("Columns after standardization:", list(df.columns))
    return df

def clean(df: pd.DataFrame):
    # keep only rows with lat/lon and date
    df = df.dropna(subset=["lat", "lon", "date"])
    # Convert numeric columns
    for num in ["brightness", "frp"]:
        if num in df.columns:
            df[num] = pd.to_numeric(df[num], errors="coerce")

    # Parse date/time
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "time" in df.columns:
        # acq_time is often integer like 1234 -> convert to HHMM
        df["time_str"] = df["time"].astype(str).str.zfill(4)
        df["hour"] = pd.to_numeric(df["time_str"].str[:2], errors="coerce").fillna(0).astype(int)
    else:
        df["hour"] = 0

    # Remove rows with invalid coordinates
    df = df[(df["lat"].between(-90, 90)) & (df["lon"].between(-180, 180))]
    df = df.reset_index(drop=True)
    print(f"After cleaning: {df.shape}")
    return df

def feature_engineer(df: pd.DataFrame):
    # time features
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["doy"] = df["date"].dt.dayofyear

    # season (simple)
    df["season"] = df["month"].apply(lambda m: "winter" if m in [12,1,2] else
                                              "spring" if m in [3,4,5] else
                                              "summer" if m in [6,7,8] else
                                              "autumn")
    # day/night encoding
    if "daynight" in df.columns:
        df["is_day"] = df["daynight"].apply(lambda x: 1 if str(x).strip().lower().startswith("d") else 0)
    else:
        # fallback from hour
        df["is_day"] = df["hour"].between(6,18).astype(int)

    # FRP / brightness derived features
    if "frp" in df.columns:
        df["frp_filled"] = df["frp"].fillna(0)
        df["frp_high"] = (df["frp_filled"] >= df["frp_filled"].median()).astype(int)
    if "brightness" in df.columns:
        df["brightness_filled"] = df["brightness"].fillna(df["brightness"].median())
        df["bright_high"] = (df["brightness_filled"] >= df["brightness_filled"].median()).astype(int)

    # Optional: lat/lon buckets for coarse spatial features (grid of 0.5 degree)
    df["lat_bin"] = (df["lat"] * 2).round(0) / 2.0
    df["lon_bin"] = (df["lon"] * 2).round(0) / 2.0

    return df

def create_label(df: pd.DataFrame):
    """
    Create a binary label for 'high risk' using FRP or brightness.
    By default: high risk = frp >= median OR brightness >= median
    You can change this rule later.
    """
    if "frp_filled" in df.columns and df["frp_filled"].sum() > 0:
        frp_med = df["frp_filled"].median()
        df["label"] = (df["frp_filled"] >= frp_med).astype(int)
    elif "brightness_filled" in df.columns:
        b_med = df["brightness_filled"].median()
        df["label"] = (df["brightness_filled"] >= b_med).astype(int)
    else:
        # fallback: label all as 1 (should not happen)
        df["label"] = 1
    return df

def run():
    df = load_and_standardize()
    df = clean(df)
    df = feature_engineer(df)
    df = create_label(df)
    OUTFILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTFILE, index=False)
    print(f"Processed data saved to: {OUTFILE} (shape: {df.shape})")

if __name__ == "__main__":
    run()
