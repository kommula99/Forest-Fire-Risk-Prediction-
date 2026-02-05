import pandas as pd
import numpy as np
from pathlib import Path

# file paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
INFILE = DATA_DIR / "merged_viirs.csv"
OUTFILE = DATA_DIR / "processed_viirs.csv"

CHUNK_SIZE = 150000  # smaller chunk size = safer on MacBook Air RAM

# column name patterns
COLS = {
    "lat": ["latitude", "LAT", "Lat"],
    "lon": ["longitude", "LON", "Lon"],
    "date": ["acq_date", "acqDate", "acq_date_utc"],
    "time": ["acq_time", "acqTime"],
    "brightness": ["brightness", "bright_ti4", "bright_ti5"],
    "frp": ["frp", "FRP"],
    "daynight": ["daynight", "day_night"]
}

def detect_columns(header):
    mapping = {}
    for key, options in COLS.items():
        for col in options:
            if col in header:
                mapping[key] = col
                break
    return mapping

def pass_1_get_medians(cols_map):
    print("STEP 1: Computing medians for FRP and Brightness...")
    frp_vals = []
    bright_vals = []
    
    for chunk in pd.read_csv(INFILE, chunksize=CHUNK_SIZE):
        if "frp" in cols_map:
            frp_vals.append(chunk[cols_map["frp"]].dropna().astype(float).values)
        if "brightness" in cols_map:
            bright_vals.append(chunk[cols_map["brightness"]].dropna().astype(float).values)

    frp_med = float(np.median(np.concatenate(frp_vals))) if frp_vals else 0
    bright_med = float(np.median(np.concatenate(bright_vals))) if bright_vals else 0

    print(f"Computed FRP median = {frp_med}")
    print(f"Computed Brightness median = {bright_med}")
    return frp_med, bright_med


def process_chunk(chunk, cols_map, frp_med, bright_med):
    # rename to standard names
    rename_map = {real: key for key, real in cols_map.items()}
    chunk = chunk.rename(columns=rename_map)

    # parse date / time
    chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce")
    if "time" in chunk.columns:
        chunk["time"] = chunk["time"].astype(str).str.zfill(4)
        chunk["hour"] = chunk["time"].str[:2].astype(int)
    else:
        chunk["hour"] = 0

    # numeric cleanup
    chunk["frp"] = pd.to_numeric(chunk.get("frp", 0), errors="coerce").fillna(0)
    chunk["brightness"] = pd.to_numeric(chunk.get("brightness", 0), errors="coerce").fillna(bright_med)

    # remove invalid coordinates
    chunk = chunk[
        chunk["lat"].between(-90, 90) &
        chunk["lon"].between(-180, 180)
    ]

    # feature engineering
    chunk["year"] = chunk["date"].dt.year
    chunk["month"] = chunk["date"].dt.month
    chunk["day"] = chunk["date"].dt.day
    chunk["doy"] = chunk["date"].dt.dayofyear

    chunk["season"] = chunk["month"].apply(
        lambda m: "winter" if m in [12,1,2] else
                  "spring" if m in [3,4,5] else
                  "summer" if m in [6,7,8] else
                  "autumn"
    )

    chunk["is_day"] = chunk.get("daynight", "").astype(str).str.startswith("D").astype(int)

    chunk["frp_high"] = (chunk["frp"] >= frp_med).astype(int)
    chunk["bright_high"] = (chunk["brightness"] >= bright_med).astype(int)

    chunk["label"] = ((chunk["frp_high"] == 1) | (chunk["bright_high"] == 1)).astype(int)

    # final output columns
    return chunk[[
        "lat","lon","date","hour","year","month","day","doy","season",
        "is_day","frp","brightness","frp_high","bright_high","label"
    ]]


def run():
    print("Detecting columns...")
    header = pd.read_csv(INFILE, nrows=0).columns.tolist()
    cols_map = detect_columns(header)
    print("Column mapping:", cols_map)

    print("Running Pass 1 (median calculations)...")
    frp_med, bright_med = pass_1_get_medians(cols_map)

    print("Running Pass 2 (streaming + writing output)...")
    first = True
    total = 0

    for chunk in pd.read_csv(INFILE, chunksize=CHUNK_SIZE):
        df = process_chunk(chunk, cols_map, frp_med, bright_med)
        if df.empty:
            continue

        if first:
            df.to_csv(OUTFILE, index=False, mode="w")
            first = False
        else:
            df.to_csv(OUTFILE, index=False, mode="a", header=False)

        total += len(df)
        print(f"Written rows: {total}")

    print("DONE! Processed file saved to:", OUTFILE)


if __name__ == "__main__":
    run()
