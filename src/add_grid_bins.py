# src/add_grid_bins.py
"""
Add lat_bin and lon_bin columns to processed_viirs.csv in a chunked, memory-safe way.
Writes output to data/processed_viirs_with_bins.csv
"""
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
INFILE = DATA_DIR / "processed_viirs.csv"
OUTFILE = DATA_DIR / "processed_viirs_with_bins.csv"
CHUNK_SIZE = 200000

def process_chunk(chunk):
    # Ensure lat/lon exist
    if not {"lat", "lon"}.issubset(chunk.columns):
        raise KeyError("lat/lon columns are required in processed_viirs.csv")

    # compute bins: 0.5 degree grid (adjust if you prefer different)
    chunk["lat_bin"] = (chunk["lat"] * 2).round(0) / 2.0
    chunk["lon_bin"] = (chunk["lon"] * 2).round(0) / 2.0

    # reorder to include new bins (keep all columns)
    return chunk

def run():
    if not INFILE.exists():
        raise FileNotFoundError(f"Input not found: {INFILE}")

    first = True
    total = 0
    reader = pd.read_csv(INFILE, chunksize=CHUNK_SIZE, parse_dates=["date"])
    for i, chunk in enumerate(reader):
        out = process_chunk(chunk)
        if first:
            out.to_csv(OUTFILE, index=False, mode="w")
            first = False
        else:
            out.to_csv(OUTFILE, index=False, header=False, mode="a")
        total += len(out)
        if (i+1) % 5 == 0:
            print(f"Processed {(i+1)*CHUNK_SIZE} input rows, written total {total}")
    print("Done. Wrote:", OUTFILE, " Total rows written:", total)

if __name__ == "__main__":
    run()
