# src/create_future_label.py
"""
Create a future-risk label per grid cell (lat_bin, lon_bin) and date.
Label = 1 if the same grid has ANY fire in the next FUTURE_DAYS days (e.g., 7).
Two phases:
 - Phase A: extract unique (grid_id, date) rows from processed_viirs.csv (streamed)
 - Phase B: for each grid, compute label by checking if there is any date in (date+1 .. date+FUTURE_DAYS)
 - Phase C: merge labels back into processed_viirs.csv in chunks to produce processed_with_future_label.csv
"""
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PROC_IN = DATA_DIR / "processed_viirs.csv"
GRID_DATES_FILE = DATA_DIR / "unique_grid_dates.csv"
GRID_LABEL_FILE = DATA_DIR / "grid_date_labels.csv"
OUTFILE = DATA_DIR / "processed_with_future_label.csv"

CHUNK_SIZE = 200000
FUTURE_DAYS = 7  # you can change this to 7/14/30

def phase_a_extract_unique_grid_dates():
    print("Phase A: scanning processed_viirs.csv to build unique grid-date pairs...")
    cols = ["lat_bin", "lon_bin", "date"]
    reader = pd.read_csv(PROC_IN, usecols=lambda c: c in cols, parse_dates=["date"], chunksize=CHUNK_SIZE)
    # We'll write unique pairs to a file as we go
    first = True
    for i, chunk in enumerate(reader):
        # Ensure lat_bin/lon_bin exist
        if not {"lat_bin", "lon_bin", "date"}.issubset(chunk.columns):
            raise KeyError("lat_bin/lon_bin/date required in processed_viirs.csv (run preprocess with grid bins enabled).")
        # create grid_id and reduce
        chunk["grid_id"] = chunk["lat_bin"].astype(str) + "_" + chunk["lon_bin"].astype(str)
        chunk_small = chunk[["grid_id", "date"]].drop_duplicates()
        if first:
            chunk_small.to_csv(GRID_DATES_FILE, index=False, mode="w")
            first = False
        else:
            chunk_small.to_csv(GRID_DATES_FILE, index=False, mode="a", header=False)
        if (i+1) % 5 == 0:
            print(f"  scanned {(i+1)*CHUNK_SIZE} rows")
    print("Phase A done. Now deduplicating the collected grid-date rows (may take some time)...")
    # load the file, deduplicate and sort
    df = pd.read_csv(GRID_DATES_FILE, parse_dates=["date"])
    df = df.drop_duplicates().reset_index(drop=True)
    df = df.sort_values(["grid_id", "date"]).reset_index(drop=True)
    df.to_csv(GRID_DATES_FILE, index=False)
    print(f"Unique grid-date pairs saved to {GRID_DATES_FILE} (rows: {len(df)})")
    return df

def phase_b_compute_future_labels(df_grid_dates: pd.DataFrame):
    print("Phase B: computing future labels per grid_id...")
    # df_grid_dates has columns: grid_id, date
    out_rows = []
    # group by grid
    for grid, gdf in df_grid_dates.groupby("grid_id"):
        dates = np.array(gdf["date"].values, dtype="datetime64[D]")  # day precision
        n = len(dates)
        if n == 0:
            continue
        # for each index, find any future date within FUTURE_DAYS
        # we can use searchsorted
        for i, d in enumerate(dates):
            # first index strictly > d
            j = np.searchsorted(dates, d + np.timedelta64(1, 'D'), side='left')
            # last index <= d + FUTURE_DAYS
            k = np.searchsorted(dates, d + np.timedelta64(FUTURE_DAYS, 'D'), side='right') - 1
            label = 1 if (j <= k and j < n) else 0
            out_rows.append((grid, pd.Timestamp(d).strftime("%Y-%m-%d"), label))
    out_df = pd.DataFrame(out_rows, columns=["grid_id", "date", "future_label"])
    out_df["date"] = pd.to_datetime(out_df["date"])
    print(f"Computed future labels for {len(out_df)} grid-date pairs.")
    out_df.to_csv(GRID_LABEL_FILE, index=False)
    print(f"Saved grid-date labels to {GRID_LABEL_FILE}")
    return out_df

def phase_c_merge_labels_back(grid_labels: pd.DataFrame):
    print("Phase C: merging labels back into processed_viirs.csv (streamed)...")
    # convert grid_labels to dict for fast lookup: dict[grid_id] = {date: label}
    print("Building lookup dictionary in memory (may be large but typically smaller than full dataset)...")
    grid_labels["date_str"] = grid_labels["date"].dt.strftime("%Y-%m-%d")
    lookup = {}
    for _, row in grid_labels.iterrows():
        gid = row["grid_id"]
        dstr = row["date_str"]
        val = int(row["future_label"])
        if gid not in lookup:
            lookup[gid] = {}
        lookup[gid][dstr] = val

    reader = pd.read_csv(PROC_IN, chunksize=CHUNK_SIZE, parse_dates=["date"])
    first = True
    total = 0
    for i, chunk in enumerate(reader):
        chunk["grid_id"] = chunk["lat_bin"].astype(str) + "_" + chunk["lon_bin"].astype(str)
        chunk["date_str"] = chunk["date"].dt.strftime("%Y-%m-%d")
        # default label 0
        chunk["future_label"] = 0
        # vectorized-ish: iterate unique grid_ids present in chunk
        for gid in chunk["grid_id"].unique():
            if gid in lookup:
                gid_mask = chunk["grid_id"] == gid
                # get mapping for this gid
                mapping = lookup[gid]
                # for dates present in chunk for this gid
                dates = chunk.loc[gid_mask, "date_str"]
                # map date_str -> label if exists
                chunk.loc[gid_mask, "future_label"] = dates.map(lambda x: mapping.get(x, 0)).astype(int)
        # write chunk
        out_cols = [c for c in chunk.columns if c != "date_str"]  # keep original columns + future_label
        out_df = chunk[out_cols]
        if first:
            out_df.to_csv(OUTFILE, index=False, mode="w")
            first = False
        else:
            out_df.to_csv(OUTFILE, index=False, header=False, mode="a")
        total += len(out_df)
        if (i+1) % 5 == 0:
            print(f"  processed { (i+1)*CHUNK_SIZE } rows; written total {total}")
    print("Phase C done. Final processed file with future_label:", OUTFILE)

def main():
    df_grid = phase_a_extract_unique_grid_dates()
    grid_labels = phase_b_compute_future_labels(df_grid)
    phase_c_merge_labels_back(grid_labels)

if __name__ == "__main__":
    main()
