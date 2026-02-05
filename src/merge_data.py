"""
merge_data.py
Merge multiple VIIRS CSV downloads into a single CSV.
Place the downloaded CSVs in the data/ folder and name them:
 - viirs_2023.csv
 - viirs_2024.csv
(or change the file names below)
"""
import pandas as pd
from pathlib import Path

DATA_DIR = Path("../data").resolve()
OUTFILE = DATA_DIR / "merged_viirs.csv"

# Edit these names if your downloaded files have different names
INPUT_FILES = [
    DATA_DIR / "viirs_2023.csv",
    DATA_DIR / "viirs_2024.csv",
]

def load_file(path: Path):
    print(f"Loading {path} ...")
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Place downloaded CSV into data/ and rename accordingly.")
    # many NASA CSVs have a leading header row or comment lines - pandas handles usual CSVs
    df = pd.read_csv(path)
    print(f" - shape: {df.shape}")
    return df

def merge():
    dfs = []
    for f in INPUT_FILES:
        dfs.append(load_file(f))
    merged = pd.concat(dfs, ignore_index=True)
    print(f"Merged dataframe shape: {merged.shape}")
    # optional: remove exact duplicate rows
    merged = merged.drop_duplicates().reset_index(drop=True)
    print(f"After dedup: {merged.shape}")
    OUTFILE.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTFILE, index=False)
    print(f"Saved merged file to: {OUTFILE}")

if __name__ == "__main__":
    merge()
