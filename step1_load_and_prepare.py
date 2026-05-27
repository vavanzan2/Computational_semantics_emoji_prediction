# =============================================================================
# step1_load_and_prepare.py
#
# GOAL: Load the Excel spreadsheet, clean and sort the data,
#       and save the result to data/df2.csv for the next steps.
# =============================================================================

import os
import re
import pandas as pd

from config import DATA_PATH, PATHS


def turn_order(x):
    """ Converts 'group-turn' (e.g., '12-3') into a number for correct sorting."""
    s = str(x)
    m = re.match(r"(\d+)-(\d+)", s)
    if m:
        return int(m.group(1)) * 1000 + int(m.group(2))
    return 0


def run():
    print("[Step 1] Loading data ...")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"File not found: '{DATA_PATH}'\n"
            "Check if DATA_PATH in config.py looks for the correct file."
        )

    df = pd.read_excel(DATA_PATH)
    print(f"  File loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # Validation: required columns
    required_cols = ["Group ID", "Turntype", "Turn No.", "Text"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in the input file: {missing}")

    # Cleaning and sorting
    df["Group ID"] = df["Group ID"].astype(str)
    df["Turntype"] = df["Turntype"].astype(str).str.lower().str.strip()
    df["order"]    = df["Turn No."].map(turn_order)

    df = df.sort_values(["Group ID", "order"]).reset_index(drop=True)

    # Save intermediate result
    os.makedirs(os.path.dirname(PATHS["df2"]), exist_ok=True)
    df.to_csv(PATHS["df2"], index=False)
    print(f"  Data prepared and saved in: {PATHS['df2']}")

    return df


if __name__ == "__main__":
    run()
