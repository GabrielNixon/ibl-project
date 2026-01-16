import os
import glob
import numpy as np
import pandas as pd

SHARD_DIR = "/Users/gabrielnixonraj/Desktop/ibl_project/data/ibl_trials_sessions_500"
OUT_PARQUET = "/Users/gabrielnixonraj/Desktop/ibl_project/data/ibl_trials_339sessions_with_stim.parquet"

def compute_signed_contrast(t):
    cols = set(t.columns)

    if "stim_signed_contrast" in cols and t["stim_signed_contrast"].notna().any():
        return t["stim_signed_contrast"].astype(float)

    if ("contrastRight" in cols) and ("contrastLeft" in cols):
        cr = t["contrastRight"].astype(float).fillna(0.0)
        cl = t["contrastLeft"].astype(float).fillna(0.0)
        return cr - cl

    if ("signedContrast" in cols):
        return t["signedContrast"].astype(float)

    if ("contrast" in cols) and ("stim_side" in cols):
        c = t["contrast"].astype(float).fillna(0.0)
        s = t["stim_side"].astype(float).fillna(0.0)
        return c * s

    return pd.Series(np.nan, index=t.index)

def main():
    files = sorted(glob.glob(os.path.join(SHARD_DIR, "*.parquet")))
    if not files:
        raise SystemExit(f"No parquet shards found in {SHARD_DIR}")

    rows = []
    for i, fp in enumerate(files, 1):
        eid = os.path.splitext(os.path.basename(fp))[0]
        t = pd.read_parquet(fp)

        if "choice" not in t.columns or "feedbackType" not in t.columns:
            continue

        stim = compute_signed_contrast(t)

        out = pd.DataFrame({
            "eid": eid,
            "t": np.arange(len(t), dtype=np.int32),
            "choice": pd.to_numeric(t["choice"], errors="coerce"),
            "feedbackType": pd.to_numeric(t["feedbackType"], errors="coerce"),
            "stim_signed_contrast": pd.to_numeric(stim, errors="coerce"),
        })
        rows.append(out)

        if i % 50 == 0:
            print(f"processed {i}/{len(files)}")

    df = pd.concat(rows, ignore_index=True)
    df.to_parquet(OUT_PARQUET, index=False)

    stim_non_nan = float(df["stim_signed_contrast"].notna().mean())
    stim_non_zero = float((df["stim_signed_contrast"].fillna(0.0) != 0.0).mean())

    print("Saved:", OUT_PARQUET)
    print("rows:", len(df), "sessions:", df["eid"].nunique())
    print("stim non-NaN frac:", stim_non_nan)
    print("stim non-zero frac:", stim_non_zero)
    print("contrast min/max (non-NaN):", float(df["stim_signed_contrast"].min()), float(df["stim_signed_contrast"].max()))

if __name__ == "__main__":
    main()
