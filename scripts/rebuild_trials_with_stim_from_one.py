import numpy as np
import pandas as pd
from one.api import ONE

IN_PARQUET = "/Users/gabrielnixonraj/Desktop/ibl_project/data/ibl_trials_339sessions.parquet"
OUT_PARQUET = "/Users/gabrielnixonraj/Desktop/ibl_project/data/ibl_trials_339sessions_with_stim.parquet"

def signed_contrast(trials):
    cols = set(trials.keys()) if hasattr(trials, "keys") else set(trials.columns)

    if "contrastRight" in cols and "contrastLeft" in cols:
        cr = pd.Series(trials["contrastRight"]).astype(float).fillna(0.0)
        cl = pd.Series(trials["contrastLeft"]).astype(float).fillna(0.0)
        return cr - cl

    if "signedContrast" in cols:
        return pd.Series(trials["signedContrast"]).astype(float)

    if "stim_side" in cols and "contrast" in cols:
        s = pd.Series(trials["stim_side"]).astype(float).fillna(0.0)
        c = pd.Series(trials["contrast"]).astype(float).fillna(0.0)
        return s * c

    if "stimSide" in cols and "contrast" in cols:
        s = pd.Series(trials["stimSide"]).astype(float).fillna(0.0)
        c = pd.Series(trials["contrast"]).astype(float).fillna(0.0)
        return s * c

    return pd.Series(np.nan, index=np.arange(len(trials["choice"])))

def main():
    df = pd.read_parquet(IN_PARQUET)
    eids = df["eid"].astype(str).drop_duplicates().tolist()
    one = ONE(silent=True)

    out_rows = []
    wrote = 0
    failed = 0

    for i, eid in enumerate(eids, 1):
        try:
            tr = one.load_object(eid, "trials")
            L = len(tr["choice"])
            stim = signed_contrast(tr)

            out = pd.DataFrame({
                "eid": eid,
                "t": np.arange(L, dtype=np.int32),
                "choice": pd.to_numeric(tr["choice"], errors="coerce"),
                "feedbackType": pd.to_numeric(tr["feedbackType"], errors="coerce"),
                "stim_signed_contrast": pd.to_numeric(stim, errors="coerce"),
            })
            out_rows.append(out)
            wrote += 1
        except Exception as e:
            failed += 1
            if failed <= 5:
                print("FAILED eid:", eid, "err:", str(e))

        if i % 25 == 0:
            print(f"processed {i}/{len(eids)} wrote={wrote} failed={failed}")

    full = pd.concat(out_rows, ignore_index=True)
    full.to_parquet(OUT_PARQUET, index=False)

    stim_non_nan = float(full["stim_signed_contrast"].notna().mean())
    stim_non_zero = float((full["stim_signed_contrast"].fillna(0.0) != 0.0).mean())

    print("Saved:", OUT_PARQUET)
    print("rows:", len(full), "sessions:", full["eid"].nunique())
    print("stim non-NaN frac:", stim_non_nan)
    print("stim non-zero frac:", stim_non_zero)
    if full["stim_signed_contrast"].notna().any():
        s = full["stim_signed_contrast"].dropna()
        print("stim min/max:", float(s.min()), float(s.max()))

if __name__ == "__main__":
    main()
