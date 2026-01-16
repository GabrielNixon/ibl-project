from one.api import ONE
import pandas as pd
import os

PARQUET = "/Users/gabrielnixonraj/Desktop/ibl_project/data/ibl_trials_339sessions.parquet"
OUTCSV  = "/Users/gabrielnixonraj/Desktop/ibl_project/meta/sessions_meta.csv"

def main():
    df = pd.read_parquet(PARQUET)
    eids = df["eid"].drop_duplicates().tolist()

    one = ONE(silent=True)

    existing = None
    done = set()
    if os.path.exists(OUTCSV):
        existing = pd.read_csv(OUTCSV)
        done = set(existing["eid"].astype(str).tolist())

    rows = []
    total = len(eids)

    for i, eid in enumerate(eids, 1):
        if str(eid) in done:
            if i % 25 == 0:
                print(f"progress: {i}/{total} (skipping cached rows)")
            continue

        try:
            d = one.get_details(eid)
            rows.append({
                "eid": str(eid),
                "subject": d.get("subject", None),
                "lab": d.get("lab", None),
                "start_time": d.get("start_time", None),
            })
        except Exception as e:
            rows.append({
                "eid": str(eid),
                "subject": None,
                "lab": None,
                "start_time": None,
                "error": str(e),
            })

        if len(rows) >= 25:
            new = pd.DataFrame(rows)
            if existing is None and not os.path.exists(OUTCSV):
                new.to_csv(OUTCSV, index=False)
                existing = new
            else:
                new.to_csv(OUTCSV, mode="a", header=False, index=False)
                existing = pd.concat([existing, new], ignore_index=True)
            rows = []
            print(f"progress: {i}/{total} wrote csv")

    if rows:
        new = pd.DataFrame(rows)
        if existing is None and not os.path.exists(OUTCSV):
            new.to_csv(OUTCSV, index=False)
            existing = new
        else:
            new.to_csv(OUTCSV, mode="a", header=False, index=False)
            existing = pd.concat([existing, new], ignore_index=True)

    meta = pd.read_csv(OUTCSV)
    meta = meta.drop_duplicates("eid")
    meta.to_csv(OUTCSV, index=False)

    print("Saved:", OUTCSV)
    print("rows:", len(meta))
    print("unique subjects:", int(meta["subject"].nunique()))
    print("unique labs:", int(meta["lab"].nunique()))
    print("missing subject rows:", int(meta["subject"].isna().sum()))

if __name__ == "__main__":
    main()
