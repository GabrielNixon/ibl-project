import numpy as np
import pandas as pd

PARQUET = "/Users/gabrielnixonraj/Desktop/ibl_project/data/ibl_trials_339sessions.parquet"
META_CSV = "/Users/gabrielnixonraj/Desktop/ibl_project/meta/sessions_meta.csv"
OUT_NPZ = "/Users/gabrielnixonraj/Desktop/ibl_project/data/ibl_sequences_T800.npz"

T = 800

def main():
    df = pd.read_parquet(PARQUET)
    meta = pd.read_csv(META_CSV)

    df["eid"] = df["eid"].astype(str)
    meta["eid"] = meta["eid"].astype(str)

    df = df.merge(meta[["eid", "subject"]], on="eid", how="left")

    eids = df["eid"].drop_duplicates().tolist()
    N = len(eids)

    x = np.zeros((N, T, 3), dtype=np.float32)
    y = np.zeros((N, T), dtype=np.int64)
    mask = np.zeros((N, T), dtype=np.float32)

    subj = []
    eid_list = []

    for i, eid in enumerate(eids):
        g = df[df["eid"] == eid].sort_values("t")
        stim = g["stim_signed_contrast"].to_numpy()
        choice = g["choice"].to_numpy()
        reward = (g["feedbackType"].to_numpy() > 0).astype(np.float32)

        stim = np.nan_to_num(stim, nan=0.0).astype(np.float32)
        choice = np.nan_to_num(choice, nan=0.0).astype(np.float32)

        L = min(len(g), T)

        prev_choice = np.zeros(L, dtype=np.float32)
        prev_reward = np.zeros(L, dtype=np.float32)
        prev_choice[1:] = choice[:L-1]
        prev_reward[1:] = reward[:L-1]

        x[i, :L, 0] = stim[:L]
        x[i, :L, 1] = prev_choice
        x[i, :L, 2] = prev_reward

        y[i, :L] = (choice[:L] > 0).astype(np.int64)
        mask[i, :L] = 1.0

        subj.append(g["subject"].iloc[0] if "subject" in g.columns else None)
        eid_list.append(eid)

        if (i + 1) % 25 == 0:
            print(f"processed {i+1}/{N}")

    np.savez_compressed(
        OUT_NPZ,
        x=x,
        y=y,
        mask=mask,
        eid=np.array(eid_list, dtype=object),
        subject=np.array(subj, dtype=object),
    )
    print("Saved:", OUT_NPZ)
    print("x:", x.shape, "y:", y.shape, "mask:", mask.shape)

if __name__ == "__main__":
    main()
