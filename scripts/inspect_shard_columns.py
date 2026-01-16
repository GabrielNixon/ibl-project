import glob, os
import pandas as pd
from collections import Counter

SHARD_DIR = "/Users/gabrielnixonraj/Desktop/ibl_project/data/ibl_trials_sessions_500"

def main():
    files = sorted(glob.glob(os.path.join(SHARD_DIR, "*.parquet")))
    print("n_shards:", len(files))
    ctr = Counter()
    for fp in files[:30]:
        t = pd.read_parquet(fp)
        ctr.update(t.columns.tolist())
    print("columns seen in first 30 shards:")
    for k, v in ctr.most_common():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
