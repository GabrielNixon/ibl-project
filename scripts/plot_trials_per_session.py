import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

PARQUET = "/Users/gabrielnixonraj/Desktop/ibl_project/data/ibl_trials_339sessions.parquet"
OUTPNG  = "/Users/gabrielnixonraj/Desktop/ibl_project/figures/trials_per_session_hist.png"

def main():
    df = pd.read_parquet(PARQUET)
    n = df.groupby("eid").size()

    print("sessions:", int(n.shape[0]))
    print("min:", int(n.min()))
    print("p25:", int(np.percentile(n, 25)))
    print("median:", int(np.percentile(n, 50)))
    print("p75:", int(np.percentile(n, 75)))
    print("max:", int(n.max()))
    print("mean:", float(n.mean()))

    plt.figure(figsize=(7, 4))
    plt.hist(n.values, bins=40)
    plt.xlabel("Trials per session")
    plt.ylabel("Number of sessions")
    plt.title("Distribution of session lengths (trials/session)")
    plt.tight_layout()
    plt.savefig(OUTPNG, dpi=200)
    plt.show()
    print("Saved:", OUTPNG)

if __name__ == "__main__":
    main()
