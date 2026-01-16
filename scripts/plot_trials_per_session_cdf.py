import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PARQUET = "/Users/gabrielnixonraj/Desktop/ibl_project/data/ibl_trials_339sessions.parquet"
OUT = "/Users/gabrielnixonraj/Desktop/ibl_project/figures/trials_per_session_cdf.png"

def main():
    df = pd.read_parquet(PARQUET)

    trials_per_session = df.groupby("eid").size().values
    x = np.sort(trials_per_session)
    y = np.arange(1, len(x) + 1) / len(x)

    plt.figure(figsize=(7, 4))
    plt.plot(x, y)
    plt.xlabel("Trials per session")
    plt.ylabel("CDF")
    plt.title("CDF of session length (trials/session)")

    # Cutoff overlays
    for cutoff in [600, 700, 800]:
        plt.axvline(cutoff, linestyle="--")
        plt.text(
            cutoff + 5,
            0.05,
            f"{cutoff}",
            rotation=90,
            va="bottom",
            fontsize=9
        )

    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT, dpi=200)
    plt.show()

    print("Saved:", OUT)

if __name__ == "__main__":
    main()
