import pandas as pd
import matplotlib.pyplot as plt

META = "/Users/gabrielnixonraj/Desktop/ibl_project/meta/sessions_meta.csv"
OUT  = "/Users/gabrielnixonraj/Desktop/ibl_project/figures/sessions_per_rat.png"

def main():
    meta = pd.read_csv(META)
    sess_per_rat = meta.groupby("subject").size().sort_values(ascending=False)

    plt.figure(figsize=(8, 4))
    plt.bar(sess_per_rat.index.astype(str), sess_per_rat.values)
    plt.xlabel("Rat (subject)")
    plt.ylabel("Sessions")
    plt.title("Sessions per rat (IBL ChoiceWorld)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(OUT, dpi=200)
    plt.show()
    print("Saved:", OUT)

if __name__ == "__main__":
    main()
