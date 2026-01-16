import pandas as pd
import numpy as np
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", type=str, required=True)
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)

    print("==== FILE ====")
    print("rows:", len(df))
    print("cols:", list(df.columns))
    print("sessions (unique eids):", df["eid"].nunique())

    print("\n==== VALUE SANITY ====")
    print("choice unique (sample):", sorted(df["choice"].dropna().unique().tolist())[:10])
    print("feedbackType unique (sample):", sorted(df["feedbackType"].dropna().unique().tolist())[:10])
    print("choice NaN frac:", float(df["choice"].isna().mean()))
    print("feedbackType NaN frac:", float(df["feedbackType"].isna().mean()))

    has_stim_col = "stim_signed_contrast" in df.columns
    print("has stim_signed_contrast column:", bool(has_stim_col))
    if has_stim_col:
        stim_non_nan = float(df["stim_signed_contrast"].notna().mean())
        print("stim non-NaN frac:", stim_non_nan)
        stim_vals = df["stim_signed_contrast"].dropna().astype(float).unique()
        stim_vals = np.sort(stim_vals)
        print("stim unique count:", int(len(stim_vals)))
        print("stim unique (head):", stim_vals[:10].tolist())
        print("stim unique (tail):", stim_vals[-10:].tolist())

    print("\n==== TRIAL COUNTS PER SESSION ====")
    counts = df.groupby("eid").size().sort_values(ascending=False)
    print("sessions:", int(counts.shape[0]))
    print("trials/session summary:")
    print("  min:", int(counts.min()))
    print("  p25:", int(counts.quantile(0.25)))
    print("  median:", int(counts.quantile(0.50)))
    print("  p75:", int(counts.quantile(0.75)))
    print("  max:", int(counts.max()))
    print("  mean:", float(counts.mean()))

    print(f"\nTop {args.topk} largest sessions (eid, n_trials):")
    for eid, n in counts.head(args.topk).items():
        print(eid, int(n))

    print(f"\nTop {args.topk} smallest sessions (eid, n_trials):")
    for eid, n in counts.tail(args.topk).items():
        print(eid, int(n))

    print("\n==== COMPLETENESS CHECKS ====")
    t_ok = df.groupby("eid")["t"].apply(lambda x: x.is_monotonic_increasing).mean()
    print("frac sessions with nondecreasing t:", float(t_ok))

    per_eid_nan = df.groupby("eid")[["choice", "feedbackType"]].apply(lambda g: g.isna().mean())
    print("median NaN frac per session (choice):", float(per_eid_nan["choice"].median()))
    print("median NaN frac per session (feedbackType):", float(per_eid_nan["feedbackType"].median()))

if __name__ == "__main__":
    main()
