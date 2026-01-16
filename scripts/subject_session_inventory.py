from one.api import ONE
import pandas as pd
import argparse

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", type=str, required=True)
    ap.add_argument("--batch_size", type=int, default=200)
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    eids = df["eid"].unique().tolist()

    one = ONE(silent=True)

    records = []
    total = len(eids)
    done = 0

    for batch in chunks(eids, args.batch_size):
        # Alyx filter: sessions with id in this batch
        ses_list = one.alyx.rest('sessions', 'list', id__in=",".join(batch))

        for ses in ses_list:
            records.append({
                "eid": ses["id"],
                "subject": ses.get("subject", None),
                "lab": ses.get("lab", None)
            })

        done += len(batch)
        print(f"metadata batches: {done}/{total}")

    meta = pd.DataFrame(records).drop_duplicates("eid")
    merged = df[["eid"]].drop_duplicates().merge(meta, on="eid", how="left")

    print("\n==== SUBJECT / SESSION INVENTORY ====")
    print("sessions total:", int(merged.shape[0]))
    print("unique subjects (rats):", int(merged["subject"].nunique()))
    print("unique labs:", int(merged["lab"].nunique()))

    print("\n==== SESSIONS PER RAT ====")
    sess_per_rat = merged.groupby("subject").size().sort_values()
    print("min sessions per rat:", int(sess_per_rat.min()))
    print("median sessions per rat:", int(sess_per_rat.median()))
    print("mean sessions per rat:", float(sess_per_rat.mean()))
    print("max sessions per rat:", int(sess_per_rat.max()))

    print("\nTop 10 rats by session count:")
    print(sess_per_rat.tail(10).to_string())

    print("\nBottom 10 rats by session count:")
    print(sess_per_rat.head(10).to_string())

    print("\n==== SESSION SIZE (TRIALS) ====")
    trials_per_session = df.groupby("eid").size()
    print("min trials/session:", int(trials_per_session.min()))
    print("median trials/session:", int(trials_per_session.median()))
    print("mean trials/session:", float(trials_per_session.mean()))
    print("max trials/session:", int(trials_per_session.max()))

    missing_subject = merged["subject"].isna().sum()
    if missing_subject:
        print("\nWARNING: sessions missing subject metadata:", int(missing_subject))

if __name__ == "__main__":
    main()
