"""Merge meta/*.json into one nodules.csv with summary columns per nodule."""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()
recs = []
for f in sorted((a.out / "meta").glob("*.json")):
    recs += json.loads(f.read_text())
df = pd.DataFrame(recs)
df["mal_mean"] = df["malignancy"].apply(np.mean)
df["mal_median"] = df["malignancy"].apply(np.median)
df["mal_std"] = df["malignancy"].apply(lambda x: float(np.std(x)))
df.to_csv(a.out / "nodules.csv", index=False)
print(df[["n_readers", "diameter_mm", "mal_mean", "mal_std"]].describe())
print("\nreaders per nodule:\n", df["n_readers"].value_counts().sort_index())
print("\nmean-malignancy histogram (rounded):\n", df["mal_mean"].round().value_counts().sort_index())
print("\nclusters with >4 annotations (suspicious):", int(df["cluster_too_big"].sum()))
