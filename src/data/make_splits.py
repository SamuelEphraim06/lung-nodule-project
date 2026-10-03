"""Patient-level, stratified 5-fold split. Run once, commit splits.csv, never regenerate.

Eligible nodules : >= MIN_READERS readers AND not a suspicious oversized cluster (> 4 annotations).
Stratum          : benign (mean<3) / indeterminate (mean==3, within --margin) / malignant (mean>3)
bin_label        : 0 benign, 1 malignant, -1 indeterminate (excluded from binary metrics,
                   still usable for soft-label training)
Protocol         : for test fold k -> validation fold (k+1) % 5 (calibration / thresholds),
                   remaining 3 folds -> training.
Grouping is by patient_id, so a patient (and both of a 2-scan patient's scans) is in ONE fold.
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, required=True, help="folder containing nodules.csv")
ap.add_argument("--min-readers", type=int, default=3)
ap.add_argument("--margin", type=float, default=0.0, help="|mean-3| <= margin -> indeterminate")
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()

df = pd.read_csv(a.out / "nodules.csv")
n0 = len(df)
df = df[(df.n_readers >= a.min_readers) & (~df.cluster_too_big.astype(bool))].copy()
print(f"eligible nodules: {len(df)} of {n0}  (>= {a.min_readers} readers, cluster <= 4 annotations)")

d = df.mal_mean - 3
df["stratum"] = np.where(d.abs() <= a.margin, "indeterminate", np.where(d < 0, "benign", "malignant"))
df["bin_label"] = df.stratum.map({"benign": 0, "malignant": 1, "indeterminate": -1})

skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=a.seed)
df["fold"] = -1
for k, (_, te) in enumerate(skf.split(df, df.stratum, groups=df.patient_id)):
    df.iloc[te, df.columns.get_loc("fold")] = k

# ---- integrity checks -------------------------------------------------------------------
assert (df.fold >= 0).all()
assert df.groupby("patient_id").fold.nunique().max() == 1, "a patient appears in >1 fold!"
print("\nnodules per fold x stratum:")
print(pd.crosstab(df.fold, df.stratum, margins=True))
print("\npatients per fold:", df.groupby("fold").patient_id.nunique().to_dict())

cols = ["patient_id", "scan_idx", "nodule_idx", "patch_file", "n_readers", "diameter_mm",
        "mal_mean", "mal_std", "stratum", "bin_label", "fold"]
df[cols].to_csv(a.out / "splits.csv", index=False)
print("\nsaved", a.out / "splits.csv")
