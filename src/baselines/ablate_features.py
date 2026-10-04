"""Feature-group ablation: does anything BEYOND nodule size help?

Same frozen 5-fold protocol as run_radiomics.py (test=k, val=(k+1)%5, train=rest), XGBoost, hard labels.
Reuses results/features.csv (no re-extraction). Reports pooled out-of-fold AUROC with a patient-level
bootstrap CI, and the PAIRED bootstrap difference against the size-only feature set.
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("--data", type=Path, required=True)
ap.add_argument("--results", type=Path, required=True)
ap.add_argument("--boot", type=int, default=2000)
a = ap.parse_args()

sp = pd.read_csv(a.data / "splits.csv"); sp = sp[sp.bin_label >= 0].reset_index(drop=True)
F = pd.read_csv(a.results / "features.csv").set_index("patch_file").loc[sp.patch_file].reset_index(drop=True)
y, fold, pat = sp.bin_label.values, sp.fold.values, sp.patient_id.values

SIZE = ["diameter_reader_mm", "volume_mm3", "eq_diameter_mm", "surface_area_mm2"]
GROUPS = {
    "shape (non-size)": ["sphericity", "elongation", "flatness"],
    "intensity": [c for c in F.columns if c.startswith("hu_") or c.startswith("frac_")],
    "boundary/context": ["shell_mean", "shell_std", "edge_contrast", "background_mean", "core_mean"],
    "texture (GLCM)": [c for c in F.columns if c.startswith("glcm_")],
}
ALL = [c for c in F.columns if c != "mask_fallback"]
CONFIGS = {"size features only": SIZE}
for k, v in GROUPS.items():
    CONFIGS[f"size + {k}"] = SIZE + v
CONFIGS["all features"] = ALL
CONFIGS["all EXCEPT size"] = [c for c in ALL if c not in SIZE]
for k, v in CONFIGS.items():
    missing = [c for c in v if c not in F.columns]; assert not missing, (k, missing)


def oof_predictions(cols):
    p = np.zeros(len(y))
    for k in range(5):
        te, va = fold == k, fold == (k + 1) % 5; tr = ~(te | va)
        m = xgb.XGBClassifier(n_estimators=600, learning_rate=0.03, max_depth=3, subsample=0.8,
                              colsample_bytree=0.7, min_child_weight=3, reg_lambda=2.0,
                              eval_metric="logloss", early_stopping_rounds=40, random_state=42)
        X = F[cols].values
        m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
        p[te] = m.predict_proba(X[te])[:, 1]
    return p


preds = {name: oof_predictions(cols) for name, cols in CONFIGS.items()}

rng = np.random.default_rng(0)
ids = np.unique(pat); idx = {g: np.where(pat == g)[0] for g in ids}
boots = [np.concatenate([idx[g] for g in rng.choice(ids, len(ids))]) for _ in range(a.boot)]
boots = [b for b in boots if len(np.unique(y[b])) == 2]
ref = preds["size features only"]
ref_b = np.array([roc_auc_score(y[b], ref[b]) for b in boots])

rows = []
for name, p in preds.items():
    bs = np.array([roc_auc_score(y[b], p[b]) for b in boots]); d = bs - ref_b
    rows.append(dict(feature_set=name, n_features=len(CONFIGS[name]), auroc=round(roc_auc_score(y, p), 3),
                     ci95=f"{np.percentile(bs, 2.5):.3f}-{np.percentile(bs, 97.5):.3f}",
                     delta_vs_size=round(float(d.mean()), 3),
                     delta_ci95=f"{np.percentile(d, 2.5):+.3f} to {np.percentile(d, 97.5):+.3f}"))
tab = pd.DataFrame(rows)
print(tab.to_string(index=False))
tab.to_csv(a.results / "ablation_feature_groups.csv", index=False)
print("\nsaved", a.results / "ablation_feature_groups.csv")
print("Read 'delta_vs_size': a CI that excludes 0 means that feature group adds reliable information over size alone.")
