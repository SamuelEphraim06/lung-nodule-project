"""Baselines on the frozen 5-fold patient-level splits (test=k, val=(k+1)%5, train=rest).

  A) size-only logistic regression (sanity floor: how far does nodule size alone get you?)
  B) radiomics features + XGBoost
Hard binary labels (benign 0 / malignant 1); indeterminate nodules (mean==3) are excluded here.
Outputs: features.csv (cache), oof_predictions.csv, metrics.json
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from .features import compute_features
from .metrics import summary, sens_at_spec, patient_bootstrap_auc

ap = argparse.ArgumentParser()
ap.add_argument("--data", type=Path, required=True, help="folder with splits.csv and patches/")
ap.add_argument("--out", type=Path, default=Path("results"))
ap.add_argument("--n-jobs", type=int, default=-1)
a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=True)

sp = pd.read_csv(a.data / "splits.csv")
sp = sp[sp.bin_label >= 0].reset_index(drop=True)
print(f"binary-labelled nodules: {len(sp)}  (malignant {int(sp.bin_label.sum())})")


def one(row):
    z = np.load(a.data / row.patch_file)
    return compute_features(z["ct"], z["mask"], row.diameter_mm)

fcache = a.out / "features.csv"
if fcache.exists():
    feats = pd.read_csv(fcache)
else:
    print("extracting features ...")
    rows = list(sp.itertuples())
    feats = pd.DataFrame(Parallel(n_jobs=a.n_jobs, batch_size=8)(delayed(one)(r) for r in rows))
    feats.insert(0, "patch_file", sp.patch_file.values)
    feats.to_csv(fcache, index=False)
feats = feats.set_index("patch_file").loc[sp.patch_file].reset_index(drop=True)
print("features:", feats.shape[1] - 0, "| nodules whose mask was empty (fallback used):", int(feats.mask_fallback.sum()))
FEATURE_COLS = [c for c in feats.columns if c != "patch_file"]


def platt(p_val, y_val, p_test):
    """Recalibrate on the validation fold only."""
    lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))).reshape(-1, 1)
    m = LogisticRegression(C=1e6).fit(lg(p_val), y_val)
    return m.predict_proba(lg(p_test))[:, 1]


oof, imps = [], []
for k in range(5):
    te, va = sp.fold == k, sp.fold == (k + 1) % 5
    tr = ~(te | va)
    y = sp.bin_label.values
    # ---- A) size-only ------------------------------------------------------------------------
    Xs = np.log(sp[["diameter_mm"]].values)
    mA = LogisticRegression().fit(Xs[tr], y[tr])
    pA_va, pA_te = mA.predict_proba(Xs[va])[:, 1], mA.predict_proba(Xs[te])[:, 1]
    # ---- B) XGBoost on radiomics ---------------------------------------------------------------
    X = feats[FEATURE_COLS].values
    mB = xgb.XGBClassifier(n_estimators=600, learning_rate=0.03, max_depth=3, subsample=0.8,
                           colsample_bytree=0.7, min_child_weight=3, reg_lambda=2.0,
                           eval_metric="logloss", early_stopping_rounds=40, random_state=42)
    mB.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
    pB_va, pB_te = mB.predict_proba(X[va])[:, 1], mB.predict_proba(X[te])[:, 1]
    pBc_va = platt(pB_va, y[va], pB_va); pBc_te = platt(pB_va, y[va], pB_te)
    imps.append(pd.Series(mB.get_booster().get_score(importance_type="gain")).reindex(
        [f"f{i}" for i in range(len(FEATURE_COLS))]).fillna(0).values)

    for name, pv, pt in [("size_only", pA_va, pA_te), ("xgb_raw", pB_va, pB_te), ("xgb_platt", pBc_va, pBc_te)]:
        s, spc, thr = sens_at_spec(y[va], pv, y[te], pt)
        d = sp[te][["patient_id", "patch_file", "fold", "bin_label", "mal_mean"]].copy()
        d["model"], d["p"], d["sens_at_val_spec90"], d["spec_at_val_spec90"] = name, pt, s, spc
        oof.append(d)
    print(f"fold {k}: test n={int(te.sum())}  AUROC size={summary(y[te], pA_te)['auroc']:.3f}  xgb={summary(y[te], pB_te)['auroc']:.3f}  (best iter {mB.best_iteration})")

oof = pd.concat(oof); oof.to_csv(a.out / "oof_predictions.csv", index=False)
res = {}
print("\n=== pooled out-of-fold results (mean per-fold sens/spec at the val-chosen 90%-spec threshold) ===")
for name, d in oof.groupby("model"):
    s = summary(d.bin_label.values, d.p.values)
    s["auroc_ci95_patient_bootstrap"] = patient_bootstrap_auc(d.bin_label.values, d.p.values, d.patient_id.values)
    s["per_fold_auroc"] = [float(summary(g.bin_label.values, g.p.values)["auroc"]) for _, g in d.groupby("fold")]
    s["sens_at_spec90"] = float(d.groupby("fold").sens_at_val_spec90.first().mean())
    s["spec_achieved"] = float(d.groupby("fold").spec_at_val_spec90.first().mean())
    strict = d[(d.mal_mean <= 2.5) | (d.mal_mean >= 3.5)]            # label-noise sensitivity check
    s["auroc_strict_labels"] = float(summary(strict.bin_label.values, strict.p.values)["auroc"]); s["n_strict"] = int(len(strict))
    res[name] = s
    print(f"{name:10s} AUROC {s['auroc']:.3f} (95% CI {s['auroc_ci95_patient_bootstrap'][0]:.3f}-{s['auroc_ci95_patient_bootstrap'][1]:.3f}) | "
          f"AUPRC {s['auprc']:.3f} | Brier {s['brier']:.3f} | ECE {s['ece']:.3f} | sens@~90spec {s['sens_at_spec90']:.3f} (spec {s['spec_achieved']:.3f}) | "
          f"strict-label AUROC {s['auroc_strict_labels']:.3f} (n={s['n_strict']})")
imp = pd.Series(np.mean(imps, 0), index=FEATURE_COLS).sort_values(ascending=False)
print("\ntop 10 features by mean XGBoost gain:\n", imp.head(10).round(2).to_string())
json.dump(res, open(a.out / "metrics.json", "w"), indent=1)
print("\nsaved", a.out / "metrics.json")
