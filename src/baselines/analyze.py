"""Deeper analysis of saved out-of-fold predictions (no retraining).

  1. paired patient-level bootstrap of the AUROC difference between two models
  2. performance inside size quartiles (does the model add anything BEYOND nodule size?)
  3. reliability diagram (calibration)
  4. selective prediction: accuracy vs coverage when the most uncertain cases are deferred
  5. does model uncertainty track radiologist disagreement? (preliminary RQ3)
Works for ANY model that writes oof_predictions.csv in the same format (use --models to choose).
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("--data", type=Path, required=True, help="folder with splits.csv")
ap.add_argument("--results", type=Path, required=True, help="folder with oof_predictions.csv")
ap.add_argument("--ref", default="size_only"); ap.add_argument("--main", default="xgb_raw")
ap.add_argument("--boot", type=int, default=2000)
a = ap.parse_args()
figs = a.results / "figs"; figs.mkdir(exist_ok=True)

oof = pd.read_csv(a.results / "oof_predictions.csv")
sp = pd.read_csv(a.data / "splits.csv")[["patch_file", "diameter_mm", "mal_std", "n_readers"]]
oof = oof.merge(sp, on="patch_file", how="left")
models = list(oof.model.unique()); out = {}
P = {m: oof[oof.model == m].set_index("patch_file") for m in models}
ref, main = P[a.ref], P[a.main].loc[P[a.ref].index]
y, grp = ref.bin_label.values, ref.patient_id.values

# 1 --- paired bootstrap ----------------------------------------------------------------------------
rng = np.random.default_rng(0)
ids = np.unique(grp); idx = {g: np.where(grp == g)[0] for g in ids}; d = []
for _ in range(a.boot):
    pick = np.concatenate([idx[g] for g in rng.choice(ids, len(ids))])
    if len(np.unique(y[pick])) == 2:
        d.append(roc_auc_score(y[pick], main.p.values[pick]) - roc_auc_score(y[pick], ref.p.values[pick]))
lo, hi = np.percentile(d, [2.5, 97.5])
out["delta_auroc"] = dict(main=a.main, ref=a.ref, mean=float(np.mean(d)), ci95=[float(lo), float(hi)], p_le_0=float(np.mean(np.array(d) <= 0)))
print(f"1) paired bootstrap  AUROC({a.main}) - AUROC({a.ref}) = {np.mean(d):+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  "
      f"(share of resamples <= 0: {np.mean(np.array(d) <= 0):.3f})")

# 2 --- size quartiles ----------------------------------------------------------------------------------
q = pd.qcut(ref.diameter_mm, 4, duplicates="drop"); rows = []
for b_, ix in ref.groupby(q, observed=True).groups.items():
    s, m = ref.loc[ix], main.loc[ix]
    ok = s.bin_label.nunique() == 2
    rows.append(dict(diameter_mm=str(b_), n=len(s), malignant_rate=round(s.bin_label.mean(), 3),
                     auroc_size_only=round(roc_auc_score(s.bin_label, s.p), 3) if ok else None,
                     auroc_main=round(roc_auc_score(m.bin_label, m.p), 3) if ok else None))
tab = pd.DataFrame(rows); out["size_quartiles"] = rows
print("\n2) performance inside size quartiles (within a bin, size barely varies, so AUROC shows what the model adds beyond size):")
print(tab.to_string(index=False))

# 3 --- reliability diagram -------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(4.8, 4.8)); ax.plot([0, 1], [0, 1], "k--", lw=1)
for m in models:
    d_ = P[m]; b_ = pd.qcut(d_.p, 8, duplicates="drop")
    g = d_.groupby(b_, observed=True).agg(p=("p", "mean"), y=("bin_label", "mean"))
    ax.plot(g.p, g.y, "o-", label=m, ms=4)
ax.set_xlabel("predicted probability"); ax.set_ylabel("observed malignant fraction"); ax.legend(); ax.set_title("Reliability (8 equal-count bins)")
plt.tight_layout(); plt.savefig(figs / "reliability.png", dpi=130); plt.close()

# 4 --- selective prediction ----------------------------------------------------------------------------------
conf = np.abs(main.p.values - 0.5); order = np.argsort(-conf)
correct = ((main.p.values > 0.5).astype(int) == y)[order]; cov, acc = [], []
for c in np.arange(1.0, 0.29, -0.05):
    n = int(round(c * len(y))); cov.append(c); acc.append(correct[:n].mean())
out["selective"] = {f"{c:.2f}": float(x) for c, x in zip(cov, acc) if abs(c * 20 - round(c * 20)) < 1e-9 and round(c * 100) % 10 == 0}
fig, ax = plt.subplots(figsize=(5, 4)); ax.plot(np.array(cov) * 100, np.array(acc) * 100, "o-", ms=4)
ax.axhline(correct.mean() * 100, color="gray", ls="--", lw=1, label="no deferral")
ax.set_xlabel("coverage (% of nodules answered)"); ax.set_ylabel("accuracy on answered (%)"); ax.invert_xaxis(); ax.legend()
ax.set_title(f"Defer-to-radiologist ({a.main})"); plt.tight_layout(); plt.savefig(figs / "selective.png", dpi=130); plt.close()
print("\n4) accuracy vs coverage (defer the least confident):")
for k, v in out["selective"].items(): print(f"   coverage {float(k)*100:3.0f}% -> accuracy {v*100:5.1f}%")

# 5 --- uncertainty vs radiologist disagreement ------------------------------------------------------------------
p = np.clip(main.p.values, 1e-6, 1 - 1e-6); ent = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
r1 = spearmanr(ent, main.mal_std.values); r2 = spearmanr(ent, -np.abs(main.mal_mean.values - 3))
out["uncertainty_vs_disagreement"] = dict(spearman_entropy_vs_reader_std=float(r1.statistic), p1=float(r1.pvalue),
                                          spearman_entropy_vs_closeness_to_3=float(r2.statistic), p2=float(r2.pvalue))
print(f"\n5) Spearman(model uncertainty, reader std)            = {r1.statistic:+.3f} (p={r1.pvalue:.2g})")
print(f"   Spearman(model uncertainty, closeness of mean to 3) = {r2.statistic:+.3f} (p={r2.pvalue:.2g})")
print("   (note: reader std and distance-to-3 are themselves related; interpret both)")
json.dump(out, open(a.results / "analysis.json", "w"), indent=1); print("\nsaved", a.results / "analysis.json", "and figs/")
