"""Plot centre slices + mask outline for a few patches, with all readers' ratings in the title."""
import argparse, json
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--n", type=int, default=6)
a = ap.parse_args()

recs = []
for f in sorted((a.out / "meta").glob("*.json")):
    recs += json.loads(f.read_text())
recs = recs[: a.n]
fig, ax = plt.subplots(len(recs), 3, figsize=(9, 3 * len(recs)), squeeze=False)
for r, rec in enumerate(recs):
    d = np.load(a.out / rec["patch_file"]); ct, m = d["ct"], d["mask"]
    c = ct.shape[0] // 2
    views = [ct[:, :, c], ct[:, c, :], ct[c, :, :]]
    mviews = [m[:, :, c], m[:, c, :], m[c, :, :]]
    for k in range(3):
        ax[r, k].imshow(np.clip(views[k], -1000, 400).T if k else np.clip(views[k], -1000, 400),
                        cmap="gray", vmin=-1000, vmax=400)
        mv = mviews[k].T if k else mviews[k]
        if mv.any():
            ax[r, k].contour(mv, levels=[0.5], colors="r", linewidths=0.8)
        ax[r, k].axis("off")
    ax[r, 0].set_title(f"{rec['patient_id']} n{rec['nodule_idx']} | {rec['diameter_mm']:.1f}mm | "
                       f"malig={rec['malignancy']}", fontsize=8, loc="left")
plt.tight_layout(); plt.savefig(a.out / "sanity.png", dpi=110)
print("saved", a.out / "sanity.png")
