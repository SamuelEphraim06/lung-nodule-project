"""3D ResNet-10 (MedicalNet-pretrained) nodule-malignancy classifier on the frozen 5-fold protocol.

Protocol (identical to the radiomics baseline): test fold k, validation fold (k+1)%5, rest = training.
  --label hard : train on benign(0)/malignant(1) nodules only            (mean rating < 3 / > 3)
  --label soft : train on ALL eligible nodules (incl. indeterminate) with a soft target
                 soft = mean over readers of [1 if rating>3, 0.5 if rating==3, 0 if rating<3]
Evaluation is always on the binary-labelled test nodules, so models are directly comparable.
Early stopping + threshold selection use the validation fold only. Platt scaling is fitted on the
validation fold (note: the same fold also drives early stopping, as in the baseline).

Smoke test (1 fold, 2 epochs, tiny subset; checks the whole path in about a minute):
  python -m src.models.train_cnn --data /content/lidc_patches --out /content/results_cnn --smoke
Real run (about 5 folds x ~40 epochs):
  python -m src.models.train_cnn --data /content/lidc_patches --out /content/results_cnn --label hard
  python -m src.models.train_cnn --data /content/lidc_patches --out /content/results_cnn --label soft
"""
import argparse, ast, copy, time
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from ..baselines.metrics import summary, sens_at_spec

STORE, CROP = 80, 64          # patches are stored 80^3 in RAM; 64^3 crops are fed to the network


def load_table(data: Path):
    sp = pd.read_csv(data / "splits.csv")
    nd = pd.read_csv(data / "nodules.csv", usecols=["patch_file", "malignancy"])
    nd["ratings"] = nd.malignancy.apply(lambda s: np.asarray(ast.literal_eval(s)))
    sp = sp.merge(nd[["patch_file", "ratings"]], on="patch_file", how="left")
    sp["soft"] = sp.ratings.apply(lambda v: float(np.mean((v > 3) * 1.0 + (v == 3) * 0.5)))
    return sp


def load_patches(data: Path, files):
    X = np.zeros((len(files), STORE, STORE, STORE), dtype=np.int16)
    o = (96 - STORE) // 2
    for i, f in enumerate(files):
        ct = np.load(data / f)["ct"]
        X[i] = ct[o:o + STORE, o:o + STORE, o:o + STORE]
    return X


def normalise(crop):
    x = (np.clip(crop.astype(np.float32), -1000, 400) + 1000) / 1400.0   # 0..1, air = 0
    return (x - 0.3) / 0.3                                               # fixed scaling keeps absolute density cues


class PatchSet(Dataset):
    def __init__(self, X, idx, y, train):
        self.X, self.idx, self.y, self.train = X, idx, y, train
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        j = self.idx[i]; v = self.X[j]
        if self.train:
            off = np.random.randint(0, STORE - CROP + 1, 3)
            v = v[off[0]:off[0]+CROP, off[1]:off[1]+CROP, off[2]:off[2]+CROP]
            v = np.transpose(v, np.random.permutation(3))
            for ax in range(3):
                if np.random.rand() < 0.5: v = np.flip(v, ax)
            x = normalise(np.ascontiguousarray(v))
            x = x + np.random.normal(0, 0.03, x.shape).astype(np.float32)
        else:
            o = (STORE - CROP) // 2
            x = normalise(v[o:o+CROP, o:o+CROP, o:o+CROP])
        return torch.from_numpy(np.ascontiguousarray(x))[None], torch.tensor(self.y[j], dtype=torch.float32)


class Net(nn.Module):
    def __init__(self, pretrained=True, drop=0.3):
        super().__init__()
        from monai.networks.nets import resnet10      # MedicalNet weights are fetched from Hugging Face
        self.backbone = resnet10(pretrained=pretrained, spatial_dims=3, n_input_channels=1,
                                 feed_forward=False, shortcut_type="B", bias_downsample=False)
        self.drop, self.head = nn.Dropout(drop), nn.Linear(512, 1)
    def forward(self, x):
        return self.head(self.drop(self.backbone(x))).squeeze(1)


@torch.no_grad()
def predict(model, loader, device, tta):
    model.eval(); out = []
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        views = [x] + ([x.flip(2), x.flip(3), x.flip(4)] if tta else [])
        out.append(torch.stack([model(v).float() for v in views]).mean(0).cpu())
    return torch.cat(out).numpy()


def bce(logits, y):
    return float(nn.functional.binary_cross_entropy_with_logits(torch.tensor(logits), torch.tensor(y, dtype=torch.float32)))


def run_fold(k, sp, X, a, device):
    te, va = (sp.fold == k).values, (sp.fold == (k + 1) % 5).values; tr = ~(te | va)
    lab = (sp.bin_label >= 0).values
    train_idx = np.where(tr & (lab if a.label == "hard" else True))[0]
    val_idx, test_idx = np.where(va & lab)[0], np.where(te & lab)[0]
    ind_idx = np.where(te & ~lab)[0]                                   # indeterminate test nodules
    y_train = (sp.bin_label.clip(lower=0).values.astype(float) if a.label == "hard" else sp.soft.values)
    y_eval = sp.bin_label.clip(lower=0).values.astype(float)
    if a.smoke:
        train_idx, val_idx, test_idx, ind_idx = train_idx[:48], val_idx[:32], test_idx[:32], ind_idx[:8]

    mk = lambda idx, y, tr_: DataLoader(PatchSet(X, idx, y, tr_), batch_size=a.bs, shuffle=tr_, drop_last=tr_ and len(idx) > a.bs,
                                         num_workers=a.workers if tr_ else 0, pin_memory=True,
                                         persistent_workers=tr_ and a.workers > 0)
    dl_tr, dl_va, dl_te, dl_in = mk(train_idx, y_train, True), mk(val_idx, y_eval, False), mk(test_idx, y_eval, False), \
        (mk(ind_idx, y_eval, False) if len(ind_idx) else None)

    torch.manual_seed(a.seed * 100 + k); np.random.seed(a.seed * 100 + k)
    model = Net(pretrained=not a.scratch).to(device)
    opt = torch.optim.AdamW([{"params": model.backbone.parameters(), "lr": a.lr},
                             {"params": model.head.parameters(), "lr": a.lr * 10}], weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    amp = device.type == "cuda"; scaler = torch.amp.GradScaler("cuda", enabled=amp)
    best, best_state, bad, log = 1e9, None, 0, []
    for ep in range(a.epochs):
        model.train(); t0, tl, n = time.time(), 0.0, 0
        for x, y in dl_tr:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                logit = model(x)
            loss = nn.functional.binary_cross_entropy_with_logits(logit.float(), y)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tl += loss.item() * len(y); n += len(y)
        sched.step()
        lv = predict(model, dl_va, device, False); yv = y_eval[val_idx]
        vl = bce(lv, yv); vauc = roc_auc_score(yv, lv) if len(np.unique(yv)) == 2 else float("nan")
        log.append(dict(fold=k, epoch=ep, train_loss=tl / max(n, 1), val_loss=vl, val_auroc=vauc))
        print(f"  fold {k} ep {ep:02d} train {tl/max(n,1):.3f} | val loss {vl:.3f} auroc {vauc:.3f} | {time.time()-t0:.0f}s", flush=True)
        if vl < best - 1e-4: best, bad, best_state = vl, 0, copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= a.patience: print("  early stop"); break
    model.load_state_dict(best_state)
    lv, lt = predict(model, dl_va, device, a.tta), predict(model, dl_te, device, a.tta)
    li = predict(model, dl_in, device, a.tta) if dl_in is not None else np.array([])
    return dict(val_idx=val_idx, test_idx=test_idx, ind_idx=ind_idx, lv=lv, lt=lt, li=li, log=log)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--label", choices=["hard", "soft"], default="hard")
    ap.add_argument("--scratch", action="store_true", help="random init instead of MedicalNet weights")
    ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--bs", type=int, default=16); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=2); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tta", action="store_true", help="average logits over 3 flips at test time")
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    if a.smoke: a.epochs, a.folds, a.workers = 2, [0], 0
    a.out.mkdir(parents=True, exist_ok=True)
    tag = a.tag or f"cnn_{a.label}{'_scratch' if a.scratch else ''}_s{a.seed}" + ("_smoke" if a.smoke else "")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, "| tag:", tag)
    torch.backends.cudnn.benchmark = True

    sp = load_table(a.data)
    print("loading patches into RAM ...", flush=True); t0 = time.time()
    X = load_patches(a.data, sp.patch_file.values); print(f"  {X.nbytes/1e9:.1f} GB in {time.time()-t0:.0f}s")

    rows, ind_rows, logs = [], [], []
    f_oof, f_ind, f_log = (a.out / f"oof_predictions_{tag}.csv", a.out / f"oof_indeterminate_{tag}.csv",
                           a.out / f"train_log_{tag}.csv")
    done = set()
    if f_oof.exists() and not a.smoke:                       # RESUME after a disconnect
        prev = pd.read_csv(f_oof); rows.append(prev); done = set(prev.fold.unique().tolist())
        if f_ind.exists(): ind_rows.append(pd.read_csv(f_ind))
        if f_log.exists(): logs += pd.read_csv(f_log).to_dict("records")
        print("resuming; folds already saved:", sorted(done))
    for k in a.folds:
        if k in done:
            print(f"fold {k}: already done, skipping"); continue
        r = run_fold(k, sp, X, a, device); logs += r["log"]
        yv, yt = sp.bin_label.values[r["val_idx"]].astype(int), sp.bin_label.values[r["test_idx"]].astype(int)
        platt = LogisticRegression(C=1e6).fit(r["lv"].reshape(-1, 1), yv)
        sig = lambda z: 1 / (1 + np.exp(-z))
        variants = {f"{tag}_raw": (sig(r["lv"]), sig(r["lt"])),
                    f"{tag}_platt": (platt.predict_proba(r["lv"].reshape(-1, 1))[:, 1], platt.predict_proba(r["lt"].reshape(-1, 1))[:, 1])}
        for name, (pv, pt) in variants.items():
            s, spc, _ = sens_at_spec(yv, pv, yt, pt)
            d = sp.iloc[r["test_idx"]][["patient_id", "patch_file", "fold", "bin_label", "mal_mean"]].copy()
            d["model"], d["p"], d["sens_at_val_spec90"], d["spec_at_val_spec90"] = name, pt, s, spc
            rows.append(d)
        if len(r["li"]):
            d = sp.iloc[r["ind_idx"]][["patient_id", "patch_file", "fold", "bin_label", "mal_mean"]].copy()
            d["model"], d["p"] = f"{tag}_platt", platt.predict_proba(r["li"].reshape(-1, 1))[:, 1]; ind_rows.append(d)
        print(f"fold {k}: test AUROC raw {summary(yt, variants[tag + '_raw'][1])['auroc']:.3f}", flush=True)
        pd.concat(rows).to_csv(f_oof, index=False)       # saved after every fold
        pd.DataFrame(logs).to_csv(f_log, index=False)
        if ind_rows: pd.concat(ind_rows).to_csv(f_ind, index=False)
    oof = pd.concat(rows)
    for name, g in oof.groupby("model"):
        s = summary(g.bin_label.values, g.p.values); print(f"{name}: AUROC {s['auroc']:.3f} AUPRC {s['auprc']:.3f} ECE {s['ece']:.3f} (n={s['n']})")


if __name__ == "__main__":
    main()
