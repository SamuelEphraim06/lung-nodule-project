"""Torch-free data helpers for the CNN (kept separate so they can be unit-tested without PyTorch)."""
import ast
from pathlib import Path
import numpy as np, pandas as pd

STORE = 80          # patches are kept in RAM as 80^3 voxels (1 mm); crops of <= 80 are fed to the network


def load_table(data: Path):
    sp = pd.read_csv(data / "splits.csv")
    nd = pd.read_csv(data / "nodules.csv", usecols=["patch_file", "malignancy"])
    nd["ratings"] = nd.malignancy.apply(lambda s: np.asarray(ast.literal_eval(s)))
    sp = sp.merge(nd[["patch_file", "ratings"]], on="patch_file", how="left")
    # soft target: mean over readers of [1 if rating>3, 0.5 if ==3, 0 if <3]
    sp["soft"] = sp.ratings.apply(lambda v: float(np.mean((v > 3) * 1.0 + (v == 3) * 0.5)))
    return sp


def load_patches(data: Path, files, with_mask=False):
    X = np.zeros((len(files), STORE, STORE, STORE), dtype=np.int16)
    M = np.zeros((len(files), STORE, STORE, STORE), dtype=np.uint8) if with_mask else None
    o = (96 - STORE) // 2
    sl = (slice(o, o + STORE),) * 3
    for i, f in enumerate(files):
        z = np.load(data / f)
        X[i] = z["ct"][sl]
        if with_mask: M[i] = z["mask"][sl]
    return X, M


def normalise(crop):
    x = (np.clip(crop.astype(np.float32), -1000, 400) + 1000) / 1400.0   # 0..1, air = 0
    return (x - 0.3) / 0.3                                               # fixed scaling keeps absolute density cues


def make_sample(X, M, j, train, crop):
    """Return a float32 array (C, crop, crop, crop); C = 1 (CT) or 2 (CT + mask).
    Training: random crop position, random axis permutation and flips -- applied IDENTICALLY to CT and mask."""
    v = X[j]; m = None if M is None else M[j]
    if train:
        off = np.random.randint(0, STORE - crop + 1, 3)
        sl = tuple(slice(o, o + crop) for o in off)
        v = v[sl]; m = None if m is None else m[sl]
        perm = np.random.permutation(3)
        v = np.transpose(v, perm); m = None if m is None else np.transpose(m, perm)
        for ax in range(3):
            if np.random.rand() < 0.5:
                v = np.flip(v, ax); m = None if m is None else np.flip(m, ax)
        x = normalise(np.ascontiguousarray(v))
        x = x + np.random.normal(0, 0.03, x.shape).astype(np.float32)
    else:
        o = (STORE - crop) // 2; sl = (slice(o, o + crop),) * 3
        x = normalise(v[sl]); m = None if m is None else m[sl]
    ch = [x] + ([np.ascontiguousarray(m).astype(np.float32)] if m is not None else [])
    return np.ascontiguousarray(np.stack(ch))
