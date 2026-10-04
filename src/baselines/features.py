"""Hand-crafted radiomics-style features from one nodule patch (1 mm isotropic, HU).

Dependencies: numpy, scipy, scikit-image only (PyRadiomics does not build on current Python).
Feature families: shape, first-order intensity, boundary/context, 2D-GLCM texture on 3 orthogonal planes.
NOTE: the shape/size features come from the radiologists' majority-vote mask, so this baseline
assumes a segmentation is available (a limitation to state in the report).
"""
import numpy as np
from scipy import ndimage as ndi
from scipy.stats import skew, kurtosis
from skimage.measure import marching_cubes, mesh_surface_area

CROP = 64          # analysis window (mm); nodules are mostly < 15 mm
LEVELS = 32


def _fallback_mask(shape, diameter_mm):
    c = np.array(shape) // 2
    g = np.indices(shape)
    r2 = sum((g[i] - c[i]) ** 2 for i in range(3))
    return (r2 <= (max(diameter_mm, 3.0) / 2) ** 2).astype(np.uint8)


def _center_crop(a, size):
    s = (np.array(a.shape) - size) // 2
    return a[s[0]:s[0]+size, s[1]:s[1]+size, s[2]:s[2]+size]


def _glcm_feats(img2d, roi2d):
    """GLCM over ROI pixels only (level 0 = outside ROI, dropped). Averaged over 4 angles, distance 1."""
    if roi2d.sum() < 6:
        return None
    hu = np.clip(img2d, -1000, 400)
    q = np.floor((hu + 1000) / 1400 * (LEVELS - 1e-6)).astype(int) + 1
    q = np.where(roi2d, q, 0)
    P = np.zeros((LEVELS + 1, LEVELS + 1))
    for dy, dx in [(0, 1), (1, 1), (1, 0), (1, -1)]:
        a = q[max(0, -dy):q.shape[0] - max(0, dy), max(0, -dx):q.shape[1] - max(0, dx)]
        b = q[max(0, dy):, max(0, dx):][:a.shape[0], :a.shape[1]] if dx >= 0 else q[max(0, dy):, :q.shape[1] + dx][:a.shape[0], :a.shape[1]]
        a = a[:b.shape[0], :b.shape[1]]
        m = (a > 0) & (b > 0)
        np.add.at(P, (a[m], b[m]), 1)
    P = P + P.T
    P = P[1:, 1:]
    s = P.sum()
    if s == 0:
        return None
    P /= s
    i, j = np.indices(P.shape)
    mu_i, mu_j = (i * P).sum(), (j * P).sum()
    sd_i, sd_j = np.sqrt(((i - mu_i) ** 2 * P).sum()), np.sqrt(((j - mu_j) ** 2 * P).sum())
    return dict(
        contrast=((i - j) ** 2 * P).sum(), homogeneity=(P / (1 + (i - j) ** 2)).sum(),
        energy=(P ** 2).sum(), entropy=-(P[P > 0] * np.log2(P[P > 0])).sum(),
        correlation=((i - mu_i) * (j - mu_j) * P).sum() / (sd_i * sd_j + 1e-9))


def compute_features(ct, mask, diameter_mm):
    ct = _center_crop(ct.astype(np.float32), CROP)
    mask = _center_crop(mask, CROP).astype(bool)
    f = {}
    used_fallback = mask.sum() < 8
    if used_fallback:
        mask = _fallback_mask(ct.shape, diameter_mm).astype(bool)
    f["mask_fallback"] = int(used_fallback)

    # ---- shape --------------------------------------------------------------------------
    V = float(mask.sum())
    f["volume_mm3"] = V
    f["eq_diameter_mm"] = (6 * V / np.pi) ** (1 / 3)
    f["diameter_reader_mm"] = float(diameter_mm)
    try:
        verts, faces, _, _ = marching_cubes(np.pad(mask, 1).astype(np.float32), 0.5)
        A = mesh_surface_area(verts, faces)
        f["surface_area_mm2"] = A
        f["sphericity"] = (np.pi ** (1 / 3)) * (6 * V) ** (2 / 3) / A
    except Exception:
        f["surface_area_mm2"], f["sphericity"] = np.nan, np.nan
    coords = np.argwhere(mask).astype(float)
    if len(coords) > 3:
        ev = np.sort(np.linalg.eigvalsh(np.cov(coords.T)))[::-1]
        ev = np.maximum(ev, 1e-9)
        f["elongation"] = float(np.sqrt(ev[1] / ev[0])); f["flatness"] = float(np.sqrt(ev[2] / ev[0]))
    else:
        f["elongation"] = f["flatness"] = np.nan

    # ---- first-order intensity inside the mask --------------------------------------------
    v = np.clip(ct[mask], -1000, 400)
    for name, val in zip(["p10", "p25", "p50", "p75", "p90"], np.percentile(v, [10, 25, 50, 75, 90])):
        f[f"hu_{name}"] = float(val)
    f.update(hu_mean=float(v.mean()), hu_std=float(v.std()), hu_min=float(v.min()), hu_max=float(v.max()),
             hu_skew=float(skew(v)) if v.std() > 0 else 0.0, hu_kurt=float(kurtosis(v)) if v.std() > 0 else 0.0)
    h, _ = np.histogram(v, bins=32, range=(-1000, 400)); p = h[h > 0] / h.sum()
    f["hu_entropy"] = float(-(p * np.log2(p)).sum())
    f["frac_ggo"] = float(((v > -750) & (v < -300)).mean())      # ground-glass-like voxels
    f["frac_solid"] = float((v > -100).mean())
    f["frac_dense"] = float((v > 130).mean())                    # calcification-like

    # ---- boundary / context -----------------------------------------------------------------
    dil = ndi.binary_dilation(mask, iterations=3)
    shell = dil & ~mask
    outside = ~dil
    if shell.any():
        f["shell_mean"] = float(np.clip(ct[shell], -1000, 400).mean())
        f["shell_std"] = float(np.clip(ct[shell], -1000, 400).std())
        f["edge_contrast"] = f["hu_mean"] - f["shell_mean"]
    else:
        f["shell_mean"] = f["shell_std"] = f["edge_contrast"] = np.nan
    f["background_mean"] = float(np.clip(ct[outside], -1000, 400).mean()) if outside.any() else np.nan
    erode = ndi.binary_erosion(mask, iterations=2)
    f["core_mean"] = float(np.clip(ct[erode], -1000, 400).mean()) if erode.any() else f["hu_mean"]

    # ---- 2D GLCM texture on three orthogonal planes through the mask centroid ---------------------
    c = np.round(coords.mean(0)).astype(int)
    acc = {}
    for ax in range(3):
        sl = [slice(None)] * 3; sl[ax] = c[ax]
        g = _glcm_feats(ct[tuple(sl)], mask[tuple(sl)])
        if g:
            for k, val in g.items():
                acc.setdefault(k, []).append(val)
    for k in ["contrast", "homogeneity", "energy", "entropy", "correlation"]:
        f[f"glcm_{k}"] = float(np.mean(acc[k])) if k in acc else np.nan
    return f
