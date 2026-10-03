"""Synthetic test: anisotropic, NON-uniform slice spacing; a sphere at a known place must land at patch centre."""
import numpy as np
from src.data.geometry import crop_patch

rng = np.random.default_rng(0)
ps = 0.7                                   # in-plane mm
zv = np.cumsum(np.r_[0, rng.choice([1.0, 1.25, 2.5], 119)])  # non-uniform z
H = W = 200
vol = np.full((H, W, len(zv)), -1000, dtype=np.int16)

# sphere radius 6 mm centred at physical (y=60mm, x=90mm, z=zv[50]+0.4)
cy, cx, cz = 60.0, 90.0, zv[50] + 0.4
I, J = np.meshgrid(np.arange(H) * ps, np.arange(W) * ps, indexing="ij")
for k, z in enumerate(zv):
    d2 = (I - cy) ** 2 + (J - cx) ** 2 + (z - cz) ** 2
    vol[:, :, k][d2 <= 36] = 0

center_ijk = (cy / ps, cx / ps, np.interp(cz, zv, np.arange(len(zv))))
p = crop_patch(vol, ps, zv, center_ijk, size=48, order=1)
inside = np.argwhere(p > -500)
com = inside.mean(0)
print("patch centre voxel:", 24, "| sphere centre of mass in patch:", np.round(com, 2))
assert np.all(np.abs(com - 24) < 1.0), "sphere not centred -> geometry bug"
# extent should be ~ +-6 voxels (1 mm iso) in ALL axes
ext = inside.max(0) - inside.min(0) + 1
print("extent (voxels, expect ~12-13 each):", ext)
assert np.all(np.abs(ext - 12.5) <= 2)
# out-of-scan padding
edge = crop_patch(vol, ps, zv, (100, 100, 2.0), size=48)
assert edge.min() == -1024 or edge.min() == -1000
print("OK")
