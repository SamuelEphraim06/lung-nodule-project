"""Pure-numpy/scipy patch cropping. No pylidc needed, so it is unit-testable.

Convention (same as pylidc): volume is indexed [i (row), j (col), k (slice)],
with k following increasing z. Patches are resampled to 1 mm isotropic voxels
around a centre given in (fractional) volume-index coordinates.
"""
import numpy as np
from scipy.ndimage import map_coordinates


def crop_patch(vol, pixel_spacing, slice_zvals, center_ijk, size=96, order=1, cval=-1024):
    """Return a (size, size, size) patch at 1 mm spacing centred on center_ijk.

    vol           : 3D array [i, j, k]
    pixel_spacing : in-plane spacing in mm (rows == cols in LIDC)
    slice_zvals   : increasing z positions (mm) of each slice, length vol.shape[2]
    center_ijk    : (i, j, k) float centre in index space
    The patch centre voxel is index size//2 along every axis.
    Regions outside the scan are filled with cval (air).
    """
    zv = np.asarray(slice_zvals, dtype=np.float64)
    ks = np.arange(len(zv), dtype=np.float64)
    offs = np.arange(size, dtype=np.float64) - size // 2  # mm offsets

    ci, cj, ck = center_ijk
    z_center = np.interp(ck, ks, zv)  # handles non-uniform slice spacing

    ii = ci + offs / pixel_spacing
    jj = cj + offs / pixel_spacing

    z_t = z_center + offs
    kk = np.interp(z_t, zv, ks)
    kk[(z_t < zv[0]) | (z_t > zv[-1])] = -1e6  # outside scan -> cval

    I, J, K = np.meshgrid(ii, jj, kk, indexing="ij")
    return map_coordinates(vol, [I, J, K], order=order, mode="constant", cval=cval)
