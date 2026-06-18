"""Pure mask geometry for cardioview — no torch / vtk / IO, so it's cheaply unit-testable."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import label as cc_label


def keep_largest(binary: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component (drops model false-positive islands)."""
    lab, n = cc_label(binary)
    if n <= 1:
        return binary
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0  # ignore background
    return lab == sizes.argmax()


def nearest_index(values, target) -> int:
    """Index of the value closest to `target` (e.g. map an ED/ES frame to a sampled frame)."""
    return min(range(len(values)), key=lambda i: abs(values[i] - target))


def bbox_slices(mask_bool: np.ndarray, spacing, margin_mm: float = 12.0) -> tuple[slice, ...]:
    """Tight bounding box of the True voxels + a mm margin per axis, clamped to the array."""
    sl = []
    for ax, n in enumerate(mask_bool.shape):
        idx = np.any(mask_bool, axis=tuple(a for a in range(mask_bool.ndim) if a != ax)).nonzero()[0]
        if len(idx) == 0:
            sl.append(slice(0, n))
            continue
        pad = int(round(margin_mm / spacing[ax]))
        sl.append(slice(max(0, idx[0] - pad), min(n, idx[-1] + 1 + pad)))
    return tuple(sl)
