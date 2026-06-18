"""Synthetic cardiac-like volumes + masks — lets the whole pipeline run before real data.

Label convention matches ACDC: 0 background, 1 LV blood pool, 2 myocardium, 3 RV.
"""
import numpy as np

LABELS = {"bg": 0, "lv": 1, "myo": 2, "rv": 3}


def make_volume(shape=(16, 128, 128), spacing=(8.0, 1.5, 1.5), lv_radius=18,
                myo_thick=6, rv_offset=34, rv_radius=16, seed=0):
    """Return (image, mask, spacing).

    Concentric LV cavity + myocardium ring + an RV blob, restricted to a mid-slab
    (apex/base left empty, as in real short-axis stacks). spacing is (z, y, x) mm.
    """
    rng = np.random.default_rng(seed)
    D, H, W = shape
    zz, yy, xx = np.mgrid[0:D, 0:H, 0:W]
    cy, cx = H / 2, W / 2
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    mask = np.zeros(shape, dtype=np.uint8)
    mask[r <= lv_radius] = LABELS["lv"]
    mask[(r > lv_radius) & (r <= lv_radius + myo_thick)] = LABELS["myo"]
    rv = np.sqrt((yy - cy) ** 2 + (xx - (cx + rv_offset)) ** 2)
    mask[rv <= rv_radius] = LABELS["rv"]
    mask[: D // 6] = 0
    mask[-D // 6:] = 0

    image = np.zeros(shape, dtype=np.float32)
    image[mask == LABELS["lv"]] = 0.90
    image[mask == LABELS["myo"]] = 0.50
    image[mask == LABELS["rv"]] = 0.85
    image += rng.normal(0, 0.05, shape).astype(np.float32)
    return image, mask, np.asarray(spacing, dtype=np.float32)


def ed_es_pair(seed=0, **kw):
    """End-diastolic (large LV) + end-systolic (small LV) volumes, for EF."""
    ed = make_volume(lv_radius=20, seed=seed, **kw)
    es = make_volume(lv_radius=13, seed=seed + 1, **kw)
    return ed, es
