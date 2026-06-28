"""The per-subject preprocessing transform: load -> resample in-plane -> (N4) -> z-score.

ACDC is highly anisotropic (in-plane ~1.4-1.6 mm, slice 10 mm -> 6-7x). The standard recipe for a
2D model: resample only the in-plane axes to a common spacing, leave slices alone, then z-score
per volume (intensity is uncalibrated and varies frame to frame). Masks resample with
nearest-neighbour so labels stay integer (no interpolation across 0/1/2/3).

Pure: returns arrays, no disk I/O. The consolidated store (data/store.py) calls this and owns the
on-disk processed/<dataset>/<paramkey>/ layout + caching.
"""
from pathlib import Path

import numpy as np

from core.config import DEFAULT_INPLANE
from cardioseg.data.mri.acdc import load_ed_es
from core.types import Image, Spacing, Volume

# In-plane resample target (mm). ACDC/M&M in-plane is ~1.2-1.6 mm; 1.5 is the common grid the 2D
# model trains on. The single source of truth. Slices (z) are left untouched (2D-model convention).
TARGET_INPLANE = DEFAULT_INPLANE
ZSCORE_EPS = 1e-6                            # guards div-by-zero on a flat (all-air) volume


def zscore(img: Image, eps: float = ZSCORE_EPS) -> Image:
    """Per-volume z-score on the whole array (uncalibrated intensity -> zero-mean)."""
    img = img.astype(np.float32)
    return (img - img.mean()) / (img.std() + eps)


def resample_inplane(
    arr: Volume, spacing: Spacing, target_inplane: float = TARGET_INPLANE, is_mask: bool = False
) -> tuple[Volume, Spacing]:
    """Resample H,W (not D) of a [D, H, W] array to target_inplane mm.

    Returns (resampled [D, H', W'], new_spacing (z, target, target)). Slices (z)
    are preserved — a 2D-model convention on anisotropic cardiac data. Masks use
    order=0 to keep labels integer.
    """
    from scipy.ndimage import zoom

    zsp, ysp, xsp = spacing
    fy, fx = ysp / target_inplane, xsp / target_inplane
    order = 0 if is_mask else 1
    out = zoom(arr, (1.0, fy, fx), order=order)
    if is_mask:
        out = np.rint(out).astype(np.uint8)
    new_spacing = (zsp, target_inplane, target_inplane)
    return out, new_spacing


def preprocess_case(
    patient_dir: str | Path, target_inplane: float = TARGET_INPLANE,
    loader=load_ed_es, n4: bool = False, n4_params=None,
) -> dict:
    """Load + resample + (N4) + z-score one subject's ED/ES. PURE (no disk I/O).

    Returns dict: ed_img, ed_gt, es_img, es_gt (each [D, H, W]), spacing (z,y,x), group, patient.
    `loader` is the dataset adapter's load_ed_es (labels already canonical). `n4` runs N4 bias-field
    correction (resample -> N4 -> z-score) with `n4_params` (an N4Cfg; defaults if None), physical +
    per-scan. The store (data/store.py) writes the result to disk; this just computes it.
    """
    patient_dir = Path(patient_dir)
    d = loader(patient_dir)
    sp = d["spacing"]
    out = {"group": d.get("group"), "patient": patient_dir.name}
    new_sp = sp
    for tag in ("ED", "ES"):
        if tag not in d:
            continue
        img, isp = resample_inplane(d[tag]["img"], sp, target_inplane, is_mask=False)
        gt, _ = resample_inplane(d[tag]["gt"], sp, target_inplane, is_mask=True)
        if n4:
            from cardioseg.preprocessing.normalization.n4 import n4_bias
            from core.hparams import N4Cfg
            p = n4_params or N4Cfg()
            img = n4_bias(img, isp, shrink=p.shrink, iters=tuple(p.iters), fwhm=p.fwhm)
        out[f"{tag.lower()}_img"] = zscore(img)
        out[f"{tag.lower()}_gt"] = gt
        new_sp = isp
    out["spacing"] = np.asarray(new_sp, dtype=np.float32)
    return out
