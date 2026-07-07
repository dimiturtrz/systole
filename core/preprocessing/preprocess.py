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
from scipy.ndimage import zoom

from core.config import DEFAULT_INPLANE, DEFAULT_SIZE
from core.preprocessing.n4 import N4Cfg, n4_bias
from core.preprocessing.nyul import transform as nyul_transform
from core.types import Image, Slice2D, Spacing, Volume

# In-plane resample target (mm). ACDC/M&M in-plane is ~1.2-1.6 mm; 1.5 is the common grid the 2D
# model trains on. The single source of truth. Slices (z) are left untouched (2D-model convention).
TARGET_INPLANE = DEFAULT_INPLANE
SIZE = DEFAULT_SIZE                          # square grid the 2D model runs on (== DataCfg.size)
ZSCORE_EPS = 1e-6                            # guards div-by-zero on a flat (all-air) volume
_MIN_ANCHOR_PX = 50                          # min blood/air px to trust the two-point anchor (else z-score)


def fit_square(arr: Slice2D, size: int, pad_value: float = 0) -> Slice2D:
    """Centre pad/crop a [H, W] array to [size, size] — the model-grid fit, used by both the
    training Dataset and inference (predict_volume stacks fit-squared slices)."""
    h, w = arr.shape
    out = np.full((size, size), pad_value, dtype=arr.dtype)
    # source crop window (centred)
    sh, sw = max(0, (h - size) // 2), max(0, (w - size) // 2)
    src = arr[sh:sh + size, sw:sw + size]
    ch, cw = src.shape
    dh, dw = (size - ch) // 2, (size - cw) // 2
    out[dh:dh + ch, dw:dw + cw] = src
    return out


def stack_slices(slices, size: int, pad_value: float = 0, dtype=None) -> np.ndarray:
    """fit_square each [H,W] slice to the model grid and stack -> [D, size, size]. The one-liner the
    eval modules repeat for GT/label volumes; `dtype` casts the stack (uint8/int64 label maps)."""
    out = np.stack([fit_square(s, size, pad_value) for s in slices])
    return out.astype(dtype) if dtype is not None else out


def blood_anchor(img: Image, gt, blood=(1, 3), eps: float = ZSCORE_EPS) -> Image:
    """Two-point affine intensity normalization: background-air -> 0, blood pool -> 1. Composition-robust
    harmonization anchored on PHYSICAL references (air + blood, present on every scan, meaning the same
    thing) instead of z-score's FOV-sensitive mean/std. Measured to halve cross-vendor tissue-level
    spread vs z-score (bd h8k). Blood level is per-volume; falls back to z-score if blood/air absent.
    ORACLE when `gt` is ground truth (upper bound); at inference a coarse blood/air seg estimates it."""
    bm = np.isin(gt, blood); am = gt == 0
    if int(bm.sum()) < _MIN_ANCHOR_PX or int(am.sum()) < _MIN_ANCHOR_PX:
        return zscore(img)                                   # no anchor available -> safe fallback
    b = float(img[bm].mean()); a = float(np.median(img[am]))
    return (img - a) / ((b - a) + eps)


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
    zsp, ysp, xsp = spacing
    fy, fx = ysp / target_inplane, xsp / target_inplane
    order = 0 if is_mask else 1
    out = zoom(arr, (1.0, fy, fx), order=order)
    if is_mask:
        out = np.rint(out).astype(np.uint8)
    new_spacing = (zsp, target_inplane, target_inplane)
    return out, new_spacing


def preprocess_case(  # noqa: PLR0913  independent preprocessing inputs
    patient_dir: str | Path, loader, target_inplane: float = TARGET_INPLANE,
    n4: bool = False, n4_params=None, nyul_standard=None, norm: str = "zscore",
) -> dict:
    """Load + resample + (N4) + (Nyúl) + z-score one subject's ED/ES. PURE (no disk I/O).

    Returns dict: ed_img, ed_gt, es_img, es_gt (each [D, H, W]), spacing (z,y,x), group, patient.
    `loader` (required) is the dataset adapter's load_ed_es (labels already canonical) — injected so
    this kernel never imports the data layer. `n4` runs N4 bias-field correction (resample -> N4 ->
    z-score) with `n4_params`. `nyul_standard` (a fitted standard landmark vector, or None) applies
    Nyúl histogram standardization BEFORE z-score — harmonization to a cohort standard (qfz). The store
    (data/store.py) writes the result to disk; this just computes it.
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
            p = n4_params or N4Cfg()
            img = n4_bias(img, isp, shrink=p.shrink, iters=tuple(p.iters), fwhm=p.fwhm)
        if nyul_standard is not None:
            img = nyul_transform(img, np.asarray(nyul_standard))
        out[f"{tag.lower()}_img"] = blood_anchor(img, gt) if norm == "blood" else zscore(img)
        out[f"{tag.lower()}_gt"] = gt
        new_sp = isp
    out["spacing"] = np.asarray(new_sp, dtype=np.float32)
    return out
