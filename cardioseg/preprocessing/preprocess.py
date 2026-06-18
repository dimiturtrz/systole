"""Preprocess ACDC frames for segmentation, with a param-keyed disk cache.

ACDC is highly anisotropic (in-plane ~1.4-1.6 mm, slice 10 mm -> 6-7x). The
standard recipe for a 2D model: resample only the in-plane axes to a common
spacing, leave slices alone, then z-score normalize per volume (intensity is
uncalibrated and varies frame to frame). Masks resample with nearest-neighbour
so labels stay integer (no interpolation across 0/1/2/3).

Processed arrays are cached under CARDIAC_PROCESSED_ROOT (default
D:/data/processed/mri/acdc), keyed by the preprocessing params, so re-runs with
the same params are instant and different params never collide.
"""
import os
from pathlib import Path

import numpy as np

from cardioseg.data.mri.data import load_ed_es
from cardioseg.types import Image, Spacing, Volume

PROCESSED_ROOT = os.environ.get("CARDIAC_PROCESSED_ROOT", "D:/data/processed/mri/acdc")


def zscore(img: Image, eps: float = 1e-6) -> Image:
    """Per-volume z-score on the whole array (uncalibrated intensity -> zero-mean)."""
    img = img.astype(np.float32)
    return (img - img.mean()) / (img.std() + eps)


def resample_inplane(
    arr: Volume, spacing: Spacing, target_inplane: float = 1.5, is_mask: bool = False
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


def _cache_path(patient_name, target_inplane):
    tag = f"inplane{str(target_inplane).replace('.', 'p')}"
    return Path(PROCESSED_ROOT) / tag / f"{patient_name}.npz"


def preprocess_case(
    patient_dir: str | Path, target_inplane: float = 1.5, use_cache: bool = True
) -> dict:
    """Load + resample + normalize a patient's ED/ES. Returns dict with keys
    ed_img, ed_gt, es_img, es_gt (each [D, H, W]), spacing (z,y,x), group.
    Caches to disk per params.
    """
    patient_dir = Path(patient_dir)
    cache = _cache_path(patient_dir.name, target_inplane)
    if use_cache and cache.exists():
        z = np.load(cache, allow_pickle=True)
        return {k: z[k] for k in z.files} | {"group": str(z["group"])}

    d = load_ed_es(patient_dir)
    sp = d["spacing"]
    out = {"group": d.get("group"), "patient": patient_dir.name}
    new_sp = sp
    for tag in ("ED", "ES"):
        if tag not in d:
            continue
        img, isp = resample_inplane(d[tag]["img"], sp, target_inplane, is_mask=False)
        gt, _ = resample_inplane(d[tag]["gt"], sp, target_inplane, is_mask=True)
        out[f"{tag.lower()}_img"] = zscore(img)
        out[f"{tag.lower()}_gt"] = gt
        new_sp = isp
    out["spacing"] = np.asarray(new_sp, dtype=np.float32)

    if use_cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **out)
    return out


if __name__ == "__main__":
    import argparse

    from cardioseg.data.mri.data import acdc_cases

    ap = argparse.ArgumentParser()
    ap.add_argument("--inplane", type=float, default=1.5)
    ap.add_argument("--n", type=int, default=0, help="0 = all patients")
    args = ap.parse_args()

    cases = acdc_cases()
    cases = cases if args.n == 0 else cases[: args.n]
    print(f"preprocessing {len(cases)} cases -> {PROCESSED_ROOT} (inplane={args.inplane}mm)")
    for pd in cases:
        r = preprocess_case(pd, target_inplane=args.inplane)
        print(f"  {r['patient']:11} {r['group']:5} ed{tuple(r['ed_img'].shape)} "
              f"spacing={tuple(round(float(s),2) for s in r['spacing'])}")
