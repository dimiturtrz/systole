"""N4 bias-field correction (scan-bucket, knowable, physical).

Coil/receive inhomogeneity makes a smooth multiplicative brightness gradient across each scan —
the same tissue reads brighter on one side. N4 (Tustison 2010) estimates that smooth log-bias
field and divides it out. Per-scan, image-derived — no training, no reference, no assumption about
other scanners. Stateless: runs at preprocess time, result cached.

Order in the pipeline: resample -> **N4** -> intensity-norm (z-score/Nyúl). N4 first because it's
spatial (fixes *where* the brightness drifts); intensity-norm is global (fixes the overall scale).
"""
from __future__ import annotations

import numpy as np

from cardioseg.types import Spacing, Volume


def n4_bias(vol: Volume, spacing: Spacing | None = None, shrink: int = 4,
            iters=(50, 50, 50), fwhm: float = 0.15) -> Volume:
    """N4-correct one [D, H, W] volume. Returns the corrected volume (same shape/dtype-ish float32).

    `shrink` downsamples for the fit (speed); the estimated field is applied at full resolution.
    Otsu foreground mask so air doesn't drive the fit. Falls back to the input on any ITK error.
    """
    import SimpleITK as sitk

    arr = vol.astype(np.float32)
    img = sitk.GetImageFromArray(arr)                  # [D,H,W] -> (x,y,z) internally
    if spacing is not None:
        img.SetSpacing(tuple(float(s) for s in spacing[::-1]))   # (z,y,x) -> sitk (x,y,z)
    try:
        mask = sitk.OtsuThreshold(img, 0, 1, 200)
        sh = [shrink, shrink, max(1, shrink // 2)]      # shrink less along z (few, thick slices)
        corr = sitk.N4BiasFieldCorrectionImageFilter()
        corr.SetMaximumNumberOfIterations(list(iters))
        corr.SetBiasFieldFullWidthAtHalfMaximum(fwhm)
        corr.Execute(sitk.Shrink(img, sh), sitk.Shrink(mask, sh))   # fit on the shrunk volume
        log_field = corr.GetLogBiasFieldAsImage(img)    # but evaluate the field at full res
        out = img / sitk.Exp(log_field)
        return sitk.GetArrayFromImage(out).astype(np.float32)
    except Exception:
        return arr                                      # ITK hiccup -> pass through, never break the run
