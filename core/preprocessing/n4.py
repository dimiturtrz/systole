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

from core.types import Spacing, Volume


def n4_bias(vol: Volume, spacing: Spacing | None = None, shrink: int = 4,
            iters=(50, 50, 50), fwhm: float = 0.15) -> Volume:
    """N4-correct one [D,H,W] volume — SimpleITK (the reference, correct implementation).

    NOTE: `n4_gpu` (pure-torch, GPU) is built and fast (~60 ms) but currently UNDER-CORRECTS vs
    SimpleITK (the histogram-sharpening needs tuning) — so it is NOT wired in here yet. Once it's
    parity-validated against SimpleITK it replaces this on the hot path (bd cardiac-seg-mdc).
    """
    return _n4_sitk(vol, spacing, shrink, iters, fwhm)


def _smooth3d(v, sigma: float):
    """Separable 3D Gaussian blur (the smooth bias field) via conv3d. v: [D,H,W] tensor."""
    import torch
    import torch.nn.functional as F
    k = max(3, int(6 * sigma) | 1)
    r = torch.arange(k, device=v.device, dtype=v.dtype) - k // 2
    g = torch.exp(-0.5 * (r / sigma) ** 2); g = g / g.sum()
    x = v[None, None]
    for dim in (2, 3, 4):
        shape = [1, 1, 1, 1, 1]; shape[dim] = k
        pad = [0, 0, 0, 0, 0, 0]; pad[(4 - dim) * 2] = pad[(4 - dim) * 2 + 1] = k // 2
        x = F.conv3d(F.pad(x, pad, mode="replicate"), g.view(shape))
    return x[0, 0]


def n4_gpu(vol: Volume, spacing: Spacing | None = None, device: str = "cuda",
           iters: int = 8, bins: int = 200, fwhm: float = 0.15) -> Volume:
    """N4 bias-field correction in pure torch (runs on `device`; CUDA = fast, no custom kernels).

    The N4 mechanism: in the log domain, iterate {sharpen the intensity histogram (deconvolve a
    Gaussian PSF) -> map each voxel to its expected uncorrupted value -> the residual is the
    estimated bias -> smooth it (the field) -> subtract}. Histogram sharpening is what separates
    bias from real low-frequency anatomy (vs a naive low-pass). All ops are tensor ops.
    """
    import torch

    x = torch.as_tensor(vol, dtype=torch.float32, device=device)
    pos = x[x > 0]
    if pos.numel() < 16:
        return vol.astype("float32")
    mask = x > 0.1 * pos.mean()                                 # crude foreground (air ~ 0)
    logx = torch.log(torch.clamp(x, min=1e-6))
    sigma = max(8.0, 0.12 * max(x.shape[-2:]))                  # broad in-plane smoothness (bias is low-freq)
    field = torch.zeros_like(logx)

    for _ in range(iters):
        u = logx - field
        um = u[mask]
        lo, hi = float(um.min()), float(um.max())
        if hi - lo < 1e-6:
            break
        centers = torch.linspace(lo, hi, bins, device=device)
        h = torch.histc(um, bins=bins, min=lo, max=hi)
        # Gaussian PSF in intensity space; Wiener-deconvolve the histogram (sharpen)
        sig_b = max(1.0, (fwhm * (hi - lo)) / ((hi - lo) / bins) / 2.355)
        off = torch.arange(bins, device=device) - bins // 2
        psf = torch.exp(-0.5 * (off / sig_b) ** 2); psf = psf / psf.sum()
        H, G = torch.fft.rfft(h), torch.fft.rfft(torch.fft.ifftshift(psf))
        e = torch.fft.irfft(H * G.conj() / (G.abs() ** 2 + 1e-2 * (G.abs() ** 2).max()), n=bins).clamp(min=0)
        # expected-value lookup E(bin) = soft mean of centers weighted by gaussian*e  (bins x bins, cheap)
        W = torch.exp(-0.5 * ((centers[:, None] - centers[None, :]) / (sig_b * (hi - lo) / bins)) ** 2) * e[None, :]
        e_lut = (W * centers[None, :]).sum(1) / (W.sum(1) + 1e-8)
        # map each voxel u -> E via linear interp over e_lut
        idx = ((u - lo) / (hi - lo) * (bins - 1)).clamp(0, bins - 1)
        i0 = idx.floor().long(); frac = idx - i0; i1 = (i0 + 1).clamp(max=bins - 1)
        E = e_lut[i0] * (1 - frac) + e_lut[i1] * frac
        resid = torch.where(mask, u - E, torch.zeros_like(u))
        field = field + _smooth3d(resid, sigma)                 # accumulate the smooth bias

    return torch.exp(logx - field).cpu().numpy().astype("float32")


def _n4_sitk(vol: Volume, spacing: Spacing | None = None, shrink: int = 4,
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
