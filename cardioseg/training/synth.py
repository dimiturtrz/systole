"""SynthSeg-style synthetic-image generation from labels (Tier 2, bd cardiac-seg-bgc).

The *invent* force of domain generalization — the third sibling to `augment.py` (*diversify*: perturb
real pixels) and `preprocessing/normalization/` (*strip*: remove vendor variance). Here we throw away
the real intensities entirely and PAINT each label class with a freshly-sampled Gaussian, so every
call produces a brand-new contrast. Train on a mix of these and the model can't latch onto any one
scanner's appearance — it must segment by anatomy/shape -> contrast-AGNOSTIC, generalizes to unseen
vendors (the cross-vendor lane). Anatomy is real (the ACDC mask); only the picture is invented.

Pattern: SynthSeg (Billot et al. 2023, Med Image Analysis). Pipeline per call: per-label GMM paint ->
smooth bias field -> random blur (resolution variation) -> Rician noise -> z-score (match the real
preprocessed input distribution). Geometry (flip/rotate/scale of BOTH image+mask) is `augment.py`'s
job and runs after, so this module keeps the mask untouched and does intensity only.

GPU-batched, per-sample, vectorized — same idioms as `augment.py` (no python loop, torch global RNG;
seed via torch.manual_seed for repro). Config = the injected SynthCfg.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from core.hparams import SynthCfg

from .augment import _gaussian_kernel


def synthesize_from_labels(mask: torch.Tensor, cfg: SynthCfg, n_classes: int) -> torch.Tensor:
    """Generate a synthetic z-scored image from an integer label mask.

    mask [B,H,W] long (canonical labels 0..n_classes-1) -> img [B,1,H,W] float, z-scored per sample.
    Each class gets its own per-sample mean (U[cfg.mu]) and texture std (U[cfg.sigma]) -> a fresh
    random contrast every call. Background (label 0) is painted too (it's tissue/air with its own
    invented intensity), so the net sees a full image, not a masked one.
    """
    b = mask.shape[0]
    dev = mask.device
    oh = F.one_hot(mask.long(), n_classes).permute(0, 3, 1, 2).float()      # [B,C,H,W]

    # --- per-label GMM paint: each (sample, class) gets a random mean + texture std ---
    mu_lo, mu_hi = cfg.mu
    sg_lo, sg_hi = cfg.sigma
    mu = torch.rand(b, n_classes, device=dev) * (mu_hi - mu_lo) + mu_lo      # [B,C]
    sg = torch.rand(b, n_classes, device=dev) * (sg_hi - sg_lo) + sg_lo
    mu_map = (oh * mu[:, :, None, None]).sum(1, keepdim=True)                # [B,1,H,W] class mean
    sg_map = (oh * sg[:, :, None, None]).sum(1, keepdim=True)               # [B,1,H,W] class std
    img = mu_map + sg_map * torch.randn(b, 1, *mask.shape[-2:], device=dev)  # painted texture

    # --- smooth multiplicative bias field (coarse 4x4 -> bilinear upsample; the N4 dual) ---
    if cfg.bias_strength > 0:
        low = torch.rand(b, 1, 4, 4, device=dev) * 2 - 1
        field = 1.0 + cfg.bias_strength * F.interpolate(low, size=img.shape[-2:], mode="bilinear",
                                                        align_corners=False)
        img = img * field

    # --- random Gaussian blur (resolution variation); single σ per call (varies every batch) ---
    bl_lo, bl_hi = cfg.blur
    sigma = float(torch.rand(1, device=dev) * (bl_hi - bl_lo) + bl_lo)
    if sigma > 0.05:
        k = _gaussian_kernel(sigma).to(dev)
        k = k.view(1, 1, *k.shape)
        img = F.conv2d(img, k, padding=k.shape[-1] // 2)

    # --- Rician noise (MRI magnitude noise: sqrt of two independent Gaussian channels) ---
    if cfg.noise > 0:
        re = img + cfg.noise * torch.randn_like(img)
        im = cfg.noise * torch.randn_like(img)
        img = torch.sqrt(re * re + im * im)

    # --- z-score per sample (match the real preprocessed input distribution) ---
    m = img.mean((1, 2, 3), keepdim=True)
    s = img.std((1, 2, 3), keepdim=True).clamp_min(1e-6)
    return (img - m) / s
