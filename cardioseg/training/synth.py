"""SynthSeg-style synthetic-image generation from labels (Tier 2, bd cardiac-seg-bgc).

The *invent* force of domain generalization — the third sibling to `augment.py` (*diversify*: perturb
real pixels) and `preprocessing/normalization/` (*strip*: remove vendor variance). Throw away the real
intensities and PAINT each label class from a sampled Gaussian, so every call is a new contrast +
(with deform) new anatomy. Train on these and the net must segment by shape/structure, not one
scanner's appearance -> generalizes across vendors. Anatomy seeds from the ACDC mask; the picture is
invented.

Pattern: SynthSeg (Billot 2023) but with the cardiac-critical fix — cardiac labels are FOV-sparse
(heart-only), and pure-random U[0,1] per-class intensities (the brain recipe) DESTROY the one learnable
cue: blood pools are bright, myocardium dark. A net trained on randomized contrast learns synth-only
structure and fails to transfer (measured: pure-random pure-synth -> 0.24-0.32 cross-vendor Dice vs
0.86 real). The fix (DRIFTS-style per-label intensity clustering): sample each class around its REAL
measured per-class mean/std (`measure_class_stats`), with bounded jitter — realistic ordering kept,
magnitude varied for robustness. So the images actually look like cardiac MRI.

Pipeline per call: nonlinear label deform -> per-label Gaussian paint around real priors -> smooth bias
field -> random blur -> Rician noise -> z-score. Geometry (flip/rotate/scale) is `augment.py`'s job,
runs after. GPU-batched, vectorized; torch global RNG (seed for repro). Config = injected SynthCfg.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from core.hparams import SynthCfg

from .augment import _gaussian_kernel


def measure_class_stats(X: torch.Tensor, Y: torch.Tensor,
                        n_classes: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-class (mean, std) of the REAL z-scored images — the realistic intensity priors that make
    synth look like MRI. X [B,1,H,W], Y [B,H,W]. A class absent from the sample falls back to (0, 1).
    Returned tensors live on X's device."""
    flat_x, flat_y = X[:, 0].reshape(-1), Y.reshape(-1)
    means = torch.zeros(n_classes, device=X.device)
    stds = torch.ones(n_classes, device=X.device)
    for c in range(n_classes):
        v = flat_x[flat_y == c]
        if v.numel() > 10:
            means[c] = v.mean()
            stds[c] = v.std().clamp_min(1e-3)
    return means, stds


def _deform_grid(b: int, h: int, w: int, amp: float, dev) -> torch.Tensor:
    """Smooth random displacement field as a grid_sample grid [B,H,W,2]. Coarse 5x5 control points
    (U[-amp,amp]) bicubic-upsampled to full res -> low-frequency elastic warp, added to the identity
    grid. amp is in normalized [-1,1] coords (0.15 ≈ 15% of half-FOV max local shift)."""
    ctrl = (torch.rand(b, 2, 5, 5, device=dev) * 2 - 1) * amp
    disp = F.interpolate(ctrl, size=(h, w), mode="bicubic", align_corners=False)   # [B,2,H,W]
    ident = F.affine_grid(torch.eye(2, 3, device=dev).expand(b, 2, 3), (b, 1, h, w),
                          align_corners=False)                                      # [B,H,W,2]
    return ident + disp.permute(0, 2, 3, 1)


def synthesize_from_labels(mask: torch.Tensor, cfg: SynthCfg, n_classes: int,
                           priors: tuple[torch.Tensor, torch.Tensor] | None = None,
                           real_img: torch.Tensor | None = None
                           ) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a synthetic z-scored image (and its label map) from an integer label mask.

    mask [B,H,W] long (labels 0..n_classes-1) -> (img [B,1,H,W] z-scored, mask [B,H,W] long).
    cfg.deform>0 warps the labels first (new anatomy; returned mask is the warped one so the target
    stays aligned). With cfg.realistic + `priors`=(means,stds) from measure_class_stats, each class is
    painted around its REAL mean/std (jittered) — the recipe that transfers. Else legacy pure-random
    U[cfg.mu]/U[cfg.sigma] (ablation only). Background (label 0) is painted too -> full image.
    """
    b = mask.shape[0]
    dev = mask.device
    mask = mask.long()
    hybrid = cfg.keep_real_bg and real_img is not None
    if cfg.deform > 0 and not hybrid:                # hybrid: keep heart aligned with real bg's hole
        grid = _deform_grid(b, *mask.shape[-2:], cfg.deform, dev)
        mask = F.grid_sample(mask[:, None].float(), grid, mode="nearest",
                             padding_mode="border", align_corners=False)[:, 0].long()
    oh = F.one_hot(mask, n_classes).permute(0, 3, 1, 2).float()             # [B,C,H,W]

    # --- per-label Gaussian paint ---
    if cfg.realistic and priors is not None:
        pmean, pstd = priors                                                # [C] real measured priors
        mu = pmean[None, :] + cfg.jitter * torch.randn(b, n_classes, device=dev)   # ordering kept, jittered
        ss_lo, ss_hi = cfg.std_scale
        sg = pstd[None, :] * (torch.rand(b, n_classes, device=dev) * (ss_hi - ss_lo) + ss_lo)
    else:                                                                    # legacy pure-random (ablation)
        mu_lo, mu_hi = cfg.mu
        sg_lo, sg_hi = cfg.sigma
        mu = torch.rand(b, n_classes, device=dev) * (mu_hi - mu_lo) + mu_lo
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

    # --- hybrid diagnostic: paste the synth heart onto the REAL background (heart voxels = synth,
    #     everything else = real). Isolates whether bg realism is the wall for pure-synth. ---
    if hybrid:
        fg = (mask > 0)[:, None]
        img = torch.where(fg, img, real_img)

    # --- z-score per sample (match the real preprocessed input distribution) ---
    m = img.mean((1, 2, 3), keepdim=True)
    s = img.std((1, 2, 3), keepdim=True).clamp_min(1e-6)
    return (img - m) / s, mask
