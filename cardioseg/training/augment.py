"""GPU-batched training augmentation — the *diversify* force of domain generalization.

Augmentation here means **perturbing real images** so the model learns invariance to nuisance
variance (the cross-vendor / acquisition shift). It is the dual of the *strip* force, N4 /
histogram standardization in `preprocessing/normalization/` — see that package's README for the
full variance taxonomy + the strip-vs-diversify decision per factor. NB augmentation (perturb what
is real) is distinct from *synthetic generation* (invent images from labels — SynthSeg / Bloch sim);
that is a separate concern with its own home, not this file.

Two kinds, both per-sample and vectorized (no python loop, no scipy — the old per-item CPU recipe
starved the GPU):
- **geometric** (image AND mask together, one affine grid_sample) — flip/rotate/scale → shape /
  orientation invariance; helps the thin RV/myo.
- **intensity** (image only) — gamma, contrast, blur, noise → scanner/vendor contrast + resolution
  variation.

Idioms (match these when adding transforms): per-sample params via `torch.rand(b,...)`; probabilistic
application via `do_X = (rand < p)` masks (`out = do_X*transformed + (1-do_X)*orig`).

Input is **already z-scored** (post-preprocess), so intensity ops are domain-*randomization* in
z-space, not physics-exact — fine for invariance, but a true multiplicative bias field would have to
run pre-z-score. Planned MRI-physics transforms (Tier 1, `bd cardiac-seg-jp1`): smooth bias-field
modulation (the N4 dual), Rician noise (we use plain Gaussian today), k-space ghost/spike. Histogram
matching is the *targeted* harmonization method — it needs a vendor reference, so it lives in
`normalization/`, not here.

Hyperparams from the injected AugCfg. Uses torch's global RNG — seed it (torch.manual_seed) for repro.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from cardioseg.hparams import AugCfg

_GAUSS3 = torch.tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]) / 16.0  # 3x3 blur kernel


def augment_batch(
    img: torch.Tensor,
    mask: torch.Tensor,
    cfg: AugCfg | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Augment a batch on its device. img [B,1,H,W] float, mask [B,H,W] long. Returns (img, mask).

    Per-sample flip * rotate * scale via a single affine grid_sample (bilinear image, nearest
    mask so labels stay integer; out-of-frame -> 0/background). Per-sample gamma, contrast,
    gaussian noise, and an occasional 3x3 blur on the image only. Hyperparams from the injected
    AugCfg. Uses torch's global RNG — seed it (torch.manual_seed) for reproducibility.
    """
    cfg = cfg or AugCfg()
    rot_deg, scale = cfg.rot_deg, cfg.scale
    b, _, _, _ = img.shape
    dev, dt = img.device, img.dtype

    # --- geometric: per-sample affine (inverse map, as grid_sample expects) ---
    ang = (torch.rand(b, device=dev) * 2 - 1) * (rot_deg * math.pi / 180.0)
    inv = 1.0 / (torch.rand(b, device=dev) * (scale[1] - scale[0]) + scale[0])  # 1/scale
    fx = torch.where(torch.rand(b, device=dev) < 0.5, -1.0, 1.0)               # horizontal flip
    fy = torch.where(torch.rand(b, device=dev) < 0.5, -1.0, 1.0)               # vertical flip
    cos, sin = torch.cos(ang), torch.sin(ang)
    theta = torch.zeros(b, 2, 3, device=dev, dtype=dt)
    theta[:, 0, 0] = fx * cos * inv
    theta[:, 0, 1] = fx * -sin * inv
    theta[:, 1, 0] = fy * sin * inv
    theta[:, 1, 1] = fy * cos * inv
    grid = F.affine_grid(theta, img.shape, align_corners=False)
    img = F.grid_sample(img, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    mask = F.grid_sample(mask[:, None].to(dt), grid, mode="nearest", padding_mode="zeros",
                         align_corners=False)[:, 0].long()

    # --- intensity (image only), per-sample, vectorized ---
    mn = img.amin((1, 2, 3), keepdim=True)
    rng = (img.amax((1, 2, 3), keepdim=True) - mn).clamp_min(1e-6)
    g_lo, g_hi = cfg.gamma
    gamma = torch.rand(b, 1, 1, 1, device=dev) * (g_hi - g_lo) + g_lo
    do_g = (torch.rand(b, 1, 1, 1, device=dev) < cfg.gamma_p).to(dt)
    img = do_g * (((img - mn) / rng) ** gamma * rng + mn) + (1 - do_g) * img    # gamma where selected

    k = _GAUSS3.to(dev, dt).view(1, 1, 3, 3)
    do_b = (torch.rand(b, 1, 1, 1, device=dev) < cfg.blur_p).to(dt)
    img = do_b * F.conv2d(img, k, padding=1) + (1 - do_b) * img                 # occasional blur

    c_lo, c_hi = cfg.contrast
    contrast = torch.rand(b, 1, 1, 1, device=dev) * (c_hi - c_lo) + c_lo
    img = img * contrast + torch.randn_like(img) * cfg.noise                    # contrast + noise
    return img, mask
