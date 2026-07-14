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
from jaxtyping import Float, Integer
from pydantic import BaseModel, Field

from core.config import _VALIDATE
from core.types import shapecheck

_GAUSSIAN_RADIUS_SIGMAS = 3.0    # Gaussian kernel half-width in units of σ (covers ~99.7% of the mass)


class AugCfg(BaseModel):
    """GPU-batched augmentation. Geometric widths conservative; intensity widths broad to span the
    cross-vendor contrast gap. Injected into augment_batch."""
    model_config = _VALIDATE
    rot_deg: float = Field(20.0, ge=0)
    scale: tuple[float, float] = (0.85, 1.15)
    translate: float = Field(0.0, ge=0)            # per-axis random shift, FRACTION of FOV. Real ACDC
    #                                                heart-centroid offset from FOV center: radial p95 0.17
    #                                                (measured, 150 pts); synth hearts voxelize DEAD-CENTER.
    #                                                MEASURED NEUTRAL on zero-real (TEST +0.009 = noise): a
    #                                                fully-conv U-Net is translation-equivariant, so absolute
    #                                                position barely matters. OPT-IN (0.12 ~ real p95); kept
    #                                                (correct aug, defensible) but OFF so it can't silently
    #                                                perturb the flagship real run unmeasured. Derived range.
    gamma: tuple[float, float] = (0.7, 1.5)
    gamma_p: float = Field(0.3, ge=0, le=1)
    blur_p: float = Field(0.2, ge=0, le=1)
    contrast: tuple[float, float] = (0.8, 1.2)
    noise: float = Field(0.08, ge=0)
    # MRI-physics aug (Tier 1, scan bucket). bias_p=0 -> off (default = old behavior).
    bias_p: float = Field(0.0, ge=0, le=1)         # prob of a smooth bias-field modulation
    bias_strength: float = Field(0.3, ge=0)        # max +/- fractional field deviation across the FOV
    # Soft-label training: Gaussian-blur the one-hot target by this σ (voxels) so boundaries are
    # probabilistic (honest partial-volume targets). 0 = off (crisp one-hot = hard labels). Selects
    # the SoftDiceCE loss when >0. NOT fit to EF — a uniform boundary-uncertainty prior. DEFAULT 1.0:
    # soft labels are the standard recipe (better calibrated, ECE -13%, equal Dice/EF — see
    # research/deep_dives/2026-06-29_soft-labels-calibration-vs-ef.md).
    soft_label_sigma: float = Field(1.0, ge=0)

class Augmentor:
    """GPU-batched augmentation engine bound to its AugCfg: construct once with the session config, then
    `augment_batch` many batches. The per-batch img/mask stay method args; only the (configured-once)
    hyperparams are held on the instance. The pure helpers — the separable Gaussian kernel and the
    one-hot->soft target `soften` (its sigma/n_classes threaded per call) — stay static. Uses torch's
    global RNG — seed it (torch.manual_seed) for reproducibility."""

    _GAUSS3 = torch.tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]) / 16.0  # 3x3 blur kernel
    _FLIP_PROB = 0.5   # per-sample probability of a horizontal / vertical flip

    def __init__(self, cfg: AugCfg | None = None):
        self.cfg = cfg or AugCfg()                               # session hyperparams (geometric + intensity)

    @staticmethod
    @shapecheck
    def gaussian_kernel(sigma: float) -> Float[torch.Tensor, "k k"]:
        """Separable 2D Gaussian kernel (sum=1), radius 3σ."""
        r = max(1, math.ceil(_GAUSSIAN_RADIUS_SIGMAS * sigma))
        xs = torch.arange(-r, r + 1, dtype=torch.float32)
        g = torch.exp(-(xs ** 2) / (2.0 * sigma * sigma))
        g = g / g.sum()
        return torch.outer(g, g)

    @staticmethod
    @shapecheck
    def soften(mask: Integer[torch.Tensor, "b h w"], sigma: float,
               n_classes: int) -> Float[torch.Tensor, "b c h w"]:
        """Hard integer mask [B, H, W] -> soft probabilistic target [B, C, H, W] (channels sum to 1).

        One-hot, then per-class Gaussian blur (boundary-uncertainty width ≈ σ voxels), then renormalize
        so each voxel's class probs sum to 1. Result is soft only at boundaries (blurred one-hots overlap
        there) and stays ~hard in class interiors. σ is a principled width (~partial-volume scale), NOT a
        knob tuned to EF. σ<=0 -> crisp one-hot (== hard target). The honest-target representation: a
        boundary voxel is a mix, so its label is a distribution, not a 0/1."""
        oh = F.one_hot(mask.long(), n_classes).permute(0, 3, 1, 2).float()   # [B, C, H, W]
        if not sigma or sigma <= 0:
            return oh
        k = Augmentor.gaussian_kernel(sigma).to(oh.device, oh.dtype)
        pad = k.shape[-1] // 2
        kk = k.view(1, 1, *k.shape).expand(n_classes, 1, *k.shape)
        oh = F.conv2d(oh, kk, padding=pad, groups=n_classes)
        return oh / oh.sum(dim=1, keepdim=True).clamp_min(1e-6)

    @shapecheck
    def augment_batch(
        self,
        img: Float[torch.Tensor, "b 1 h w"],
        mask: Integer[torch.Tensor, "b h w"],
    ) -> tuple[Float[torch.Tensor, "b 1 h w"], Integer[torch.Tensor, "b h w"]]:
        """Augment a batch on its device. img [B,1,H,W] float, mask [B,H,W] long. Returns (img, mask).

        Per-sample flip * rotate * scale via a single affine grid_sample (bilinear image, nearest
        mask so labels stay integer; out-of-frame -> 0/background). Per-sample gamma, contrast,
        gaussian noise, and an occasional 3x3 blur on the image only. Hyperparams from this instance's
        AugCfg. Uses torch's global RNG — seed it (torch.manual_seed) for reproducibility.
        """
        cfg = self.cfg
        rot_deg, scale = cfg.rot_deg, cfg.scale
        b, _, _, _ = img.shape
        dev, dt = img.device, img.dtype

        # --- geometric: per-sample affine (inverse map, as grid_sample expects) ---
        ang = (torch.rand(b, device=dev) * 2 - 1) * (rot_deg * math.pi / 180.0)
        inv = 1.0 / (torch.rand(b, device=dev) * (scale[1] - scale[0]) + scale[0])  # 1/scale
        fx = torch.where(torch.rand(b, device=dev) < Augmentor._FLIP_PROB, -1.0, 1.0)  # horizontal flip
        fy = torch.where(torch.rand(b, device=dev) < Augmentor._FLIP_PROB, -1.0, 1.0)  # vertical flip
        cos, sin = torch.cos(ang), torch.sin(ang)
        theta = torch.zeros(b, 2, 3, device=dev, dtype=dt)
        theta[:, 0, 0] = fx * cos * inv
        theta[:, 0, 1] = fx * -sin * inv
        theta[:, 1, 0] = fy * sin * inv
        theta[:, 1, 1] = fy * cos * inv
        # translation: FOV-fraction shift -> normalized grid units ([-1,1] spans the full dim, so *2).
        # Gives the synth heart real-like position spread (see AugCfg.translate); no-op at translate=0.
        if cfg.translate > 0:
            theta[:, 0, 2] = (torch.rand(b, device=dev) * 2 - 1) * cfg.translate * 2.0
            theta[:, 1, 2] = (torch.rand(b, device=dev) * 2 - 1) * cfg.translate * 2.0
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

        k = Augmentor._GAUSS3.to(dev, dt).view(1, 1, 3, 3)
        do_b = (torch.rand(b, 1, 1, 1, device=dev) < cfg.blur_p).to(dt)
        img = do_b * F.conv2d(img, k, padding=1) + (1 - do_b) * img                 # occasional blur

        # smooth bias-field modulation (scan bucket; the N4 dual). Coarse 4x4 random field -> bilinear
        # upsample = low-freq, then multiply (1 +/- strength). On z-scored input this is a smooth
        # across-FOV contrast drift, domain-randomization not physics-exact.
        do_bf = (torch.rand(b, 1, 1, 1, device=dev) < cfg.bias_p).to(dt)
        low = torch.rand(b, 1, 4, 4, device=dev, dtype=dt) * 2 - 1
        field = 1.0 + cfg.bias_strength * F.interpolate(low, size=img.shape[-2:], mode="bilinear",
                                                        align_corners=False)
        img = do_bf * (img * field) + (1 - do_bf) * img

        c_lo, c_hi = cfg.contrast
        contrast = torch.rand(b, 1, 1, 1, device=dev) * (c_hi - c_lo) + c_lo
        img = img * contrast + torch.randn_like(img) * cfg.noise                    # contrast + noise
        return img, mask
