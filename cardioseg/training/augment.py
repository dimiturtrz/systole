"""GPU-batched training augmentation.

The old augmentation ran scipy rotate/zoom/gaussian_filter **per item, on the CPU, inside the
DataLoader workers** — the throughput bottleneck that starved the GPU. This does the same recipe
on the whole batch on the GPU in torch: one affine grid_sample for flip+rotate+scale, vectorized
intensity jitter. No python per-item loop, no scipy.

Geometric (image AND mask together) teaches shape/orientation invariance — helps the thin RV/myo.
Intensity (image only) simulates the scanner/vendor contrast + resolution variation that drives
the cross-dataset gap.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

_GAUSS3 = torch.tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]) / 16.0  # 3x3 blur kernel


def augment_batch(
    img: torch.Tensor,
    mask: torch.Tensor,
    rot_deg: float = 20.0,
    scale: tuple[float, float] = (0.85, 1.15),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Augment a batch on its device. img [B,1,H,W] float, mask [B,H,W] long. Returns (img, mask).

    Per-sample flip * rotate * scale via a single affine grid_sample (bilinear image, nearest
    mask so labels stay integer; out-of-frame -> 0/background). Per-sample gamma, contrast,
    gaussian noise, and an occasional 3x3 blur on the image only. Uses torch's global RNG —
    seed it (torch.manual_seed) for reproducibility.
    """
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
    gamma = torch.rand(b, 1, 1, 1, device=dev) * 0.8 + 0.7                      # 0.7-1.5
    do_g = (torch.rand(b, 1, 1, 1, device=dev) < 0.3).to(dt)
    img = do_g * (((img - mn) / rng) ** gamma * rng + mn) + (1 - do_g) * img    # gamma where selected

    k = _GAUSS3.to(dev, dt).view(1, 1, 3, 3)
    do_b = (torch.rand(b, 1, 1, 1, device=dev) < 0.2).to(dt)
    img = do_b * F.conv2d(img, k, padding=1) + (1 - do_b) * img                 # occasional blur

    contrast = torch.rand(b, 1, 1, 1, device=dev) * 0.4 + 0.8                   # 0.8-1.2
    img = img * contrast + torch.randn_like(img) * 0.08                         # contrast + noise
    return img, mask
