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


def _ellipse(gy, gx, cy, cx, ry, rx, soft=0.04):
    """Soft ellipse membership in [0,1] over a coordinate grid. cy/cx/ry/rx are [B,1,1] (per-sample),
    gy/gx are [1,H,W] in normalized [-1,1] coords. Sigmoid edge of width ~soft."""
    d = ((gy - cy) / ry) ** 2 + ((gx - cx) / rx) ** 2
    return torch.sigmoid((1.0 - d) / soft)


def _procedural_thorax(b: int, h: int, w: int, dev) -> torch.Tensor:
    """Randomized synthetic thorax background [B,1,H,W] (z-ish scale): body ellipse (mid intensity) with
    a brighter fat/muscle rim, two dark lung fields flanking the centre, a spine blob, and smooth
    low-frequency texture. NOT photorealistic — it supplies the STRUCTURES the segmenter confuses the
    heart (esp. thin RV) with: dark lung next door, bright chest wall. All intensities/positions sampled
    per-sample so contrast stays vendor-agnostic. Heart is painted on top by the caller."""
    gy = torch.linspace(-1, 1, h, device=dev).view(1, h, 1)
    gx = torch.linspace(-1, 1, w, device=dev).view(1, 1, w)
    r = lambda lo, hi: (torch.rand(b, 1, 1, device=dev) * (hi - lo) + lo)        # per-sample U[lo,hi]

    air = r(-1.3, -0.7)                                                          # dark outside the body
    body_i, fat_i, lung_i = r(-0.2, 0.3), r(0.7, 1.4), r(-1.4, -0.9)
    body = _ellipse(gy, gx, r(-0.1, 0.1), r(-0.1, 0.1), r(0.75, 0.95), r(0.8, 1.0))
    inner = _ellipse(gy, gx, r(-0.1, 0.1), r(-0.1, 0.1), r(0.6, 0.78), r(0.62, 0.82))
    img = air + (body_i - air) * body + (fat_i - body_i) * (body - inner)        # fat rim = body minus inner
    # two dark lung fields flanking the centre (heart sits between them)
    for sx in (-1.0, 1.0):
        lung = _ellipse(gy, gx, r(-0.15, 0.15), sx * r(0.3, 0.5), r(0.3, 0.5), r(0.18, 0.32)) * inner
        img = img + (lung_i - img) * lung
    # spine: small bright blob low-centre
    spine = _ellipse(gy, gx, r(0.55, 0.75), r(-0.08, 0.08), r(0.08, 0.14), r(0.08, 0.14))
    img = img + (r(0.5, 1.2) - img) * spine
    # smooth low-frequency texture
    low = torch.rand(b, 1, 5, 5, device=dev) * 2 - 1
    img = img + 0.15 * F.interpolate(low, size=(h, w), mode="bicubic", align_corners=False)[:, 0]
    return img.unsqueeze(1)                                                      # [B,1,H,W]


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
    # --- extend the label map to the whole FOV when partitioning the background ---
    # Heart-only labels leave the bg as one flat blob (the pure-synth wall). Split it by REAL per-slice
    # intensity into bg_tiers tissue tiers (dark lung -> bright fat) -> the bg gets REAL anatomical
    # shapes (lungs land where they are), painted from per-tier priors. n_paint = heart + bg tiers.
    pmean, pstd = priors if priors is not None else (None, None)
    n_paint = n_classes
    ext = mask
    if cfg.bg_mode == "partition" and real_img is not None and not hybrid:
        K = cfg.bg_tiers
        thr = torch.linspace(-1.0, 1.0, K - 1, device=dev)                  # K-1 thresholds -> K bins
        tier = torch.bucketize(real_img[:, 0].contiguous(), thr)            # [B,H,W] in 0..K-1
        ext = torch.where(mask == 0, n_classes + tier, mask)               # bg -> n_classes+tier
        n_paint = n_classes + K
        # per-tier priors from the REAL bg intensity (shape from partition, appearance from these stats)
        bg_mean = torch.zeros(K, device=dev)
        bg_std = torch.full((K,), 0.5, device=dev)
        rv = real_img[:, 0]
        for k in range(K):
            v = rv[(mask == 0) & (tier == k)]
            if v.numel() > 10:
                bg_mean[k] = v.mean()
                bg_std[k] = v.std().clamp_min(1e-3)
        if pmean is not None:
            pmean = torch.cat([pmean, bg_mean]); pstd = torch.cat([pstd, bg_std])
    oh = F.one_hot(ext, n_paint).permute(0, 3, 1, 2).float()                # [B,n_paint,H,W]

    # --- per-label Gaussian paint ---
    if cfg.realistic and pmean is not None:
        mu = pmean[None, :] + cfg.jitter * torch.randn(b, n_paint, device=dev)     # ordering kept, jittered
        ss_lo, ss_hi = cfg.std_scale
        sg = pstd[None, :] * (torch.rand(b, n_paint, device=dev) * (ss_hi - ss_lo) + ss_lo)
    else:                                                                    # legacy pure-random (ablation)
        mu_lo, mu_hi = cfg.mu
        sg_lo, sg_hi = cfg.sigma
        mu = torch.rand(b, n_paint, device=dev) * (mu_hi - mu_lo) + mu_lo
        sg = torch.rand(b, n_paint, device=dev) * (sg_hi - sg_lo) + sg_lo
    mu_map = (oh * mu[:, :, None, None]).sum(1, keepdim=True)                # [B,1,H,W] class mean
    sg_map = (oh * sg[:, :, None, None]).sum(1, keepdim=True)               # [B,1,H,W] class std
    # partial volume: blur the class-MEAN map so boundary voxels are tissue mixes (real finite-voxel
    # averaging), not hard label edges. Texture (sg) added after, so interiors keep their grain.
    if cfg.pv_sigma > 0:
        kpv = _gaussian_kernel(cfg.pv_sigma).to(dev)
        kpv = kpv.view(1, 1, *kpv.shape)
        mu_map = F.conv2d(mu_map, kpv, padding=kpv.shape[-1] // 2)
    img = mu_map + sg_map * torch.randn(b, 1, *mask.shape[-2:], device=dev)  # painted texture

    # --- procedural thorax background: replace the bg blob with structured surroundings (dark lungs,
    #     fat rim, spine) the net confuses the heart with. Heart classes keep their paint. ---
    if cfg.bg_mode == "thorax" and not hybrid:
        thorax = _procedural_thorax(b, *mask.shape[-2:], dev)
        img = torch.where((mask > 0)[:, None], img, thorax)

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

    # --- k-space PSF: real MRI resolution = finite k-space sampling. Low-pass by keeping the central
    #     cfg.kspace fraction of frequencies (fft -> window -> ifft) = sinc PSF + slight Gibbs ringing,
    #     more physical than a Gaussian blur. ---
    if 0 < cfg.kspace < 1:
        H, W = img.shape[-2:]
        f = torch.fft.fftshift(torch.fft.fft2(img), dim=(-2, -1))
        ch, cw = int(H * cfg.kspace / 2), int(W * cfg.kspace / 2)
        win = torch.zeros_like(f.real)
        win[..., H // 2 - ch:H // 2 + ch + 1, W // 2 - cw:W // 2 + cw + 1] = 1.0
        img = torch.fft.ifft2(torch.fft.ifftshift(f * win, dim=(-2, -1))).real

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
