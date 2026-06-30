"""Physics-based synthetic-image generation from labels (bd cardiac-seg-bgc / 276).

The *invent* force of domain generalization — sibling to `augment.py` (*diversify*: perturb real
pixels) and `preprocessing/normalization/` (*strip*: remove vendor variance). Throw away the real
intensities and PAINT each label class by its tissue's bSSFP SIGNAL (core.data.mri_physics) under
per-sample swept sequence params (TR/flip) and FIELD strength (1.5T/3T). Contrast is PHYSICAL, not
fitted; sweeping the sequence/field sweeps it along the real cross-vendor manifold. Train on these and
the net must segment by shape/structure, not one scanner's appearance.

Why physical (not statistical): scanner differences ARE physical (sequence, field). The earlier
statistical recipe (paint around measured per-class means) wins this specific test by ~0.04 Dice, but
partly by riding a flow artifact (RV≠cav intensity); physics is the correct, general model — chosen on
principle, not the metric (see bd 276).

Cardiac labels are FOV-sparse (heart-only); the background is split by REAL per-slice intensity into
tissue tiers (real SHAPES), painted by tissue too -> whole-FOV physical synth. Pipeline per call:
deform -> bg partition -> bSSFP paint -> partial-volume -> bias -> blur -> k-space PSF -> Rician noise
-> z-score. Geometry (flip/rotate/scale) is `augment.py`'s job, after. GPU-batched, vectorized; torch
global RNG (seed for repro). Config = injected SynthCfg.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from core.hparams import SynthCfg

from .augment import _gaussian_kernel


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
                           real_img: torch.Tensor | None = None
                           ) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a synthetic z-scored image (and its label map) from an integer label mask.

    mask [B,H,W] long (labels 0..n_classes-1) -> (img [B,1,H,W] z-scored, mask [B,H,W] long).
    cfg.deform>0 warps the labels first (new anatomy; returned mask is the warped one so the target
    stays aligned). Each class is painted by the bSSFP SIGNAL of its tissue (mri_physics) under
    per-sample swept TR/flip/field -> physical, vendor-randomized contrast. With bg_mode='partition'
    the background is split by REAL per-slice intensity into tissue tiers (real SHAPES) and painted by
    tissue too -> whole-FOV physical synth. `real_img` supplies those bg shapes (and the hybrid bg).
    """
    import math
    from .mri_physics import bssfp_signal, tissue_params

    b = mask.shape[0]
    dev = mask.device
    mask = mask.long()
    hybrid = cfg.keep_real_bg and real_img is not None
    if cfg.deform > 0 and not hybrid:                # hybrid: keep heart aligned with real bg's hole
        grid = _deform_grid(b, *mask.shape[-2:], cfg.deform, dev)
        mask = F.grid_sample(mask[:, None].float(), grid, mode="nearest",
                             padding_mode="border", align_corners=False)[:, 0].long()
    # --- extend the label map to the whole FOV: split the bg by REAL per-slice intensity into bg_tiers
    #     tissue tiers (lungs/fat land where they really are = real SHAPES), each painted by tissue. ---
    n_paint = n_classes
    ext = mask
    if cfg.bg_mode == "partition" and real_img is not None and not hybrid:
        K = cfg.bg_tiers
        thr = torch.linspace(-1.0, 1.0, K - 1, device=dev)                  # K-1 thresholds -> K bins
        tier = torch.bucketize(real_img[:, 0].contiguous(), thr)            # [B,H,W] in 0..K-1
        ext = torch.where(mask == 0, n_classes + tier, mask)               # bg -> n_classes+tier
        n_paint = n_classes + K
    oh = F.one_hot(ext, n_paint).permute(0, 3, 1, 2).float()                # [B,n_paint,H,W]

    # --- physical paint: each class' intensity = balanced-SSFP signal from its tissue T1/T2/PD under
    #     per-sample swept sequence params (TR, flip) and FIELD strength (1.5T/3T = cross-vendor axis). ---
    # tissue params per available field -> [n_fields, n_paint]; pick one field per sample
    params = [tissue_params(n_classes, n_paint - n_classes, float(f), dev) for f in cfg.fields]
    t1s = torch.stack([p[0] for p in params]); t2s = torch.stack([p[1] for p in params])
    pds = torch.stack([p[2] for p in params])                               # [n_fields, n_paint]
    fi = torch.randint(len(cfg.fields), (b,), device=dev)                    # per-sample field index
    t1, t2, pd = t1s[fi], t2s[fi], pds[fi]                                  # [B, n_paint]
    tr = torch.rand(b, 1, device=dev) * (cfg.tr_ms[1] - cfg.tr_ms[0]) + cfg.tr_ms[0]
    fl = torch.rand(b, 1, device=dev) * (cfg.flip_deg[1] - cfg.flip_deg[0]) + cfg.flip_deg[0]
    mu = bssfp_signal(t1, t2, pd, tr, fl * math.pi / 180.0)                  # [B, n_paint]
    mu = mu + cfg.jitter * mu.abs().mean() * torch.randn(b, n_paint, device=dev)   # residual jitter
    sg = mu.abs() * cfg.texture                                              # within-class texture
    mu_map = (oh * mu[:, :, None, None]).sum(1, keepdim=True)                # [B,1,H,W] class mean
    sg_map = (oh * sg[:, :, None, None]).sum(1, keepdim=True)               # [B,1,H,W] class std
    # partial volume: blur the class-MEAN map so boundary voxels are tissue mixes (real finite-voxel
    # averaging), not hard label edges. Texture (sg) added after, so interiors keep their grain.
    if cfg.pv_sigma > 0:
        kpv = _gaussian_kernel(cfg.pv_sigma).to(dev)
        kpv = kpv.view(1, 1, *kpv.shape)
        mu_map = F.conv2d(mu_map, kpv, padding=kpv.shape[-1] // 2)
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
