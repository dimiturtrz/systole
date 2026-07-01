"""Physics-based synthetic-image generation from labels (bd cardiac-seg-bgc / 276).

The *invent* force of domain generalization — sibling to `augment.py` (*diversify*: perturb real
pixels) and `preprocessing/normalization/` (*strip*: remove vendor variance). Throw away the real
intensities and PAINT each label class by its tissue's bSSFP SIGNAL (core.data.static.mri_physics) under
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
from pydantic import BaseModel, Field

from core.config import _VALIDATE

from .augment import _gaussian_kernel


class SynthCfg(BaseModel):
    """Physics-based synthetic-image generation from labels (SynthSeg-style, bd cardiac-seg-bgc/276).
    Discard real intensities; paint every class by the bSSFP SIGNAL EQUATION from its tissue T1/T2/PD
    (core.data.dynamic.mri_physics) under per-sample swept sequence params (TR/flip/field) -> a physically-
    plausible, vendor-randomized contrast every call -> a contrast-AGNOSTIC model. ONE paint path (no
    stats/random branches): contrast is physical, not fitted. Anatomy is the real mask (deform invents
    more); background gets real SHAPES via intensity partition, painted by tissue too. synth_p=0 -> off
    (pure real-image training). Refs: SynthSeg (Billot 2023); bSSFP Freeman–Hill."""
    model_config = _VALIDATE
    synth_p: float = Field(0.5, ge=0, le=1)         # fraction of in-batch samples replaced by synth.
    #                                                DEFAULT 0.5 = the MINIMUM synthetic share in training
    #                                                (owner mandate): synth stays in the pipeline. Costs
    #                                                ~0.01 cross-vendor Dice (0.864->0.852) — a robustness
    #                                                bet. synth_p=0 must be set explicitly for real-only.
    deform: float = Field(0.15, ge=0)              # nonlinear label-warp amplitude (norm coords); invents
    #                                                anatomy too (full SynthSeg). 0 = pixels-only, real mask
    # --- bSSFP sequence sweep = the physical cross-vendor contrast diversity ---
    tr_ms: tuple[float, float] = (2.8, 4.0)         # repetition time sweep (ms) — cine range
    flip_deg: tuple[float, float] = (35.0, 70.0)     # flip-angle sweep (deg)
    b0_hz: float = Field(0.0, ge=0)                 # off-resonance B0 field amplitude (Hz): smooth Δf
    #                                                field -> bSSFP BANDING (lowers+spreads long-T2 blood,
    #                                                the cav-too-bright fidelity fix). 0 = on-resonance
    fields: tuple[float, ...] = (1.5, 3.0)          # field strengths (T) sampled per-sample — T1/T2 shift
    #                                                = the dominant cross-vendor relaxation axis
    vendors: tuple[str, ...] = ("Siemens", "Philips", "GE", "Canon")   # sampled per-sample -> emitted as
    #                                                metadata so synth carries provenance + flows the same
    #                                                harmonization path as real (return_meta=True)
    jitter: float = Field(0.4, ge=0)               # residual per-class signal perturbation (extra breadth)
    texture: float = Field(0.05, ge=0)             # within-class texture: std as a fraction of |signal|
    flow: float = Field(0.0, ge=0)                 # blood-pool signal variation (flow/inflow): extra
    #                                                texture on blood classes so cav/RV aren't flat-bright
    #                                                (real cine blood spreads from flow) — fidelity lever
    blood_scale: float = Field(1.0, ge=0)          # LEGACY empirical blood-pool MEAN scale (the fidelity-
    #                                                found 1.6 that first localized the gap). Superseded by
    #                                                `inflow` (physical); kept for comparison. 1.0 = off.
    inflow: bool = False                           # entry-slice INFLOW enhancement (PHYSICAL, no magic
    #                                                fraction): per sample f_fresh = min(1, v*TR/thk) from
    #                                                blood velocity v + slice thickness thk + the derived TR,
    #                                                then blend blood toward fresh PD*sin(flip) (unsaturated
    #                                                spins entering the slice) -> the physical reason cine
    #                                                blood > bSSFP steady-state. Replaces blood_scale/the 0.15
    #                                                scalar; f emerges from physiology (mean~0.15, distributed).
    blood_v_cms: tuple[float, float] = (5.0, 90.0) # through-plane blood velocity (cm/s), PHYSIOLOGICAL:
    #                                                mid-cavity ~5-30, basal/valve-plane E-wave ~80-90
    #                                                (PMC9843884). The physical input to inflow, sampled per
    #                                                slice (position-varying) — the source of f's spread.
    slice_mm: tuple[float, float] = (6.0, 8.0)     # cine slice thickness (mm), SCMR standardized 6-8mm
    derive_acq: bool = False                       # TR/flip from physics (derive_acquisition +
    #                                                derive_flip_range, sampled across the SAR-bounded band)
    #                                                instead of the cfg.tr_ms/flip_deg ranges. OPT-IN: the
    #                                                physics-derived pipeline ~matches but doesn't beat the
    #                                                empirical ranges on cross-vendor Dice (0.673 vs 0.701),
    #                                                so it's not the default — it's the defensible/derived
    #                                                path (bd 276), enabled with inflow for the physics run.
    # --- background ---
    bg_mode: str = "partition"                      # "partition" = split bg by REAL per-slice intensity
    #                                                into bg_tiers tissue tiers (real lung/fat/muscle
    #                                                SHAPES, painted by tissue); "flat" = single bg tissue;
    #                                                "procedural" = SYNTHETIC random-field organ blobs (no
    #                                                real img) -> whole-FOV bg for ZERO-REAL synth anatomy.
    bg_tiers: int = Field(6, ge=2)                  # distinct background tissue tiers (params interpolated)
    bg_blobs: int = Field(6, ge=2)                   # procedural bg: coarse random-field grid (smaller=bigger
    #                                                organ blobs); upsampled+bucketized into bg_tiers regions
    keep_real_bg: bool = False                      # DIAGNOSTIC/hybrid: paste synth heart onto REAL bg
    #                                                (isolates bg realism; forces deform off). Not pure-synth.
    # --- physical corruption chain (all diversity knobs; each a real acquisition effect) ---
    pv_sigma: float = Field(0.0, ge=0)             # partial-volume: blur the class-MEAN map (vox)
    kspace: float = Field(0.0, ge=0, le=1)         # k-space PSF: fraction of k-space kept (sinc + Gibbs)
    bias_strength: float = Field(0.3, ge=0)         # smooth multiplicative B1/coil bias field, +/- fraction
    blur: tuple[float, float] = (0.0, 1.0)          # extra Gaussian blur σ (resolution)
    noise: float = Field(0.05, ge=0)               # Rician noise std (post-paint, pre-z-score)


def _deform_grid(b: int, h: int, w: int, amp: float, dev, steps: int = 6) -> torch.Tensor:
    """DIFFEOMORPHIC (topology-preserving) elastic warp as a grid_sample grid [B,H,W,2]. A coarse 5x5
    velocity field (U[-amp,amp], bicubic-upsampled) is INTEGRATED by scaling-and-squaring so the map
    stays invertible — it can't fold/tear the anatomy. The old `ident + disp` (non-integrated) folds at
    amplitude: measured pure-synth collapse 0.673 -> 0.13 (deform 0.4) -> 0.04 (0.6). Integration
    (SynthSeg/VoxelMorph) fixes that. amp = velocity magnitude in normalized [-1,1] coords."""
    v = (torch.rand(b, 2, 5, 5, device=dev) * 2 - 1) * amp
    v = F.interpolate(v, size=(h, w), mode="bicubic", align_corners=False)          # [B,2,H,W] velocity
    ident = F.affine_grid(torch.eye(2, 3, device=dev).expand(b, 2, 3), (b, 1, h, w),
                          align_corners=False)                                      # [B,H,W,2]
    d = (v / (2 ** steps)).permute(0, 2, 3, 1)                                      # [B,H,W,2] small displ
    for _ in range(steps):                                 # scaling-and-squaring: phi = phi ∘ phi
        sampled = F.grid_sample(d.permute(0, 3, 1, 2), ident + d, mode="bilinear",
                                padding_mode="border", align_corners=False)
        d = d + sampled.permute(0, 2, 3, 1)                # d <- d + d(x + d(x))
    return ident + d


def synthesize_from_labels(mask: torch.Tensor, cfg: SynthCfg, n_classes: int,
                           real_img: torch.Tensor | None = None, return_meta: bool = False):
    """Generate a synthetic z-scored image (and its label map) from an integer label mask.

    mask [B,H,W] long (labels 0..n_classes-1) -> (img [B,1,H,W] z-scored, mask [B,H,W] long).
    return_meta=True appends a per-sample provenance dict {vendor, field, tr, flip} — the acquisition
    each synth image simulated, so synth carries metadata like real data (harmonization / stratified eval).
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
    elif cfg.bg_mode == "procedural" and not hybrid:
        # ZERO-REAL whole-FOV bg: a per-sample coarse random field (bg_blobs x bg_blobs) upsampled to
        # smooth organ-like blobs, bucketized into K tiers -> each painted by an interpolated tissue on
        # the bg ladder (mri_physics). SynthSeg-style random-shape context: the net can't localize the
        # heart by "it's the only non-flat thing", it must learn the 3-class contrast signature. No real
        # image needed -> this is the pure-synth-anatomy background (kills the flat-bg 0.07 wall). (bwp)
        K = cfg.bg_tiers
        coarse = torch.rand(b, 1, cfg.bg_blobs, cfg.bg_blobs, device=dev)
        fieldb = F.interpolate(coarse, size=mask.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        thr = torch.linspace(0.0, 1.0, K + 1, device=dev)[1:-1]             # K-1 interior thresholds -> K bins
        tier = torch.bucketize(fieldb.contiguous(), thr)                    # [B,H,W] in 0..K-1
        ext = torch.where(mask == 0, n_classes + tier, mask)
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
    if cfg.derive_acq:
        # TR DERIVED per field (floor + jitter). FLIP sampled across the physically-plausible RANGE
        # per field (derive_flip_range: low-contrast end .. SAR cap) — domain randomization needs
        # contrast BREADTH, not the single contrast-optimal point (that measured worse; bd 276). Physics-
        # bounded (SAR ceiling), not an arbitrary global range.
        from .mri_physics import derive_flip_range, TR_RANGE_MS
        rng = torch.tensor([derive_flip_range(float(f)) for f in cfg.fields], device=dev)      # [F,2] lo,hi
        tr = TR_RANGE_MS[0] + (TR_RANGE_MS[1] - TR_RANGE_MS[0]) * torch.rand(b, 1, device=dev) # sample cited TR band
        lo, hi = rng[fi, 0:1], rng[fi, 1:2]
        fl = lo + (hi - lo) * torch.rand(b, 1, device=dev)                                     # flip diversity (FWHM band)
    else:                                                                    # legacy global ranges
        tr = torch.rand(b, 1, device=dev) * (cfg.tr_ms[1] - cfg.tr_ms[0]) + cfg.tr_ms[0]
        fl = torch.rand(b, 1, device=dev) * (cfg.flip_deg[1] - cfg.flip_deg[0]) + cfg.flip_deg[0]
    vi = torch.randint(len(cfg.vendors), (b,), device=dev)                   # per-sample vendor tag
    meta = {"vendor": [cfg.vendors[i] for i in vi.tolist()],                 # provenance for each synth
            "field": torch.tensor(cfg.fields, device=dev)[fi], "tr": tr[:, 0], "flip": fl[:, 0]}
    mu = bssfp_signal(t1, t2, pd, tr, fl * math.pi / 180.0)                  # [B, n_paint] steady-state
    mu = mu + cfg.jitter * mu.abs().mean() * torch.randn(b, n_paint, device=dev)   # residual jitter
    if cfg.inflow:                               # entry-slice inflow: f_fresh = min(1, v*TR/thk) PER SAMPLE
        from .mri_physics import blood_classes    # from physiological v/thk + derived TR (no magic fraction)
        v = torch.rand(b, 1, device=dev) * (cfg.blood_v_cms[1] - cfg.blood_v_cms[0]) + cfg.blood_v_cms[0]
        thk = torch.rand(b, 1, device=dev) * (cfg.slice_mm[1] - cfg.slice_mm[0]) + cfg.slice_mm[0]
        f = (v * tr / (100.0 * thk)).clamp(max=1.0)                          # v cm/s, tr ms, thk mm -> frac
        s_fresh = pd * torch.sin(fl * math.pi / 180.0)                       # [B, n_paint] fully-relaxed excite
        for c in blood_classes(n_classes):
            mu[:, c] = (1 - f[:, 0]) * mu[:, c] + f[:, 0] * s_fresh[:, c]     # blend blood toward fresh
    if cfg.blood_scale != 1.0:                    # legacy empirical blood-pool mean scale (superseded by inflow)
        from .mri_physics import blood_classes
        for c in blood_classes(n_classes):
            mu[:, c] = mu[:, c] * cfg.blood_scale
    sg = mu.abs() * cfg.texture                                              # within-class texture
    if cfg.flow > 0:                                                         # flow: blood pools spread
        from .mri_physics import blood_classes
        for c in blood_classes(n_classes):
            sg[:, c] = sg[:, c] + cfg.flow * mu[:, c].abs()
    mu_map = (oh * mu[:, :, None, None]).sum(1, keepdim=True)                # [B,1,H,W] class mean
    # off-resonance bSSFP banding: a smooth Δf (B0) field -> per-pixel signal drop near dphi=±π,
    # strongest for long-T2 blood -> lowers+spreads the over-bright cavity (the cav-fidelity fix).
    # Applied as the signal RATIO band(φ)/band(0) so per-class jitter/flow in mu_map are preserved.
    if cfg.b0_hz > 0:
        from .mri_physics import banding
        t2m = (oh * t2[:, :, None, None]).sum(1, keepdim=True)               # per-pixel T2
        low = torch.rand(b, 1, 4, 4, device=dev) * 2 - 1
        df = cfg.b0_hz * F.interpolate(low, size=mask.shape[-2:], mode="bilinear", align_corners=False)
        phi = 2 * math.pi * df * (tr[:, :, None, None] / 1000.0)             # off-resonance per TR (rad)
        mu_map = mu_map * banding(t2m, tr[:, :, None, None], phi)
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
    img = (img - m) / s
    return (img, mask, meta) if return_meta else (img, mask)   # opt-in provenance (vendor/field/tr/flip)
