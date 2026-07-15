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

import math
from abc import ABC, abstractmethod
from typing import Annotated, Literal

import torch
import torch.nn.functional as F
from jaxtyping import Float, Integer
from pydantic import BaseModel, Field, model_validator

from core.config import _VALIDATE
from core.data.static.mri.base import Vendor
from core.types import shapecheck

from .augment import Augmentor
from .mri_physics import TORSO_BG, TR_RANGE_MS, MriPhysics
from .mrxcat import FOV_TISSUE


# Acquisition strategy as a discriminated union: each variant BUILDS its Acquisition (cfg.acq.build()).
# build() bodies resolve the strategy classes (defined lower) at call-time.
class AcquisitionCfg(BaseModel):
    """Per-sample scanner-settings strategy cfg. Subclass build() returns the Acquisition."""
    model_config = _VALIDATE

    def build(self) -> "Acquisition":
        raise NotImplementedError


class LegacyAcqCfg(AcquisitionCfg):
    """TR/flip uniform over the cfg global ranges (tr_ms, flip_deg)."""
    mode: Literal["legacy"] = "legacy"

    def build(self):
        return LegacyAcq()


class RandomizedAcqCfg(AcquisitionCfg):
    """Physics-bounded domain randomization (derive_flip_range over the SAR-bounded band)."""
    mode: Literal["randomized"] = "randomized"

    def build(self):
        return RandomizedAcq()


class MatchedAcqCfg(AcquisitionCfg):
    """Paint TO one target scanner — no randomization (bd 7pto)."""
    mode: Literal["matched"] = "matched"
    match_field: float = 1.5                        # target field (T; nearest in `fields`)
    match_tr_ms: float = 3.0                        # target repetition time (ms)
    match_flip_deg: float = 50.0                    # target flip angle (deg)
    match_vendor: str = ""                          # target vendor tag ("" -> first of vendors)

    def build(self):
        return MatchedAcq(self.match_field, self.match_tr_ms, self.match_flip_deg, self.match_vendor)


AnyAcqCfg = Annotated[LegacyAcqCfg | RandomizedAcqCfg | MatchedAcqCfg, Field(discriminator="mode")]
ACQ_VARIANTS = {c.model_fields["mode"].default: c for c in (LegacyAcqCfg, RandomizedAcqCfg, MatchedAcqCfg)}


# Background strategy union: each variant BUILDS its Background (cfg.bg.build()), holding its own params.
class BackgroundCfg(BaseModel):
    """FOV-extension / bg-painting strategy cfg. Subclass build() returns the Background."""
    model_config = _VALIDATE

    def build(self) -> "Background":
        raise NotImplementedError


class FlatBgCfg(BackgroundCfg):
    """Single background tissue tier."""
    mode: Literal["flat"] = "flat"

    def build(self):
        return FlatBg()


class ProceduralBgCfg(BackgroundCfg):
    """Random-field organ blobs bucketized into the physical TORSO_BG tissues (zero-real whole-FOV)."""
    mode: Literal["procedural"] = "procedural"
    bg_blobs: int = Field(6, ge=2)                  # coarse random-field grid (smaller = bigger organ blobs)

    def build(self):
        return ProceduralBg(self.bg_blobs)


class PartitionBgCfg(BackgroundCfg):
    """Real per-slice intensity partitioned into bg tiers (real SHAPES, painted by tissue)."""
    mode: Literal["partition"] = "partition"
    bg_tiers: int = Field(6, ge=2)

    def build(self):
        return PartitionBg(self.bg_tiers)


class HybridBgCfg(BackgroundCfg):
    """Paint heart on flat bg; composite the real (heart-excised) background back in."""
    mode: Literal["hybrid"] = "hybrid"

    def build(self):
        return HybridBg()


class MrxcatBgCfg(BackgroundCfg):
    """Whole-FOV MRXCAT tissue map (mask IS the 8-class FOV; paint all tissues)."""
    mode: Literal["mrxcat"] = "mrxcat"

    def build(self):
        return FovBg()


AnyBgCfg = Annotated[FlatBgCfg | ProceduralBgCfg | PartitionBgCfg | HybridBgCfg | MrxcatBgCfg,
                     Field(discriminator="mode")]
BG_VARIANTS = {c.model_fields["mode"].default: c
               for c in (FlatBgCfg, ProceduralBgCfg, PartitionBgCfg, HybridBgCfg, MrxcatBgCfg)}


class SynthCfg(BaseModel):
    """Physics-based synthetic-image generation from labels (SynthSeg-style, bd cardiac-seg-bgc/276).
    Discard real intensities; paint every class by the bSSFP SIGNAL EQUATION from its tissue T1/T2/PD
    (core.data.dynamic.mri_physics) under per-sample swept sequence params (TR/flip/field) -> a physically-
    plausible, vendor-randomized contrast every call -> a contrast-AGNOSTIC model. ONE paint path (no
    stats/random branches): contrast is physical, not fitted. Anatomy is the real mask (deform invents
    more); background gets real SHAPES via intensity partition, painted by tissue too. synth_p=0 -> off
    (pure real-image training). Refs: SynthSeg (Billot 2023); bSSFP Freeman–Hill."""
    model_config = _VALIDATE

    @model_validator(mode="before")
    @classmethod
    def _lift_legacy(cls, v):
        """Back-compat: old config.json had flat acq_mode/match_* and bg_mode/bg_tiers/bg_blobs; fold them
        into the `acq` / `bg` discriminated unions so pre-refactor configs (registered models) still load."""
        if not isinstance(v, dict):
            return v
        if "acq" not in v and "acq_mode" in v:
            v = dict(v)
            acq: dict = {"mode": v.pop("acq_mode")}
            for k in ("match_field", "match_tr_ms", "match_flip_deg", "match_vendor"):
                if k in v:
                    acq[k] = v.pop(k)
            v["acq"] = acq
        if "bg" not in v and "bg_mode" in v:
            v = dict(v)
            bg: dict = {"mode": v.pop("bg_mode")}
            for k in ("bg_tiers", "bg_blobs"):
                if k in v:
                    val = v.pop(k)
                    if not (k == "bg_tiers" and bg["mode"] == "procedural"):   # procedural dropped bg_tiers (fixed torso composition)
                        bg[k] = val
            v["bg"] = bg
        return v
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
    vendors: tuple[str, ...] = tuple(Vendor)   # sampled per-sample -> emitted as
    #                                                metadata so synth carries provenance + flows the same
    #                                                harmonization path as real (return_meta=True)
    jitter: float = Field(0.4, ge=0)               # residual per-class signal perturbation (extra breadth)
    tissue_spread: float = Field(0.0, ge=0)        # per-sample PHYSICAL T1/T2 sampling of heart tissues
    #                                                over the literature band (mri_physics.TISSUE_RANGE),
    #                                                lerped from the point by this (0=off/point, 1=full
    #                                                band). Replaces decorrelated jitter with a physically-
    #                                                constrained sweep (bd 04bh); blood cavities split by O2.
    texture: float = Field(0.05, ge=0)             # within-class texture: std as a fraction of |signal|
    flow: float = Field(0.0, ge=0)                 # blood-pool signal variation (flow/inflow): extra
    #                                                texture on blood classes so cav/RV aren't flat-bright
    #                                                (real cine blood spreads from flow) — fidelity lever
    blood_scale: float = Field(1.0, ge=0)          # LEGACY empirical blood-pool MEAN scale (the fidelity-
    #                                                found 1.6 that first localized the gap). Superseded by
    #                                                `inflow` (physical); kept for comparison. 1.0 = off.
    inflow: bool = False                           # entry-slice INFLOW enhancement: f_fresh=min(1,v*TR/thk),
    #                                                blend blood toward fresh PD*sin(flip). GAIN-ONLY: it
    #                                                models fresh-spin brightening but NOT the compensating
    #                                                flow-dephasing loss + papillary/trabecular PV, so ON it
    #                                                over-brightens in a correct scene (blood +3.9z vs real
    #                                                +1.6). Measured real blood (+1.66) sits BELOW steady-state
    #                                                (+2.04) -> net loss slightly wins; OFF is closer to real.
    #                                                DEFAULT off pending a dephasing-loss term (bd). Zero-real
    #                                                Dice 0.554->0.613 turning it off w/ torso-composition bg.
    blood_v_cms: tuple[float, float] = (5.0, 90.0) # through-plane blood velocity (cm/s), PHYSIOLOGICAL:
    #                                                mid-cavity ~5-30, basal/valve-plane E-wave ~80-90
    #                                                (PMC9843884). The physical input to inflow, sampled per
    #                                                slice (position-varying) — the source of f's spread.
    slice_mm: tuple[float, float] = (6.0, 8.0)     # cine slice thickness (mm), SCMR standardized 6-8mm
    acq: AnyAcqCfg = Field(default_factory=LegacyAcqCfg)   # scanner-settings strategy (legacy/randomized/matched)
    bg: AnyBgCfg = Field(default_factory=PartitionBgCfg)   # FOV/bg strategy (flat/procedural/partition/hybrid/mrxcat)
    # --- physical corruption chain (all diversity knobs; each a real acquisition effect) ---
    pv_sigma: float = Field(0.0, ge=0)             # partial-volume: blur the class-MEAN map (vox)
    kspace: float = Field(0.0, ge=0, le=1)         # k-space PSF: fraction of k-space kept (sinc + Gibbs)
    bias_strength: float = Field(0.3, ge=0)         # smooth multiplicative B1/coil bias field, +/- fraction
    blur: tuple[float, float] = (0.0, 1.0)          # extra Gaussian blur σ (resolution)
    noise: float = Field(0.05, ge=0)               # Rician noise std (post-paint, pre-z-score)


class Background(ABC):
    """FOV-fill STRATEGY: turn a heart-only label map into a whole-FOV paint map (`extend`) and
    optionally composite the painted image against a real image (`compose`). One impl per bg mode —
    the painter holds NO bg if/elif; it just calls the strategy make_background(cfg) returns."""
    needs_real_img: bool = False
    keeps_heart_aligned: bool = False                        # True -> painter must NOT deform (heart

    @abstractmethod                                          #         must line up with the real hole)
    def extend(self, mask: Integer[torch.Tensor, "*b *grid"], n_classes: int, dev,
               real_img: Float[torch.Tensor, "*b 1 *h *w"] | None = None) -> tuple[Integer[torch.Tensor, "*b *grid"], int]:
        """-> (ext [B,H,W] long, n_paint). bg pixels (label 0) may be relabeled to n_classes+tier."""

    def compose(self, img: Float[torch.Tensor, "*b 1 *h *w"], mask: Integer[torch.Tensor, "*b *grid"],
                real_img: Float[torch.Tensor, "*b 1 *h *w"] | None) -> Float[torch.Tensor, "*b 1 *h *w"]:
        return img                                           # default: painter output is final

    def paint_params(self, n_classes: int, n_paint: int, field: float, dev):
        """(T1,T2,PD) [n_paint] for the extended classes at `field`. Default = heart classes + bg-ladder
        tiers (`tissue_params`). A strategy that assigns each class an EXPLICIT tissue (FovBg) overrides."""
        return MriPhysics.tissue_params(n_classes, n_paint - n_classes, field, dev)

    def seg_target(self, mask: Integer[torch.Tensor, "*b *grid"]) -> Integer[torch.Tensor, "*b *grid"]:
        """The SEG label map from the (painted) label map. Default: the paint map IS the target. A
        strategy whose paint map carries EXTRA non-target classes (FovBg's whole-FOV tissues) restricts
        it to the segmentation classes — so paint and train target decouple with no caller change."""
        return mask


class FlatBg(Background):
    """Single background tissue (bg stays label 0 -> painted by the muscle fallback)."""
    def extend(self, mask, n_classes, dev, real_img=None):
        return mask, n_classes


class _TierBg(Background):
    """bg = K tissue tiers over the FOV; subclasses supply the per-pixel tier field in 0..K-1."""
    def __init__(self, k_tiers: int):
        self.k_tiers = k_tiers

    @abstractmethod
    def _field(self, mask, dev, real_img):
        ...

    def extend(self, mask, n_classes, dev, real_img=None):
        if self.needs_real_img and real_img is None:
            return mask, n_classes                           # no bg source -> fall back to flat
        tier = self._field(mask, dev, real_img)
        ext = torch.where(mask == 0, n_classes + tier, mask)
        return ext, n_classes + self.k_tiers


class ProceduralBg(_TierBg):
    """ZERO-REAL whole-FOV: a coarse random field (blobs x blobs) upsampled to smooth organ-like blobs,
    bucketized into the TORSO_BG tissues at their PHYSICAL area fractions (dark air -> muscle) and painted
    by literature bSSFP. Random SHAPES (zero-real diversity) but a torso-correct intensity HISTOGRAM, so
    the per-image z-score lands the heart classes at real levels (fixes the myo-too-dark mean-gap that the
    old equal-tier ladder caused: it over-weighted bright tissue -> image mean too high)."""
    def __init__(self, blobs: int):
        super().__init__(len(TORSO_BG))
        self.blobs = blobs

    def _field(self, mask, dev, real_img=None):
        b = mask.shape[0]
        coarse = torch.rand(b, 1, self.blobs, self.blobs, device=dev)
        fb = F.interpolate(coarse, size=mask.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        return torch.bucketize(fb.contiguous(), MriPhysics.torso_thresholds(dev))  # physical area fractions

    def paint_params(self, n_classes, n_paint, field, dev):
        return MriPhysics.torso_paint_params(n_classes, field, dev)               # named torso tissues


class PartitionBg(_TierBg):
    """Split a REAL image's per-pixel intensity into K tiers = real background SHAPES. The real image
    MUST have its heart excised (excise_heart) or a different heart leaks into the tiers (bd mirs)."""
    needs_real_img = True

    def _field(self, mask, dev, real_img):
        thr = torch.linspace(-1.0, 1.0, self.k_tiers - 1, device=dev)       # K-1 thresholds -> K bins
        return torch.bucketize(real_img[:, 0].contiguous(), thr)


class HybridBg(Background):
    """Paste the painted synth heart into a REAL image (heart voxels = synth, rest = real). The real
    image MUST be heart-excised first, else its own heart survives unlabeled (bd mirs)."""
    needs_real_img = True
    keeps_heart_aligned = True

    def extend(self, mask, n_classes, dev, real_img=None):
        return mask, n_classes                               # paint heart + flat bg; bg replaced in compose

    def compose(self, img, mask, real_img):
        if real_img is None:
            return img
        fg = (mask > 0)[:, None]
        return torch.where(fg, img, real_img)


class FovBg(Background):
    """MRXCAT WHOLE-FOV (bd q4ww): the input map is ALREADY a whole-FOV tissue map
    (`mrxcat.to_tissue_map`, classes 0..7 = `FOV_TISSUE`), so there's no bg to invent — every class is
    painted by its named tissue (`paint_params` → `named_tissue_params`). The heart classes (1/2/3)
    coincide with the canonical seg labels, so the training target is recovered downstream as the FOV map
    restricted to {1,2,3}. `mask` passed to the painter IS this FOV map; n_classes stays the model's 4
    (blood/inflow logic keys on cavities 1/3, unchanged)."""
    def extend(self, mask, n_classes, dev, real_img=None):
        return mask.long(), len(FOV_TISSUE)                  # mask is the FOV tissue map; paint all 8

    def paint_params(self, n_classes, n_paint, field, dev):
        return MriPhysics.named_tissue_params([FOV_TISSUE[c] for c in range(n_paint)], field, dev)

    def seg_target(self, mask):
        return mask.where(mask <= 3, torch.zeros_like(mask))  # noqa: PLR2004  FOV 4..7 (organs)→bg; heart 1..3 kept




class Acquisition(ABC):
    """Per-sample SCANNER SETTINGS strategy: pick field index (into cfg.fields), TR (ms), flip (deg),
    and vendor index (into cfg.vendors). One impl per mode — the painter holds no acq if/else."""
    @abstractmethod
    def sample(self, b: int, cfg: SynthCfg, dev):
        """-> (fi [b] long, tr [b,1], fl [b,1], vi [b] long)."""


class LegacyAcq(Acquisition):
    """TR/flip uniform over the cfg global ranges (tr_ms, flip_deg). Field/vendor uniform-random."""
    def sample(self, b, cfg, dev):
        fi = torch.randint(len(cfg.fields), (b,), device=dev)
        tr = torch.rand(b, 1, device=dev) * (cfg.tr_ms[1] - cfg.tr_ms[0]) + cfg.tr_ms[0]
        fl = torch.rand(b, 1, device=dev) * (cfg.flip_deg[1] - cfg.flip_deg[0]) + cfg.flip_deg[0]
        vi = torch.randint(len(cfg.vendors), (b,), device=dev)
        return fi, tr, fl, vi


class RandomizedAcq(Acquisition):
    """Physics-bounded domain randomization: TR over the cited cine band, flip across the per-field
    SAR-bounded FWHM contrast range (derive_flip_range). Breadth, not the single optimal point."""
    def sample(self, b, cfg, dev):
        fi = torch.randint(len(cfg.fields), (b,), device=dev)
        rng = torch.tensor([MriPhysics.derive_flip_range(float(f)) for f in cfg.fields], device=dev)   # [F,2] lo,hi
        tr = TR_RANGE_MS[0] + (TR_RANGE_MS[1] - TR_RANGE_MS[0]) * torch.rand(b, 1, device=dev)
        lo, hi = rng[fi, 0:1], rng[fi, 1:2]
        fl = lo + (hi - lo) * torch.rand(b, 1, device=dev)
        vi = torch.randint(len(cfg.vendors), (b,), device=dev)
        return fi, tr, fl, vi


class MatchedAcq(Acquisition):
    """Paint TO one target scanner (from MatchedAcqCfg): fixed field (nearest available), TR, flip,
    vendor — no randomization. For known-deployment training + the match-vs-randomize test (bd 7pto)."""
    def __init__(self, field: float, tr_ms: float, flip_deg: float, vendor: str):
        self.field, self.tr_ms, self.flip_deg, self.vendor = field, tr_ms, flip_deg, vendor

    def sample(self, b, cfg, dev):
        fields = torch.tensor(cfg.fields, device=dev)
        fidx = int(torch.argmin((fields - self.field).abs()))               # nearest available field
        vidx = cfg.vendors.index(self.vendor) if self.vendor in cfg.vendors else 0
        fi = torch.full((b,), fidx, device=dev, dtype=torch.long)
        vi = torch.full((b,), vidx, device=dev, dtype=torch.long)
        tr = torch.full((b, 1), float(self.tr_ms), device=dev)
        fl = torch.full((b, 1), float(self.flip_deg), device=dev)
        return fi, tr, fl, vi


_MIN_BLUR_SIGMA = 0.05   # below this a Gaussian blur is a no-op -> skip the conv


class SynthPainter:
    """Physics-based synth PAINTER: label mask -> z-scored bSSFP image (`synthesize_from_labels`), plus the
    two pipeline helpers it composes — the diffeomorphic label warp (`_deform_grid`) and heart excision
    for clean backgrounds (`excise_heart`, also the entry point train.py calls on real images)."""

    @staticmethod
    @shapecheck
    def _deform_grid(b: int, h: int, w: int, amp: float, dev, steps: int = 6) -> Float[torch.Tensor, "b h w 2"]:  # noqa: PLR0913  geometry params (b,h,w,amp,steps — independent)
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

    @staticmethod
    @shapecheck
    def excise_heart(img: Float[torch.Tensor, "b 1 h w"], gt: Integer[torch.Tensor, "b h w"],
                     iters: int = 40) -> Float[torch.Tensor, "b 1 h w"]:
        """Remove the heart from a REAL image so it can serve as a clean BACKGROUND for a DIFFERENT
        (synthetic) heart. Zero the heart pixels (gt>0) and inpaint the hole by iterative neighbour
        diffusion (masked 3x3 mean, growing the known region inward) — otherwise the real image's own
        heart survives OUTSIDE the pasted synth-heart mask, unlabeled, and the net trains against a
        phantom second heart (bd mirs). img [B,1,H,W], gt [B,H,W] (0=bg). Returns a heart-free copy."""
        hole = (gt > 0)[:, None].float()                         # [B,1,H,W] region to erase+fill
        out = img * (1.0 - hole)
        known = 1.0 - hole
        k = torch.ones(1, 1, 3, 3, device=img.device) / 9.0
        for _ in range(iters):
            num = F.conv2d(out * known, k, padding=1)
            den = F.conv2d(known, k, padding=1).clamp_min(1e-6)
            out = torch.where(known.bool(), out, num / den)      # fill only still-unknown pixels
            known = (F.conv2d(known, k, padding=1) > 0).float()  # grow known region 1px into the hole
        return out

    @staticmethod
    @shapecheck
    def synthesize_from_labels(mask: Integer[torch.Tensor, "*b h w"], cfg: SynthCfg, n_classes: int,  # noqa: C901, PLR0912, PLR0915
                               real_img: Float[torch.Tensor, "*b 1 h w"] | None = None, *, return_meta: bool = False):
        # complexity noqa above: a linear bSSFP pipeline of optional physical effects (mu/sg/oh threaded), not
        # control-flow complexity — splitting would fragment one signal model into ~8-tensor helpers. Kept whole.
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
        b = mask.shape[0]
        dev = mask.device
        mask = mask.long()
        bg = cfg.bg.build()                              # bg STRATEGY (flat/procedural/partition/hybrid/mrxcat)
        if cfg.deform > 0 and not bg.keeps_heart_aligned:   # hybrid keeps heart aligned with the real hole
            grid = SynthPainter._deform_grid(b, *mask.shape[-2:], cfg.deform, dev)
            mask = F.grid_sample(mask[:, None].float(), grid, mode="nearest",
                                 padding_mode="border", align_corners=False)[:, 0].long()
        # extend the heart-only labels to the whole FOV (bg tissue tiers / real SHAPES / random blobs — the
        # chosen strategy decides). partition/hybrid assume real_img is heart-excised (bd mirs).
        ext, n_paint = bg.extend(mask, n_classes, dev, real_img)
        oh = F.one_hot(ext, n_paint).permute(0, 3, 1, 2).float()                # [B,n_paint,H,W]
    
        # --- physical paint: each class' intensity = balanced-SSFP signal from its tissue T1/T2/PD under
        #     per-sample swept sequence params (TR, flip) and FIELD strength (1.5T/3T = cross-vendor axis). ---
        # tissue params per available field -> [n_fields, n_paint]; pick one field per sample
        params = [bg.paint_params(n_classes, n_paint, float(f), dev) for f in cfg.fields]  # strategy owns tissues
        t1s = torch.stack([p[0] for p in params]); t2s = torch.stack([p[1] for p in params])
        pds = torch.stack([p[2] for p in params])                               # [n_fields, n_paint]
        # acquisition STRATEGY: per-sample field/TR/flip/vendor (legacy ranges / physics-randomized / matched)
        fi, tr, fl, vi = cfg.acq.build().sample(b, cfg, dev)
        t1, t2, pd = t1s[fi], t2s[fi], pds[fi]                                  # [B, n_paint]
        if cfg.tissue_spread > 0:                                               # physical per-sample T1/T2 sweep
            t1, t2 = MriPhysics.sample_heart_tissue(t1, t2, fi, cfg.fields, n_classes, cfg.tissue_spread)
        meta = {"vendor": [cfg.vendors[i] for i in vi.tolist()],                 # provenance for each synth
                "field": torch.tensor(cfg.fields, device=dev)[fi], "tr": tr[:, 0], "flip": fl[:, 0]}
        mu = MriPhysics.bssfp_signal(t1, t2, pd, tr, fl * math.pi / 180.0)       # [B, n_paint] steady-state
        mu = mu + cfg.jitter * mu.abs().mean() * torch.randn(b, n_paint, device=dev)   # residual jitter
        if cfg.inflow:                               # entry-slice inflow: f_fresh = min(1, v*TR/thk) PER SAMPLE
            v = torch.rand(b, 1, device=dev) * (cfg.blood_v_cms[1] - cfg.blood_v_cms[0]) + cfg.blood_v_cms[0]
            thk = torch.rand(b, 1, device=dev) * (cfg.slice_mm[1] - cfg.slice_mm[0]) + cfg.slice_mm[0]
            f = (v * tr / (100.0 * thk)).clamp(max=1.0)                          # v cm/s, tr ms, thk mm -> frac
            s_fresh = pd * torch.sin(fl * math.pi / 180.0)                       # [B, n_paint] fully-relaxed excite
            for c in MriPhysics.blood_classes(n_classes):
                mu[:, c] = (1 - f[:, 0]) * mu[:, c] + f[:, 0] * s_fresh[:, c]     # blend blood toward fresh
        if cfg.blood_scale != 1.0:                    # legacy empirical blood-pool mean scale (superseded by inflow)
            for c in MriPhysics.blood_classes(n_classes):
                mu[:, c] = mu[:, c] * cfg.blood_scale
        sg = mu.abs() * cfg.texture                                              # within-class texture
        if cfg.flow > 0:                                                         # flow: blood pools spread
            for c in MriPhysics.blood_classes(n_classes):
                sg[:, c] = sg[:, c] + cfg.flow * mu[:, c].abs()
        mu_map = (oh * mu[:, :, None, None]).sum(1, keepdim=True)                # [B,1,H,W] class mean
        # off-resonance bSSFP banding: a smooth Δf (B0) field -> per-pixel signal drop near dphi=±π,
        # strongest for long-T2 blood -> lowers+spreads the over-bright cavity (the cav-fidelity fix).
        # Applied as the signal RATIO band(φ)/band(0) so per-class jitter/flow in mu_map are preserved.
        if cfg.b0_hz > 0:
            t2m = (oh * t2[:, :, None, None]).sum(1, keepdim=True)               # per-pixel T2
            low = torch.rand(b, 1, 4, 4, device=dev) * 2 - 1
            df = cfg.b0_hz * F.interpolate(low, size=mask.shape[-2:], mode="bilinear", align_corners=False)
            phi = 2 * math.pi * df * (tr[:, :, None, None] / 1000.0)             # off-resonance per TR (rad)
            mu_map = mu_map * MriPhysics.banding(t2m, tr[:, :, None, None], phi)
        sg_map = (oh * sg[:, :, None, None]).sum(1, keepdim=True)               # [B,1,H,W] class std
        # partial volume: blur the class-MEAN map so boundary voxels are tissue mixes (real finite-voxel
        # averaging), not hard label edges. Texture (sg) added after, so interiors keep their grain.
        if cfg.pv_sigma > 0:
            kpv = Augmentor.gaussian_kernel(cfg.pv_sigma).to(dev)
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
        if sigma > _MIN_BLUR_SIGMA:
            k = Augmentor.gaussian_kernel(sigma).to(dev)
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
    
        # post-paint composite (strategy-specific: hybrid pastes the synth heart into the real image;
        # every other strategy is a no-op). real_img must be heart-excised for this to be clean (bd mirs).
        img = bg.compose(img, mask, real_img)
    
        # --- z-score per sample (match the real preprocessed input distribution) ---
        m = img.mean((1, 2, 3), keepdim=True)
        s = img.std((1, 2, 3), keepdim=True).clamp_min(1e-6)
        img = (img - m) / s
        mask = bg.seg_target(mask)                                 # decouple paint map from seg target (FovBg)
        return (img, mask, meta) if return_meta else (img, mask)   # opt-in provenance (vendor/field/tr/flip)
