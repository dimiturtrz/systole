"""Hyperparameters as injectable, *validated* configs (pydantic v2) — one typed source of truth.

Each component takes its config as an argument (dependency injection): `build_unet(ModelCfg)`,
`augment_batch(x, m, AugCfg)`, `train_seg(TrainCfg)`. Defaults live here; a run is fully described by
one `TrainCfg`, serialized to `runs/<run>/config.json` (provenance + reproducibility).

Why pydantic (not plain dataclasses): the fields cross a trust boundary — `--set` overrides and
loaded config.json are user input. Bounds (`gamma_p∈[0,1]`, `val_frac∈(0,1)`, `loss.kind` enum, …)
+ `validate_assignment` mean a bad value is rejected AT LOAD with a clear error, instead of silently
training 6 min on garbage. That's the boundary where validation pays (cardiac-seg-8y9); internal
structs we fully control stay plain.

Separate from `config.py` (paths.yaml = machine-specific *where data lives*); this is run-specific.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from core.config import DEFAULT_INPLANE, DEFAULT_SIZE, KNOWN_DATASETS

_VALIDATE = ConfigDict(validate_assignment=True)   # setattr (used by --set) re-validates the field


class ModelCfg(BaseModel):
    """U-Net shape (MONAI). Injected into build_unet."""
    model_config = _VALIDATE
    spatial_dims: Literal[2, 3] = 2
    in_channels: int = Field(1, ge=1)
    out_channels: int = Field(4, ge=2)             # bg / RV / LV-myo / LV-cav
    channels: tuple[int, ...] = (16, 32, 64, 128, 256)
    strides: tuple[int, ...] = (2, 2, 2, 2)
    res_units: int = Field(2, ge=0)
    dropout: float = Field(0.0, ge=0, le=1)        # 0 by default: dropout 0.1/0.2 regressed EF ~2pp on this
    #                                                already-regularized small net (heavy aug + early stop), no
    #                                                Dice gain — boundary/volume precision is dropout-fragile. See bp4.


class AugCfg(BaseModel):
    """GPU-batched augmentation. Geometric widths conservative; intensity widths broad to span the
    cross-vendor contrast gap. Injected into augment_batch."""
    model_config = _VALIDATE
    rot_deg: float = Field(20.0, ge=0)
    scale: tuple[float, float] = (0.85, 1.15)
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


class LossCfg(BaseModel):
    """Segmentation loss. dice_ce = MONAI Dice+CE (region baseline). dice_ce_tversky adds an FP-penalty
    (beta>alpha discourages over-seg). dice_ce_her adds a pure-GPU erosion-Hausdorff boundary term.
    dice_ce_hd = Hausdorff-DT (CPU-bound on Windows: needs cucim = Linux-only)."""
    model_config = _VALIDATE
    kind: Literal["dice_ce", "dice_ce_tversky", "dice_ce_her", "dice_ce_hd"] = "dice_ce"
    tversky_alpha: float = Field(0.3, ge=0, le=1)  # FN weight
    tversky_beta: float = Field(0.7, ge=0, le=1)   # FP weight (> alpha = punish over-seg)
    tversky_lambda: float = Field(1.0, ge=0)
    her_weight: float = Field(0.5, ge=0)
    her_alpha: float = Field(2.0, ge=0)            # distance exponent ((k+1)^alpha per erosion level)
    her_erosions: int = Field(10, ge=1)
    her_warmup: int = Field(5, ge=0)
    her_ramp: int = Field(5, ge=1)
    hd_weight: float = Field(0.01, ge=0)
    hd_warmup: int = Field(15, ge=0)
    hd_ramp: int = Field(5, ge=1)


class N4Cfg(BaseModel):
    """N4 bias-field correction params (the SimpleITK path, preprocessing/normalization/n4.py).
    Serialized inside DataCfg so a run with n4=True fully records what it ran (+ keys its cache)."""
    model_config = _VALIDATE
    shrink: int = Field(4, ge=1)               # downsample factor for the field fit (speed)
    iters: tuple[int, ...] = (50, 50, 50)      # per-level fitting iterations
    fwhm: float = Field(0.15, gt=0)            # bias-field FWHM


class DataCfg(BaseModel):
    """The data + the split, as criteria over the cloud (no named splits). Load `sources`; hold out
    everything matching `test_datasets` (whole dataset) OR `test_vendors` (by vendor); train/val =
    the rest, labelled. The criteria ARE the split — serialized to config.json. Defaults = the
    generalization split (ACDC centre-shift + Canon unseen-vendor)."""
    model_config = _VALIDATE
    sources: tuple[str, ...] = KNOWN_DATASETS
    # Split = criteria over the cloud. TEST = unseen vendors (Canon + GE) held out entirely, plus
    # cmrxmotion as a whole (single-vendor Siemens motion-robustness set — must be held out by
    # dataset, else it'd silently join Siemens train). VAL = ACDC (a held-out centre/protocol) — a
    # real domain-shift tuning signal that is NOT test, so aug/calibration are tuned without peeking
    # at test. TRAIN = the rest (Siemens + Philips).
    test_datasets: tuple[str, ...] = ("cmrxmotion",)
    test_vendors: tuple[str, ...] = ("Canon", "GE")
    val_datasets: tuple[str, ...] = ("acdc",)        # held-out domain for val (empty -> random val_frac)
    val_vendors: tuple[str, ...] = ()
    inplane: float = Field(DEFAULT_INPLANE, gt=0)
    n4: bool = False
    n4_params: N4Cfg = Field(default_factory=N4Cfg)   # only applied when n4=True; recorded regardless
    val_frac: float = Field(0.2, gt=0, lt=1)
    size: int = Field(DEFAULT_SIZE, ge=32)


class TrainCfg(BaseModel):
    """The whole run. model_dump() -> config.json = full provenance."""
    model_config = _VALIDATE
    data: DataCfg = Field(default_factory=DataCfg)
    model: ModelCfg = Field(default_factory=ModelCfg)
    aug: AugCfg = Field(default_factory=AugCfg)
    loss: LossCfg = Field(default_factory=LossCfg)
    epochs: int = Field(128, ge=1)                 # ceiling; early stopping ends sooner
    batch: int = Field(64, ge=1)
    lr: float = Field(1e-3, gt=0)
    patience: int = Field(20, ge=1)
    workers: int = Field(6, ge=0)                  # store consolidation only (DataLoader is workers=0)
    seed: int = Field(0, ge=0)
    n_patients: int = Field(0, ge=0)               # debug cap (0 = all)
    device: Optional[str] = None
    out_dir: Optional[str] = None


def _coerce(val: str, cur):
    """Parse an override string to the current field's type (tuples via literal_eval). pydantic's
    validate_assignment then enforces the bounds when we setattr the result."""
    if isinstance(cur, bool):
        return val.lower() in ("1", "true", "yes", "on")
    if isinstance(cur, int):
        return int(val)
    if isinstance(cur, float):
        return float(val)
    if isinstance(cur, (tuple, list)):
        return ast.literal_eval(val)
    if cur is None:                                       # Optional[str] field (device, out_dir)
        return None if val.lower() in ("none", "null", "") else val
    return val


def apply_overrides(cfg: TrainCfg, items: list[str]) -> TrainCfg:
    """Apply `a.b=val` dotted overrides in place (e.g. 'aug.gamma_p=0.5', 'data.test_vendors=(\"GE\",)').
    Each setattr is validated (validate_assignment) -> an out-of-bounds/typo value raises immediately."""
    for it in items or []:
        key, _, val = it.partition("=")
        *parents, leaf = key.strip().split(".")
        obj = cfg
        for p in parents:
            obj = getattr(obj, p)
        setattr(obj, leaf, _coerce(val.strip(), getattr(obj, leaf)))
    return cfg


def to_json(cfg: TrainCfg, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(cfg.model_dump_json(indent=2))


def from_json(path: str | Path) -> TrainCfg:
    return TrainCfg.model_validate_json(Path(path).read_text())
