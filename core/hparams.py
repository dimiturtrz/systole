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

from pydantic import BaseModel, Field, model_validator

from core.config import _VALIDATE
from core.data.dynamic.augment import AugCfg
from core.data.dynamic.generator import GeneratorCfg
from core.data.dynamic.synth import SynthCfg
from core.data.static.store import DataCfg
from core.losses import (
    LOSS_VARIANTS,
    AnyLossCfg,
    DiceCECfg,
    DiceCEHDCfg,
    DiceCEHERCfg,
    DiceCETverskyCfg,
    LossCfg,
)
from core.model import ModelCfg
from core.preprocessing.n4 import N4Cfg

# Config classes live with the class they configure (ModelCfg→core.model, AugCfg→augment,
# SynthCfg→synth, DataCfg→store, N4Cfg→preprocessing.n4, GeneratorCfg→generator, LossCfg+variants→
# core.losses so each cfg builds its own loss). Re-exported here so `from core.hparams import X` still
# resolves. TrainCfg (whole-run composition root) stays here.
__all__ = ["ModelCfg", "AugCfg", "SynthCfg", "N4Cfg", "DataCfg", "GeneratorCfg", "LossCfg",
           "AnyLossCfg", "DiceCECfg", "DiceCETverskyCfg", "DiceCEHERCfg", "DiceCEHDCfg", "LOSS_VARIANTS",
           "TrainCfg", "apply_overrides", "to_json", "from_json"]


# LossCfg (discriminated union) + variants live in core.losses, co-located with the loss classes so
# each cfg can BUILD its own loss (cfg.build()). Imported above; re-exported for back-compat.


class TrainCfg(BaseModel):
    """The whole run = master config. Two halves mirroring the runtime objects: `generator` (data
    engine) + `model`/`loss` (consumer), plus run params. model_dump() -> config.json = full
    provenance; one TrainCfg fully describes a run."""
    model_config = _VALIDATE

    @model_validator(mode="before")
    @classmethod
    def _lift_flat(cls, v):
        """Back-compat: old config.json had data/aug/synth at the top level; lift them under
        `generator` so pre-refactor runs (registered models, cached configs) still load."""
        if isinstance(v, dict) and "generator" not in v and any(k in v for k in ("data", "aug", "synth")):
            v = dict(v)
            v["generator"] = {k: v.pop(k) for k in ("data", "aug", "synth") if k in v}
        return v

    generator: GeneratorCfg = Field(default_factory=GeneratorCfg)
    model: ModelCfg = Field(default_factory=ModelCfg)
    loss: AnyLossCfg = Field(default_factory=DiceCECfg)
    epochs: int = Field(128, ge=1)                 # ceiling; early stopping ends sooner
    batch: int = Field(64, ge=1)
    lr: float = Field(1e-3, gt=0)
    patience: int = Field(20, ge=1)
    workers: int = Field(6, ge=0)                  # store consolidation only (DataLoader is workers=0)
    seed: int = Field(0, ge=0)
    n_patients: int = Field(0, ge=0)               # debug cap (0 = all)
    # EF/volume-consistency auxiliary lane: after warmup, each epoch adds `ef_lambda`·vol_loss (soft
    # EDV/ESV vs GT, over `ef_subjects` sampled subjects) INTO one seg gradient step — a NUDGE, not a
    # co-equal vote (seg is dense signal, EF is 2 scalars/patient). vol_loss is DIMENSIONLESS (relative
    # to EDV_gt) so ef_lambda is O(1) + scale-robust, not a unit-coupled magic number. ef_warmup lets
    # seg establish first (like the HD/HER aux losses). 0 = off. (cardioseg.training.ef_lane)
    ef_lambda: float = Field(0.0, ge=0)
    ef_warmup: int = Field(4, ge=0)
    ef_subjects: int = Field(16, ge=1)
    # ef_learn: retire the hand-set ef_lambda — LEARN the seg-vs-EF balance via Kendall uncertainty
    # weighting (2 log-variances trained with the model). Self-balancing, no magic weight. Needs
    # ef_lambda>0 only as the on-switch (its value is ignored when ef_learn=True).
    ef_learn: bool = False
    # ef_kaggle: also feed the vol lane the Kaggle EF-only cases (1140, no masks) as EF-RATIO weak
    # supervision — the original point (turn "unusable" EF-only data into LV-cav volume signal).
    ef_kaggle: bool = False
    ef_kaggle_subjects: int = Field(4, ge=1)       # Kaggle cine is ~16x a labeled patient -> sample few
    device: Optional[str] = None
    out_dir: Optional[str] = None
    # Where the preloaded slice tensors live during training — the speed/capacity tradeoff
    # (disk < ram < vram capacity; on-demand < cpu < gpu speed). "gpu": resident in VRAM, epochs are
    # pure-GPU (fastest; capped by VRAM — the cardiac set is ~3GB, fits easily). "cpu": resident in
    # RAM, batches copied to the GPU per step (for sets too big for VRAM). The loop is identical —
    # it does .to(device) either way (a no-op when already on GPU).
    residency: Literal["gpu", "cpu"] = "gpu"


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
        key = key.strip()
        if key == "loss.kind":                    # union: switch the loss VARIANT (a setattr can't cross types)
            cfg.loss = LOSS_VARIANTS[val.strip()]()
            continue
        *parents, leaf = key.split(".")
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
