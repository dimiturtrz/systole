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
from core.model import ModelCfg
from core.preprocessing.n4 import N4Cfg
from core.data.static.store import DataCfg
from core.data.dynamic.augment import AugCfg
from core.data.dynamic.synth import SynthCfg
from core.data.dynamic.generator import GeneratorCfg

# Config classes now live with the class they configure (ModelCfg→core.model, AugCfg→augment,
# SynthCfg→synth, DataCfg→store, N4Cfg→preprocessing.n4, GeneratorCfg→generator). Re-exported here
# so `from core.hparams import ModelCfg` etc. still resolves. LossCfg + TrainCfg stay the composition
# root: LossCfg's builder is in cardioseg (core can't import it), TrainCfg is the whole-run root.
__all__ = ["ModelCfg", "AugCfg", "SynthCfg", "N4Cfg", "DataCfg", "GeneratorCfg", "LossCfg",
           "TrainCfg", "apply_overrides", "to_json", "from_json"]


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
