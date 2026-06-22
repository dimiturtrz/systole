"""Hyperparameters as injectable dataclasses — one typed source of truth per concern.

Each component takes its config as an argument (dependency injection): `build_unet(ModelCfg)`,
`augment_batch(x, m, AugCfg)`, `train_seg(TrainCfg)`. Defaults live here (not scattered across
argparse / module constants / function signatures), so a run is fully described by one `TrainCfg`
— serialize it to `runs/<run>/config.json` and the run is reproducible + the model card has provenance.

Separate from `config.py` (paths.yaml): that's machine-specific *where data lives*; this is
run-specific *experiment hyperparams*. Different lifetimes, different files.
"""
from __future__ import annotations

import ast
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class ModelCfg:
    """U-Net shape (MONAI). Injected into build_unet."""
    spatial_dims: int = 2
    in_channels: int = 1
    out_channels: int = 4                       # bg / RV / LV-myo / LV-cav
    channels: tuple = (16, 32, 64, 128, 256)
    strides: tuple = (2, 2, 2, 2)
    res_units: int = 2


@dataclass
class AugCfg:
    """GPU-batched augmentation. Geometric widths conservative (canonical orientation); intensity
    widths broad to span the cross-vendor contrast gap. Injected into augment_batch."""
    rot_deg: float = 20.0
    scale: tuple = (0.85, 1.15)
    gamma: tuple = (0.7, 1.5)
    gamma_p: float = 0.3
    blur_p: float = 0.2
    contrast: tuple = (0.8, 1.2)
    noise: float = 0.08


@dataclass
class LossCfg:
    """Segmentation loss. dice_ce = MONAI Dice+CE (region, the baseline). dice_ce_hd adds a
    Hausdorff-DT boundary term (λ·HD), ramped in over `hd_warmup` epochs (HD losses diverge early) —
    targets the ES boundary over-segmentation that region losses are blind to (the EF-bias lever)."""
    kind: str = "dice_ce"               # dice_ce | dice_ce_tversky | dice_ce_hd
    # Tversky FP-penalty (dice_ce_tversky): beta>alpha penalizes false positives harder ->
    # discourages over-segmentation (the ES cavity over-fill). Pure GPU region loss, no warmup.
    tversky_alpha: float = 0.3          # FN weight
    tversky_beta: float = 0.7           # FP weight (> alpha = punish over-seg)
    tversky_lambda: float = 1.0         # weight of the Tversky term added to Dice+CE
    # Hausdorff-ER (dice_ce_her): erosion-based Hausdorff surrogate, pure-torch GPU (no DT/cucim).
    # Targets FAR boundary errors (stray voxels, loose RV boundary). warmup -> ramp like HD.
    her_weight: float = 0.5
    her_alpha: float = 2.0              # distance exponent ((k+1)^alpha per erosion level)
    her_erosions: int = 10             # erosion iterations (depth of distance proxy)
    her_warmup: int = 5
    her_ramp: int = 5
    # Hausdorff-DT (dice_ce_hd): NOTE CPU-bound on Windows (needs cucim, Linux-only) -> slow/unstable.
    hd_weight: float = 0.01
    hd_warmup: int = 15
    hd_ramp: int = 5


@dataclass
class DataCfg:
    """What data + how it's split. battery=True pools `sources` and holds out ACDC+Canon (a split
    query); battery=False trains on `train_dataset` and tests on `test`."""
    battery: bool = True
    sources: tuple = ("acdc", "mnm2", "mnms1")  # store datasets to load for the battery pool
    train_dataset: str = "mnm2"                 # named mode (battery=False)
    test: str = "acdc"                          # named mode: acdc|mnm2|mnms1|canon|none
    inplane: float = 1.5
    n4: bool = False
    val_frac: float = 0.2
    size: int = 256


@dataclass
class TrainCfg:
    """The whole run. asdict(this) -> config.json = full provenance."""
    data: DataCfg = field(default_factory=DataCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    aug: AugCfg = field(default_factory=AugCfg)
    loss: LossCfg = field(default_factory=LossCfg)
    epochs: int = 128                           # ceiling; early stopping ends sooner
    batch: int = 64
    lr: float = 1e-3
    patience: int = 20
    workers: int = 6                            # store consolidation only (DataLoader is workers=0)
    seed: int = 0
    n_patients: int = 0                         # debug cap (0 = all)
    device: str | None = None
    out_dir: str | None = None


def _coerce(val: str, cur):
    """Coerce an override string to the current field's type (bool/int/float/tuple via literal_eval)."""
    if isinstance(cur, bool):
        return val.lower() in ("1", "true", "yes", "on")
    if isinstance(cur, int):
        return int(val)
    if isinstance(cur, float):
        return float(val)
    if isinstance(cur, (tuple, list)):
        return type(cur)(ast.literal_eval(val))
    return val


def apply_overrides(cfg: TrainCfg, items: list[str]) -> TrainCfg:
    """Apply `a.b=val` dotted overrides in place (e.g. 'aug.gamma_p=0.5', 'model.channels=(32,64,128)')."""
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
    Path(path).write_text(json.dumps(asdict(cfg), indent=2))


def from_json(path: str | Path) -> TrainCfg:
    d = json.loads(Path(path).read_text())
    nested = ("data", "model", "aug", "loss")
    return TrainCfg(data=DataCfg(**d["data"]), model=ModelCfg(**d["model"]),
                    aug=AugCfg(**d["aug"]), loss=LossCfg(**d.get("loss", {})),
                    **{k: v for k, v in d.items() if k not in nested})
