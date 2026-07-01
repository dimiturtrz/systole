"""MONAI U-Net factory (2D or 3D) + run loading. Shape comes from an injected ModelCfg
(defined here, next to build_unet); load_run rebuilds it from a run's saved config so the
architecture always matches the weights."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from core.config import _VALIDATE


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
    norm: str = "instance"                         # feature normalization: 'instance' (default; MONAI's,
    #                                                per-instance -> self-harmonizes contrast/scale, OOD-robust),
    #                                                'batch' (train running stats, NOT per-instance), 'none'
    #                                                (no self-norm). Ablation knob for the harmonization test
    #                                                (bd h8k: does input harmonization matter without instance-norm?).


def build_unet(cfg: ModelCfg | None = None):
    """4-class U-Net (bg, RV, myo, LV-cav) from a ModelCfg. Default cfg = 2D slice-wise."""
    from monai.networks.nets import UNet
    cfg = cfg or ModelCfg()
    return UNet(
        spatial_dims=cfg.spatial_dims,
        in_channels=cfg.in_channels,
        out_channels=cfg.out_channels,
        channels=tuple(cfg.channels),
        strides=tuple(cfg.strides),
        num_res_units=cfg.res_units,
        dropout=cfg.dropout,          # enables MC-dropout uncertainty at inference (iq7)
        norm=(None if cfg.norm == "none" else cfg.norm),   # 'instance' (default) | 'batch' | 'none' (ablation)
    )


def resolve_device(preferred: str | None = None) -> str:
    """Torch device string: explicit `preferred`, else 'cuda' if available, else 'cpu'."""
    import torch
    return preferred or ("cuda" if torch.cuda.is_available() else "cpu")


def load_run(run, device: str | None = None):
    """Load a trained run into eval mode. The architecture is rebuilt from the run's saved
    config.json (so weights can't mismatch a wrong default arch); older runs without a config
    fall back to the default ModelCfg. Returns (model, cfg | None, device)."""
    import torch
    run = Path(run)
    cfg_path = run / "config.json"
    cfg = None
    if cfg_path.exists():
        from core.hparams import from_json
        cfg = from_json(cfg_path)
    device = resolve_device(device)
    model = build_unet(cfg.model if cfg else None).to(device)
    model.load_state_dict(torch.load(run / "model.pth", map_location=device))
    model.eval()
    return model, cfg, device
