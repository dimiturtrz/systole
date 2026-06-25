"""MONAI U-Net factory (2D or 3D) + run loading. Shape comes from an injected ModelCfg
(cardioseg.hparams); load_run rebuilds it from a run's saved config so the architecture
always matches the weights."""

from pathlib import Path

from cardioseg.hparams import ModelCfg


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
        from cardioseg.hparams import from_json
        cfg = from_json(cfg_path)
    device = resolve_device(device)
    model = build_unet(cfg.model if cfg else None).to(device)
    model.load_state_dict(torch.load(run / "model.pth", map_location=device))
    model.eval()
    return model, cfg, device
