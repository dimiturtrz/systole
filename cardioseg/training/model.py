"""MONAI U-Net factory (2D or 3D). Shape comes from an injected ModelCfg (cardioseg.hparams)."""

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
