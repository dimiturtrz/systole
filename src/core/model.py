"""MONAI U-Net factory (2D or 3D)."""


def build_unet(spatial_dims=3, in_channels=1, out_channels=4,
               channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2)):
    """4-class U-Net (bg, LV, myo, RV). Set spatial_dims=2 for slice-wise."""
    from monai.networks.nets import UNet
    return UNet(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=channels,
        strides=strides,
        num_res_units=2,
    )
