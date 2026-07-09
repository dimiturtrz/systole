"""core.model tests: build_unet (MONAI U-Net factory) + resolve_device.

A tiny real net is built (cheap) to check the factory wires ModelCfg into the right output
shape; device resolution is exercised as pure branching (explicit > cuda-if-available > cpu).
"""
import torch

from core.model import Model, ModelCfg


def test_build_unet_default_is_2d_4class():
    """Default ModelCfg -> 2D 4-class net: [B,1,H,W] -> [B,4,H,W] (bg/RV/myo/LV-cav)."""
    net = Model.build_unet()
    x = torch.zeros(2, 1, 32, 32)
    y = net(x)
    assert y.shape == (2, 4, 32, 32)


def test_build_unet_honors_cfg_channels():
    """A custom cfg (in/out channels) is threaded through to the net's I/O shape."""
    cfg = ModelCfg(in_channels=1, out_channels=2, channels=(8, 16), strides=(2,))
    net = Model.build_unet(cfg)
    y = net(torch.zeros(1, 1, 32, 32))
    assert y.shape == (1, 2, 32, 32)


def test_resolve_device_explicit_wins(monkeypatch):
    """An explicit device string is returned verbatim (no cuda probe)."""
    assert Model.resolve_device("cpu") == "cpu"
    assert Model.resolve_device("mps") == "mps"


def test_resolve_device_falls_back_to_cpu(monkeypatch):
    """No preference + no cuda -> 'cpu'."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert Model.resolve_device() == "cpu"


def test_resolve_device_picks_cuda_when_available(monkeypatch):
    """No preference + cuda available -> 'cuda'."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert Model.resolve_device() == "cuda"
