"""core.run tests: load_run rebuilds a trained run (weights + saved TrainCfg) into an eval-mode net.

The mlflow registry, torch weight file, and Hparams deserialization are faked at the core.run
module boundary — only load_run's own branching is exercised: architecture rebuilt from the run's
config.json when present, default ModelCfg fallback when absent, weights loaded, net put in eval,
and the resolved device threaded through.
"""
import torch.nn as nn

import core.run as run_mod
from core.run import Run


class _FakeNet(nn.Module):
    """A net stub recording load_state_dict / eval / to for the assertions."""
    def __init__(self):
        super().__init__()
        self.loaded = self.evaled = None
        self.device = None

    def load_state_dict(self, state_dict, *args, **kwargs):
        self.loaded = state_dict

    def eval(self):
        self.evaled = True
        return self

    def to(self, *args, **kwargs):
        self.device = args[0] if args else kwargs.get("device")
        return self


def _patch(monkeypatch, net, *, cfg):
    """Fake the three heavy collaborators load_run calls: build_unet, resolve_device, torch.load,
    and (when cfg) Hparams.from_json."""
    seen = {}
    monkeypatch.setattr(run_mod.Model, "resolve_device", lambda d: d or "cpu")
    monkeypatch.setattr(run_mod.Model, "build_unet", lambda c: (seen.update(built_with=c), net)[1])
    monkeypatch.setattr(run_mod.torch, "load", lambda p, map_location=None: {"w": 1})
    if cfg is not None:
        monkeypatch.setattr(run_mod.Hparams, "from_json", lambda p: cfg)
    return seen


def test_load_run_rebuilds_arch_from_config(monkeypatch, tmp_path):
    """config.json present -> arch rebuilt from its .model; weights loaded; eval mode; cfg returned."""
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.pth").write_bytes(b"w")
    cfg = type("Cfg", (), {"model": object()})()
    net = _FakeNet()
    seen = _patch(monkeypatch, net, cfg=cfg)

    model, out_cfg, device = Run.load_run(tmp_path, "cpu")
    assert model is net
    assert out_cfg is cfg
    assert device == "cpu"
    assert seen["built_with"] is cfg.model         # arch from the saved cfg, not default
    assert net.loaded == {"w": 1} and net.evaled and net.device == "cpu"


def test_load_run_falls_back_to_default_cfg(monkeypatch, tmp_path):
    """No config.json -> cfg is None and build_unet gets None (default ModelCfg)."""
    (tmp_path / "model.pth").write_bytes(b"w")
    net = _FakeNet()
    seen = _patch(monkeypatch, net, cfg=None)

    model, out_cfg, device = Run.load_run(tmp_path)
    assert out_cfg is None
    assert seen["built_with"] is None              # default arch
    assert device == "cpu" and net.evaled
