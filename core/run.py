"""load_run: rebuild a trained run (weights + saved TrainCfg) into an eval-mode model. It needs BOTH
core.model (architecture) and core.hparams (TrainCfg deserialization), so it lives above both — the
extraction that removes core.model's back-dependency on core.hparams (the former circular import)."""
from __future__ import annotations

from pathlib import Path

import torch

from core.hparams import from_json
from core.model import build_unet, resolve_device


def load_run(run, device: str | None = None):
    """Load a trained run into eval mode. The architecture is rebuilt from the run's saved
    config.json (so weights can't mismatch a wrong default arch); older runs without a config
    fall back to the default ModelCfg. Returns (model, cfg | None, device)."""
    run = Path(run)
    cfg_path = run / "config.json"
    cfg = from_json(cfg_path) if cfg_path.exists() else None
    device = resolve_device(device)
    model = build_unet(cfg.model if cfg else None).to(device)
    model.load_state_dict(torch.load(run / "model.pth", map_location=device))
    model.eval()
    return model, cfg, device
