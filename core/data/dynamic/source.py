"""DynamicSource — synthetic training data behind the same seam as StaticSource.

A generation node: shapes come from an anatomy pool (Rodero SSM label maps), intensities are painted by
the Generator from physical bounds (bg_mode + the swept bSSFP contrast). Input-dependence is a spectrum,
set by `seed`:
  - seed=None  -> ZERO real input: label maps only, image = zeros, every row force-painted (synth_p
                  irrelevant — force_synth overrides it). The magnum-opus "train synth, test real" arm.
  - seed=Source -> COMPOSITE: the seed's real (X,Y) unioned with the synth-anatomy rows; only the synth
                  rows are force-painted (real rows repaint with prob synth_p). Augmentation (bd pwih).

`materialize` returns the SAME triple as StaticSource (X, Y, force_synth) — so the train loop is one
polymorphic call, no real-vs-synth if. `bg_mode` rides on the source (the painter strategy the Generator
looks up); train.py applies it to the generator cfg before building the engine.
"""
from __future__ import annotations

import torch

from core.data.dynamic import anatomy as _anatomy
from core.data.dynamic.generator import Generator
from core.data.dynamic.synth import AnyBgCfg, ProceduralBgCfg
from core.data.ingest.source import Source


class DynamicSource:
    kind = "dynamic"

    def __init__(self, pool: str, bg: AnyBgCfg | None = None, synth_p: float = 1.0,
                 seed: Source | None = None, note: str = ""):
        self.pool = str(pool)
        self.bg = bg or ProceduralBgCfg()      # whole-FOV synthetic organ field (zero-real goalpost, bd bwp)
        self.synth_p = synth_p
        self.seed = seed
        self._note = note

    def _resident(self, size: int, device: str):
        """(X, Y, force_synth) resident tensors for this source's Generator. Zero-input: N = pool size,
        X = zeros, all force. Seeded: seed's real ++ synth-anatomy rows, force only the synth ones."""
        Ys = torch.as_tensor(_anatomy.load_pool(self.pool), dtype=torch.long, device=device)    # [M,H,W] labels
        Xsy = torch.zeros((Ys.shape[0], 1, size, size), device=device)                 # no real pixels
        if self.seed is None:                                                          # zero real input
            return Xsy, Ys, torch.ones(Ys.shape[0], dtype=torch.bool, device=device)
        Xr, Yr = self.seed.resident(size, device)                                      # composite: real ++ synth
        X = torch.cat([Xr, Xsy]); Y = torch.cat([Yr, Ys])
        fs = torch.cat([torch.zeros(Xr.shape[0], dtype=torch.bool, device=device),
                        torch.ones(Ys.shape[0], dtype=torch.bool, device=device)])
        return X, Y, fs

    def train_gen(self, size: int, device: str, gen_cfg, n_classes: int):
        """The source's own batch engine. The painter (bg_mode) + synth fraction ride on a COPY of the
        generator cfg — no global mutation. force_synth is internal (never in the public interface)."""
        X, Y, fs = self._resident(size, device)
        synth = gen_cfg.synth.model_copy(update={"bg": self.bg, "synth_p": self.synth_p})
        cfg = gen_cfg.model_copy(update={"synth": synth})
        return Generator(cfg, X, Y, n_classes, device, force_synth=fs)

    def provenance(self) -> dict:
        return {"kind": self.kind, "pool": self.pool, "bg": self.bg.mode, "synth_p": self.synth_p,
                "note": self._note, "seed": (self.seed.provenance() if self.seed else None)}
