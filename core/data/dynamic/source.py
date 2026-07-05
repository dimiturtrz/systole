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

from core.data.source import Source

DEFAULT_POOL_BG = "procedural"          # whole-FOV synthetic organ field (zero-real goalpost, bd bwp)


class DynamicSource:
    kind = "dynamic"

    def __init__(self, pool: str, bg_mode: str = DEFAULT_POOL_BG, synth_p: float = 1.0,
                 seed: Source | None = None, note: str = ""):
        self.pool = str(pool)
        self.bg_mode = bg_mode
        self.synth_p = synth_p
        self.seed = seed
        self._note = note

    def materialize(self, size: int, device: str):
        """(X [N,1,H,W], Y [N,H,W], force_synth [N] bool). Zero-input: N = pool size, X = zeros, all
        force. Seeded: real (from seed) ++ synth-anatomy rows, force only the synth ones."""
        import torch
        from core.data.dynamic.anatomy import load_pool
        Ys = torch.as_tensor(load_pool(self.pool), dtype=torch.long, device=device)    # [M,H,W] labels
        Xsy = torch.zeros((Ys.shape[0], 1, size, size), device=device)                 # no real pixels
        if self.seed is None:                                                          # zero real input
            fs = torch.ones(Ys.shape[0], dtype=torch.bool, device=device)
            return Xsy, Ys, fs
        Xr, Yr, _ = self.seed.materialize(size, device)                                # composite: real ++ synth
        X = torch.cat([Xr, Xsy]); Y = torch.cat([Yr, Ys])
        fs = torch.cat([torch.zeros(Xr.shape[0], dtype=torch.bool, device=device),
                        torch.ones(Ys.shape[0], dtype=torch.bool, device=device)])
        return X, Y, fs

    def provenance(self) -> dict:
        return {"kind": self.kind, "pool": self.pool, "bg_mode": self.bg_mode, "synth_p": self.synth_p,
                "note": self._note, "seed": (self.seed.provenance() if self.seed else None)}
