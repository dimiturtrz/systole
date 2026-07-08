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
from core.data.dynamic.generator import CompositeGenerator, Generator
from core.data.dynamic.synth import AnyBgCfg, ProceduralBgCfg
from core.data.ingest.source import Source


class DynamicSource:
    kind = "dynamic"

    def __init__(self, pool: str, bg: AnyBgCfg | None = None, synth_p: float = 1.0,  # noqa: PLR0913
                 seed: Source | None = None, note: str = "", cap: int | None = None):
        self.pool = str(pool)
        self.bg = bg or ProceduralBgCfg()      # whole-FOV synthetic organ field (zero-real goalpost, bd bwp)
        self.synth_p = synth_p
        self.seed = seed
        self._note = note
        # cap = max resident slices this source contributes (deterministic subsample). The GPU-resident
        # preload is VRAM-bounded (synth_main: the full 42k-slice pool does NOT fit the 32 GB card, only
        # ~10k does) — a composite of several big pools must cap each so the UNION stays resident-sized.
        self.cap = cap

    def _resident(self, size: int, device: str):
        """(X, Y, force_synth) resident tensors for this source's Generator. Zero-input: N = pool size,
        X = zeros, all force. Seeded: seed's real ++ synth-anatomy rows, force only the synth ones."""
        Ys = torch.as_tensor(_anatomy.load_pool(self.pool), dtype=torch.long, device=device)    # [M,H,W] labels
        if self.cap is not None and Ys.shape[0] > self.cap:                             # VRAM-bound the resident
            keep = torch.randperm(Ys.shape[0], generator=torch.Generator().manual_seed(0))[:self.cap]
            Ys = Ys[keep.to(Ys.device)]
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
        return {"kind": self.kind, "pool": self.pool, "cap": self.cap, "bg": self.bg.mode,
                "synth_p": self.synth_p, "note": self._note,
                "seed": (self.seed.provenance() if self.seed else None)}


class CompositeSource:
    """The composite training set as a UNION OF SOURCES — each a clean single-origin generation node
    (SSM pool, label-space pathology, MRXCAT, learned), each keeping its OWN painter/bg, not one pool
    frankensteined into one generator. `train_gen` builds each child's generator and unions them behind
    the same batch() seam (CompositeGenerator). This is the "each source enters the DAG at a different
    point with a different control degree" composition (bd cumw/uch6)."""

    kind = "composite"

    def __init__(self, sources, note: str = ""):
        self.sources = tuple(sources)
        if not self.sources:
            raise ValueError("CompositeSource needs at least one source")
        self._note = note

    def train_gen(self, size: int, device: str, gen_cfg, n_classes: int):
        return CompositeGenerator([s.train_gen(size, device, gen_cfg, n_classes) for s in self.sources])

    def provenance(self) -> dict:
        return {"kind": self.kind, "note": self._note,
                "sources": [s.provenance() for s in self.sources]}
