"""The training data engine — one object that turns the resident real slices into collapsed,
ready-to-train batches. The dual of the model: `Generator(GeneratorCfg)` produces batches, the model
consumes them. Everything that shapes a batch lives here — synthetic-from-labels generation (SynthSeg),
real-pixel augmentation, soft-label targets — so the train loop is just glue:

    gen = Generator(cfg.generator, Xtr, Ytr, n_classes, device)
    for idx in batches:
        x, yt = gen.batch(idx)          # collapsed: values realized, target softened
        loss = loss_fn(model(x), yt)

"Collapsed" = the batch is concrete tensors: priors sampled, partition done, deform/paint/augment all
realized; no distributions leak to the caller. Real / synth / mixed are all just this one object under
different SynthCfg — and a future learned (GAN) generator slots in behind the same `batch()` seam.
"""
from __future__ import annotations

import torch
from jaxtyping import Bool, Float, Integer
from pydantic import BaseModel, Field

from core.config import _VALIDATE
from core.data.static.store import DataCfg
from core.types import shapecheck

from .augment import AugCfg
from .pipeline import Batch, Pipeline
from .synth import SynthCfg


class GeneratorCfg(BaseModel):
    """The data-engine config: everything that makes a training BATCH (vs the model that consumes it).
    Composes the data/split (`data`), real-pixel perturbation (`aug`), and synthetic-from-labels
    generation (`synth`). Drives the Generator. Swapping the data side of a run = editing this subtree;
    a future GAN generator slots in behind the same seam."""
    model_config = _VALIDATE
    data: DataCfg = Field(default_factory=DataCfg)
    aug: AugCfg = Field(default_factory=AugCfg)
    synth: SynthCfg = Field(default_factory=SynthCfg)


class Generator:
    """Produces collapsed training batches from resident real tensors. Holds the data on its device;
    `batch(idx)` indexes, optionally replaces a per-sample fraction with synthetic-from-labels images,
    augments (real-pixel perturbation), and softens the target — returning (x [B,1,H,W], yt) ready for
    the model. synth_p=0 -> pure real; synth_p=1 -> pure synthetic; in between -> per-sample mix."""

    @shapecheck
    def __init__(self, cfg: GeneratorCfg, X: Float[torch.Tensor, "n 1 h w"], Y: Integer[torch.Tensor, "n h w"],  # noqa: PLR0913  constructor init inputs (Generator holds the state)
                 n_classes: int, device: str, force_synth: Bool[torch.Tensor, "*n"] | None = None,
                 valid: Bool[torch.Tensor, "n c"] | None = None):
        self.cfg = cfg
        self.X, self.Y = X, Y
        self.n = X.shape[0]                       # resident slice count — the epoch-loop seam (also CompositeGenerator)
        self.device = device
        # force_synth [N] bool: rows that MUST be painted synthetic every batch (e.g. synth-anatomy masks
        # with no real pixels), aligned to X/Y. Real rows still repaint with prob synth_p. (bd pwih)
        self.force_synth = force_synth
        # valid [N,C] bool: per-slice class-validity for partial-label training (None = all valid). Rides
        # to the loss via Batch.valid; the Generator just slices + carries it (no transform touches it).
        self.valid = valid
        # The transform recipe: synth-replace -> augment -> soften. A composable op list, not an
        # if-ladder — sweepable (physically-constrained diversity), and each op is unit-testable.
        self.pipeline = Pipeline.build(cfg, n_classes)

    @shapecheck
    def batch(self, idx: Integer[torch.Tensor, "*b"], *, pin: bool = False):
        """Collapsed batch for the resident indices: build the Batch, run it through the pipeline
        (index real -> synth replace -> augment -> soften), return (x, yt, valid). `valid` is None
        unless the source is partial-label (then [B,C] for the loss)."""
        b = Batch(x=self.X[idx].to(self.device, non_blocking=pin),
                  y=self.Y[idx].to(self.device, non_blocking=pin).long(),
                  force=None if self.force_synth is None else self.force_synth[idx].to(self.device),
                  valid=None if self.valid is None else self.valid[idx].to(self.device))
        for t in self.pipeline:
            b = t(b)
        return b.x, b.yt, b.valid


class CompositeGenerator:
    """Union of child generators behind the SAME batch() seam — the composite training set as a list of
    sources, each with its OWN painter (bg/synth_p), NOT one pool crammed into one generator. A global
    index space [0, Σnᵢ) maps to (child, local); a batch may mix rows from several children, each painting
    its own via its own pipeline. Sampling is proportional to child size (a bigger pool → more batches).
    `valid` is None (composite is for synth sources — partial-label real isn't composed here)."""

    def __init__(self, gens: list[Generator]) -> None:
        self.gens = gens
        self.device = gens[0].device
        self.valid: Bool[torch.Tensor, "n c"] | None = None
        sizes = torch.tensor([g.n for g in gens])
        self.n = int(sizes.sum())
        self.offsets = torch.cat([torch.zeros(1, dtype=torch.long), sizes.cumsum(0)])   # [len+1] boundaries

    @shapecheck
    def batch(self, idx: Integer[torch.Tensor, "*b"], *, pin: bool = False):
        """Route the global indices to their child generators, paint each child's rows with its own
        pipeline, and concatenate — one collapsed batch of mixed-source samples."""
        xs: list[torch.Tensor] = []
        ys: list[torch.Tensor] = []
        for c, g in enumerate(self.gens):
            lo, hi = int(self.offsets[c]), int(self.offsets[c + 1])
            m = (idx >= lo) & (idx < hi)
            if not bool(m.any()):
                continue
            x, y, _ = g.batch(idx[m] - lo, pin=pin)
            xs.append(x); ys.append(y)
        return torch.cat(xs), torch.cat(ys), None
