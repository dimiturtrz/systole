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
from pydantic import BaseModel, Field

from core.config import _VALIDATE
from core.data.static.store import DataCfg

from .augment import AugCfg
from .synth import SynthCfg
from .pipeline import Batch, build_pipeline


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

    def __init__(self, cfg: GeneratorCfg, X: torch.Tensor, Y: torch.Tensor,
                 n_classes: int, device: str, force_synth: torch.Tensor | None = None,
                 valid: torch.Tensor | None = None):
        self.cfg = cfg
        self.X, self.Y = X, Y
        self.device = device
        # force_synth [N] bool: rows that MUST be painted synthetic every batch (e.g. synth-anatomy masks
        # with no real pixels), aligned to X/Y. Real rows still repaint with prob synth_p. (bd pwih)
        self.force_synth = force_synth
        # valid [N,C] bool: per-slice class-validity for partial-label training (None = all valid). Rides
        # to the loss via Batch.valid; the Generator just slices + carries it (no transform touches it).
        self.valid = valid
        # The transform recipe: synth-replace -> augment -> soften. A composable op list, not an
        # if-ladder — sweepable (physically-constrained diversity), and each op is unit-testable.
        self.pipeline = build_pipeline(cfg, n_classes)

    @property
    def synth_on(self) -> bool:
        """Whether synth painting can fire this Generator (synth_p>0, or any forced-synth row)."""
        return self.cfg.synth.synth_p > 0 or (self.force_synth is not None and bool(self.force_synth.any()))

    def batch(self, idx: torch.Tensor, pin: bool = False):
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
