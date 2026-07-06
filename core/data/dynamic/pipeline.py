"""Batch transform pipeline — the Generator's monolith `batch()` decomposed into composable ops.

A batch is built (index resident tensors) then passed through an ordered list of Transforms, each
`(Batch) -> Batch` (MONAI-style). The op list IS the recipe: synth-from-labels replace -> real-pixel
augment -> soften target. Pluggable + sweepable — which directly serves "physically-constrained
diversity" (the transform list is the manifold you sweep), and it kills the hardcoded if-ladder in
Generator.batch. Same order + math as before -> bit-identical batches.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from .augment import AugCfg, augment_batch, soften
from .synth import SynthCfg, synthesize_from_labels


@dataclass
class Batch:
    """The mutable batch flowing through the pipeline. `x` image, `y` hard mask, `force` = per-row
    bool 'must be painted synthetic' (None = none), `yt` = the (soft) target the model trains on."""
    x: torch.Tensor                    # [B,1,H,W] f32
    y: torch.Tensor                    # [B,H,W] long
    force: torch.Tensor | None = None  # [B] bool
    yt: torch.Tensor | None = None     # [B,C,H,W] or [B,1,H,W], set by Soften


class Transform(Protocol):
    def __call__(self, b: Batch) -> Batch: ...


class SynthReplace:
    """Per-row replace image+mask with synthetic-from-labels (physics contrast). Fires on a `synth_p`
    fraction, plus any `force`-marked rows (synth-anatomy with no real pixels). No-op when off."""

    def __init__(self, cfg: SynthCfg, n_classes: int):
        self.cfg, self.n_classes = cfg, n_classes

    def __call__(self, b: Batch) -> Batch:
        on = self.cfg.synth_p > 0 or (b.force is not None and bool(b.force.any()))
        if not on:
            return b
        xs, ys = synthesize_from_labels(b.y, self.cfg, self.n_classes, real_img=b.x)
        pick = torch.rand(b.x.shape[0], device=b.x.device) < self.cfg.synth_p
        if b.force is not None:
            pick = pick | b.force
        do = pick.float()[:, None, None, None]
        b.x = do * xs + (1 - do) * b.x
        b.y = torch.where(do[:, 0, 0, 0].bool()[:, None, None], ys, b.y)
        return b


class Augment:
    """GPU-batched real-pixel augmentation (geometry + intensity), on the hard mask (nearest interp)."""

    def __init__(self, cfg: AugCfg):
        self.cfg = cfg

    def __call__(self, b: Batch) -> Batch:
        b.x, b.y = augment_batch(b.x, b.y, self.cfg)
        return b


class Soften:
    """Final target: soft probabilistic mask (sigma>0) or the hard mask with a channel dim (sigma=0)."""

    def __init__(self, sigma: float, n_classes: int):
        self.sigma, self.n_classes = sigma, n_classes

    def __call__(self, b: Batch) -> Batch:
        b.yt = soften(b.y, self.sigma, self.n_classes) if self.sigma > 0 else b.y[:, None]
        return b


def build_pipeline(cfg, n_classes: int) -> list[Transform]:
    """The default recipe: synth-replace -> augment -> soften. cfg = GeneratorCfg (synth + aug)."""
    return [SynthReplace(cfg.synth, n_classes), Augment(cfg.aug), Soften(cfg.aug.soft_label_sigma, n_classes)]
