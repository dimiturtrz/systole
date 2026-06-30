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

from core.hparams import GeneratorCfg

from .augment import augment_batch, soften
from .synth import synthesize_from_labels, measure_class_stats


class Generator:
    """Produces collapsed training batches from resident real tensors. Holds the data on its device;
    `batch(idx)` indexes, optionally replaces a per-sample fraction with synthetic-from-labels images,
    augments (real-pixel perturbation), and softens the target — returning (x [B,1,H,W], yt) ready for
    the model. synth_p=0 -> pure real; synth_p=1 -> pure synthetic; in between -> per-sample mix."""

    def __init__(self, cfg: GeneratorCfg, X: torch.Tensor, Y: torch.Tensor,
                 n_classes: int, device: str):
        self.cfg = cfg
        self.X, self.Y = X, Y
        self.n_classes = n_classes
        self.device = device
        self.synth_on = cfg.synth.synth_p > 0
        self.soft_sigma = cfg.aug.soft_label_sigma
        # Realistic intensity priors: per-class mean/std measured ONCE from the real slices, so synth
        # paints classes around their true distribution (blood bright, myo dark) instead of random.
        self.priors = None
        if self.synth_on and cfg.synth.realistic:
            self.priors = tuple(t.to(device) for t in measure_class_stats(X, Y, n_classes))

    def batch(self, idx: torch.Tensor, pin: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Collapsed batch for the given resident indices. Order: index real -> per-sample synth replace
        (invent image+anatomy from labels) -> augment (geometry+intensity on real pixels) -> soften."""
        x = self.X[idx].to(self.device, non_blocking=pin)            # [B,1,H,W] f32
        y = self.Y[idx].to(self.device, non_blocking=pin).long()     # [B,H,W]
        if self.synth_on:
            # invent image+anatomy from labels for a per-sample fraction (synth_p=1 -> pure synth). With
            # bg partition the synth target is the warped mask -> blend both x and y per sample. Paint
            # BEFORE augment so affine geometry warps synth picture + mask together.
            xs, ys = synthesize_from_labels(y, self.cfg.synth, self.n_classes, self.priors, real_img=x)
            do = (torch.rand(x.shape[0], 1, 1, 1, device=self.device) < self.cfg.synth.synth_p).float()
            x = do * xs + (1 - do) * x
            y = torch.where(do[:, 0, 0, 0].bool()[:, None, None], ys, y)
        x, y = augment_batch(x, y, self.cfg.aug)                     # GPU-batched real-pixel aug
        # soften AFTER augment (aug stays on the hard mask, nearest-interp) -> soft target last
        yt = soften(y, self.soft_sigma, self.n_classes) if self.soft_sigma > 0 else y[:, None]
        return x, yt
