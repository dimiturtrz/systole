"""Differentiable volume-consistency loss from SOFT EDV/ESV volumes (torch).

The dimensionless EDV/ESV Huber that supervises LV-cav VOLUME — what EF depends on and per-pixel Dice
is blind to. The EF lane builds the soft per-subject EDV/ESV (segment-summed model forward, see
cardioseg.training.ef_lane) and passes them here.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from jaxtyping import Float

from core.shapecheck import shapecheck


class VolLoss:
    """Namespace for the dimensionless EDV/ESV volume-consistency loss (the free helper folded in as a
    staticmethod): supervises LV-cav VOLUME — what EF depends on, which per-pixel Dice is blind to."""

    @staticmethod
    @shapecheck
    def vol_loss(edv_pred: Float[torch.Tensor, "*k"], esv_pred: Float[torch.Tensor, "*k"], edv_gt, esv_gt,
                 delta: float = 0.1) -> Float[torch.Tensor, ""]:
        """DIMENSIONLESS volume-consistency loss — both volumes normalized by the (stable, >0) GT EDV, so
        the mL scale cancels: spacing / heart-size / dataset invariant, and ~the same magnitude as Dice+CE
        (so its weight is O(1), not a unit-coupled magic number). Huber (robust to the odd mis-scaled
        patient); `delta` is now in RELATIVE units (0.1 = a 10% volume error is the L2/L1 knee). Normalizing
        both by EDV_gt (not each by its own GT) avoids a blow-up when ESV_gt -> 0, and matches EF's own
        framing (EF = 1 - ESV/EDV). Supervises LV-cav VOLUME — what EF depends on, which per-pixel Dice is
        blind to.

        BATCHED: edv_pred/esv_pred (and edv_gt/esv_gt) may be scalars OR [K] per-subject vectors — Huber
        mean-reduces over all 2K elements, identical to per-subject loss then averaged. Lets the EF lane
        do one segment-summed forward for a whole sampled batch instead of a python loop over subjects."""
        edv_gt = torch.as_tensor(edv_gt, dtype=edv_pred.dtype, device=edv_pred.device)
        esv_gt = torch.as_tensor(esv_gt, dtype=edv_pred.dtype, device=edv_pred.device)
        pred = torch.stack([edv_pred, esv_pred]) / edv_gt              # [2] or [2,K] — dimensionless
        tgt = torch.stack([torch.ones_like(edv_gt), esv_gt / edv_gt])
        return F.huber_loss(pred, tgt, delta=delta)
