"""Differentiable chamber volumes + EF from SOFT masks (torch).

The numpy core.measure readout done on PROBABILITIES so it can be a loss: LV blood-pool volume =
Σ p[cav] × voxel-volume over a patient's slice stack (the soft Riemann sum, core.measure.
expected_volume_ml). On a one-hot (GT) mask this equals core.measure exactly — the sanity that lets
us trust it as a supervision signal. EDV/ESV need the WHOLE stack (all a patient's ED / ES slices),
so this is computed per-volume, not per-slice.
"""
from __future__ import annotations

import torch

from core.measure import voxel_volume_ml
from core.data.static.labels import LV_CAV


def soft_lv_volume(probs: torch.Tensor, spacing, lv_label: int = LV_CAV) -> torch.Tensor:
    """LV-cav volume (mL), differentiable, from soft probs [N, C, H, W] (a patient's ED or ES stack) =
    Σ p[lv_label] × voxel-volume. Spacing (z,y,x) mm is a constant scale — grad flows through `probs`."""
    return probs[:, lv_label].sum() * voxel_volume_ml(spacing)


def soft_ef(ed_probs: torch.Tensor, es_probs: torch.Tensor, spacing, lv_label: int = LV_CAV):
    """(EF %, EDV mL, ESV mL) — differentiable — from ED/ES soft-prob stacks [N,C,H,W]. EF = (EDV-ESV)/
    EDV; a ratio, so spacing cancels there but is carried for the absolute volumes."""
    edv = soft_lv_volume(ed_probs, spacing, lv_label)
    esv = soft_lv_volume(es_probs, spacing, lv_label)
    ef = (edv - esv) / edv * 100.0 if float(edv) > 0 else edv.new_tensor(float("nan"))
    return ef, edv, esv


def vol_loss(edv_pred: torch.Tensor, esv_pred: torch.Tensor, edv_gt, esv_gt,
             delta: float = 0.1) -> torch.Tensor:
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
    import torch.nn.functional as F
    edv_gt = torch.as_tensor(edv_gt, dtype=edv_pred.dtype, device=edv_pred.device)
    esv_gt = torch.as_tensor(esv_gt, dtype=edv_pred.dtype, device=edv_pred.device)
    pred = torch.stack([edv_pred, esv_pred]) / edv_gt              # [2] or [2,K] — dimensionless
    tgt = torch.stack([torch.ones_like(edv_gt), esv_gt / edv_gt])
    return F.huber_loss(pred, tgt, delta=delta)
