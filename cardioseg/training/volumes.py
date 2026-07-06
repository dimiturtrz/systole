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
