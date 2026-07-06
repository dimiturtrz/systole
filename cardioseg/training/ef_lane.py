"""Volume-consistency training lane — per-patient EF supervision, the auxiliary objective to seg.

EDV/ESV need a patient's whole SAX stack, so this runs per-VOLUME (not per-slice, a different
granularity than the seg lane): load a subject's ED+ES stacks, forward WITH grad -> soft EDV/ESV ->
vol_loss vs the GT-derived volumes. On labeled data the target is the GT mask's volume (core.measure);
for EF-only data (Kaggle, later) it's the EF csv. Runs a sampled handful of subjects per epoch.
"""
from __future__ import annotations

import torch

from core.data.static.store import load_arrays
from core.preprocessing.preprocess import fit_square
from core.measure import label_volume_ml
from core.data.static.labels import LV_CAV
from .volumes import soft_lv_volume, vol_loss


def _stack(vol, size: int, device: str) -> torch.Tensor:
    """[D,H,W] numpy -> [D,1,size,size] on device (grid-fit, no augmentation)."""
    slices = [torch.from_numpy(fit_square(vol[z], size, 0.0)) for z in range(vol.shape[0])]
    return torch.stack(slices)[:, None].to(device)


def volume_loss_batch(model, npz_paths, size: int, device: str, delta: float = 10.0,
                      amp: bool = True) -> torch.Tensor | None:
    """Mean volume-consistency loss over the given labeled subjects (GT-derived EDV/ESV targets).
    Grad-enabled per-patient forward. Skips subjects missing ED/ES or with no cavity. None if empty."""
    total, n = None, 0
    for p in npz_paths:
        c = load_arrays(p)
        if "ed_img" not in c or "es_img" not in c:
            continue
        spacing = tuple(float(s) for s in c["spacing"])
        edv_gt = label_volume_ml(c["ed_gt"], LV_CAV, spacing)
        esv_gt = label_volume_ml(c["es_gt"], LV_CAV, spacing)
        if edv_gt <= 0:
            continue
        with torch.autocast("cuda", enabled=amp):
            ed = model(_stack(c["ed_img"], size, device)).softmax(1)
            es = model(_stack(c["es_img"], size, device)).softmax(1)
        loss = vol_loss(soft_lv_volume(ed.float(), spacing),
                        soft_lv_volume(es.float(), spacing), edv_gt, esv_gt, delta)
        total = loss if total is None else total + loss
        n += 1
    return (total / n) if n else None
