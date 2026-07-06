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


def volume_loss_batch(model, npz_paths, size: int, device: str, delta: float = 0.1,
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


def _zscore(s):
    import numpy as np
    s = s.astype(np.float32)
    return (s - s.mean()) / (s.std() + 1e-6)


def kaggle_ef_loss(model, cases, ef_targets: dict, size: int, device: str, amp: bool = True,
                   delta: float = 0.1):
    """EF-RATIO consistency on Kaggle (EF-only, NO masks). EF is spacing-invariant, so it sidesteps
    Kaggle's ambiguous slice-spacing: predict LV-cav over the cine, per-phase volume (any unit) ->
    EDV=max-phase, ESV=min-phase -> EF -> Huber vs the csv EF. Efficiency: a NO-GRAD pass over all
    (location,phase) slices finds the ED/ES phases, then only those 2 phases are forwarded WITH grad
    (grad cost ~ a labeled patient, not 16x). Weak supervision from 1140 EF-labeled cases. valid={}
    for these (no dense mask) so the seg loss never touches them."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    from core.data.static.mri.kaggle_dsb import load_sax
    total, n = None, 0
    for c in cases:
        t = ef_targets.get(c.name)
        if not t or not t.get("ef"):
            continue
        sax = load_sax(c)
        if not sax:
            continue
        P = min(v.shape[0] for v, _, _ in sax)                       # common phase count
        X = torch.from_numpy(np.array([[fit_square(_zscore(vol[p]), size, 0.0) for p in range(P)]
                                       for vol, _, _ in sax])).to(device)          # [L, P, H, W]
        L = X.shape[0]
        with torch.no_grad(), torch.autocast("cuda", enabled=amp):
            pv = model(X.reshape(L * P, 1, size, size)).softmax(1)[:, LV_CAV]       # [L*P, H, W]
            phase_vol = pv.float().sum((1, 2)).view(L, P).sum(0)                    # [P] cavity vol / phase
        ed_p, es_p = int(phase_vol.argmax()), int(phase_vol.argmin())
        with torch.autocast("cuda", enabled=amp):                                  # grad on ED/ES phases only
            ed = model(X[:, ed_p].unsqueeze(1)).softmax(1)[:, LV_CAV].float().sum()
            es = model(X[:, es_p].unsqueeze(1)).softmax(1)[:, LV_CAV].float().sum()
        ef_pred = (ed - es) / ed * 100.0 if float(ed) > 0 else ed.new_tensor(0.0)
        loss = F.huber_loss(ef_pred / 100, ef_pred.new_tensor(t["ef"] / 100), delta=delta)
        total = loss if total is None else total + loss
        n += 1
    return (total / n) if n else None
