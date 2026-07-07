"""The FIT operator (bd ncph/ixea) — analysis-by-synthesis in reverse.

The forward painter (synth.py) SAMPLEs an image from parameters θ. This inverts it: given a REAL scan
and its segmentation, find the θ that best REGENERATES it, by gradient descent through a deterministic,
differentiable render (mri_physics.bssfp_signal is pure torch). Two uses:
  1. product — the recovered θ is an interpretable physics estimate of the scan (qMRI / digital twin);
  2. probe   — low recon error means the forward physics SPANS this scan; the residual says what's missing.

Identifiability — the probe's KEY finding (bd ixea, stronger than expected): the heart has only TWO
tissue levels (blood, myocardium; RV-cav==LV-cav==blood). Under uncalibrated MRI intensity you can only
compare AFTER a gain/bias normalization — and an affine map takes two levels onto two levels EXACTLY for
*any* acquisition. So the single acquisition signal (the blood/myo contrast ratio) is normalized away:
acquisition is **not identifiable at all** from one frame's heart region — not merely "TR/flip trade
off", even a single param is unrecoverable. Breaking the degeneracy needs one of: (a) MULTIPLE
acquisitions of the same anatomy (varied flip/TR — real qMRI; bd 5ev5), (b) ABSOLUTE-calibrated
intensity (which uncalibrated cardiac MRI doesn't give — the whole domain problem), or (c) ≥3 known
tissue levels (bg is unknown here). So the FIT loop is mechanically correct (differentiable, converges)
but the digital-twin needs multi-acquisition input; one-frame heart fit is degenerate by construction.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .mri_physics import bssfp_signal, tissue_params


def render_heart(seg: torch.Tensor, tr: torch.Tensor, flip_deg: torch.Tensor, n_classes: int,
                 field: float, device) -> torch.Tensor:
    """Deterministic, differentiable bSSFP paint of the HEART classes (bg=0, excluded from any loss).
    seg [B,H,W] long; tr/flip_deg [B,1]; -> signal [B,1,H,W]. Tissue T1/T2/PD at literature values
    (tissue_params, no bg tiers). Differentiable wrt tr and flip_deg."""
    t1, t2, pd = tissue_params(n_classes, 0, field, device)          # [n_classes]
    mu = bssfp_signal(t1[None], t2[None], pd[None], tr, flip_deg * math.pi / 180.0)   # [B, n_classes]
    oh = F.one_hot(seg.clamp(min=0), n_classes).permute(0, 3, 1, 2).float()           # [B,n,H,W]
    return (oh * mu[:, :, None, None]).sum(1, keepdim=True)                           # [B,1,H,W]


def _standardize(v: torch.Tensor) -> torch.Tensor:
    """Zero-mean unit-std — compares the CONTRAST pattern, not absolute gain/bias (already normalized)."""
    return (v - v.mean()) / v.std().clamp_min(1e-6)


def fit_acquisition(real_img: torch.Tensor, seg: torch.Tensor, n_classes: int, field: float = 1.5,
                    fit_params: tuple[str, ...] = ("flip",), tr0: float = 3.0, flip0: float = 50.0,
                    steps: int = 400, lr: float = 0.5, device: str = "cpu") -> dict:
    """FIT the acquisition θ (subset in `fit_params`, default flip only = identifiable) to ONE real scan
    given its seg, by Adam on MSE of the standardized heart-region render vs real. real_img [B,1,H,W]
    z-scored; seg [B,H,W]. Returns fitted tr/flip (deg), final recon loss, and the recon image."""
    real = real_img.to(device)
    seg = seg.to(device).long()
    heart = (seg > 0)[:, None]                                       # [B,1,H,W] — loss only where tissue known
    if not heart.any():
        raise ValueError("segmentation has no foreground; nothing to fit")
    tr = torch.tensor([[tr0]], device=device, requires_grad="tr" in fit_params)
    flip = torch.tensor([[flip0]], device=device, requires_grad="flip" in fit_params)
    params = [p for p, name in ((tr, "tr"), (flip, "flip")) if name in fit_params]
    if not params:
        raise ValueError(f"fit_params must include 'tr' and/or 'flip'; got {fit_params}")
    opt = torch.optim.Adam(params, lr=lr)
    th = _standardize(real[heart])                                  # target contrast pattern (const)
    loss = torch.tensor(float("nan"))
    for _ in range(steps):
        opt.zero_grad()
        r = render_heart(seg, tr, flip, n_classes, field, device)
        loss = ((_standardize(r[heart]) - th) ** 2).mean()
        loss.backward()
        opt.step()
        with torch.no_grad():                                        # physical bounds (SAR / cine)
            tr.clamp_(2.0, 6.0)
            flip.clamp_(5.0, 90.0)
    with torch.no_grad():
        recon = render_heart(seg, tr, flip, n_classes, field, device)
    return {"tr": float(tr.detach()), "flip": float(flip.detach()), "recon_loss": float(loss.detach()),
            "fit_params": fit_params, "recon": recon.detach()}


def _main():
    """Fit acquisition to one real ACDC slice (given its GT) and report recon + the identifiability check
    (fit flip-only vs fit tr+flip from two inits -> same recon, different params = under-determined)."""
    import argparse
    from pathlib import Path

    import numpy as np
    ap = argparse.ArgumentParser(description="FIT probe: recover acquisition from a real scan (bd ixea).")
    ap.add_argument("--npz", required=True, help="processed ACDC case npz (ed_img/ed_gt/...)")
    ap.add_argument("--slice", type=int, default=None, help="slice index (default: largest-fg ED slice)")
    ap.add_argument("--field", type=float, default=1.5)
    ap.add_argument("--out", default=None, help="montage PNG (real | recon)")
    a = ap.parse_args()
    d = np.load(a.npz)
    img, gt = d["ed_img"], d["ed_gt"]
    k = a.slice if a.slice is not None else int(np.argmax([(gt[i] > 0).sum() for i in range(gt.shape[0])]))
    ti = torch.as_tensor(img[k], dtype=torch.float32)[None, None]
    ti = (ti - ti.mean()) / ti.std().clamp_min(1e-6)               # z-score like the training input
    tg = torch.as_tensor(gt[k], dtype=torch.long)[None]
    n = int(gt.max()) + 1
    flip_only = fit_acquisition(ti, tg, n, field=a.field, fit_params=("flip",))
    both_a = fit_acquisition(ti, tg, n, field=a.field, fit_params=("tr", "flip"), tr0=2.5, flip0=30.0)
    both_b = fit_acquisition(ti, tg, n, field=a.field, fit_params=("tr", "flip"), tr0=5.0, flip0=70.0)
    print(f"slice {k}  n_classes {n}")
    print(f"  flip-only : flip={flip_only['flip']:.1f}  (tr fixed {3.0})  recon_loss={flip_only['recon_loss']:.4f}")
    print(f"  tr+flip #1: tr={both_a['tr']:.2f} flip={both_a['flip']:.1f}  recon_loss={both_a['recon_loss']:.4f}")
    print(f"  tr+flip #2: tr={both_b['tr']:.2f} flip={both_b['flip']:.1f}  recon_loss={both_b['recon_loss']:.4f}")
    print("  (tr+flip: similar recon, different params => under-determined from one frame — bd 5ev5)")
    if a.out or True:
        from PIL import Image
        heart = (tg[0] > 0).numpy()
        def show(t):
            v = t[0, 0].numpy().copy()
            m = v[heart]
            v = (v - m.mean()) / (m.std() + 1e-6)
            v = np.clip((v + 2) / 4, 0, 1); v[~heart] = 0
            return (v * 255).astype(np.uint8)
        montage = np.concatenate([show(ti), show(flip_only["recon"])], axis=1)
        out = a.out or (str(Path(a.npz).with_suffix("")) + "_fit.png")
        Image.fromarray(montage).save(out)
        print(f"  wrote {out}  (real | recon, heart region)")


if __name__ == "__main__":
    _main()
