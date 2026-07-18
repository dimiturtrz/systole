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

`fit_tissue` (bd 5ev5) extends the fit from acquisition to per-class tissue CONTRAST (T2 + relative PD)
over a same-session {ED, ES} set, and exposes the degeneracy as a measured IDENTIFIABILITY MARGIN via the
`absolute` switch: calibrated (absolute levels) recovers params (synthetic margin ~0.23); uncalibrated
(one joint affine removed, what real cardiac forces) collapses the margin to ~0 — the fit returns the
literature prior. So the tissue-level twin is under-determined on standard cine; it needs calibrated or
varied-flip/TR (qMRI) input, which ACDC/M&M don't provide.
"""
from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float, Integer
from PIL import Image

from .mri_physics import MriPhysics

log = logging.getLogger("cardioseg.inverse")


class Inverse:
    """The FIT operator as a namespace: differentiable heart render + acquisition fit (bd ncph/ixea)."""

    @staticmethod
    def render_heart(  # noqa: PLR0913  physics params (independent scalars)
        seg: Integer[torch.Tensor, "*b *grid"], tr: Float[torch.Tensor, "*b 1"],
        flip_deg: Float[torch.Tensor, "*b 1"], n_classes: int,
        field: float, device: str,
    ) -> Float[torch.Tensor, "*b 1 *h *w"]:
        """Deterministic, differentiable bSSFP paint of the HEART classes (bg=0, excluded from any loss).
        seg [B,H,W] long; tr/flip_deg [B,1]; -> signal [B,1,H,W]. Tissue T1/T2/PD at literature values
        (tissue_params, no bg tiers). Differentiable wrt tr and flip_deg."""
        t1, t2, pd = MriPhysics.tissue_params(n_classes, 0, field, device)          # [n_classes]
        mu = MriPhysics.bssfp_signal(t1[None], t2[None], pd[None], tr, flip_deg * math.pi / 180.0)   # [B, n_classes]
        oh = F.one_hot(seg.clamp(min=0), n_classes).permute(0, 3, 1, 2).float()           # [B,n,H,W]
        return (oh * mu[:, :, None, None]).sum(1, keepdim=True)                           # [B,1,H,W]

    @staticmethod
    def _standardize(v: Float[torch.Tensor, "..."]) -> Float[torch.Tensor, "..."]:
        """Zero-mean unit-std — compares the CONTRAST pattern, not absolute gain/bias (already normalized)."""
        return (v - v.mean()) / v.std().clamp_min(1e-6)

    @staticmethod
    def render_heart_params(  # noqa: PLR0913  physics params (independent tensors)
        seg: Integer[torch.Tensor, "*b *grid"], t1: Float[torch.Tensor, "n"], t2: Float[torch.Tensor, "n"],
        pd: Float[torch.Tensor, "n"], tr: Float[torch.Tensor, "1"], flip_deg: Float[torch.Tensor, "1"],
    ) -> Float[torch.Tensor, "*b 1 *h *w"]:
        """Differentiable bSSFP paint from EXPLICIT per-class tissue params (vs render_heart's fixed table) —
        the tissue params are the fit variables. seg [B,H,W]; t1/t2/pd [n_classes]; -> [B,1,H,W]."""
        mu = MriPhysics.bssfp_signal(t1[None], t2[None], pd[None], tr, flip_deg * math.pi / 180.0)   # [1,n]
        oh = F.one_hot(seg.clamp(min=0), t1.shape[0]).permute(0, 3, 1, 2).float()                    # [B,n,H,W]
        return (oh * mu[:, :, None, None]).sum(1, keepdim=True)

    @staticmethod
    def _recon_set(segs: Integer[torch.Tensor, "k *h *w"], t1: Float[torch.Tensor, "n"], t2: Float[torch.Tensor, "n"], pd: Float[torch.Tensor, "n"], tr: Float[torch.Tensor, "1"], flip: Float[torch.Tensor, "1"]) -> Float[torch.Tensor, "k 1 *h *w"]:  # noqa: PLR0913
        """Per-frame differentiable render of a same-session set with shared tissue+acquisition params."""
        return torch.cat([Inverse.render_heart_params(segs[k:k + 1], t1, t2, pd, tr, flip)
                          for k in range(segs.shape[0])])

    @staticmethod
    def fit_tissue(imgs: Float[torch.Tensor, "k 1 *h *w"], segs: Integer[torch.Tensor, "k *h *w"],  # noqa: PLR0913
                   n_classes: int, field: float = 1.5, *, absolute: bool = False, prior_w: float = 1.0,
                   steps: int = 600, lr: float = 0.02, device: str = "cpu") -> dict[str, object]:
        """FIT per-class tissue CONTRAST (T2 + relative PD) + shared flip to a same-session frame SET (bd
        5ev5). `absolute` is the identifiability switch: True = calibrated levels (loss on raw signal, the
        5ev5 shared-absolute-scale regime where the 2 heart levels are 2 constraints → params recoverable);
        False = uncalibrated (loss on the CONTRAST after removing one joint affine — what real cardiac MRI
        forces — where 2 levels + a 2-dof affine leave 0 contrast dof → params degenerate, the prior wins).
        Literature TISSUE_RANGE priors (weight `prior_w`) regularize. Returns fitted params, per-frame recon,
        and the IDENTIFIABILITY MARGIN (max |log-deviation| off prior — ~0 on uncalibrated real = the twin is
        under-determined without calibration/qMRI, confirming ixea)."""
        imgs, segs = imgs.to(device), segs.to(device).long()
        heart = (segs > 0)[:, None]                                      # [K,1,H,W]
        if not heart.any():
            raise ValueError("segmentation set has no foreground; nothing to fit")
        t1p, t2p, pdp = MriPhysics.tissue_params(n_classes, 0, field, device)   # prior points [n]
        tr = torch.tensor([3.0], device=device)
        target = imgs[heart] if absolute else Inverse._standardize(imgs[heart])   # joint (set-wide) scale
        log_t2 = torch.zeros(n_classes, device=device, requires_grad=True)      # T2 = prior * exp(log_t2)
        log_pd = torch.zeros(n_classes, device=device, requires_grad=True)      # PD = prior * exp(log_pd)
        flip = torch.tensor([50.0], device=device, requires_grad=True)
        opt = torch.optim.Adam([log_t2, log_pd, flip], lr=lr)
        loss = torch.tensor(float("nan"))
        for _ in range(steps):
            opt.zero_grad()
            recon = Inverse._recon_set(segs, t1p, t2p * torch.exp(log_t2), pdp * torch.exp(log_pd), tr, flip)
            pred = recon[heart] if absolute else Inverse._standardize(recon[heart])
            data = ((pred - target) ** 2).mean()
            prior = prior_w * (log_t2 ** 2 + log_pd ** 2).mean()         # pull to literature (log-space, 0=prior)
            loss = data + prior
            loss.backward()
            opt.step()
            with torch.no_grad():
                flip.clamp_(5.0, 90.0)
        with torch.no_grad():
            t2, pd = t2p * torch.exp(log_t2), pdp * torch.exp(log_pd)
            recon = Inverse._recon_set(segs, t1p, t2, pd, tr, flip)
            pred = recon[heart] if absolute else Inverse._standardize(recon[heart])
            data_final = float(((pred - target) ** 2).mean())
            margin = float((log_t2.abs() + log_pd.abs()).max())         # max |log-deviation| off prior
        return {"t1": t1p.tolist(), "t2": t2.detach().tolist(), "pd": pd.detach().tolist(), "flip": float(flip),
                "recon_loss": data_final, "margin": margin, "recon": recon.detach(), "k": imgs.shape[0]}

    @staticmethod
    def fit_acquisition(  # noqa: PLR0913  physics params (independent scalars)
        real_img: Float[torch.Tensor, "*b 1 *h *w"], seg: Integer[torch.Tensor, "*b *grid"],
        n_classes: int, field: float = 1.5,
        fit_params: tuple[str, ...] = ("flip",), tr0: float = 3.0, flip0: float = 50.0,
        steps: int = 400, lr: float = 0.5, device: str = "cpu",
    ) -> dict[str, object]:
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
        th = Inverse._standardize(real[heart])                          # target contrast pattern (const)
        loss = torch.tensor(float("nan"))
        for _ in range(steps):
            opt.zero_grad()
            r = Inverse.render_heart(seg, tr, flip, n_classes, field, device)
            loss = ((Inverse._standardize(r[heart]) - th) ** 2).mean()
            loss.backward()
            opt.step()
            with torch.no_grad():                                        # physical bounds (SAR / cine)
                tr.clamp_(2.0, 6.0)
                flip.clamp_(5.0, 90.0)
        with torch.no_grad():
            recon = Inverse.render_heart(seg, tr, flip, n_classes, field, device)
        return {"tr": float(tr.detach()), "flip": float(flip.detach()), "recon_loss": float(loss.detach()),
                "fit_params": fit_params, "recon": recon.detach()}

    @staticmethod
    def add_args(ap: argparse.ArgumentParser) -> None:
        ap.add_argument("--npz", required=True, help="processed ACDC case npz (ed_img/ed_gt/...)")
        ap.add_argument("--mode", choices=("acquisition", "tissue"), default="acquisition",
                        help="acquisition=fit TR/flip (ixea); tissue=fit per-class T2/PD on the ED+ES set (5ev5)")
        ap.add_argument("--slice", type=int, default=None, help="slice index (default: largest-fg ED slice)")
        ap.add_argument("--field", type=float, default=1.5)
        ap.add_argument("--out", default=None, help="montage PNG (real | recon)")

    @staticmethod
    def _largest_fg(gt: np.ndarray) -> int:
        return int(np.argmax([(gt[i] > 0).sum() for i in range(gt.shape[0])]))

    @staticmethod
    def run_tissue(args: argparse.Namespace) -> None:
        """5ev5: fit per-class tissue contrast on the same-session {ED, ES} set (shared receiver gain, joint-
        normalized). Reports the IDENTIFIABILITY MARGIN under real (uncalibrated) vs the calibrated regime —
        on uncalibrated cardiac the margin collapses to ~0 (2 heart levels + a joint affine → the prior wins)."""
        d = np.load(args.npz)
        n = int(max(d["ed_gt"].max(), d["es_gt"].max())) + 1
        frames = [(d["ed_img"], d["ed_gt"]), (d["es_img"], d["es_gt"])]
        sl = [(im[k], gt[k]) for im, gt in frames for k in (Inverse._largest_fg(gt),)]
        imgs = torch.stack([torch.as_tensor(im, dtype=torch.float32) for im, _ in sl])[:, None]
        segs = torch.stack([torch.as_tensor(gt, dtype=torch.long) for _, gt in sl])
        real = Inverse.fit_tissue(imgs, segs, n, field=args.field, absolute=False)
        calib = Inverse.fit_tissue(imgs, segs, n, field=args.field, absolute=True)
        log.info(f"tissue FIT on {{ED,ES}} (n_classes {n}, K={real['k']})")
        log.info(f"  real (uncalibrated, joint affine): recon_loss={real['recon_loss']:.5f}  "
                 f"MARGIN={real['margin']:.3f}  t2={[round(x, 1) for x in real['t2']]}")
        log.info(f"  calibrated (absolute levels)      : recon_loss={calib['recon_loss']:.5f}  "
                 f"MARGIN={calib['margin']:.3f}  t2={[round(x, 1) for x in calib['t2']]}")
        log.info("  margin~0 under real = tissue params NOT identifiable from uncalibrated cardiac (bd ixea/5ev5)")

    @staticmethod
    def run(args: argparse.Namespace) -> None:
        if args.mode == "tissue":
            Inverse.run_tissue(args)
            return
        d = np.load(args.npz)
        img, gt = d["ed_img"], d["ed_gt"]
        k = args.slice if args.slice is not None else int(np.argmax([(gt[i] > 0).sum() for i in range(gt.shape[0])]))
        ti = torch.as_tensor(img[k], dtype=torch.float32)[None, None]
        ti = (ti - ti.mean()) / ti.std().clamp_min(1e-6)               # z-score like the training input
        tg = torch.as_tensor(gt[k], dtype=torch.long)[None]
        n = int(gt.max()) + 1
        flip_only = Inverse.fit_acquisition(ti, tg, n, field=args.field, fit_params=("flip",))
        both_a = Inverse.fit_acquisition(ti, tg, n, field=args.field, fit_params=("tr", "flip"), tr0=2.5, flip0=30.0)
        both_b = Inverse.fit_acquisition(ti, tg, n, field=args.field, fit_params=("tr", "flip"), tr0=5.0, flip0=70.0)
        log.info(f"slice {k}  n_classes {n}")
        log.info(f"  flip-only : flip={flip_only['flip']:.1f}  (tr fixed {3.0})  "
                 f"recon_loss={flip_only['recon_loss']:.4f}")
        log.info(f"  tr+flip #1: tr={both_a['tr']:.2f} flip={both_a['flip']:.1f}  "
                 f"recon_loss={both_a['recon_loss']:.4f}")
        log.info(f"  tr+flip #2: tr={both_b['tr']:.2f} flip={both_b['flip']:.1f}  "
                 f"recon_loss={both_b['recon_loss']:.4f}")
        log.info("  (tr+flip: similar recon, different params => under-determined from one frame — bd 5ev5)")
        heart = (tg[0] > 0).numpy()
        def show(t: torch.Tensor) -> np.ndarray:
            v = t[0, 0].numpy().copy()
            m = v[heart]
            v = (v - m.mean()) / (m.std() + 1e-6)
            v = np.clip((v + 2) / 4, 0, 1); v[~heart] = 0
            return (v * 255).astype(np.uint8)
        montage = np.concatenate([show(ti), show(flip_only["recon"])], axis=1)
        out = args.out or (str(Path(args.npz).with_suffix("")) + "_fit.png")
        Image.fromarray(montage).save(out)
        log.info(f"  wrote {out}  (real | recon, heart region)")
