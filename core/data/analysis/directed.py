"""Directed generation — fit the synth ENVELOPE to a target vendor's UNLABELED images (bd ncph/6i8g).

Param-recovery is identifiability-blocked (inverse.py: the heart has two tissue levels and uncalibrated
MRI leaves a free affine gain, so a single frame's params are degenerate). But the degeneracy is on the
PARAMS, not the APPEARANCE — many params render the same image. So we fit the observable ENVELOPE instead:
the whole-FOV radial power spectrum (blur / k-space PSF / noise set its high-frequency roll-off) to a
target vendor's real images, with the heart CONTRAST held at the physics prior (leak-free — no our-data
statistics enter the tissue model). No masks are touched: the fit target is the target domain's image
PSD, the defining input of unsupervised domain adaptation.

This turns the random domain-randomization painter into a DIRECTED generator aimed at one deployment
vendor — the test of whether the ~0.17 randomization tax (bd xmcf: synth's flat white high-frequency
plateau vs real's colored roll-off) is recoverable by matching the target's spectral envelope. The
seeded fit/hold-out partition keeps the fitted vendor slices disjoint from the xvx0 test arm.
"""
from __future__ import annotations

import argparse
import json
import logging
from itertools import product
from pathlib import Path
from typing import Any

import polars as pl
import torch
from jaxtyping import Float

from core.data.dynamic.anatomy import Anatomy
from core.data.dynamic.dataset import ACDCSliceDataset
from core.data.dynamic.synth import SynthCfg, SynthPainter
from core.data.ingest.splits.synth_main import POOL, SynthMain
from core.data.static import splits
from core.data.static.store.build import Build as store
from core.hparams import Hparams, TrainCfg
from core.model import Model

log = logging.getLogger("cardioseg.directed")

# Envelope search grid — the leak-free knobs that set the whole-FOV spectral roll-off (heart contrast
# stays at the physics prior). blur = resolution Gaussian σ; kspace = fraction of k-space kept (0 = off);
# noise = Rician std; bandlimited = add noise BEFORE the low-pass (colored, like acquisition) vs white.
_BLUR_SIGMAS: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5)
_KSPACE_KEEP: tuple[float, ...] = (0.0, 0.9, 0.75)
_NOISE_STDS: tuple[float, ...] = (0.02, 0.035, 0.05, 0.07)
_BANDLIMITED: tuple[bool, ...] = (False, True)

_PSD_BINS = 48         # radial-profile bins (rotational average of |FFT|^2)
_FIT_MASKS = 128       # pool masks painted per envelope candidate (PSD is envelope-, not shape-, driven)
_EPS = 1e-12


class Directed:
    """Fit the painter envelope to a target vendor's image PSD (bd 6i8g). Holds the base synth cfg +
    n_classes + device + paint seed (shared state); fit() takes the target PSD and a mask batch."""

    def __init__(self, cfg: SynthCfg, n_classes: int, device: str, seed: int = 0) -> None:
        self.cfg, self.n_classes, self.device, self.seed = cfg, n_classes, device, seed

    @staticmethod
    def radial_psd(imgs: Float[torch.Tensor, "n 1 h w"], n_bins: int = _PSD_BINS) -> Float[torch.Tensor, "b"]:
        """Batch-mean radial power spectrum, normalized to sum 1 over k>0 — a spectral SHAPE, not a scale
        (the per-image z-score already fixes total energy). Rotationally averaged |FFT|^2 in `n_bins`
        radial bins with the DC bin dropped. Pure -> unit-testable."""
        f = torch.fft.fftshift(torch.fft.fft2(imgs.float()), dim=(-2, -1))
        power = (f.abs() ** 2).mean(dim=(0, 1))                          # [H,W] mean over batch + channel
        h, w = power.shape
        yy, xx = torch.meshgrid(torch.arange(h, device=power.device) - h // 2,
                                torch.arange(w, device=power.device) - w // 2, indexing="ij")
        r = torch.sqrt(yy.float() ** 2 + xx.float() ** 2)
        bins = torch.clamp((r / (min(h, w) // 2) * n_bins).long(), max=n_bins - 1).reshape(-1)
        prof = torch.zeros(n_bins, device=power.device).scatter_add_(0, bins, power.reshape(-1))
        count = torch.zeros(n_bins, device=power.device).scatter_add_(0, bins, torch.ones_like(power.reshape(-1)))
        prof = (prof / count.clamp_min(1))[1:]                           # per-bin mean power, drop DC
        return prof / prof.sum().clamp_min(_EPS)

    @staticmethod
    def psd_distance(a: Float[torch.Tensor, "b"], b: Float[torch.Tensor, "b"]) -> float:
        """MSE of the log10 radial-PSD profiles — spectral-shape mismatch (the xmcf white-plateau gap).
        Log-space so the high-frequency decades (where synth's plateau lives) count, not just the low-k
        bulk. Pure -> unit-testable."""
        return float(((torch.log10(a.clamp_min(_EPS)) - torch.log10(b.clamp_min(_EPS))) ** 2).mean())

    def synth_psd(self, masks: Float[torch.Tensor, "n h w"], cfg: SynthCfg) -> Float[torch.Tensor, "b"]:
        """Radial PSD of a synth batch painted from `masks` under `cfg`. Reseed per call so every envelope
        candidate sees the same RNG stream — the PSD delta is the envelope, not paint noise."""
        torch.manual_seed(self.seed)
        img, *_ = SynthPainter.synthesize_from_labels(masks, cfg, self.n_classes)
        return Directed.radial_psd(img)

    def fit(self, target_psd: Float[torch.Tensor, "b"], masks: Float[torch.Tensor, "n h w"]) -> dict[str, Any]:
        """Grid-search the envelope knobs to minimize PSD distance to `target_psd`. Returns the best
        override set + the full ranked table (deterministic, reproducible from the seed)."""
        rows: list[dict[str, Any]] = []
        for sigma, keep, noise, bandlimited in product(_BLUR_SIGMAS, _KSPACE_KEEP, _NOISE_STDS, _BANDLIMITED):
            cfg = self.cfg.model_copy(update={"blur": (sigma, sigma), "kspace": keep,
                                              "noise": noise, "noise_bandlimited": bandlimited})
            dist = Directed.psd_distance(self.synth_psd(masks, cfg), target_psd)
            rows.append({"blur": sigma, "kspace": keep, "noise": noise,
                         "bandlimited": bandlimited, "psd_dist": round(dist, 5)})
        rows.sort(key=lambda row: row["psd_dist"])
        return {"best": rows[0], "table": rows}

    @staticmethod
    def add_args(ap: argparse.ArgumentParser) -> None:
        ap.add_argument("--vendor", required=True,
                        help="target vendor to fit the envelope to (GE/Canon/Siemens/Philips) — its label value")
        ap.add_argument("--fit-frac", type=float, default=0.5,
                        help="seeded fraction of the vendor's slices used to FIT; the rest is held out for xvx0")
        ap.add_argument("--max-slices", type=int, default=1500, help="cap on fit slices (PSD is sample-agnostic)")
        ap.add_argument("--seed", type=int, default=0, help="paint + partition seed (reproducible fit/holdout)")
        ap.add_argument("--out", default=None, help="write the fitted config + PSD table JSON here")

    @staticmethod
    def _target_psd(args: argparse.Namespace, data_cfg: Any, size: int,
                    device: str) -> tuple[Float[torch.Tensor, "b"], int, int]:
        """Load the target vendor's real images, take the seeded fit partition, return (target PSD, n_fit,
        n_holdout) — the hold-out count is what the xvx0 test arm gets (disjoint from the fit set)."""
        meta = store.load_cfg(data_cfg).filter(pl.col("labelled") & (pl.col("vendor") == args.vendor))
        real, *_ = ACDCSliceDataset.load_to_gpu([p for p in splits.Splits.paths(meta)], size, "cpu")
        n = int(real.shape[0])
        if n == 0:
            raise ValueError(f"no labelled slices for vendor {args.vendor!r} in the cloud")
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(args.seed))
        n_fit = int(n * args.fit_frac)
        fit = real[perm[:n_fit]][:args.max_slices].to(device)
        return Directed.radial_psd(fit), int(fit.shape[0]), n - n_fit

    @staticmethod
    def run(args: argparse.Namespace) -> dict[str, Any]:
        cfg = TrainCfg()
        Hparams.apply_overrides(cfg, [])
        device = Model.resolve_device(None)
        synth_cfg = cfg.generator.synth
        synth_cfg.synth_p = 1.0
        n_classes, size = cfg.model.out_channels, cfg.generator.data.size
        target_psd, n_fit, n_holdout = Directed._target_psd(args, cfg.generator.data, size, device)
        masks = torch.as_tensor(Anatomy.load_pool(SynthMain.pool(POOL)), dtype=torch.long)
        perm = torch.randperm(masks.shape[0], generator=torch.Generator().manual_seed(args.seed))
        mask_batch = masks[perm[:_FIT_MASKS]].to(device)
        engine = Directed(synth_cfg, n_classes, device, seed=args.seed)
        generic = synth_cfg.model_copy()
        generic_dist = Directed.psd_distance(engine.synth_psd(mask_batch, generic), target_psd)
        fit = engine.fit(target_psd, mask_batch)
        best = fit["best"]
        overrides = (f"generator.synth.blur='({best['blur']},{best['blur']})' "
                     f"generator.synth.kspace={best['kspace']} generator.synth.noise={best['noise']} "
                     f"generator.synth.noise_bandlimited={best['bandlimited']}")
        result = {"vendor": args.vendor, "n_fit": n_fit, "n_holdout": n_holdout, "seed": args.seed,
                  "generic_psd_dist": round(generic_dist, 5), "directed_best": best,
                  "psd_improvement": round(generic_dist - best["psd_dist"], 5),
                  "train_set_override": overrides, "table_top5": fit["table"][:5]}
        log.info(json.dumps(result, indent=2))
        if args.out:
            Path(args.out).write_text(json.dumps({**result, "table": fit["table"]}, indent=2))
            log.info(f"wrote {args.out}")
        return result
