"""Synth fidelity — how close is the generator to real, per class, and where does it break?

A single mean-Dice on a synth-trained model hides *what's wrong* with the images. This measures the
images directly: paint synth from REAL masks, then compare the per-class intensity DISTRIBUTION of
synth vs real (Wasserstein-1 / quantile distance, z-score units). A large per-class distance localizes
the break — "myo contrast is off by X", "background tier Y mismatched" — and tells us whether the
physics generator *can* mimic real with small error or is hitting an architectural ceiling (-> GAN).

Pure `wasserstein1d` is unit-tested; `synth_real_distance` needs the generator + a real (X, Y) set.
Companion to attribution.py (what the MODEL learns) — this is what the DATA looks like.
"""
from __future__ import annotations

import copy
import json
import logging

import numpy as np
import polars as pl
import torch

from core.data.dynamic.dataset import ACDCSliceDataset
from core.data.dynamic.synth import MatchedAcqCfg, SynthPainter
from core.data.static import splits
from core.data.static.labels import CLASSES
from core.data.static.store.build import Build as store
from core.hparams import Hparams, TrainCfg
from core.model import Model

log = logging.getLogger("cardioseg.synth_fidelity")

_NAMES = ["bg"] + [nm for nm, _ in CLASSES.values()]

_MIN_DPRIME_PTS = 50    # fewer sample points than this -> d'/std estimate too unstable (SD noisy)
_MIN_VENDOR_SUBJECTS = 100  # skip a vendor with fewer real subjects when building spread bands


# --- separability (d'): a DIFFERENT axis from W1. W1 asks "is synth FAITHFUL to real?"; d' asks "are
#     the classes DISTINGUISHABLE?" (the learnability bug-check — a net can't segment a boundary it can't
#     see). Signal-detection d'(A,B) = |mu_A-mu_B| / sqrt(.5(var_A+var_B)); affine-invariant, so z-score
#     safe. PER-SLICE (avg over slices) is the honest axis: it isolates WITHIN-image separability (what a
#     2D net actually sees), removing the across-slice acquisition-sweep spread that POOLING conflates in.
_PAIRS = [(2, 3), (2, 1), (2, 0), (1, 3), (1, 0), (3, 0)]     # (myo,cav)(myo,rv)(myo,bg)(rv,cav)(rv,bg)(cav,bg)

# --- variance attribution: WHERE does synth's per-class spread come from, and does it match REAL's? The
#     generator's diversity is a SUM of knobs. Split each class's spread into POOLED std (across-slice =
#     inter-sample DIVERSITY: the acquisition sweep, jitter, inflow, bias) vs mean PER-SLICE std (within-
#     image TEXTURE: texture, noise). Toggle each knob OFF and measure the drop = that knob's marginal
#     contribution. REAL's per-class std is the TARGET a physical knob must reproduce. This decides whether
#     an unphysical knob (jitter) is REDUNDANT with the physical sweep or supplies spread physics must
#     replace (-> literature T1/T2/PD sampling sized to exactly that gap). No training.
_KNOB_OFF = {"jitter": {"jitter": 0.0}, "texture": {"texture": 0.0}, "noise": {"noise": 0.0},
             "inflow": {"inflow": False}, "bias": {"bias_strength": 0.0}, "blur": {"blur": (0.0, 0.0)},
             "sweep": {"acq": MatchedAcqCfg()}}     # matched acq = single fixed field/TR/flip -> sweep off


class SynthFidelity:
    """Synth-vs-real distribution analysis: how close the generator's painted intensity matches real,
    per class. Holds the generation cfg + n_classes + device + size (shared STATE); methods take only
    the real (X,Y) set / meta that varies. (bd 01fh: cfg/n_classes/device/size were threaded as args.)"""

    def __init__(self, cfg, n_classes: int, device: str, size: int):
        self.cfg, self.n_classes, self.device, self.size = cfg, n_classes, device, size

    @staticmethod
    def wasserstein1d(a: torch.Tensor, b: torch.Tensor, q: int = 100) -> float:
        """1-D Wasserstein-1 (earth-mover) distance between two samples via quantile differences:
        mean |Q_a(t) - Q_b(t)| over q evenly-spaced quantiles t. Sample-size-agnostic. Pure -> testable."""
        if a.numel() == 0 or b.numel() == 0:
            return float("nan")
        cap = 100_000                                       # torch.quantile caps input size; subsample above
        subsample = lambda v: v[torch.randperm(v.numel(), device=v.device)[:cap]] if v.numel() > cap else v
        quantiles = torch.linspace(0, 1, q, device=a.device)
        return float((torch.quantile(subsample(a).float(), quantiles) - torch.quantile(subsample(b).float(), quantiles)).abs().mean())

    @staticmethod
    def dprime(a: torch.Tensor, b: torch.Tensor) -> float:
        """Separability of two 1-D samples. |mean diff| / pooled SD; nan if either < 50 pts (SD unstable)."""
        if a.numel() < _MIN_DPRIME_PTS or b.numel() < _MIN_DPRIME_PTS:
            return float("nan")
        pooled_std = (0.5 * (a.var() + b.var())).sqrt()
        return float((a.mean() - b.mean()).abs() / pooled_std) if float(pooled_std) > 0 else float("nan")

    @staticmethod
    def _pair_dprime(x1c: torch.Tensor, ym: torch.Tensor, n_classes: int) -> dict:
        dprimes = {}
        for i, j in _PAIRS:
            if i < n_classes and j < n_classes:
                dprimes[f"{_NAMES[i]}|{_NAMES[j]}"] = round(SynthFidelity.dprime(x1c[ym == i], x1c[ym == j]), 3)
        return dprimes

    @staticmethod
    def separability(X: torch.Tensor, Y: torch.Tensor, cfg, n_classes: int, device: str) -> dict:
        """Per-class-pair d' for REAL vs SYNTH (synth painted from the same masks). Both POOLED (all pixels)
        and PER-SLICE (mean of per-slice d' — the within-image, net's-eye axis). ratio synth/real < 1 =
        synth under-separates that boundary. Real is the achievable bar (real images ARE segmentable)."""
        Xs, _ = SynthPainter.synthesize_from_labels(Y.to(device), cfg, n_classes, real_img=X.to(device))
        Xr, Xs = X[:, 0].to(device), Xs[:, 0]
        labels = Y.to(device)
        def per_slice(x1c):
            dprime_lists = {}
            for i in range(x1c.shape[0]):
                for pair_name, value in SynthFidelity._pair_dprime(x1c[i].reshape(-1), labels[i].reshape(-1), n_classes).items():
                    dprime_lists.setdefault(pair_name, []).append(value)
            return {pair_name: round(float(np.nanmean(values)), 3) for pair_name, values in dprime_lists.items()}
        result = {}
        for level, real_dprimes, synth_dprimes in (("pooled", SynthFidelity._pair_dprime(Xr.reshape(-1), labels.reshape(-1), n_classes),
                                    SynthFidelity._pair_dprime(Xs.reshape(-1), labels.reshape(-1), n_classes)),
                                   ("per_slice", per_slice(Xr), per_slice(Xs))):
            ratio = {pair_name: round(synth_dprimes[pair_name] / real_dprimes[pair_name], 3) if real_dprimes[pair_name] == real_dprimes[pair_name] and real_dprimes[pair_name] > 0
                     else float("nan") for pair_name in real_dprimes}
            result[level] = {"real": real_dprimes, "synth": synth_dprimes, "ratio_synth_over_real": ratio}
        return result

    @staticmethod
    def _spread(x1c: torch.Tensor, ym: torch.Tensor, n_classes: int) -> dict:
        """Per-class (pooled std across all pixels, mean per-slice std) — DIVERSITY vs within-image TEXTURE."""
        flat_intensities, flat_labels = x1c.reshape(-1), ym.reshape(-1)
        pooled = {c: float(flat_intensities[flat_labels == c].std()) if int((flat_labels == c).sum()) > _MIN_DPRIME_PTS else float("nan")
                  for c in range(n_classes)}
        per_slice_stds = {c: [] for c in range(n_classes)}
        for i in range(x1c.shape[0]):
            slice_intensities, slice_labels = x1c[i].reshape(-1), ym[i].reshape(-1)
            for c in range(n_classes):
                if int((slice_labels == c).sum()) > _MIN_DPRIME_PTS:
                    per_slice_stds[c].append(float(slice_intensities[slice_labels == c].std()))
        per_slice_means = {c: round(float(np.nanmean(values)), 3) if values else float("nan") for c, values in per_slice_stds.items()}
        return {"pooled": {_NAMES[c]: round(pooled[c], 3) for c in range(n_classes)},
                "per_slice": {_NAMES[c]: per_slice_means[c] for c in range(n_classes)}}

    @staticmethod
    def real_spread_bands(meta, n_classes: int, size: int, max_per_vendor: int = 800) -> dict:
        """Per-VENDOR real per-class pooled σ, and the σ BAND (min..max across vendors) + all-pooled. The
        fair target for a DIVERSITY synth: real spread isn't one number, it's a RANGE across scanners. Synth
        σ ABOVE the band's max on a class = genuinely wider than any real vendor (candidate over-spread);
        WITHIN the band = just covering the vendor range (domain randomization working, not a defect).
        DIAGNOSTIC ONLY — never a tuning target (fitting synth to these = leaking real/test-vendor stats)."""
        per_vendor = {}
        for vendor in sorted(meta.get_column("vendor").unique().to_list()):
            vendor_meta = meta.filter(pl.col("vendor") == vendor)
            X, Y = ACDCSliceDataset.load_to_gpu(splits.Splits.paths(vendor_meta), size, "cpu")
            n = int(X.shape[0])
            if n < _MIN_VENDOR_SUBJECTS:
                continue
            if n > max_per_vendor:
                generator = torch.Generator().manual_seed(0)
                sample_indices = torch.randperm(n, generator=generator)[:max_per_vendor]
                X, Y = X[sample_indices], Y[sample_indices]
            per_vendor[vendor] = SynthFidelity._spread(X[:, 0], Y, n_classes)["pooled"]
        band = {}
        for c in range(n_classes):
            values = [per_vendor[vendor][_NAMES[c]] for vendor in per_vendor
                    if per_vendor[vendor][_NAMES[c]] == per_vendor[vendor][_NAMES[c]]]
            band[_NAMES[c]] = {"min": round(min(values), 3), "max": round(max(values), 3)} if values else {}
        return {"per_vendor": per_vendor, "band_across_vendors": band}

    def distance(self, X: torch.Tensor, Y: torch.Tensor, q: int = 100) -> dict:
        """Per-class Wasserstein-1 between real and synth intensity distributions. Synth is painted from the
        SAME real masks (so regions match); compares what each class LOOKS like. Returns per-class distances
        (z-units) + the mean and worst class — the break localized."""
        Xs, _ = SynthPainter.synthesize_from_labels(Y.to(self.device), self.cfg, self.n_classes, real_img=X.to(self.device))
        real_intensities, synth_intensities = X[:, 0].reshape(-1).cpu(), Xs[:, 0].reshape(-1).cpu()
        labels = Y.reshape(-1).cpu()
        # W1 decomposed: LOCATION = |mean diff| (a z-shift, e.g. from bg composition / normalization),
        # SHAPE = W1 of the mean-centered distributions (genuine signal-model mismatch). Tells whether the
        # gap is fixable by matching composition (location) or needs better blood physics (shape).
        distances, loc, shape = {}, {}, {}
        for c in range(self.n_classes):
            class_mask = labels == c
            real_class, synth_class = real_intensities[class_mask], synth_intensities[class_mask]
            distances[_NAMES[c]] = round(self.wasserstein1d(real_class, synth_class, q), 3)
            if real_class.numel() and synth_class.numel():
                loc[_NAMES[c]] = round(abs(float(real_class.mean()) - float(synth_class.mean())), 3)
                shape[_NAMES[c]] = round(self.wasserstein1d(real_class - real_class.mean(), synth_class - synth_class.mean(), q), 3)
            else:
                loc[_NAMES[c]] = shape[_NAMES[c]] = float("nan")
        values = [value for value in distances.values() if not np.isnan(value)]              # drop NaN (absent classes)
        worst = max(distances, key=lambda name: (distances[name] if not np.isnan(distances[name]) else -1))
        return {"per_class_w1": distances, "location": loc, "shape": shape,
                "mean_w1": round(sum(values) / len(values), 3) if values else float("nan"),
                "worst_class": worst}

    def variance(self, X: torch.Tensor, Y: torch.Tensor, field: float | None = None) -> dict:
        """Per-class spread: REAL (target) vs SYNTH baseline vs each knob toggled OFF (marginal Δ). `field`
        pins a single field strength (1.5/3.0) to stratify out the field axis; None = full sweep. Reuses the
        generator — no fitting, no training. The lens for 'is jitter's variance physical-replaceable?'."""
        X_device, Y_device = X[:, 0].to(self.device), Y.to(self.device)
        base = copy.deepcopy(self.cfg)
        if field is not None:
            base.fields = (field,)
        def synth_spread(c):
            Xs, _ = SynthPainter.synthesize_from_labels(Y_device, c, self.n_classes, real_img=X.to(self.device))
            return self._spread(Xs[:, 0], Y_device, self.n_classes)
        result = {"field": field or "sweep",
               "real_target": self._spread(X_device, Y_device, self.n_classes),
               "synth_baseline": synth_spread(base)}
        deltas = {}
        for name, overrides in _KNOB_OFF.items():
            knob_cfg = copy.deepcopy(base)
            for field_name, value in overrides.items():
                setattr(knob_cfg, field_name, value)
            off_spread = synth_spread(knob_cfg)
            deltas[name] = {level: {class_name: round(result["synth_baseline"][level][class_name] - off_spread[level][class_name], 3)
                                  for class_name in off_spread[level] if result["synth_baseline"][level][class_name] == result["synth_baseline"][level][class_name]
                                  and off_spread[level][class_name] == off_spread[level][class_name]}
                            for level in ("pooled", "per_slice")}
        result["knob_delta"] = deltas
        return result

    def by_vendor(self, meta, q: int = 100, min_slices: int = 200, max_slices: int = 1200) -> dict:
        """Per-vendor synth-vs-real distance — does the blood-level (location) gap differ by SCANNER?
        Groups the (labelled) real cases by vendor, paints synth from each vendor's masks, measures the
        per-class W1 / location / shape for each. A vendor-dependent blood location gap says the fix must
        be machine-conditioned (per-vendor inflow), not one global scalar; a flat gap says a single term
        is enough. Thin vendors (< min_slices) are skipped (W1 noisy on tiny samples); large vendors are
        random-subsampled to max_slices (W1 is sample-size-agnostic, and it bounds GPU memory + makes
        vendors comparable). Reuses `self.distance` per subset — the pure metric stays the source."""
        out = {}
        for vendor in sorted(meta.get_column("vendor").unique().to_list()):
            vendor_meta = meta.filter(pl.col("vendor") == vendor)
            X, Y = ACDCSliceDataset.load_to_gpu(splits.Splits.paths(vendor_meta), self.size, "cpu")
            n = int(X.shape[0])
            if n < min_slices:
                out[vendor] = {"skipped": f"{n} slices < {min_slices}"}
                continue
            if n > max_slices:
                generator = torch.Generator().manual_seed(0)
                sample_indices = torch.randperm(n, generator=generator)[:max_slices]
                X, Y = X[sample_indices], Y[sample_indices]
            out[vendor] = {"n_slices": n, "n_used": int(X.shape[0]),
                      **self.distance(X, Y, q)}
        return out


    @staticmethod
    def add_args(ap):
        ap.add_argument("--mode", choices=("distance", "separability", "variance"), default="distance")
        ap.add_argument("--set", nargs="*", default=[], dest="overrides", help="synth cfg overrides, e.g. synth.bg_tiers=8")
        ap.add_argument("--by-vendor", action="store_true", help="distance mode: break down per scanner vendor")
        ap.add_argument("--by-field", action="store_true", help="variance mode: stratify by field (1.5/3T) too")
        ap.add_argument("--val-only", action="store_true", help="target = acdc-val only (default: ALL labelled "
                        "real, all vendors — the fair multi-vendor spread for a diversity synth; DIAGNOSTIC, "
                        "not a tuning target)")
        ap.add_argument("--max-slices", type=int, default=2500, help="cap on pooled real slices (σ/W1 are "
                        "sample-size-agnostic; bounds VRAM)")

    @staticmethod
    def run(args):  # pragma: no cover
        cfg = TrainCfg()
        Hparams.apply_overrides(cfg, [f"generator.{o}" if o.startswith("synth.") else o for o in args.overrides])
        cfg.generator.synth.synth_p = 1.0
        device = Model.resolve_device(None)
        data_cfg = cfg.generator.data
        synth_cfg, n_classes = cfg.generator.synth, cfg.model.out_channels
        fidelity = SynthFidelity(synth_cfg, n_classes, device, data_cfg.size)
        meta = store.load_cfg(data_cfg)                          # ALL preprocessing params (nyul/norm too)
        if args.mode == "distance" and args.by_vendor:
            log.info(json.dumps(fidelity.by_vendor(meta.filter(pl.col("labelled"))), indent=2))
            return
        # real target: ALL labelled real (all vendors) by default — the multi-vendor manifold synth should
        # cover — vs a single cohort (--val-only). Compare-to-all-data is DIAGNOSTIC coverage, not tuning.
        if args.val_only:
            real_df = splits.ModelSplit(data_cfg, meta).val          # coded split's val if set, else criteria
        else:
            real_df = meta.filter(pl.col("labelled"))
        X, Y = ACDCSliceDataset.load_to_gpu(splits.Splits.paths(real_df), data_cfg.size, "cpu")
        n = int(X.shape[0])
        if n > args.max_slices:
            sample_indices = torch.randperm(n, generator=torch.Generator().manual_seed(0))[:args.max_slices]
            X, Y = X[sample_indices], Y[sample_indices]
        if args.mode == "separability":
            result = SynthFidelity.separability(X, Y, synth_cfg, n_classes, device)
        elif args.mode == "variance":
            fields = (None, 1.5, 3.0) if args.by_field else (None,)
            result = {str(field or "sweep"): fidelity.variance(X, Y, field=field) for field in fields}
            result["real_vendor_bands"] = SynthFidelity.real_spread_bands(meta.filter(pl.col("labelled")), n_classes, data_cfg.size)
        else:
            result = fidelity.distance(X, Y)
        log.info(json.dumps(result, indent=2))
