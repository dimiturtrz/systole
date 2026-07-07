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

import argparse
import copy
import json
import logging

import numpy as np
import polars as pl
import torch

from core.data.dynamic.dataset import load_to_gpu
from core.data.dynamic.synth import synthesize_from_labels
from core.data.static import splits, store
from core.data.static.labels import CLASSES
from core.hparams import TrainCfg, apply_overrides
from core.model import resolve_device
from core.obs import setup

log = logging.getLogger("cardioseg.synth_fidelity")

_NAMES = ["bg"] + [nm for nm, _ in CLASSES.values()]

_MIN_DPRIME_PTS = 50    # fewer sample points than this -> d'/std estimate too unstable (SD noisy)
_MIN_VENDOR_SUBJECTS = 100  # skip a vendor with fewer real subjects when building spread bands


def wasserstein1d(a: torch.Tensor, b: torch.Tensor, q: int = 100) -> float:
    """1-D Wasserstein-1 (earth-mover) distance between two samples via quantile differences:
    mean |Q_a(t) - Q_b(t)| over q evenly-spaced quantiles t. Sample-size-agnostic. Pure -> testable."""
    if a.numel() == 0 or b.numel() == 0:
        return float("nan")
    cap = 100_000                                       # torch.quantile caps input size; subsample above
    sub = lambda v: v[torch.randperm(v.numel(), device=v.device)[:cap]] if v.numel() > cap else v
    qs = torch.linspace(0, 1, q, device=a.device)
    return float((torch.quantile(sub(a).float(), qs) - torch.quantile(sub(b).float(), qs)).abs().mean())


# --- separability (d'): a DIFFERENT axis from W1. W1 asks "is synth FAITHFUL to real?"; d' asks "are
#     the classes DISTINGUISHABLE?" (the learnability bug-check — a net can't segment a boundary it can't
#     see). Signal-detection d'(A,B) = |mu_A-mu_B| / sqrt(.5(var_A+var_B)); affine-invariant, so z-score
#     safe. PER-SLICE (avg over slices) is the honest axis: it isolates WITHIN-image separability (what a
#     2D net actually sees), removing the across-slice acquisition-sweep spread that POOLING conflates in.
_PAIRS = [(2, 3), (2, 1), (2, 0), (1, 3), (1, 0), (3, 0)]     # (myo,cav)(myo,rv)(myo,bg)(rv,cav)(rv,bg)(cav,bg)


def dprime(a: torch.Tensor, b: torch.Tensor) -> float:
    """Separability of two 1-D samples. |mean diff| / pooled SD; nan if either < 50 pts (SD unstable)."""
    if a.numel() < _MIN_DPRIME_PTS or b.numel() < _MIN_DPRIME_PTS:
        return float("nan")
    denom = (0.5 * (a.var() + b.var())).sqrt()
    return float((a.mean() - b.mean()).abs() / denom) if float(denom) > 0 else float("nan")


def _pair_dprime(x1c: torch.Tensor, ym: torch.Tensor, n_classes: int) -> dict:
    out = {}
    for i, j in _PAIRS:
        if i < n_classes and j < n_classes:
            out[f"{_NAMES[i]}|{_NAMES[j]}"] = round(dprime(x1c[ym == i], x1c[ym == j]), 3)
    return out


def separability(X: torch.Tensor, Y: torch.Tensor, cfg, n_classes: int, device: str) -> dict:
    """Per-class-pair d' for REAL vs SYNTH (synth painted from the same masks). Both POOLED (all pixels)
    and PER-SLICE (mean of per-slice d' — the within-image, net's-eye axis). ratio synth/real < 1 =
    synth under-separates that boundary. Real is the achievable bar (real images ARE segmentable)."""
    Xs, _ = synthesize_from_labels(Y.to(device), cfg, n_classes, real_img=X.to(device))
    Xr, Xs = X[:, 0].to(device), Xs[:, 0]
    ym = Y.to(device)
    def per_slice(x1c):
        acc = {}
        for i in range(x1c.shape[0]):
            for k, v in _pair_dprime(x1c[i].reshape(-1), ym[i].reshape(-1), n_classes).items():
                acc.setdefault(k, []).append(v)
        return {k: round(float(np.nanmean(v)), 3) for k, v in acc.items()}
    out = {}
    for lvl, real_v, syn_v in (("pooled", _pair_dprime(Xr.reshape(-1), ym.reshape(-1), n_classes),
                                _pair_dprime(Xs.reshape(-1), ym.reshape(-1), n_classes)),
                               ("per_slice", per_slice(Xr), per_slice(Xs))):
        ratio = {k: round(syn_v[k] / real_v[k], 3) if real_v[k] == real_v[k] and real_v[k] > 0
                 else float("nan") for k in real_v}
        out[lvl] = {"real": real_v, "synth": syn_v, "ratio_synth_over_real": ratio}
    return out


# --- variance attribution: WHERE does synth's per-class spread come from, and does it match REAL's? The
#     generator's diversity is a SUM of knobs. Split each class's spread into POOLED std (across-slice =
#     inter-sample DIVERSITY: the acquisition sweep, jitter, inflow, bias) vs mean PER-SLICE std (within-
#     image TEXTURE: texture, noise). Toggle each knob OFF and measure the drop = that knob's marginal
#     contribution. REAL's per-class std is the TARGET a physical knob must reproduce. This decides whether
#     an unphysical knob (jitter) is REDUNDANT with the physical sweep or supplies spread physics must
#     replace (-> literature T1/T2/PD sampling sized to exactly that gap). No training.
_KNOB_OFF = {"jitter": {"jitter": 0.0}, "texture": {"texture": 0.0}, "noise": {"noise": 0.0},
             "inflow": {"inflow": False}, "bias": {"bias_strength": 0.0}, "blur": {"blur": (0.0, 0.0)},
             "sweep": {"acq_mode": "matched"}}      # matched = single fixed field/TR/flip -> sweep off


def _spread(x1c: torch.Tensor, ym: torch.Tensor, n_classes: int) -> dict:
    """Per-class (pooled std across all pixels, mean per-slice std) — DIVERSITY vs within-image TEXTURE."""
    xr, yr = x1c.reshape(-1), ym.reshape(-1)
    pooled = {c: float(xr[yr == c].std()) if int((yr == c).sum()) > _MIN_DPRIME_PTS else float("nan")
              for c in range(n_classes)}
    ps = {c: [] for c in range(n_classes)}
    for i in range(x1c.shape[0]):
        xi, yi = x1c[i].reshape(-1), ym[i].reshape(-1)
        for c in range(n_classes):
            if int((yi == c).sum()) > _MIN_DPRIME_PTS:
                ps[c].append(float(xi[yi == c].std()))
    perslice = {c: round(float(np.nanmean(v)), 3) if v else float("nan") for c, v in ps.items()}
    return {"pooled": {_NAMES[c]: round(pooled[c], 3) for c in range(n_classes)},
            "per_slice": {_NAMES[c]: perslice[c] for c in range(n_classes)}}


def real_spread_bands(meta, n_classes: int, size: int, max_per_vendor: int = 800) -> dict:
    """Per-VENDOR real per-class pooled σ, and the σ BAND (min..max across vendors) + all-pooled. The
    fair target for a DIVERSITY synth: real spread isn't one number, it's a RANGE across scanners. Synth
    σ ABOVE the band's max on a class = genuinely wider than any real vendor (candidate over-spread);
    WITHIN the band = just covering the vendor range (domain randomization working, not a defect).
    DIAGNOSTIC ONLY — never a tuning target (fitting synth to these = leaking real/test-vendor stats)."""
    per_vendor = {}
    for v in sorted(meta.get_column("vendor").unique().to_list()):
        sub = meta.filter(pl.col("vendor") == v)
        X, Y = load_to_gpu(splits.paths(sub), size, "cpu")
        n = int(X.shape[0])
        if n < _MIN_VENDOR_SUBJECTS:
            continue
        if n > max_per_vendor:
            g = torch.Generator().manual_seed(0)
            idx = torch.randperm(n, generator=g)[:max_per_vendor]
            X, Y = X[idx], Y[idx]
        per_vendor[v] = _spread(X[:, 0], Y, n_classes)["pooled"]
    band = {}
    for c in range(n_classes):
        vals = [per_vendor[v][_NAMES[c]] for v in per_vendor
                if per_vendor[v][_NAMES[c]] == per_vendor[v][_NAMES[c]]]
        band[_NAMES[c]] = {"min": round(min(vals), 3), "max": round(max(vals), 3)} if vals else {}
    return {"per_vendor": per_vendor, "band_across_vendors": band}


class SynthFidelity:
    """Synth-vs-real distribution analysis: how close the generator's painted intensity matches real,
    per class. Holds the generation cfg + n_classes + device + size (shared STATE); methods take only
    the real (X,Y) set / meta that varies. (bd 01fh: cfg/n_classes/device/size were threaded as args.)"""

    def __init__(self, cfg, n_classes: int, device: str, size: int):
        self.cfg, self.n_classes, self.device, self.size = cfg, n_classes, device, size

    def distance(self, X: torch.Tensor, Y: torch.Tensor, q: int = 100) -> dict:
        """Per-class Wasserstein-1 between real and synth intensity distributions. Synth is painted from the
        SAME real masks (so regions match); compares what each class LOOKS like. Returns per-class distances
        (z-units) + the mean and worst class — the break localized."""
        Xs, _ = synthesize_from_labels(Y.to(self.device), self.cfg, self.n_classes, real_img=X.to(self.device))
        rx, sx = X[:, 0].reshape(-1).cpu(), Xs[:, 0].reshape(-1).cpu()
        ym = Y.reshape(-1).cpu()
        # W1 decomposed: LOCATION = |mean diff| (a z-shift, e.g. from bg composition / normalization),
        # SHAPE = W1 of the mean-centered distributions (genuine signal-model mismatch). Tells whether the
        # gap is fixable by matching composition (location) or needs better blood physics (shape).
        dist, loc, shape = {}, {}, {}
        for c in range(self.n_classes):
            m = ym == c
            r, s = rx[m], sx[m]
            dist[_NAMES[c]] = round(wasserstein1d(r, s, q), 3)
            if r.numel() and s.numel():
                loc[_NAMES[c]] = round(abs(float(r.mean()) - float(s.mean())), 3)
                shape[_NAMES[c]] = round(wasserstein1d(r - r.mean(), s - s.mean(), q), 3)
            else:
                loc[_NAMES[c]] = shape[_NAMES[c]] = float("nan")
        vals = [v for v in dist.values() if v == v]                  # drop NaN (absent classes)
        worst = max(dist, key=lambda k: (dist[k] if dist[k] == dist[k] else -1))
        return {"per_class_w1": dist, "location": loc, "shape": shape,
                "mean_w1": round(sum(vals) / len(vals), 3) if vals else float("nan"),
                "worst_class": worst}

    def variance(self, X: torch.Tensor, Y: torch.Tensor, field: float | None = None) -> dict:
        """Per-class spread: REAL (target) vs SYNTH baseline vs each knob toggled OFF (marginal Δ). `field`
        pins a single field strength (1.5/3.0) to stratify out the field axis; None = full sweep. Reuses the
        generator — no fitting, no training. The lens for 'is jitter's variance physical-replaceable?'."""
        Xdev, Ydev = X[:, 0].to(self.device), Y.to(self.device)
        base = copy.deepcopy(self.cfg)
        if field is not None:
            base.fields = (field,)
        def synth_spread(c):
            Xs, _ = synthesize_from_labels(Ydev, c, self.n_classes, real_img=X.to(self.device))
            return _spread(Xs[:, 0], Ydev, self.n_classes)
        out = {"field": field or "sweep",
               "real_target": _spread(Xdev, Ydev, self.n_classes),
               "synth_baseline": synth_spread(base)}
        deltas = {}
        for name, ov in _KNOB_OFF.items():
            c = copy.deepcopy(base)
            for k, v in ov.items():
                setattr(c, k, v)
            off = synth_spread(c)
            deltas[name] = {lvl: {cl: round(out["synth_baseline"][lvl][cl] - off[lvl][cl], 3)
                                  for cl in off[lvl] if out["synth_baseline"][lvl][cl] == out["synth_baseline"][lvl][cl]
                                  and off[lvl][cl] == off[lvl][cl]}
                            for lvl in ("pooled", "per_slice")}
        out["knob_delta"] = deltas
        return out

    def by_vendor(self, meta, q: int = 100, min_slices: int = 200, max_slices: int = 1200) -> dict:
        """Per-vendor synth-vs-real distance — does the blood-level (location) gap differ by SCANNER?
        Groups the (labelled) real cases by vendor, paints synth from each vendor's masks, measures the
        per-class W1 / location / shape for each. A vendor-dependent blood location gap says the fix must
        be machine-conditioned (per-vendor inflow), not one global scalar; a flat gap says a single term
        is enough. Thin vendors (< min_slices) are skipped (W1 noisy on tiny samples); large vendors are
        random-subsampled to max_slices (W1 is sample-size-agnostic, and it bounds GPU memory + makes
        vendors comparable). Reuses `self.distance` per subset — the pure metric stays the source."""
        out = {}
        for v in sorted(meta.get_column("vendor").unique().to_list()):
            sub = meta.filter(pl.col("vendor") == v)
            X, Y = load_to_gpu(splits.paths(sub), self.size, "cpu")
            n = int(X.shape[0])
            if n < min_slices:
                out[v] = {"skipped": f"{n} slices < {min_slices}"}
                continue
            if n > max_slices:
                g = torch.Generator().manual_seed(0)
                idx = torch.randperm(n, generator=g)[:max_slices]
                X, Y = X[idx], Y[idx]
            out[v] = {"n_slices": n, "n_used": int(X.shape[0]),
                      **self.distance(X, Y, q)}
        return out


def _main():
    ap = argparse.ArgumentParser(description="Synth fidelity: per-class synth-vs-real intensity analysis. "
                                 "distance=W1 faithfulness | separability=d' distinguishability | "
                                 "variance=per-knob spread attribution vs real (all on VAL, test untouched).")
    ap.add_argument("--mode", choices=("distance", "separability", "variance"), default="distance")
    ap.add_argument("--set", nargs="*", default=[], dest="overrides", help="synth cfg overrides, e.g. synth.bg_tiers=8")
    ap.add_argument("--by-vendor", action="store_true", help="distance mode: break down per scanner vendor")
    ap.add_argument("--by-field", action="store_true", help="variance mode: stratify by field (1.5/3T) too")
    ap.add_argument("--val-only", action="store_true", help="target = acdc-val only (default: ALL labelled "
                    "real, all vendors — the fair multi-vendor spread for a diversity synth; DIAGNOSTIC, "
                    "not a tuning target)")
    ap.add_argument("--max-slices", type=int, default=2500, help="cap on pooled real slices (σ/W1 are "
                    "sample-size-agnostic; bounds VRAM)")
    a = ap.parse_args()
    setup()
    cfg = TrainCfg()
    apply_overrides(cfg, [f"generator.{o}" if o.startswith("synth.") else o for o in a.overrides])
    cfg.generator.synth.synth_p = 1.0
    device = resolve_device(None)
    d = cfg.generator.data
    sc, nc = cfg.generator.synth, cfg.model.out_channels
    fid = SynthFidelity(sc, nc, device, d.size)
    meta = store.load_cfg(d)                          # ALL preprocessing params (nyul/norm too)
    if a.mode == "distance" and a.by_vendor:
        log.info(json.dumps(fid.by_vendor(meta.filter(pl.col("labelled"))), indent=2))
        return
    # real target: ALL labelled real (all vendors) by default — the multi-vendor manifold synth should
    # cover — vs a single cohort (--val-only). Compare-to-all-data is DIAGNOSTIC coverage, not tuning.
    if a.val_only:
        real_df = splits.model_val(d, meta)          # coded split's val if set, else criteria
    else:
        real_df = meta.filter(pl.col("labelled"))
    X, Y = load_to_gpu(splits.paths(real_df), d.size, "cpu")
    n = int(X.shape[0])
    if n > a.max_slices:
        idx = torch.randperm(n, generator=torch.Generator().manual_seed(0))[:a.max_slices]
        X, Y = X[idx], Y[idx]
    if a.mode == "separability":
        s = separability(X, Y, sc, nc, device)
    elif a.mode == "variance":
        fields = (None, 1.5, 3.0) if a.by_field else (None,)
        s = {str(f or "sweep"): fid.variance(X, Y, field=f) for f in fields}
        s["real_vendor_bands"] = real_spread_bands(meta.filter(pl.col("labelled")), nc, d.size)
    else:
        s = fid.distance(X, Y)
    log.info(json.dumps(s, indent=2))


if __name__ == "__main__":
    _main()
