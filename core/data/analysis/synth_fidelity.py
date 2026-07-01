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

import json
from pathlib import Path

import torch

from core.data.static.labels import CLASSES

_NAMES = ["bg"] + [nm for nm, _ in CLASSES.values()]


def wasserstein1d(a: torch.Tensor, b: torch.Tensor, q: int = 100) -> float:
    """1-D Wasserstein-1 (earth-mover) distance between two samples via quantile differences:
    mean |Q_a(t) - Q_b(t)| over q evenly-spaced quantiles t. Sample-size-agnostic. Pure -> testable."""
    if a.numel() == 0 or b.numel() == 0:
        return float("nan")
    cap = 100_000                                       # torch.quantile caps input size; subsample above
    sub = lambda v: v[torch.randperm(v.numel(), device=v.device)[:cap]] if v.numel() > cap else v
    qs = torch.linspace(0, 1, q, device=a.device)
    return float((torch.quantile(sub(a).float(), qs) - torch.quantile(sub(b).float(), qs)).abs().mean())


def synth_real_distance(X: torch.Tensor, Y: torch.Tensor, cfg, n_classes: int, device: str,
                        q: int = 100) -> dict:
    """Per-class Wasserstein-1 between real and synth intensity distributions. Synth is painted from the
    SAME real masks (so regions match); compares what each class LOOKS like. Returns per-class distances
    (z-units) + the mean and worst class — the break localized."""
    from core.data.dynamic.synth import synthesize_from_labels
    Xs, _ = synthesize_from_labels(Y.to(device), cfg, n_classes, real_img=X.to(device))
    rx, sx = X[:, 0].reshape(-1).cpu(), Xs[:, 0].reshape(-1).cpu()
    ym = Y.reshape(-1).cpu()
    # W1 decomposed: LOCATION = |mean diff| (a z-shift, e.g. from bg composition / normalization),
    # SHAPE = W1 of the mean-centered distributions (genuine signal-model mismatch). Tells whether the
    # gap is fixable by matching composition (location) or needs better blood physics (shape).
    dist, loc, shape = {}, {}, {}
    for c in range(n_classes):
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


def by_vendor(meta, cfg, n_classes: int, device: str, size: int, q: int = 100,
              min_slices: int = 200, max_slices: int = 1200) -> dict:
    """Per-vendor synth-vs-real distance — does the blood-level (location) gap differ by SCANNER?
    Groups the (labelled) real cases by vendor, paints synth from each vendor's masks, measures the
    per-class W1 / location / shape for each. A vendor-dependent blood location gap says the fix must
    be machine-conditioned (per-vendor inflow), not one global scalar; a flat gap says a single term
    is enough. Thin vendors (< min_slices) are skipped (W1 noisy on tiny samples); large vendors are
    random-subsampled to max_slices (W1 is sample-size-agnostic, and it bounds GPU memory + makes
    vendors comparable). Reuses `synth_real_distance` per subset — the pure metric stays the source."""
    import polars as pl
    import torch
    from core.data.dynamic.dataset import load_to_gpu
    from core.data.static import splits
    out = {}
    for v in sorted(meta.get_column("vendor").unique().to_list()):
        sub = meta.filter(pl.col("vendor") == v)
        X, Y = load_to_gpu(splits.paths(sub), size, "cpu")
        n = int(X.shape[0])
        if n < min_slices:
            out[v] = {"skipped": f"{n} slices < {min_slices}"}
            continue
        if n > max_slices:
            g = torch.Generator().manual_seed(0)
            idx = torch.randperm(n, generator=g)[:max_slices]
            X, Y = X[idx], Y[idx]
        out[v] = {"n_slices": n, "n_used": int(X.shape[0]),
                  **synth_real_distance(X, Y, cfg, n_classes, device, q)}
    return out


def _main():
    import argparse
    from core.hparams import TrainCfg
    from core.data.static import store, splits
    from core.data.dynamic.dataset import load_to_gpu
    from core.model import resolve_device
    from core.obs import setup

    ap = argparse.ArgumentParser(description="Synth fidelity: per-class synth-vs-real distribution distance.")
    ap.add_argument("--set", nargs="*", default=[], dest="overrides", help="synth cfg overrides, e.g. synth.bg_tiers=8")
    ap.add_argument("--by-vendor", action="store_true", help="break the distance down per scanner vendor "
                    "(is the blood-level gap machine-dependent?) over all labelled cases")
    a = ap.parse_args()
    setup()
    from core.hparams import apply_overrides
    import polars as pl
    cfg = TrainCfg()
    apply_overrides(cfg, [f"generator.{o}" if o.startswith("synth.") else o for o in a.overrides])
    cfg.generator.synth.synth_p = 1.0
    device = resolve_device(None)
    d = cfg.generator.data
    meta = store.load(list(d.sources), inplane=d.inplane, n4=d.n4)
    if a.by_vendor:
        s = by_vendor(meta.filter(pl.col("labelled")), cfg.generator.synth,
                      cfg.model.out_channels, device, d.size)
    else:
        _, va, _ = splits.make_split(meta, d.test_datasets, d.test_vendors, d.val_frac, 0)
        X, Y = load_to_gpu(splits.paths(va), d.size, "cpu")
        s = synth_real_distance(X, Y, cfg.generator.synth, cfg.model.out_channels, device)
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    _main()
