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

from core.labels import CLASSES

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
    from cardioseg.data.synth import synthesize_from_labels
    Xs, _ = synthesize_from_labels(Y.to(device), cfg, n_classes, real_img=X.to(device))
    rx, sx = X[:, 0].reshape(-1).cpu(), Xs[:, 0].reshape(-1).cpu()
    ym = Y.reshape(-1).cpu()
    dist = {}
    for c in range(n_classes):
        m = ym == c
        dist[_NAMES[c]] = round(wasserstein1d(rx[m], sx[m], q), 3)
    vals = [v for v in dist.values() if v == v]                  # drop NaN (absent classes)
    worst = max(dist, key=lambda k: (dist[k] if dist[k] == dist[k] else -1))
    return {"per_class_w1": dist, "mean_w1": round(sum(vals) / len(vals), 3) if vals else float("nan"),
            "worst_class": worst}


def _main():
    import argparse
    from core.hparams import TrainCfg
    from core.data import store, splits
    from cardioseg.data.dataset import load_to_gpu
    from core.model import resolve_device
    from core.obs import setup

    ap = argparse.ArgumentParser(description="Synth fidelity: per-class synth-vs-real distribution distance.")
    ap.add_argument("--set", nargs="*", default=[], dest="overrides", help="synth cfg overrides, e.g. synth.bg_tiers=8")
    a = ap.parse_args()
    setup()
    from core.hparams import apply_overrides
    cfg = TrainCfg()
    apply_overrides(cfg, [f"generator.{o}" if o.startswith("synth.") else o for o in a.overrides])
    cfg.generator.synth.synth_p = 1.0
    device = resolve_device(None)
    d = cfg.generator.data
    meta = store.load(list(d.sources), inplane=d.inplane, n4=d.n4)
    _, va, _ = splits.make_split(meta, d.test_datasets, d.test_vendors, d.val_frac, 0)
    X, Y = load_to_gpu(splits.paths(va), d.size, "cpu")
    s = synth_real_distance(X, Y, cfg.generator.synth, cfg.model.out_channels, device)
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    _main()
