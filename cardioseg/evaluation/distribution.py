"""Error-distribution plots for the model — the "look at the distribution, not one number"
view (G4). On a chosen evaluation set:
  - KDE of per-class boundary distances (where Dice/HD summarize, this shows the shape)
  - EF Bland-Altman: scatter + bias + 95% limits of agreement, with a marginal KDE of the
    differences (the distribution outline) down the right margin
Also prints pooled HD95 / ASSD / Dice / EF-MAE. Writes PNGs to <run>/plots/.

    # flagship: the M&M-2 model evaluated on the held-out ACDC set
    python -m cardioseg.evaluation.distribution --run runs/mnm2_to_acdc --eval acdc
    # in-domain ACDC, on its seed-0 val split
    python -m cardioseg.evaluation.distribution --run runs/acdc --eval acdc --holdout
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import gaussian_kde

from cardioseg.training.model import build_unet
from cardioseg.training.dataset import fit_square, split_patients
from cardioseg.training.train import _registry
from cardioseg.preprocessing.preprocess import preprocess_case
from cardioseg.evaluation.validate import predict_volume
from cardioseg.evaluation.measure import ejection_fraction
from cardioseg.evaluation.evaluate import surface_distances, surface_metrics, dice

CLASSES = {1: ("RV", "#5b8def"), 2: ("LV-myo", "#ffca5b"), 3: ("LV-cav", "#ef5350")}
SIZE = 256


def collect(run: Path, device: str, cases, loader, cache_ns: str):
    model = build_unet(spatial_dims=2, out_channels=4).to(device)
    model.load_state_dict(torch.load(run / "model.pth", map_location=device))
    model.eval()

    dists = {c: [] for c in CLASSES}  # pooled boundary distances per class (mm)
    dice_acc = {c: [] for c in CLASSES}
    ef_gt, ef_pred = [], []
    for pd in cases:
        c = preprocess_case(pd, loader=loader, cache_ns=cache_ns)
        sp = tuple(float(s) for s in c["spacing"])
        masks = {}
        for tag in ("ED", "ES"):
            k = tag.lower()
            if f"{k}_img" not in c:
                continue
            pred = predict_volume(model, c[f"{k}_img"], SIZE, device)
            gt = np.stack([fit_square(s, SIZE, 0) for s in c[f"{k}_gt"]]).astype(np.uint8)
            masks[tag] = (pred, gt)
            if tag == "ED":
                for cl in CLASSES:
                    dists[cl].append(surface_distances(pred, gt, cl, sp))
                    dice_acc[cl].append(dice(pred, gt, cl))
        if "ED" in masks and "ES" in masks:
            ef_g, _, _ = ejection_fraction(masks["ED"][1], masks["ES"][1], sp, lv_label=3)
            ef_p, _, _ = ejection_fraction(masks["ED"][0], masks["ES"][0], sp, lv_label=3)
            ef_gt.append(ef_g)
            ef_pred.append(ef_p)
    return dists, dice_acc, np.array(ef_gt), np.array(ef_pred)


def plot_kde(dists, out: Path, label: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    xs = np.linspace(0, 12, 300)
    for cl, (name, color) in CLASSES.items():
        sd = np.concatenate([d for d in dists[cl] if d.size])
        if sd.size < 2:
            continue
        m = surface_metrics(sd)
        ax.plot(xs, gaussian_kde(sd)(xs), color=color, lw=2,
                label=f"{name}  ASSD {m['assd']:.1f} · HD95 {m['hd95']:.1f} mm")
    ax.set_xlabel("boundary distance (mm)")
    ax.set_ylabel("density")
    ax.set_title(f"Per-class boundary-distance distribution{label}")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def plot_bland_altman(ef_gt, ef_pred, out: Path, label: str):
    """Transposed Bland–Altman: difference on the x-axis, the error distribution drawn
    upright on top; bias + 95% LoA as vertical lines (and in the title)."""
    mean = (ef_gt + ef_pred) / 2
    diff = ef_pred - ef_gt
    bias, sd = float(np.mean(diff)), float(np.std(diff, ddof=1))
    lo, hi = bias - 1.96 * sd, bias + 1.96 * sd

    fig = plt.figure(figsize=(7, 5))
    gs = fig.add_gridspec(2, 1, height_ratios=(1, 3), hspace=0.05)
    axk = fig.add_subplot(gs[0])                 # upright distribution of the differences
    ax = fig.add_subplot(gs[1], sharex=axk)      # scatter vs mean EF

    # focus on the bulk so the distribution shape reads; mark hard failures off-scale
    x_lo = min(lo, float(np.percentile(diff, 2))) - 6
    x_hi = max(hi, float(np.percentile(diff, 98))) + 6
    ax.set_xlim(x_lo, x_hi)
    lines = [(bias, "-"), (hi, "--"), (lo, "--")]

    ax.axvspan(lo, hi, color="#7a9bff", alpha=0.08)            # 95% LoA band
    ax.scatter(diff, mean, c="#7a9bff", s=28, edgecolor="#3b5bbf", linewidth=0.4, zorder=3)
    ax.axvline(0, color="#ccc", lw=0.8, zorder=0)
    for x, ls in lines:
        ax.axvline(x, color="#555", ls=ls, lw=1, zorder=2)
    ax.set_xlabel("pred − GT  (EF %)")
    ax.set_ylabel("mean EF  %")
    n_off = int((diff < x_lo).sum() + (diff > x_hi).sum())
    if n_off:
        ax.text(0.02, 0.04, f"{n_off} off-scale (EF prediction collapsed)",
                transform=ax.transAxes, fontsize=8, color="#a33")

    # upright distribution outline on top
    xs = np.linspace(x_lo, x_hi, 200)
    if diff.size >= 2 and np.std(diff) > 0:
        k = gaussian_kde(diff)(xs)
        axk.fill_between(xs, 0, k, color="#7a9bff", alpha=0.35)
        axk.plot(xs, k, color="#3b5bbf", lw=1.5)
    for x, ls in lines:
        axk.axvline(x, color="#555", ls=ls, lw=1)
    axk.set_yticks([])
    axk.tick_params(labelbottom=False)
    axk.set_title(f"EF Bland–Altman{label}\nbias: {bias:+.1f}%   ·   95% LoA: [{lo:+.1f}, {hi:+.1f}]",
                  fontsize=11)

    fig.subplots_adjust(left=0.1, right=0.97, top=0.86, bottom=0.11)  # tight_layout breaks shared marginal
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return bias, sd


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="runs/mnm2_to_acdc", help="run dir with model.pth")
    ap.add_argument("--eval", default="acdc", choices=["acdc", "mnm2"], help="set to evaluate on")
    ap.add_argument("--holdout", action="store_true", help="use the seed-0 0.2 val split (in-domain runs)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    run = Path(a.run)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cases_fn, loader, ns = _registry()[a.eval]
    cases = list(cases_fn())
    if a.holdout:
        cases = split_patients(cases, 0.2, a.seed)[1]
    label = f" ({a.eval}{', held-out' if a.holdout else ''}, n={len(cases)})"
    dists, dice_acc, ef_gt, ef_pred = collect(run, device, cases, loader, ns)

    out = run / "plots"
    out.mkdir(parents=True, exist_ok=True)
    plot_kde(dists, out / "boundary_kde.png", label)
    bias, sd = plot_bland_altman(ef_gt, ef_pred, out / "ef_bland_altman.png", label)

    print(f"=== per-class surface metrics (pooled){label} ===")
    for cl, (name, _) in CLASSES.items():
        pooled = np.concatenate([d for d in dists[cl] if d.size])
        m = surface_metrics(pooled)
        print(f"  {name:7} Dice {np.mean(dice_acc[cl]):.3f}  ASSD {m['assd']:.2f}  HD95 {m['hd95']:.2f}  HD {m['hd']:.1f} mm")
    print(f"=== EF Bland-Altman: bias {bias:+.1f}% · 95% LoA [{lo_hi(bias, sd)}] · MAE {np.mean(np.abs(ef_pred - ef_gt)):.1f}%")
    print(f"plots -> {out}")


def lo_hi(bias, sd):
    return f"{bias - 1.96 * sd:+.1f}, {bias + 1.96 * sd:+.1f}"


if __name__ == "__main__":
    main()
