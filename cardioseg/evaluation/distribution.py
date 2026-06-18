"""Error-distribution plots for the model — the "look at the distribution, not one number"
view (G4). On the held-out val set:
  - KDE of per-class boundary distances (where Dice/HD summarize, this shows the shape)
  - EF Bland-Altman (bias + limits of agreement) — systematic vs random EF error
Also prints pooled HD95 / ASSD / Dice / EF-MAE. Writes PNGs to <run>/plots/.

    python -m cardioseg.evaluation.distribution --run runs/acdc
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
from cardioseg.data.mri.data import acdc_cases
from cardioseg.preprocessing.preprocess import preprocess_case
from cardioseg.evaluation.validate import predict_volume
from cardioseg.evaluation.measure import ejection_fraction
from cardioseg.evaluation.evaluate import surface_distances, surface_metrics, dice

CLASSES = {1: ("RV", "#5b8def"), 2: ("LV-myo", "#ffca5b"), 3: ("LV-cav", "#ef5350")}
SIZE = 256


def collect(run: Path, device: str):
    model = build_unet(spatial_dims=2, out_channels=4).to(device)
    model.load_state_dict(torch.load(run / "model.pth", map_location=device))
    model.eval()
    val = split_patients(list(acdc_cases()), 0.2, 0)[1]

    dists = {c: [] for c in CLASSES}  # pooled boundary distances per class (mm)
    dice_acc = {c: [] for c in CLASSES}
    ef_gt, ef_pred = [], []
    for pd in val:
        c = preprocess_case(pd)
        sp = tuple(float(s) for s in c["spacing"])
        masks = {}
        for tag in ("ED", "ES"):
            k = tag.lower()
            pred = predict_volume(model, c[f"{k}_img"], SIZE, device)
            gt = np.stack([fit_square(s, SIZE, 0) for s in c[f"{k}_gt"]]).astype(np.uint8)
            masks[tag] = (pred, gt)
            if tag == "ED":
                for cl in CLASSES:
                    dists[cl].append(surface_distances(pred, gt, cl, sp))
                    dice_acc[cl].append(dice(pred, gt, cl))
        ef_g, _, _ = ejection_fraction(masks["ED"][1], masks["ES"][1], sp, lv_label=3)
        ef_p, _, _ = ejection_fraction(masks["ED"][0], masks["ES"][0], sp, lv_label=3)
        ef_gt.append(ef_g)
        ef_pred.append(ef_p)
    return dists, dice_acc, np.array(ef_gt), np.array(ef_pred)


def plot_kde(dists, out: Path):
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
    ax.set_title("Per-class boundary-distance distribution (held-out)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def plot_bland_altman(ef_gt, ef_pred, out: Path):
    mean = (ef_gt + ef_pred) / 2
    diff = ef_pred - ef_gt
    bias, sd = float(np.mean(diff)), float(np.std(diff, ddof=1))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(mean, diff, c="#7a9bff", s=28)
    for y, ls, lbl in [(bias, "-", f"bias {bias:+.1f}"),
                       (bias + 1.96 * sd, "--", f"+1.96σ {bias + 1.96 * sd:+.1f}"),
                       (bias - 1.96 * sd, "--", f"−1.96σ {bias - 1.96 * sd:+.1f}")]:
        ax.axhline(y, color="#888", ls=ls)
        ax.text(ax.get_xlim()[1], y, " " + lbl, va="center", fontsize=8, color="#555")
    ax.axhline(0, color="#ccc", lw=0.8)
    ax.set_xlabel("mean EF (GT, pred)  %")
    ax.set_ylabel("pred − GT  (EF %)")
    ax.set_title("EF Bland–Altman (held-out)")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return bias, sd


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="runs/acdc")
    a = ap.parse_args()
    run = Path(a.run)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dists, dice_acc, ef_gt, ef_pred = collect(run, device)

    out = run / "plots"
    out.mkdir(parents=True, exist_ok=True)
    plot_kde(dists, out / "boundary_kde.png")
    bias, sd = plot_bland_altman(ef_gt, ef_pred, out / "ef_bland_altman.png")

    print("=== per-class surface metrics (pooled, held-out) ===")
    for cl, (name, _) in CLASSES.items():
        pooled = np.concatenate([d for d in dists[cl] if d.size])
        m = surface_metrics(pooled)
        print(f"  {name:7} Dice {np.mean(dice_acc[cl]):.3f}  ASSD {m['assd']:.2f}  HD95 {m['hd95']:.2f}  HD {m['hd']:.1f} mm")
    print(f"=== EF Bland-Altman: bias {bias:+.1f}% · 95% LoA [{bias - 1.96 * sd:+.1f}, {bias + 1.96 * sd:+.1f}] · MAE {np.mean(np.abs(ef_pred - ef_gt)):.1f}%")
    print(f"plots -> {out}")


if __name__ == "__main__":
    main()
