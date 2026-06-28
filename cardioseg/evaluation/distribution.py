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
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import gaussian_kde

from cardioseg.training.model import load_run, resolve_device
from cardioseg.training.dataset import fit_square, SIZE
from cardioseg.data import store, splits
from cardioseg.evaluation.validate import predict_volume
from cardioseg.evaluation.measure import ejection_fraction, LOA_Z
from cardioseg.evaluation.evaluate import surface_distances, surface_metrics, dice, CLASSES
from cardioseg.evaluation.postprocess import largest_cc_per_class


def collect(run: Path, device: str, meta_rows):
    """One record per subject: dice + boundary distances per class, EF gt/pred, and the
    stratification keys (vendor/pathology/field) straight from the store's meta — for the
    stratified views. Pure eval on the existing model (no retrain).

    `meta_rows` = iterable of meta dicts (e.g. polars df.iter_rows(named=True)) carrying `path`
    (the consolidated npz) + vendor/pathology/field_T columns.
    """
    model, _, _ = load_run(run, device)

    rows = []
    for r in meta_rows:
        c = store.load_arrays(r["path"])
        sp = tuple(float(s) for s in c["spacing"])
        ft = r.get("field_T")
        rec = {"patient": Path(r["path"]).stem,
               "pathology": r.get("pathology"), "vendor": r.get("vendor"),
               "scanner": r.get("scanner"), "country": r.get("country"),
               "continent": r.get("continent"), "motion_grade": r.get("motion_grade"),
               "field": f"{ft}T" if ft not in (None, "") else None}
        masks = {}
        sd_acc = {cl: [] for cl in CLASSES}
        di_acc = {cl: [] for cl in CLASSES}
        for tag in ("ED", "ES"):
            k = tag.lower()
            if f"{k}_img" not in c:
                continue
            pred = largest_cc_per_class(predict_volume(model, c[f"{k}_img"], SIZE, device, tta=True))
            gt = np.stack([fit_square(s, SIZE, 0) for s in c[f"{k}_gt"]]).astype(np.uint8)
            masks[tag] = (pred, gt)
            # pool BOTH phases — ES (small contracted cavity) is the harder phase; excluding it
            # made the boundary/Dice numbers optimistic.
            for cl in CLASSES:
                sd_acc[cl].append(surface_distances(pred, gt, cl, sp))
                di_acc[cl].append(dice(pred, gt, cl))
        if any(di_acc[cl] for cl in CLASSES):
            rec["sd"] = {cl: (np.concatenate(sd_acc[cl]) if sd_acc[cl] else np.array([])) for cl in CLASSES}
            rec["dice"] = {cl: float(np.mean(di_acc[cl])) for cl in CLASSES if di_acc[cl]}
        if "ED" in masks and "ES" in masks:
            rec["ef_gt"] = ejection_fraction(masks["ED"][1], masks["ES"][1], sp)[0]
            rec["ef_pred"] = ejection_fraction(masks["ED"][0], masks["ES"][0], sp)[0]
        rows.append(rec)
    return rows


def _pooled(rows):
    """Pooled dists/dice/ef arrays from per-case rows (for the total plots)."""
    dists = {c: [r["sd"][c] for r in rows if "sd" in r] for c in CLASSES}
    dice_acc = {c: [r["dice"][c] for r in rows if "dice" in r] for c in CLASSES}
    ef = [(r["ef_gt"], r["ef_pred"]) for r in rows if "ef_gt" in r]
    ef_gt = np.array([g for g, _ in ef])
    ef_pred = np.array([p for _, p in ef])
    return dists, dice_acc, ef_gt, ef_pred


def plot_kde(dists, out: Path, label: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    # data-driven x-range so the outlier tail (the point of the plot) isn't clipped at 12 mm
    pooled = [d for cl in CLASSES for d in dists[cl] if d.size]
    allsd = np.concatenate(pooled) if pooled else np.array([0.0])
    xmax = max(12.0, float(np.percentile(allsd, 99.5)))
    xs = np.linspace(0, xmax, 400)
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
    lo, hi = bias - LOA_Z * sd, bias + LOA_Z * sd

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


SMALL_N = 10   # groups below this are flagged — per-group stats are noisy


def _groups(rows, key):
    """{group_value -> rows with EF} for a stratify key, dropping rows missing the key/EF."""
    g = defaultdict(list)
    for r in rows:
        v = r.get(key)
        if v is not None and "ef_gt" in r:
            g[str(v)].append(r)
    return dict(sorted(g.items(), key=lambda kv: -len(kv[1])))   # largest group first


def strata_table(rows, key) -> dict:
    """Per-group mean Dice (per class + mean), EF MAE + bias, n. Printed + returned for JSON."""
    g = _groups(rows, key)
    if not g:
        return {}
    print(f"\n=== stratified by {key} ===")
    # GT-EF mean given alongside MAE: EF is a ratio, so absolute MAE isn't comparable across groups
    # with different cavity sizes (small-cavity HCM amplifies a fixed volume error). Dice is the
    # range-independent segmentation-quality metric; read MAE *with* the GT-EF context.
    print(f"  {'group':12} {'n':>4}  {'RV':>5} {'myo':>5} {'LVc':>5} {'mean':>5}  {'gtEF':>5} {'EF MAE':>7} {'bias':>6}")
    out = {}
    for grp, rs in g.items():
        d = {cl: np.mean([r["dice"][cl] for r in rs if "dice" in r]) for cl in CLASSES}
        diff = np.array([r["ef_pred"] - r["ef_gt"] for r in rs])
        gt_ef = float(np.mean([r["ef_gt"] for r in rs]))
        mae, bias = float(np.mean(np.abs(diff))), float(np.mean(diff))
        mean_d = float(np.mean(list(d.values())))
        flag = "  (small n)" if len(rs) < SMALL_N else ""
        print(f"  {grp:12} {len(rs):>4}  {d[1]:.3f} {d[2]:.3f} {d[3]:.3f} {mean_d:.3f}  "
              f"{gt_ef:>4.0f}% {mae:>6.1f}% {bias:>+5.1f}%{flag}")
        out[grp] = {"n": len(rs), "dice": {CLASSES[c][0]: float(d[c]) for c in CLASSES},
                    "dice_mean": mean_d, "gt_ef_mean": gt_ef, "ef_mae": mae, "ef_bias": bias}
    return out


def plot_strata(rows, key, out: Path, label: str):
    """Two panels: per-group mean Dice (bars) + per-group EF MAE (bars), n annotated,
    small-n groups hatched. The 'where does it fail' figure."""
    g = _groups(rows, key)
    if len(g) < 2:
        return
    groups = list(g)
    mean_dice = [np.mean([r["dice"][c] for r in g[gr] for c in CLASSES if "dice" in r]) for gr in groups]
    ef_mae = [float(np.mean([abs(r["ef_pred"] - r["ef_gt"]) for r in g[gr]])) for gr in groups]
    ns = [len(g[gr]) for gr in groups]
    small = [n < SMALL_N for n in ns]

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(max(5, 1.1 * len(groups)), 6))
    x = np.arange(len(groups))
    for ax, vals, ylab, col, top in ((a1, mean_dice, "mean Dice", "#5b8def", 1.0),
                                     (a2, ef_mae, "EF MAE (%)", "#ef5350", max(ef_mae))):
        bars = ax.bar(x, vals, color=col, edgecolor="#333", linewidth=0.5)
        for b, sm in zip(bars, small):
            if sm:
                b.set_hatch("///"); b.set_alpha(0.55)
        ax.set_ylim(0, top * 1.22)   # headroom so the value/n labels clear the frame + title
        for xi, v, n in zip(x, vals, ns):
            ax.text(xi, v, f"{v:.2f}\nn={n}", ha="center", va="bottom", fontsize=8)
        ax.set_ylabel(ylab)
        ax.set_xticks(x); ax.set_xticklabels(groups, rotation=20, ha="right", fontsize=9)
    a1.set_title(f"By {key}{label}   (hatched = n<{SMALL_N}, noisy)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="runs/seg", help="run dir with model.pth")
    ap.add_argument("--eval", default="acdc", choices=["acdc", "mnm2", "mnms1", "cmrxmotion", "canon"],
                    help="eval set: a dataset, or 'canon' (mnms1 vendor==Canon) — a criteria filter")
    ap.add_argument("--holdout", action="store_true", help="use the seed-0 0.2 val split (in-domain runs)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    run = Path(a.run)
    device = resolve_device()

    # eval set = a criteria filter over the consolidated store (no named splits)
    if a.eval == "canon":
        df = store.load(["mnms1"]).filter((pl.col("vendor") == "Canon") & pl.col("labelled"))
    else:
        df = store.load([a.eval]).filter(pl.col("labelled"))
        if a.holdout:
            _, df = splits.patient_val(df, 0.2, a.seed)
    label = f" ({a.eval}{', held-out' if a.holdout else ''}, n={len(df)})"
    rows = collect(run, device, df.iter_rows(named=True))
    dists, dice_acc, ef_gt, ef_pred = _pooled(rows)

    out = run / "plots"
    out.mkdir(parents=True, exist_ok=True)
    # --- total (pooled) ---
    plot_kde(dists, out / "boundary_kde.png", label)
    bias, sd = plot_bland_altman(ef_gt, ef_pred, out / "ef_bland_altman.png", label)

    print(f"=== per-class surface metrics (pooled){label} ===")
    for cl, (name, _) in CLASSES.items():
        pooled = np.concatenate([d for d in dists[cl] if d.size])
        m = surface_metrics(pooled)
        print(f"  {name:7} Dice {np.mean(dice_acc[cl]):.3f}  ASSD {m['assd']:.2f}  HD95 {m['hd95']:.2f}  HD {m['hd']:.1f} mm")
    print(f"=== EF Bland-Altman: bias {bias:+.1f}% · 95% LoA [{lo_hi(bias, sd)}] · MAE {np.mean(np.abs(ef_pred - ef_gt)):.1f}%")

    # --- stratified (only axes with >1 group present) ---
    strata = {}
    for key in ("vendor", "scanner", "pathology", "field", "country", "continent", "motion_grade"):
        if len({str(r.get(key)) for r in rows if r.get(key) is not None}) > 1:
            strata[key] = strata_table(rows, key)
            plot_strata(rows, key, out / f"strata_{key}.png", label)
    import json
    (out / "stratified.json").write_text(json.dumps(strata, indent=2))
    print(f"\nplots + stratified.json -> {out}")


def lo_hi(bias, sd):
    return f"{bias - LOA_Z * sd:+.1f}, {bias + LOA_Z * sd:+.1f}"


if __name__ == "__main__":
    main()
