"""Score any segmenter's NIfTI predictions with cardioseg's eval layer.

The 'any segmenter -> one evaluation' bridge: point it at nnU-Net's predicted masks
+ the matching ground-truth dir, and it reports the SAME Dice / HD95 / EF the
cardioseg pipeline reports for its own model. Same yardstick, swappable segmenter —
that's the whole architecture in one script.

    python -m baselines.nnunet.score \
        --pred <nnunet_output_dir> \
        --gt   <data>/nnunet/raw/Dataset029_BATTERY/labelsTs \
        --manifest <data>/nnunet/raw/Dataset029_BATTERY/ts_manifest.json

Case files are named <dataset>_<subject>_<ED|ES>.nii.gz (from convert.py), so EF pairs ED/ES.
With --manifest, also reports the battery's two axes separately (acdc centre-shift vs canon vendor).
With --out, writes baselines/nnunet/results.json (per-axis dice/hd95/EF) — the single source
cardioseg/evaluation/results.py reads, so the baseline number lives in one place (no hand-copy).
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from core.data.static.mri.base import Base
from core.evaluate import Evaluate
from core.measure import Measure

CLASSES = {1: "RV", 2: "LV-myo", 3: "LV-cav"}


def score(pred_dir: str, gt_dir: str, cases: list[str] | None = None) -> dict:
    """Score predictions vs GT. `cases` (filenames) restricts to a subset (one battery axis)."""
    pred_dir, gt_dir = Path(pred_dir), Path(gt_dir)
    hds = {c: [] for c in CLASSES}
    percase = defaultdict(lambda: {c: [] for c in CLASSES})   # pid -> class -> [per-frame dice]
    frames = defaultdict(dict)   # patient -> {tag: (pred, gt, spacing)}

    files = sorted(pred_dir.glob("*.nii.gz"))
    if cases is not None:
        keep = set(cases)
        files = [f for f in files if f.name in keep]
    for pf in files:
        gf = gt_dir / pf.name
        if not gf.exists():
            continue
        pred, sp = Base.load_nifti(pf)
        gt, _ = Base.load_nifti(gf)
        stem = pf.name[:-7]                     # strip .nii.gz
        pid = stem.rsplit("_", 1)[0] if "_" in stem else stem
        for c in CLASSES:
            percase[pid][c].append(Evaluate.dice(pred, gt, c))    # per-frame dice (macro-aggregated below)
            h = Evaluate.hd95(pred, gt, c, sp)
            if not np.isnan(h):
                hds[c].append(h)
        if "_" in stem:
            frames[pid][stem.rsplit("_", 1)[1]] = (pred, gt, sp)

    # macro Dice (matches cardioseg.evaluation): per case = mean ED+ES dice; then mean over cases.
    print("=== nnU-Net baseline, scored by cardioseg.evaluation (macro: mean over cases) ===")
    dice, hd95d, dmean = {}, {}, []          # hd95d (not hd95 — that's the imported fn used above)
    for c, name in CLASSES.items():
        per = [float(np.mean(percase[p][c])) for p in percase if percase[p][c]]
        d = float(np.mean(per)) if per else float("nan")
        dmean.append(d)
        hh = float(np.mean(hds[c])) if hds[c] else float("nan")
        dice[name] = round(d, 3)
        hd95d[name] = round(hh, 1)
        print(f"  {name:7} Dice {d:.3f}  HD95 {hh:.1f} mm")
    mean_dice = float(np.nanmean(dmean))
    print(f"  mean Dice {mean_dice:.3f}")

    diffs = []                                  # signed pred - GT (EF %), for bias + LoA
    for pid, ph in frames.items():
        if "ED" in ph and "ES" in ph:
            sp = ph["ED"][2]
            ef_p, *_ = Measure.ejection_fraction(ph["ED"][0], ph["ES"][0], sp, lv_label=3)
            ef_g, *_ = Measure.ejection_fraction(ph["ED"][1], ph["ES"][1], sp, lv_label=3)
            diffs.append(ef_p - ef_g)
    out = {"dice": {**dice, "mean": round(mean_dice, 3)}, "hd95": hd95d, "ef_mae": float("nan")}
    if diffs:
        d = np.array(diffs)
        mae, bias, sd = float(np.mean(np.abs(d))), float(d.mean()), float(d.std(ddof=1))
        lo, hi = bias - 1.96 * sd, bias + 1.96 * sd
        out.update(ef_mae=round(mae, 1), ef_bias=round(bias, 1), ef_loa=[round(lo, 1), round(hi, 1)])
        print(f"  EF MAE {mae:.1f}%  bias {bias:+.1f}%  95% LoA [{lo:+.1f}, {hi:+.1f}]  (n={len(d)})")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred", required=True, help="dir of nnU-Net predicted masks")
    ap.add_argument("--gt", required=True, help="dir of matching GT masks (e.g. labelsTs)")
    ap.add_argument("--manifest", default=None, help="ts_manifest.json -> also score per battery axis")
    ap.add_argument("--out", default=None, help="write per-axis results.json (needs --manifest), "
                    "e.g. baselines/nnunet/results.json — the single source results.py reads")
    a = ap.parse_args()
    print("\n##### OVERALL (acdc+canon) #####")
    score(a.pred, a.gt)
    per_axis = {}
    if a.manifest:
        man = json.loads(Path(a.manifest).read_text())
        axes = defaultdict(list)
        for case, info in man.items():
            axes[info["axis"]].append(f"{case}.nii.gz")
        for axis, cases in sorted(axes.items()):
            print(f"\n##### AXIS: {axis} (n={len(cases)} cases) #####")
            per_axis[axis] = score(a.pred, a.gt, cases=cases)
    if a.out:
        if not per_axis:
            ap.error("--out needs --manifest (per-axis scores)")
        out = {"_note": "nnU-Net v2 (2d, fold0, 50ep) scored by baselines/nnunet/score.py — "
               "regenerate with --out; read by cardioseg/evaluation/results.py.", **per_axis}
        Path(a.out).write_text(json.dumps(out, indent=2))
        print(f"\nwrote {a.out}: " + " · ".join(f"{ax} mean {r['dice']['mean']}" for ax, r in per_axis.items()))

    if per_axis:                                          # log the scored baseline alongside our runs
        from cardioseg.tracking import start
        trk = start("cardioseg", "nnunet-baseline", {"segmenter": "nnU-Net", "config": "2d/fold0/50ep"})
        trk.summary({ax: {"dice_mean": r["dice"]["mean"], "ef_mae": r.get("ef_mae")} for ax, r in per_axis.items()})
        trk.end()


if __name__ == "__main__":
    main()
