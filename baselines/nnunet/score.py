"""Score any segmenter's NIfTI predictions with cardioseg's eval layer.

The 'any segmenter -> one evaluation' bridge: point it at nnU-Net's predicted masks
+ the matching ground-truth dir, and it reports the SAME Dice / HD95 / EF the
cardioseg pipeline reports for its own model. Same yardstick, swappable segmenter —
that's the whole architecture in one script.

    python -m baselines.nnunet.score \
        --pred <nnunet_output_dir> \
        --gt   D:/data/volumetric/mri/nnunet/raw/Dataset029_BATTERY/labelsTs \
        --manifest D:/data/volumetric/mri/nnunet/raw/Dataset029_BATTERY/ts_manifest.json

Case files are named <dataset>_<subject>_<ED|ES>.nii.gz (from convert.py), so EF pairs ED/ES.
With --manifest, also reports the battery's two axes separately (acdc centre-shift vs canon vendor).
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from cardioseg.data.mri.base import load_nifti
from cardioseg.evaluation.evaluate import hd95
from cardioseg.evaluation.measure import ejection_fraction

CLASSES = {1: "RV", 2: "LV-myo", 3: "LV-cav"}


def score(pred_dir: str, gt_dir: str, cases: list[str] | None = None) -> dict:
    """Score predictions vs GT. `cases` (filenames) restricts to a subset (one battery axis)."""
    pred_dir, gt_dir = Path(pred_dir), Path(gt_dir)
    inter = {c: 0.0 for c in CLASSES}
    denom = {c: 0.0 for c in CLASSES}
    hds = {c: [] for c in CLASSES}
    frames = defaultdict(dict)   # patient -> {tag: (pred, gt, spacing)}

    files = sorted(pred_dir.glob("*.nii.gz"))
    if cases is not None:
        keep = set(cases)
        files = [f for f in files if f.name in keep]
    for pf in files:
        gf = gt_dir / pf.name
        if not gf.exists():
            continue
        pred, sp = load_nifti(pf)
        gt, _ = load_nifti(gf)
        for c in CLASSES:
            p, g = pred == c, gt == c
            inter[c] += 2.0 * np.logical_and(p, g).sum()
            denom[c] += p.sum() + g.sum()
            h = hd95(pred, gt, c, sp)
            if not np.isnan(h):
                hds[c].append(h)
        stem = pf.name[:-7]                     # strip .nii.gz
        if "_" in stem:
            pid, tag = stem.rsplit("_", 1)
            frames[pid][tag] = (pred, gt, sp)

    print("=== nnU-Net baseline, scored by cardioseg.evaluation ===")
    dmean = []
    for c, name in CLASSES.items():
        d = inter[c] / denom[c] if denom[c] else float("nan")
        dmean.append(d)
        hh = float(np.mean(hds[c])) if hds[c] else float("nan")
        print(f"  {name:7} Dice {d:.3f}  HD95 {hh:.1f} mm")
    mean_dice = float(np.nanmean(dmean))
    print(f"  mean Dice {mean_dice:.3f}")

    diffs = []                                  # signed pred - GT (EF %), for bias + LoA
    for pid, ph in frames.items():
        if "ED" in ph and "ES" in ph:
            sp = ph["ED"][2]
            ef_p, *_ = ejection_fraction(ph["ED"][0], ph["ES"][0], sp, lv_label=3)
            ef_g, *_ = ejection_fraction(ph["ED"][1], ph["ES"][1], sp, lv_label=3)
            diffs.append(ef_p - ef_g)
    out = {"mean_dice": mean_dice, "ef_mae": float("nan")}
    if diffs:
        d = np.array(diffs)
        mae, bias, sd = float(np.mean(np.abs(d))), float(d.mean()), float(d.std(ddof=1))
        lo, hi = bias - 1.96 * sd, bias + 1.96 * sd
        out.update(ef_mae=mae, ef_bias=bias, ef_loa=[lo, hi])
        print(f"  EF MAE {mae:.1f}%  bias {bias:+.1f}%  95% LoA [{lo:+.1f}, {hi:+.1f}]  (n={len(d)})")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred", required=True, help="dir of nnU-Net predicted masks")
    ap.add_argument("--gt", required=True, help="dir of matching GT masks (e.g. labelsTs)")
    ap.add_argument("--manifest", default=None, help="ts_manifest.json -> also score per battery axis")
    a = ap.parse_args()
    print("\n##### OVERALL (acdc+canon) #####")
    score(a.pred, a.gt)
    if a.manifest:
        man = json.loads(Path(a.manifest).read_text())
        axes = defaultdict(list)
        for case, info in man.items():
            axes[info["axis"]].append(f"{case}.nii.gz")
        for axis, cases in sorted(axes.items()):
            print(f"\n##### AXIS: {axis} (n={len(cases)} cases) #####")
            score(a.pred, a.gt, cases=cases)


if __name__ == "__main__":
    main()
