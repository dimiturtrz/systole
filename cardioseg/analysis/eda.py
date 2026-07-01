"""ACDC reality-check + first data viz.

Run BEFORE trusting any label convention. Settles the open question from
research/.../application-curriculum-and-gaps.md (sec 4): the numeric label
encoding was INFERRED. This decides it on real bytes and disambiguates LV vs RV
*geometrically* (LV cavity is the blood pool enclosed by the myocardium ring)
rather than trusting either source — which caught a flipped-label EF bug.

Usage:
    export CARDIAC_DATA_ROOT=<data>/raw/mri/acdc
    python -m core.data.static.mri.eda                 # summarize N patients + viz
    python -m core.data.static.mri.eda --patient patient001
"""
import argparse
from pathlib import Path

import numpy as np

from core.data.static.mri.acdc import (
    DATA_ROOT, acdc_cases, identify_lv_cavity, load_ed_es,
)

OUT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "_eda_out"


def summarize_patient(patient_dir):
    d = load_ed_es(patient_dir)
    sp = d["spacing"]
    print(f"\n=== {patient_dir.name} | group={d.get('group','?')} ===")
    for tag in ("ED", "ES"):
        if tag not in d:
            continue
        img, gt = d[tag]["img"], d[tag]["gt"]
        anis = max(sp) / min(sp)
        lv, scores = identify_lv_cavity(gt)
        print(f"  {tag}: shape={img.shape} spacing(z,y,x)="
              f"{tuple(round(float(s),2) for s in sp)} mm  anisotropy={anis:.1f}x")
        print(f"      labels={np.unique(gt).tolist()}  img range=[{img.min():.0f},{img.max():.0f}]")
        print(f"      myo-enclosure score={ {k: round(v,2) for k,v in scores.items()} }"
              f"  -> LV cavity = label {lv}")
    return d


def save_viz(patient_dir, d, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [t for t in ("ED", "ES") if t in d]
    if not rows:
        return
    fig, axes = plt.subplots(len(rows), 3, figsize=(9, 3 * len(rows)), squeeze=False)
    for ri, tag in enumerate(rows):
        img, gt = d[tag]["img"], d[tag]["gt"]
        D = img.shape[0]
        for ci, (name, z) in enumerate(
            (("base", int(D * 0.25)), ("mid", D // 2), ("apex", int(D * 0.75)))
        ):
            ax = axes[ri][ci]
            ax.imshow(img[z], cmap="gray")
            ax.imshow(np.ma.masked_where(gt[z] == 0, gt[z]), cmap="jet",
                      alpha=0.4, vmin=0, vmax=3)
            ax.set_title(f"{tag} {name} z={z}")
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_png, dpi=90)
    print(f"  saved {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", default=None, help="e.g. patient001")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--root", default=None)
    args = ap.parse_args()

    cases = acdc_cases(args.root)
    print(f"DATA_ROOT = {args.root or DATA_ROOT}  ({len(cases)} patients)")
    if not cases:
        print("NO patient*/ dirs found — check the layout / CARDIAC_DATA_ROOT.")
        return
    if args.patient:
        cases = [c for c in cases if c.name == args.patient] or cases[:1]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for pd in cases[: args.n]:
        d = summarize_patient(pd)
        save_viz(pd, d, OUT_DIR / f"{pd.name}_overlay.png")


if __name__ == "__main__":
    main()
