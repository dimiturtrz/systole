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
import logging
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt

from core.data.static.mri.acdc import DATA_ROOT, AcdcAdapter
from core.data.static.mri.base import Base, Phase

log = logging.getLogger("cardioseg.eda")

OUT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "_eda_out"


class Eda:
    """ACDC reality-check + first data viz — the free helpers folded in as staticmethods:
    per-patient summary (shape/spacing/labels + geometric LV-cavity disambiguation) and the
    ED/ES base/mid/apex overlay figure."""

    @staticmethod
    def summarize_patient(patient_dir):
        case_data = AcdcAdapter().load_ed_es(patient_dir)
        spacing = case_data["spacing"]
        log.info(f"\n=== {patient_dir.name} | group={case_data.get('group','?')} ===")
        for tag in Phase:
            if tag not in case_data:
                continue
            img, gt = case_data[tag]["img"], case_data[tag]["gt"]
            anisotropy = max(spacing) / min(spacing)
            lv, scores = Base.identify_lv_cavity(gt)
            log.info(f"  {tag}: shape={img.shape} spacing(z,y,x)="
                  f"{tuple(round(float(spacing_value),2) for spacing_value in spacing)} mm  anisotropy={anisotropy:.1f}x")
            log.info(f"      labels={np.unique(gt).tolist()}  img range=[{img.min():.0f},{img.max():.0f}]")
            log.info(f"      myo-enclosure score={ {label: round(score,2) for label,score in scores.items()} }"
                  f"  -> LV cavity = label {lv}")
        return case_data

    @staticmethod
    def save_viz(patient_dir, d, out_png):
        rows = [tag for tag in Phase if tag in d]
        if not rows:
            return
        fig, axes = plt.subplots(len(rows), 3, figsize=(9, 3 * len(rows)), squeeze=False)
        for ri, tag in enumerate(rows):
            img, gt = d[tag]["img"], d[tag]["gt"]
            depth = img.shape[0]
            for ci, (name, z) in enumerate(
                (("base", int(depth * 0.25)), ("mid", depth // 2), ("apex", int(depth * 0.75)))
            ):
                ax = axes[ri][ci]
                ax.imshow(img[z], cmap="gray")
                ax.imshow(np.ma.masked_where(gt[z] == 0, gt[z]), cmap="jet",
                          alpha=0.4, vmin=0, vmax=3)
                ax.set_title(f"{tag} {name} z={z}")
                ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_png, dpi=90)
        log.info(f"  saved {out_png}")


    @staticmethod
    def add_args(ap):
        ap.add_argument("--patient", default=None, help="e.g. patient001")
        ap.add_argument("--n", type=int, default=3)
        ap.add_argument("--root", default=None)

    @staticmethod
    def run(args):  # pragma: no cover
        cases = AcdcAdapter(root=args.root).cases()
        log.info(f"DATA_ROOT = {args.root or DATA_ROOT}  ({len(cases)} patients)")
        if not cases:
            log.warning("NO patient*/ dirs found — check the layout / CARDIAC_DATA_ROOT.")
            return
        if args.patient:
            cases = [case for case in cases if case.name == args.patient] or cases[:1]

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        for patient_dir in cases[: args.n]:
            case_data = Eda.summarize_patient(patient_dir)
            Eda.save_viz(patient_dir, case_data, OUT_DIR / f"{patient_dir.name}_overlay.png")
