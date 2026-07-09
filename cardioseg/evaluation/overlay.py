"""Build the segmentation-overlay hero: held-out ACDC, MRI | ground truth | prediction.

Two rows — a clean case and the worst-EF HCM case (the honest failure) — each the mid-ventricular
ED slice, chambers colored (RV blue / myo green / LV-cav red). Uses the exact shipped inference path
(validate.predict_volume + largest-CC), so the picture matches the reported numbers.

    uv run python -m cardioseg.evaluation.overlay --run runs/gen
"""
import argparse
import logging
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from core.config import FLAGSHIP_REF
from core.data.static import splits
from core.data.static.store import build as store
from core.data.static.store import load_arrays
from core.hparams import from_json
from core.inference import Inference
from core.measure import Measure
from core.model import build_unet
from core.obs import setup
from core.postprocess import Postprocess
from core.preprocessing.preprocess import stack_slices
from core.registry import resolve

log = logging.getLogger("cardioseg.overlay")


class Overlay:
    """Overlay hero-figure cores: the case-assembly + row-selection policy, decoupled from the model
    forward + savefig shell (which lives in module-level `main`). Pure selection is testable; the
    matplotlib panel render is a staticmethod carrying its own `# pragma: no cover`."""

    # 0 bg transparent, 1 RV blue, 2 myo green, 3 LV-cav red
    _CMAP = ListedColormap([(0, 0, 0, 0), (0.20, 0.45, 0.95, 0.55),
                            (0.20, 0.80, 0.35, 0.55), (0.95, 0.25, 0.25, 0.55)])

    CLEAN_GROUPS = ("DCM", "NOR", "MINF")
    HCM_GROUP = "HCM"

    @staticmethod
    def _mid_slice(gt_vol):
        """Slice index with the most foreground (mid-ventricular)."""
        return int(np.argmax([(s > 0).sum() for s in gt_vol]))

    @staticmethod
    def pick_hero_cases(cases: list[dict]) -> tuple[dict, dict]:
        """Choose the two overlay rows from scored cases (each dict has group + ef_gt/ef_pred): the
        lowest-EF-error clean case (DCM/NOR/MINF) and the WORST-EF HCM case (the honest failure). Each case
        gains an `ef_err` key. Pure selection — no model, no plot; the picture-choosing policy, testable."""
        for c in cases:
            c["ef_err"] = abs(c["ef_gt"] - c["ef_pred"])
        clean = min((c for c in cases if c["group"] in Overlay.CLEAN_GROUPS), key=lambda c: c["ef_err"])
        hcm = max((c for c in cases if c["group"] == Overlay.HCM_GROUP), key=lambda c: c["ef_err"])
        return clean, hcm

    @staticmethod
    def _panel(ax, img, mask, title):  # pragma: no cover  (matplotlib imshow render)
        ax.imshow(img, cmap="gray")
        ax.imshow(mask, cmap=Overlay._CMAP, vmin=0, vmax=3, interpolation="nearest")
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    @staticmethod
    def _case(model, path, size, device):
        case = load_arrays(path)
        spacing = tuple(float(s) for s in case["spacing"])
        pred_ed = Postprocess.largest_cc_per_class(Inference.predict_volume(model, case["ed_img"], size, device, tta=True))
        pred_es = Postprocess.largest_cc_per_class(Inference.predict_volume(model, case["es_img"], size, device, tta=True))
        gt_ed = stack_slices(case["ed_gt"], size)
        img_ed = stack_slices(case["ed_img"], size, 0.0)
        ef_p, _, _ = Measure.ejection_fraction(pred_ed, pred_es, spacing)
        gt_es = stack_slices(case["es_gt"], size)
        ef_g, _, _ = Measure.ejection_fraction(gt_ed, gt_es, spacing)
        z = Overlay._mid_slice(gt_ed)
        return dict(group=case.get("group"), img=img_ed[z], gt=gt_ed[z], pred=pred_ed[z],
                    ef_gt=ef_g, ef_pred=ef_p, name=Path(path).stem)


def main():  # pragma: no cover  (loads the model + GPU inference over ACDC + matplotlib savefig)
    setup()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default=FLAGSHIP_REF)
    ap.add_argument("--out", default="cardioseg/docs/media/seg_overlay.png")
    args = ap.parse_args()

    run = resolve(args.run)
    cfg = from_json(run / "config.json")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_unet(cfg.model).to(device)
    model.load_state_dict(torch.load(run / "model.pth", map_location=device))
    d = cfg.generator.data
    size = d.size

    meta = store.load_cfg(d)                    # the run's own preprocessing params
    val = splits.model_val(d, meta)             # the held-out VAL split (acdc in xvendor) — split-derived, not a literal
    paths = splits.paths(val)

    # score every case once: pick a clean low-error case + the worst-EF HCM case
    cases = [Overlay._case(model, p, size, device) for p in paths]
    clean, hcm = Overlay.pick_hero_cases(cases)

    fig, axes = plt.subplots(2, 3, figsize=(9, 6.2))
    for row, c in enumerate((clean, hcm)):
        tag = f"{c['name']} ({c['group']})  EF gt {c['ef_gt']:.0f}% / pred {c['ef_pred']:.0f}%"
        Overlay._panel(axes[row, 0], c["img"], np.zeros_like(c["gt"]), f"MRI — {tag}")
        Overlay._panel(axes[row, 1], c["img"], c["gt"], "ground truth")
        Overlay._panel(axes[row, 2], c["img"], c["pred"], "prediction")
    fig.suptitle("Held-out ACDC — RV (blue) · LV-myo (green) · LV-cav (red)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    log.info(f"wrote {args.out} | clean {clean['name']} {clean['group']} "
             f"| hcm {hcm['name']} err {hcm['ef_err']:.1f}")


if __name__ == "__main__":
    main()
