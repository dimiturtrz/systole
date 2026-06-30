"""Build the segmentation-overlay hero: held-out ACDC, MRI | ground truth | prediction.

Two rows — a clean case and the worst-EF HCM case (the honest failure) — each the mid-ventricular
ED slice, chambers colored (RV blue / myo green / LV-cav red). Uses the exact shipped inference path
(validate.predict_volume + largest-CC), so the picture matches the reported numbers.

    uv run python -m cardioseg.evaluation.overlay --run runs/gen
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from core.config import FLAGSHIP_REF
from core.registry import resolve
from core.hparams import from_json
from core.model import build_unet
from core.preprocessing.preprocess import fit_square
from core.data import store, splits
from core.data.store import load_arrays
from core.inference import predict_volume
from core.postprocess import largest_cc_per_class
from core.measure import ejection_fraction

# 0 bg transparent, 1 RV blue, 2 myo green, 3 LV-cav red
_CMAP = ListedColormap([(0, 0, 0, 0), (0.20, 0.45, 0.95, 0.55),
                        (0.20, 0.80, 0.35, 0.55), (0.95, 0.25, 0.25, 0.55)])


def _mid_slice(gt_vol):
    """Slice index with the most foreground (mid-ventricular)."""
    return int(np.argmax([(s > 0).sum() for s in gt_vol]))


def _panel(ax, img, mask, title):
    ax.imshow(img, cmap="gray")
    ax.imshow(mask, cmap=_CMAP, vmin=0, vmax=3, interpolation="nearest")
    ax.set_title(title, fontsize=11)
    ax.axis("off")


def _case(model, path, size, device):
    c = load_arrays(path)
    c = {k: (c[k].item() if k == "group" and hasattr(c[k], "item") else c[k]) for k in c}
    spacing = tuple(float(s) for s in c["spacing"])
    pred_ed = largest_cc_per_class(predict_volume(model, c["ed_img"], size, device, tta=True))
    pred_es = largest_cc_per_class(predict_volume(model, c["es_img"], size, device, tta=True))
    gt_ed = np.stack([fit_square(s, size, 0) for s in c["ed_gt"]])
    img_ed = np.stack([fit_square(s, size, 0.0) for s in c["ed_img"]])
    ef_p, _, _ = ejection_fraction(pred_ed, pred_es, spacing)
    gt_es = np.stack([fit_square(s, size, 0) for s in c["es_gt"]])
    ef_g, _, _ = ejection_fraction(gt_ed, gt_es, spacing)
    z = _mid_slice(gt_ed)
    return dict(group=c.get("group"), img=img_ed[z], gt=gt_ed[z], pred=pred_ed[z],
                ef_gt=ef_g, ef_pred=ef_p, name=Path(path).stem)


def main():
    import torch
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default=FLAGSHIP_REF)
    ap.add_argument("--out", default="cardioseg/docs/media/seg_overlay.png")
    a = ap.parse_args()

    run = resolve(a.run)
    cfg = from_json(run / "config.json")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_unet(cfg.model).to(device)
    model.load_state_dict(torch.load(run / "model.pth", map_location=device))
    size = cfg.data.size

    meta = store.load(["acdc"], inplane=cfg.data.inplane, n4=cfg.data.n4)
    acdc = meta.filter((meta["dataset"] == "acdc") & meta["labelled"])
    paths = splits.paths(acdc)

    # score every case once: pick a clean low-error case + the worst-EF HCM case
    cases = [_case(model, p, size, device) for p in paths]
    for c in cases:
        c["ef_err"] = abs(c["ef_gt"] - c["ef_pred"])
    clean = min((c for c in cases if c["group"] in ("DCM", "NOR", "MINF")), key=lambda c: c["ef_err"])
    hcm = max((c for c in cases if c["group"] == "HCM"), key=lambda c: c["ef_err"])

    fig, axes = plt.subplots(2, 3, figsize=(9, 6.2))
    for row, c in enumerate((clean, hcm)):
        tag = f"{c['name']} ({c['group']})  EF gt {c['ef_gt']:.0f}% / pred {c['ef_pred']:.0f}%"
        _panel(axes[row, 0], c["img"], np.zeros_like(c["gt"]), f"MRI — {tag}")
        _panel(axes[row, 1], c["img"], c["gt"], "ground truth")
        _panel(axes[row, 2], c["img"], c["pred"], "prediction")
    fig.suptitle("Held-out ACDC — RV (blue) · LV-myo (green) · LV-cav (red)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=130)
    print("wrote", a.out, "| clean", clean["name"], clean["group"],
          "| hcm", hcm["name"], f"err {hcm['ef_err']:.1f}")


if __name__ == "__main__":
    main()
