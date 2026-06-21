"""Validation for ACDC segmentation: predict full short-axis volumes, score
per-class Dice (pooled over slices) and per-patient ejection fraction vs ground
truth. Lives in evaluation/ (not training/) — measuring a model is evaluation.

EF uses each patient's own spacing. Note EF is a volume *ratio*, so a constant
spacing cancels; the per-patient spacing matters once absolute volumes (mL) are
reported, and is the honest thing to carry through regardless.
"""
from pathlib import Path

import numpy as np

from cardioseg.types import Volume

CLASS_NAMES = {1: "RV", 2: "LV-myo", 3: "LV-cav"}


def predict_volume(model, vol_img: Volume, size: int, device: str, tta: bool = False) -> Volume:
    """Predict a label map [D, size, size] for one z-scored [D, H, W] volume.

    Each slice [H, W] -> [1, 1, size, size] -> model -> argmax over the 4 class
    channels -> [size, size] label map; stacked back to [D, size, size].

    `tta` averages predictions over the 4 in-plane flips (test-time augmentation).
    """
    import torch
    from ..training.dataset import fit_square

    preds = []
    model.eval()
    with torch.no_grad():
        for z in range(vol_img.shape[0]):
            x = fit_square(vol_img[z].astype(np.float32), size, 0.0)
            x = torch.from_numpy(x)[None, None].to(device)
            logits = _tta_logits(model, x) if tta else model(x)
            preds.append(logits.argmax(1)[0].cpu().numpy().astype(np.uint8))
    return np.stack(preds)


def _tta_logits(model, x):
    """Average softmax over the 4 in-plane flips (identity, H, W, HW), un-flipping each
    output back before averaging. argmax of the sum == argmax of the mean."""
    import torch

    acc = None
    for dims in ([], [2], [3], [2, 3]):
        v = torch.flip(x, dims) if dims else x
        out = torch.softmax(model(v), dim=1)
        out = torch.flip(out, dims) if dims else out
        acc = out if acc is None else acc + out
    return acc


def validate(
    model, val_dirs: list[Path], size: int, device: str, target_inplane: float = 1.5,
    loader=None, cache_ns: str = "", postproc: bool = True, tta: bool = True, n4: bool = False,
) -> tuple[dict[int, float], list[dict]]:
    """Return (dice_per_class, ef_rows).

    dice_per_class: {1,2,3 -> Dice pooled over all val slices}.
    ef_rows: list of dicts {patient, group, ef_gt, ef_pred, edv_gt, edv_pred}.

    `loader`/`cache_ns` default to ACDC; pass mnm2.load_ed_es / "mnm2" to score an
    M&M-2 set with the same model (labels already remapped to ACDC convention).
    """
    from ..preprocessing.preprocess import preprocess_case, preprocess_many
    from ..data.mri.acdc import load_ed_es
    from .measure import ejection_fraction
    from .postprocess import largest_cc_per_class
    from .evaluate import surface_distances, surface_metrics
    from ..training.dataset import fit_square

    loader = loader or load_ed_es
    preprocess_many(val_dirs, target_inplane=target_inplane, loader=loader, cache_ns=cache_ns, n4=n4)  # parallel cache warm
    inter = {c: 0.0 for c in CLASS_NAMES}
    denom = {c: 0.0 for c in CLASS_NAMES}
    surf = {c: {"hd95": [], "assd": []} for c in CLASS_NAMES}   # per-volume boundary distances (mm)
    ef_rows = []
    for pd in val_dirs:
        c = preprocess_case(pd, target_inplane=target_inplane, loader=loader, cache_ns=cache_ns, n4=n4)
        spacing = tuple(float(s) for s in c["spacing"])      # per-patient (z,y,x)
        vols = {}
        for tag in ("ED", "ES"):
            if f"{tag.lower()}_img" not in c:
                continue
            pred = predict_volume(model, c[f"{tag.lower()}_img"], size, device, tta=tta)
            if postproc:
                pred = largest_cc_per_class(pred)
            gt = np.stack([fit_square(s, size, 0) for s in c[f"{tag.lower()}_gt"]])
            vols[tag] = (pred, gt)
            for cl in CLASS_NAMES:
                p, g = pred == cl, gt == cl
                inter[cl] += 2.0 * np.logical_and(p, g).sum()
                denom[cl] += p.sum() + g.sum()
                sd = surface_distances(pred, gt, cl, spacing)   # 3D boundary distances (mm), this volume
                if sd.size:
                    m = surface_metrics(sd)
                    surf[cl]["hd95"].append(m["hd95"]); surf[cl]["assd"].append(m["assd"])
        if "ED" in vols and "ES" in vols:
            ef_p, edv_p, _ = ejection_fraction(vols["ED"][0], vols["ES"][0], spacing, lv_label=3)
            ef_g, edv_g, _ = ejection_fraction(vols["ED"][1], vols["ES"][1], spacing, lv_label=3)
            ef_rows.append(dict(patient=pd.name, group=c.get("group"),
                                ef_gt=ef_g, ef_pred=ef_p, edv_gt=edv_g, edv_pred=edv_p))

    dice_per_class = {cl: (inter[cl] / denom[cl] if denom[cl] else float("nan"))
                      for cl in CLASS_NAMES}
    # median over volumes — robust report (HD95 already drops per-volume outliers; median across
    # cases drops the odd failed volume too, matching the "worst case decides, but report robust" line)
    surf_per_class = {cl: {"hd95": float(np.median(surf[cl]["hd95"])) if surf[cl]["hd95"] else float("nan"),
                           "assd": float(np.median(surf[cl]["assd"])) if surf[cl]["assd"] else float("nan")}
                      for cl in CLASS_NAMES}
    return dice_per_class, ef_rows, surf_per_class


def summarize(dice_per_class, ef_rows, surf_per_class=None):
    """Print the Dice table + (boundary table) + EF table, return a JSON-able metrics dict."""
    print("\n=== VAL Dice (per class, pooled over slices) ===")
    for cl, name in CLASS_NAMES.items():
        print(f"  {name:7} (label {cl}): {dice_per_class[cl]:.3f}")
    mean_dice = float(np.nanmean([dice_per_class[c] for c in CLASS_NAMES]))
    print(f"  mean: {mean_dice:.3f}")

    if surf_per_class:
        print("\n=== VAL boundary (median over volumes, mm) ===")
        for cl, name in CLASS_NAMES.items():
            print(f"  {name:7} HD95 {surf_per_class[cl]['hd95']:5.2f}  ASSD {surf_per_class[cl]['assd']:5.2f}")

    print("\n=== VAL EF: GT vs predicted ===")
    errs = []
    for r in ef_rows:
        d = abs(r["ef_gt"] - r["ef_pred"])
        errs.append(d)
        print(f"  {r['patient']:11} {str(r['group']):5}  GT {r['ef_gt']:5.1f}%  "
              f"pred {r['ef_pred']:5.1f}%  |d| {d:4.1f}")
    ef_mae = float(np.mean(errs)) if errs else float("nan")
    if errs:
        print(f"  EF MAE = {ef_mae:.1f}%  (n={len(errs)})")

    return {
        "dice": {CLASS_NAMES[c]: dice_per_class[c] for c in CLASS_NAMES},
        "dice_mean": mean_dice,
        "ef_mae": ef_mae,
        "ef_rows": ef_rows,
        "boundary": ({CLASS_NAMES[c]: surf_per_class[c] for c in CLASS_NAMES}
                     if surf_per_class else None),
    }
