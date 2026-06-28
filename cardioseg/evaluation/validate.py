"""Validation for ACDC segmentation: predict full short-axis volumes, score
per-class Dice (pooled over slices) and per-patient ejection fraction vs ground
truth. Lives in evaluation/ (not training/) — measuring a model is evaluation.

EF uses each patient's own spacing. Note EF is a volume *ratio*, so a constant
spacing cancels; the per-patient spacing matters once absolute volumes (mL) are
reported, and is the honest thing to carry through regardless.
"""
from pathlib import Path

import numpy as np

from core.types import Volume
from cardioseg.evaluation.evaluate import CLASSES

CLASS_NAMES = {k: name for k, (name, _) in CLASSES.items()}   # single source: evaluate.CLASSES


_FLIPS = ([], [2], [3], [2, 3])  # the 4 in-plane flips TTA averages over (identity, H, W, HW)


def _stack_slices(vol_img: Volume, size: int) -> np.ndarray:
    """Square-fit every slice of a [D, H, W] volume -> [D, size, size] float array (model input grid)."""
    from core.preprocessing.preprocess import fit_square
    return np.stack([fit_square(vol_img[z].astype(np.float32), size, 0.0) for z in range(vol_img.shape[0])])


def predict_volume_members(model, vol_img: Volume, size: int, device: str):
    """Run the 4 TTA flips and keep them as a cheap K-member ensemble for uncertainty decomposition.
    Returns (pred uint8 [D,size,size], mean_softmax [D,C,size,size], members [K,D,C,size,size]) on
    `device`. The members enable the aleatoric/epistemic (BALD) split; mean is their average."""
    import torch

    model.eval()
    xs = _stack_slices(vol_img, size)
    with torch.no_grad():
        x = torch.from_numpy(xs)[:, None].to(device)          # [D, 1, size, size]
        flips = []
        for dims in _FLIPS:
            p = torch.softmax(model(torch.flip(x, dims) if dims else x), dim=1)
            flips.append(torch.flip(p, dims) if dims else p)  # un-flip back before stacking
        members = torch.stack(flips)                          # [K, D, C, size, size]
        mean = members.mean(0)                                # [D, C, size, size] mean softmax
        pred = mean.argmax(1).to(torch.uint8).cpu().numpy()
    return pred, mean, members


def predict_volume_probs(model, vol_img: Volume, size: int, device: str):
    """Mean-softmax over the 4 TTA flips — the shared inference primitive. Returns
    (pred uint8 [D,size,size], mean_softmax [D,C,size,size]). See predict_volume_members for the
    per-member stack used by the uncertainty decomposition."""
    pred, mean, _ = predict_volume_members(model, vol_img, size, device)
    return pred, mean


def predict_volume(model, vol_img: Volume, size: int, device: str, tta: bool = False) -> Volume:
    """Predict a label map [D, size, size] for one z-scored [D, H, W] volume.

    `tta=True` averages over the 4 in-plane flips (delegates to predict_volume_probs); `tta=False`
    is a single batched forward. argmax of the flip-sum == argmax of the mean, so results match."""
    import torch

    if tta:
        pred, _ = predict_volume_probs(model, vol_img, size, device)
        return pred
    model.eval()
    xs = _stack_slices(vol_img, size)
    with torch.no_grad():
        x = torch.from_numpy(xs)[:, None].to(device)          # [D, 1, size, size]
        return model(x).argmax(1).cpu().numpy().astype(np.uint8)


def validate(
    model, npz_paths: list, size: int, device: str,
    postproc: bool = True, tta: bool = True,
) -> tuple[dict[int, float], list[dict]]:
    """Return (dice_per_class, ef_rows).

    dice_per_class: {1,2,3 -> Dice pooled over all val slices}.
    ef_rows: list of dicts {patient, group, ef_gt, ef_pred, edv_gt, edv_pred}.

    `npz_paths` are consolidated-subject npz files from the data store (data/store.py) — each holds
    resampled+z-scored ed/es img+gt, spacing, group. Dataset-agnostic: labels are already canonical.
    """
    from .measure import ejection_fraction
    from .postprocess import largest_cc_per_class
    from .evaluate import surface_distances, surface_metrics
    from core.preprocessing.preprocess import fit_square
    from core.data.store import load_arrays

    inter = {c: 0.0 for c in CLASS_NAMES}
    denom = {c: 0.0 for c in CLASS_NAMES}
    surf = {c: {"hd95": [], "assd": []} for c in CLASS_NAMES}   # per-volume boundary distances (mm)
    ef_rows = []
    for npz_path in npz_paths:
        c = load_arrays(npz_path)
        c = {k: (c[k].item() if k == "group" and hasattr(c[k], "item") else c[k]) for k in c}
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
            ef_p, edv_p, _ = ejection_fraction(vols["ED"][0], vols["ES"][0], spacing)
            ef_g, edv_g, _ = ejection_fraction(vols["ED"][1], vols["ES"][1], spacing)
            ef_rows.append(dict(patient=Path(npz_path).stem, group=c.get("group"),
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
