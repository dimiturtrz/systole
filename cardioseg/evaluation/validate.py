"""Validation for ACDC segmentation: predict full short-axis volumes, score
per-class Dice (pooled over slices) and per-patient ejection fraction vs ground
truth. Lives in evaluation/ (not training/) — measuring a model is evaluation.

EF uses each patient's own spacing. Note EF is a volume *ratio*, so a constant
spacing cancels; the per-patient spacing matters once absolute volumes (mL) are
reported, and is the honest thing to carry through regardless.
"""
import numpy as np

CLASS_NAMES = {1: "RV", 2: "LV-myo", 3: "LV-cav"}


def predict_volume(model, vol_img, size, device):
    """Predict a label map [D,size,size] for one z-scored [D,H,W] volume."""
    import torch
    from ..training.dataset import fit_square

    preds = []
    model.eval()
    with torch.no_grad():
        for z in range(vol_img.shape[0]):
            x = fit_square(vol_img[z].astype(np.float32), size, 0.0)
            x = torch.from_numpy(x)[None, None].to(device)
            preds.append(model(x).argmax(1)[0].cpu().numpy().astype(np.uint8))
    return np.stack(preds)


def validate(model, val_dirs, size, device, target_inplane=1.5):
    """Return (dice_per_class, ef_rows).

    dice_per_class: {1,2,3 -> Dice pooled over all val slices}.
    ef_rows: list of dicts {patient, group, ef_gt, ef_pred, edv_gt, edv_pred}.
    """
    from ..preprocessing.preprocess import preprocess_case
    from .measure import ejection_fraction
    from ..training.dataset import fit_square

    inter = {c: 0.0 for c in CLASS_NAMES}
    denom = {c: 0.0 for c in CLASS_NAMES}
    ef_rows = []
    for pd in val_dirs:
        c = preprocess_case(pd, target_inplane=target_inplane)
        spacing = tuple(float(s) for s in c["spacing"])      # per-patient (z,y,x)
        vols = {}
        for tag in ("ED", "ES"):
            if f"{tag.lower()}_img" not in c:
                continue
            pred = predict_volume(model, c[f"{tag.lower()}_img"], size, device)
            gt = np.stack([fit_square(s, size, 0) for s in c[f"{tag.lower()}_gt"]])
            vols[tag] = (pred, gt)
            for cl in CLASS_NAMES:
                p, g = pred == cl, gt == cl
                inter[cl] += 2.0 * np.logical_and(p, g).sum()
                denom[cl] += p.sum() + g.sum()
        if "ED" in vols and "ES" in vols:
            ef_p, edv_p, _ = ejection_fraction(vols["ED"][0], vols["ES"][0], spacing, lv_label=3)
            ef_g, edv_g, _ = ejection_fraction(vols["ED"][1], vols["ES"][1], spacing, lv_label=3)
            ef_rows.append(dict(patient=pd.name, group=c.get("group"),
                                ef_gt=ef_g, ef_pred=ef_p, edv_gt=edv_g, edv_pred=edv_p))

    dice_per_class = {cl: (inter[cl] / denom[cl] if denom[cl] else float("nan"))
                      for cl in CLASS_NAMES}
    return dice_per_class, ef_rows


def summarize(dice_per_class, ef_rows):
    """Print the Dice table + EF table, return a JSON-able metrics dict."""
    print("\n=== VAL Dice (per class, pooled over slices) ===")
    for cl, name in CLASS_NAMES.items():
        print(f"  {name:7} (label {cl}): {dice_per_class[cl]:.3f}")
    mean_dice = float(np.nanmean([dice_per_class[c] for c in CLASS_NAMES]))
    print(f"  mean: {mean_dice:.3f}")

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
    }
