"""Training loops. `--synthetic` runs with no real dataset; `--acdc` runs the
real end-to-end pipeline (ACDC -> preprocess cache -> 2D U-Net -> Dice + EF).

    python -m cardioseg.training.train --synthetic
    python -m cardioseg.training.train --acdc --epochs 15
"""
import argparse

import numpy as np


def train_synthetic(steps=20, device="cpu"):
    import torch
    from ..evaluation.losses import dice_ce_loss
    from ..data.mri.synth import make_volume
    from .model import build_unet

    model = build_unet().to(device)
    loss_fn = dice_ce_loss()
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    model.train()
    for step in range(steps):
        img, mask, _ = make_volume(seed=step)
        x = torch.from_numpy(img)[None, None].to(device)              # [1,1,D,H,W]
        y = torch.from_numpy(mask.astype("int64"))[None, None].to(device)
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        print(f"step {step:3d}  loss {loss.item():.4f}")
    return model


# --------------------------------------------------------------------------- #
# Real ACDC end-to-end                                                         #
# --------------------------------------------------------------------------- #
def _predict_volume(model, vol_img, size, device):
    """Predict label map [D,size,size] for one [D,H,W] z-scored volume."""
    import torch
    from .dataset import fit_square

    preds = []
    model.eval()
    with torch.no_grad():
        for z in range(vol_img.shape[0]):
            x = fit_square(vol_img[z].astype(np.float32), size, 0.0)
            x = torch.from_numpy(x)[None, None].to(device)
            logits = model(x)
            preds.append(logits.argmax(1)[0].cpu().numpy().astype(np.uint8))
    return np.stack(preds)


def evaluate_val(model, val_dirs, size, device, spacing=(10.0, 1.5, 1.5)):
    """Per-class Dice over all val slices + EF (pred vs GT) per patient."""
    from ..preprocessing.preprocess import preprocess_case
    from ..evaluation.measure import ejection_fraction
    from .dataset import fit_square

    inter = {c: 0.0 for c in (1, 2, 3)}
    denom = {c: 0.0 for c in (1, 2, 3)}
    ef_rows = []
    for pd in val_dirs:
        c = preprocess_case(pd, target_inplane=spacing[1])
        vols = {}
        for tag in ("ED", "ES"):
            if f"{tag.lower()}_img" not in c:
                continue
            pred = _predict_volume(model, c[f"{tag.lower()}_img"], size, device)
            gt = np.stack([fit_square(s, size, 0) for s in c[f"{tag.lower()}_gt"]])
            vols[tag] = (pred, gt)
            for cl in (1, 2, 3):                       # accumulate Dice numerator/denom
                p, g = pred == cl, gt == cl
                inter[cl] += 2.0 * np.logical_and(p, g).sum()
                denom[cl] += p.sum() + g.sum()
        if "ED" in vols and "ES" in vols:
            ef_p, *_ = ejection_fraction(vols["ED"][0], vols["ES"][0], spacing, lv_label=3)
            ef_g, *_ = ejection_fraction(vols["ED"][1], vols["ES"][1], spacing, lv_label=3)
            ef_rows.append((pd.name, c.get("group"), ef_g, ef_p))

    dice_per_class = {cl: (inter[cl] / denom[cl] if denom[cl] else float("nan"))
                      for cl in (1, 2, 3)}
    return dice_per_class, ef_rows


def train_acdc(epochs=15, batch=16, lr=1e-3, size=256, n_patients=0,
               val_frac=0.2, seed=0, device=None):
    import torch
    from torch.utils.data import DataLoader
    from ..evaluation.losses import dice_ce_loss
    from .model import build_unet
    from .dataset import build_splits

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} torch={torch.__version__}")

    train_ds, val_ds, train_dirs, val_dirs = build_splits(
        size=size, val_frac=val_frac, seed=seed, n_patients=n_patients)
    print(f"patients: {len(train_dirs)} train / {len(val_dirs)} val  | "
          f"slices: {len(train_ds)} train / {len(val_ds)} val")

    dl = DataLoader(train_ds, batch_size=batch, shuffle=True, drop_last=True)
    model = build_unet(spatial_dims=2, out_channels=4).to(device)
    loss_fn = dice_ce_loss()
    opt = torch.optim.Adam(model.parameters(), lr)

    for ep in range(epochs):
        model.train()
        tot = 0.0
        for x, y in dl:
            x, y = x.to(device), y[:, None].to(device)        # y -> [B,1,H,W]
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            tot += loss.item()
        print(f"epoch {ep:2d}  train_loss {tot/len(dl):.4f}")

    dice_per_class, ef_rows = evaluate_val(model, val_dirs, size, device)
    names = {1: "RV", 2: "LV-myo", 3: "LV-cav"}
    print("\n=== VAL Dice (per class, pooled over slices) ===")
    for cl in (1, 2, 3):
        print(f"  {names[cl]:7} (label {cl}): {dice_per_class[cl]:.3f}")
    print(f"  mean: {np.nanmean(list(dice_per_class.values())):.3f}")

    print("\n=== VAL EF: GT vs predicted ===")
    errs = []
    for name, grp, ef_g, ef_p in ef_rows:
        errs.append(abs(ef_g - ef_p))
        print(f"  {name:11} {str(grp):5}  GT {ef_g:5.1f}%  pred {ef_p:5.1f}%  "
              f"|d| {abs(ef_g-ef_p):4.1f}")
    if errs:
        print(f"  EF MAE = {np.mean(errs):.1f}%  (n={len(errs)})")
    return model, dice_per_class, ef_rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--acdc", action="store_true")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--n-patients", type=int, default=0, help="0=all")
    ap.add_argument("--steps", type=int, default=20)
    args = ap.parse_args()

    if args.acdc:
        train_acdc(epochs=args.epochs, batch=args.batch, n_patients=args.n_patients)
    elif args.synthetic:
        train_synthetic(steps=args.steps)
    else:
        print("pass --synthetic or --acdc")
