"""Train a 2D U-Net on ACDC end-to-end: preprocess cache -> train -> Dice + EF.

    python -m cardioseg.training.train --acdc --epochs 40
"""
import argparse
import json
from pathlib import Path


def train_acdc(epochs=40, batch=32, lr=1e-3, size=256, n_patients=0,
               val_frac=0.2, seed=0, device=None, out_dir="runs/acdc"):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from ..evaluation.losses import dice_ce_loss
    from ..evaluation.validate import validate, summarize
    from .model import build_unet
    from .dataset import build_splits

    torch.manual_seed(seed)
    np.random.seed(seed)          # augmentation uses global np.random
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} torch={torch.__version__} seed={seed}")

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

    dice_per_class, ef_rows = validate(model, val_dirs, size, device)
    metrics = summarize(dice_per_class, ef_rows)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pth")
    meta = {"epochs": epochs, "batch": batch, "lr": lr, "size": size, "seed": seed,
            "n_train": len(train_dirs), "n_val": len(val_dirs),
            "val_patients": [p.name for p in val_dirs], **metrics}
    (out / "metrics.json").write_text(json.dumps(meta, indent=2))
    print(f"\nsaved model + metrics -> {out}/")
    return model, metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--acdc", action="store_true")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--n-patients", type=int, default=0, help="0=all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/acdc")
    args = ap.parse_args()

    if args.acdc:
        train_acdc(epochs=args.epochs, batch=args.batch,
                   n_patients=args.n_patients, seed=args.seed, out_dir=args.out)
    else:
        print("pass --acdc")
