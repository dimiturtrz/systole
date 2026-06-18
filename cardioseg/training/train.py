"""Minimal training loop. `--synthetic` trains on generated data, so it runs with no real dataset.

Wire a modality loader (e.g. cardioseg/data/mri/data.py) into a real
Dataset/DataLoader once data lands.
"""
import argparse


def train_synthetic(steps=20, device="cpu"):
    import torch
    from ..data.mri.synth import make_volume
    from ..evaluation.losses import dice_ce_loss
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--steps", type=int, default=20)
    args = ap.parse_args()
    if args.synthetic:
        train_synthetic(steps=args.steps)
    else:
        print("Real-data training: implement the ACDC Dataset in "
              "cardioseg/data/mri/data.py, then wire here.")
