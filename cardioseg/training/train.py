"""Train a 2D U-Net on ACDC or M&M-2, with an optional cross-dataset test.

    python -m cardioseg.training.train --acdc --epochs 40
    python -m cardioseg.training.train --dataset mnm2 --test acdc --epochs 40

The `--test` set is held out entirely (never seen in train/val) and scored with the
same model — the honest generalization number. Train on M&M-2 (multi-vendor) / test
on ACDC (single-centre) is the strong direction; the reverse measures OOD drop.
"""
import argparse
import json
import time
from pathlib import Path

# dataset -> (cases_fn, loader, cache_ns). Loaders are dataset-agnostic; M&M-2 labels
# are remapped to the ACDC convention on load, so one model spans both.


def _val_dice(model, val_dl, device) -> float:
    """Fast batched mean foreground Dice (pooled over val slices, no TTA) — the early-stop signal."""
    import torch
    import numpy as np

    inter = {c: 0.0 for c in (1, 2, 3)}
    denom = {c: 0.0 for c in (1, 2, 3)}
    model.eval()
    with torch.no_grad():
        for x, y in val_dl:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            for c in (1, 2, 3):
                p, g = pred == c, y == c
                inter[c] += 2.0 * (p & g).sum().item()
                denom[c] += (p.sum() + g.sum()).item()
    return float(np.mean([inter[c] / denom[c] if denom[c] else 0.0 for c in (1, 2, 3)]))


def train_seg(dataset="acdc", epochs=128, batch=32, lr=1e-3, size=256, n_patients=0,
              val_frac=0.2, seed=0, device=None, out_dir=None, test=None, workers=6, patience=20):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from ..evaluation.losses import dice_ce_loss
    from ..evaluation.validate import validate, summarize
    from .model import build_unet
    from .dataset import build_splits
    from .augment import augment_batch
    from ..data.mri.registry import get_adapter

    adapter = get_adapter(dataset)
    cases_fn, loader, ns = adapter.cases, adapter.load_ed_es, adapter.cache_ns
    out_dir = out_dir or f"runs/{dataset}"

    torch.manual_seed(seed)
    np.random.seed(seed)          # augmentation uses global np.random
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True   # fixed 256^2 input -> autotune fastest convs
    print(f"device={device} torch={torch.__version__} seed={seed} dataset={dataset}")

    train_ds, val_ds, train_dirs, val_dirs = build_splits(
        size=size, val_frac=val_frac, seed=seed, n_patients=n_patients,
        cases=cases_fn(), loader=loader, cache_ns=ns)
    print(f"patients: {len(train_dirs)} train / {len(val_dirs)} val  | "
          f"slices: {len(train_ds)} train / {len(val_ds)} val")

    pin = device == "cuda"
    augment = train_ds.augment   # augmentation is GPU-batched in this loop, not in the workers
    dl = DataLoader(train_ds, batch_size=batch, shuffle=True, drop_last=True,
                    num_workers=workers, pin_memory=pin,
                    persistent_workers=workers > 0)   # cheap per-item path now -> GPU stays fed
    val_dl = DataLoader(val_ds, batch_size=batch, num_workers=workers, pin_memory=pin)
    model = build_unet(spatial_dims=2, out_channels=4).to(device)
    loss_fn = dice_ce_loss()
    opt = torch.optim.Adam(model.parameters(), lr)
    scaler = torch.amp.GradScaler("cuda", enabled=pin)   # mixed precision

    # `epochs` is a ceiling — early stopping keeps the best-val checkpoint and bails when val
    # plateaus, so the heavy-aug "needs more epochs" question is answered per-run, not hardcoded.
    best_dice, best_state, bad = -1.0, None, 0
    for ep in range(epochs):
        t0 = time.perf_counter()
        model.train()
        tot = 0.0
        for x, y in dl:
            x = x.to(device, non_blocking=pin)
            y = y.to(device, non_blocking=pin)                # [B,H,W]
            if augment:
                x, y = augment_batch(x, y)                     # GPU-batched flip/rotate/scale + intensity
            y = y[:, None]                                     # -> [B,1,H,W]
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=pin):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item()
        vd = _val_dice(model, val_dl, device)                  # fast batched slice-Dice (no TTA)
        improved = vd > best_dice + 1e-4
        if improved:
            best_dice, bad = vd, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        print(f"epoch {ep:2d}  train_loss {tot/len(dl):.4f}  val_dice {vd:.4f}"
              f"{' *' if improved else ''}  ({time.perf_counter()-t0:.1f}s)")
        if bad >= patience:
            print(f"early stop @ epoch {ep} (no val gain for {patience}); best val_dice {best_dice:.4f}")
            break
    if best_state is not None:
        model.load_state_dict(best_state)                      # evaluate/ship the best, not the last

    print(f"\n===== VALIDATION ({dataset}) =====")
    dice_per_class, ef_rows = validate(model, val_dirs, size, device, loader=loader, cache_ns=ns)
    results = {"val": summarize(dice_per_class, ef_rows)}

    # held-out cross-dataset test (the generalization number)
    if test and test != dataset:
        tadapter = get_adapter(test)
        tloader, tns = tadapter.load_ed_es, tadapter.cache_ns
        test_dirs = tadapter.cases()
        if n_patients:
            test_dirs = test_dirs[:n_patients]
        print(f"\n===== CROSS-DATASET TEST: train={dataset} -> test={test} (n={len(test_dirs)}) =====")
        tdice, tef = validate(model, test_dirs, size, device, loader=tloader, cache_ns=tns)
        results[f"test_{test}"] = summarize(tdice, tef)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pth")
    meta = {"dataset": dataset, "test": test, "epochs": epochs, "batch": batch, "lr": lr,
            "size": size, "seed": seed, "n_train": len(train_dirs), "n_val": len(val_dirs),
            "val_patients": [p.name for p in val_dirs], "results": results}
    (out / "metrics.json").write_text(json.dumps(meta, indent=2))
    print(f"\nsaved model + metrics -> {out}/")
    return model, results


# Back-compat wrapper (README + tooling call this).
def train_acdc(epochs=128, batch=32, lr=1e-3, size=256, n_patients=0,
               val_frac=0.2, seed=0, device=None, out_dir="runs/acdc"):
    return train_seg("acdc", epochs=epochs, batch=batch, lr=lr, size=size,
                     n_patients=n_patients, val_frac=val_frac, seed=seed,
                     device=device, out_dir=out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--acdc", action="store_true", help="shorthand for --dataset acdc")
    ap.add_argument("--dataset", choices=["acdc", "mnm2", "mnms1"], default=None)
    ap.add_argument("--test", choices=["acdc", "mnm2", "mnms1", "none"], default="none",
                    help="held-out cross-dataset test set")
    ap.add_argument("--epochs", type=int, default=128, help="ceiling; early stopping ends sooner")
    ap.add_argument("--patience", type=int, default=20, help="early-stop patience (epochs w/o val gain)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=6, help="DataLoader workers (0=main proc)")
    ap.add_argument("--n-patients", type=int, default=0, help="0=all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ds = args.dataset or ("acdc" if args.acdc else None)
    if not ds:
        print("pass --dataset acdc|mnm2 (or --acdc)")
    else:
        test = None if args.test == "none" else args.test
        train_seg(ds, epochs=args.epochs, batch=args.batch, workers=args.workers,
                  n_patients=args.n_patients, seed=args.seed, out_dir=args.out, test=test,
                  patience=args.patience)
