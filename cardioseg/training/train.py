"""Train a 2D U-Net from a TrainCfg (cardioseg.hparams) — one typed config in, model+metrics out.

    python -m cardioseg.training.train --out runs/gen         # default split: hold out ACDC + Canon
    python -m cardioseg.training.train --set data.test_vendors=('GE',) aug.gamma_p=0.5

The split is criteria over the data cloud (DataCfg.test_datasets / test_vendors); held-out test =
rows matching those, never seen in train/val. The full config is serialized to runs/<run>/config.json
for provenance + reproducibility — the criteria ARE the record of what was held out.
"""
import argparse
import json
import time
from pathlib import Path

from ..hparams import TrainCfg
from ..labels import FOREGROUND


def _val_dice(model, val_dl, device) -> float:
    """Fast batched mean foreground Dice (pooled over val slices, no TTA) — the early-stop signal."""
    import torch
    import numpy as np

    inter = {c: 0.0 for c in FOREGROUND}
    denom = {c: 0.0 for c in FOREGROUND}
    model.eval()
    with torch.no_grad():
        for x, y in val_dl:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            for c in FOREGROUND:
                p, g = pred == c, y == c
                inter[c] += 2.0 * (p & g).sum().item()
                denom[c] += (p.sum() + g.sum()).item()
    return float(np.mean([inter[c] / denom[c] if denom[c] else 0.0 for c in FOREGROUND]))


def train_seg(cfg: TrainCfg):
    """Train from one TrainCfg. Returns (model, results). Serializes config.json + metrics.json."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from ..evaluation.losses import build_loss
    from ..evaluation.validate import validate, summarize
    from .model import build_unet, resolve_device
    from .dataset import datasets
    from .augment import augment_batch
    from ..data import store, splits
    from ..obs import setup, timed, progress
    from ..hparams import to_json

    d = cfg.data
    out = Path(cfg.out_dir or "runs/seg")
    log = setup(out / "train.log")
    to_json(cfg, out / "config.json")          # full provenance, written up front

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)                    # augmentation uses global np.random
    device = resolve_device(cfg.device)
    torch.backends.cudnn.benchmark = True       # fixed input size -> autotune fastest convs
    log.info("device=%s torch=%s seed=%s | held out: datasets=%s vendors=%s", device,
             torch.__version__, cfg.seed, list(d.test_datasets), list(d.test_vendors))

    # split = criteria over the consolidated store (builds processed/<ds>/ if missing)
    with timed(log, "store.load + split"):
        meta = store.load(list(d.sources), inplane=d.inplane, n4=d.n4, workers=cfg.workers)
        train_df, val_df, test_df = splits.make_split(
            meta, d.test_datasets, d.test_vendors, d.val_frac, cfg.seed)
    if cfg.n_patients:                          # debug cap
        train_df, val_df = train_df.head(cfg.n_patients), val_df.head(max(1, cfg.n_patients // 4))
        test_df = test_df.head(cfg.n_patients)

    with timed(log, f"build slice datasets ({len(train_df)}+{len(val_df)} subjects)"):
        train_ds, val_ds = datasets(splits.paths(train_df), splits.paths(val_df), d.size)
    log.info("patients: %d train / %d val / %d test | slices: %d train / %d val",
             len(train_df), len(val_df), len(test_df), len(train_ds), len(val_ds))

    # Dataset is fully in RAM + augmentation is GPU-batched, so DataLoader workers=0: on Windows
    # num_workers>0 PICKLES the whole in-RAM dataset to every worker (huge stall, GPU starves).
    # cfg.workers parallelizes store CONSOLIDATION (ThreadPool), not this loop.
    pin = device == "cuda"
    augment = train_ds.augment
    dl = DataLoader(train_ds, batch_size=cfg.batch, shuffle=True, drop_last=True, num_workers=0, pin_memory=pin)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch, num_workers=0, pin_memory=pin)
    model = build_unet(cfg.model).to(device)
    loss_fn = build_loss(cfg.loss)
    opt = torch.optim.Adam(model.parameters(), cfg.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=pin)   # mixed precision

    # cfg.epochs is a ceiling — early stopping keeps the best-val checkpoint and bails on plateau.
    best_dice, best_state, bad = -1.0, None, 0
    nb = len(dl)
    for ep in range(cfg.epochs):
        t0 = time.perf_counter()
        model.train()
        if hasattr(loss_fn, "epoch"):
            loss_fn.epoch = ep                             # drives the HD-warmup ramp (dice_ce_hd)
        tot = 0.0
        for x, y in progress(dl, f"epoch {ep}", total=nb):
            x = x.to(device, non_blocking=pin)
            y = y.to(device, non_blocking=pin)                # [B,H,W]
            if augment:
                x, y = augment_batch(x, y, cfg.aug)            # GPU-batched, hyperparams from cfg.aug
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
        dt = time.perf_counter() - t0
        log.info("epoch %2d  train_loss %.4f  val_dice %.4f%s  (%.1fs, %.1f batch/s)",
                 ep, tot / nb, vd, " *" if improved else "", dt, nb / dt)
        if bad >= cfg.patience:
            log.info("early stop @ epoch %d (no val gain for %d); best val_dice %.4f", ep, cfg.patience, best_dice)
            break
    if best_state is not None:
        model.load_state_dict(best_state)                      # evaluate/ship the best, not the last

    log.info("===== VALIDATION =====")
    dice_per_class, ef_rows, surf = validate(model, splits.paths(val_df), d.size, device)
    results = {"val": summarize(dice_per_class, ef_rows, surf)}

    # held-out test = the criteria split (datasets/vendors held out)
    if len(test_df):
        log.info("===== HELD-OUT TEST: datasets=%s vendors=%s (n=%d) =====",
                 list(d.test_datasets), list(d.test_vendors), len(test_df))
        tdice, tef, tsurf = validate(model, splits.paths(test_df), d.size, device)
        results["test"] = summarize(tdice, tef, tsurf)

    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pth")
    meta = {"config": cfg.model_dump(), "n_train": len(train_df), "n_val": len(val_df),
            "val_patients": val_df.get_column("subject_id").to_list(), "results": results}
    (out / "metrics.json").write_text(json.dumps(meta, indent=2))
    log.info("saved model + config + metrics -> %s/", out)

    # auto-finalize: per-run model card + ONNX export. Both guarded — a missing optional dep or a
    # hiccup logs and moves on, never fails a finished training run.
    try:
        from ..modelcard import generate
        generate(out)
        log.info("wrote %s/MODEL_CARD.md", out)
    except Exception as e:
        log.warning("model card skipped: %s", e)
    try:
        from .export_onnx import export
        export(out, splits.paths(val_df)[0])               # ONNX + INT8, parity-gated
    except ImportError:
        log.info("ONNX export skipped (pip install .[export] for onnxruntime)")
    except Exception as e:
        log.warning("ONNX export skipped: %s", e)
    return model, results


if __name__ == "__main__":
    from ..hparams import apply_overrides

    # Defaults = the generalization split (hold out ACDC + Canon). Change the split via the criteria
    # on DataCfg with --set, e.g. legacy train M&M-2 -> test ACDC:
    #   --set data.sources=('mnm2','acdc') data.test_datasets=('acdc',) data.test_vendors=()
    ap = argparse.ArgumentParser(description="train a 2D U-Net from a TrainCfg (split = DataCfg criteria)")
    ap.add_argument("--epochs", type=int); ap.add_argument("--batch", type=int)
    ap.add_argument("--patience", type=int); ap.add_argument("--workers", type=int)
    ap.add_argument("--seed", type=int); ap.add_argument("--n-patients", type=int, dest="n_patients")
    ap.add_argument("--n4", action="store_true"); ap.add_argument("--out", default=None)
    ap.add_argument("--set", nargs="*", default=[], dest="overrides",
                    help="deep cfg overrides, e.g. data.test_vendors=('GE',) aug.gamma_p=0.5")
    a = ap.parse_args()

    cfg = TrainCfg()
    for attr in ("epochs", "batch", "patience", "workers", "seed", "n_patients"):
        if getattr(a, attr) is not None:
            setattr(cfg, attr, getattr(a, attr))
    if a.n4:
        cfg.data.n4 = True
    if a.out:
        cfg.out_dir = a.out
    apply_overrides(cfg, a.overrides)
    train_seg(cfg)
