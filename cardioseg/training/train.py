"""Train a 2D U-Net from a TrainCfg (core.hparams) — one typed config in, model+metrics out.

    python -m cardioseg.training.train --out runs/gen         # default split: hold out ACDC + Canon
    python -m cardioseg.training.train --set generator.data.test_vendors=('GE',) generator.aug.gamma_p=0.5

The split is criteria over the data cloud (DataCfg.test_datasets / test_vendors); held-out test =
rows matching those, never seen in train/val. The full config is serialized to runs/<run>/config.json
for provenance + reproducibility — the criteria ARE the record of what was held out.
"""
import argparse
import json
import time
from pathlib import Path

from core.hparams import TrainCfg
from core.data.static.labels import FOREGROUND
from ..tracking import track_run


def _val_dice(model, Ximg, Ymsk, batch: int, device) -> float:
    """Fast batched mean foreground Dice (pooled over val slices, no TTA) — the early-stop signal.
    Ximg/Ymsk are the resident val tensors; .to(device) is a no-op when they're already on the GPU."""
    import torch
    import numpy as np

    inter = {c: 0.0 for c in FOREGROUND}
    denom = {c: 0.0 for c in FOREGROUND}
    model.eval()
    with torch.no_grad():
        for i in range(0, Ximg.shape[0], batch):
            x = Ximg[i:i + batch].to(device)
            y = Ymsk[i:i + batch].to(device).long()
            pred = model(x).argmax(1)
            for c in FOREGROUND:
                p, g = pred == c, y == c
                inter[c] += 2.0 * (p & g).sum().item()
                denom[c] += (p.sum() + g.sum()).item()
    return float(np.mean([inter[c] / denom[c] if denom[c] else 0.0 for c in FOREGROUND]))


def train_seg(cfg: TrainCfg, alias: str | None = None, quick: bool = False):
    """Train from one TrainCfg. Returns (model, results). Builds artifacts in a gitignored staging
    dir, then registers the complete set (model.pth + config + metrics + onnx + card) to the mlflow
    model registry — the sole model store. `alias='production'` makes this run the flagship."""
    import numpy as np
    import torch
    from .losses import build_loss
    from ..evaluation.validate import validate, summarize
    from core.model import build_unet, resolve_device
    from core.data.dynamic.dataset import load_to_gpu
    from core.data.dynamic.generator import Generator
    from core.data.static import store, splits
    from core.obs import setup, timed, progress
    from core.hparams import to_json

    d = cfg.generator.data
    # staging dir (gitignored) — artifacts build here, then get registered to mlflow (the store).
    # NOT a permanent runs/ dir; cfg.out_dir still works for an explicit local copy.
    out = Path(cfg.out_dir or ".staging/run")
    out.mkdir(parents=True, exist_ok=True)
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
        meta = store.load(list(d.sources), inplane=d.inplane, n4=d.n4, n4_params=d.n4_params,
                          workers=cfg.workers, nyul=d.nyul, norm=d.norm)
        train_df, val_df, test_df = splits.make_split(
            meta, d.test_datasets, d.test_vendors, d.val_frac, cfg.seed,
            val_datasets=d.val_datasets, val_vendors=d.val_vendors)
    if cfg.n_patients:                          # debug cap
        train_df, val_df = train_df.head(cfg.n_patients), val_df.head(max(1, cfg.n_patients // 4))
        test_df = test_df.head(cfg.n_patients)

    # Preload ALL slices into device memory (VRAM): after this, the epoch loop is pure GPU — index a
    # permutation, augment, train; zero per-epoch CPU/disk/host↔device copy. The slice set fits the
    # card (~3 GB at 256px). No DataLoader/workers (which on Windows pickle the whole RAM dataset per
    # worker and starve the GPU). Prefer fast all-GPU epochs over disk-streamed ones.
    pin = device == "cuda"
    # residency = where the preloaded tensors live (gpu=VRAM-resident / cpu=RAM, copied per batch).
    # gpu only makes sense with a cuda device; fall back to cpu residency otherwise.
    data_device = device if (cfg.residency == "gpu" and device == "cuda") else "cpu"
    with timed(log, f"preload slices (residency={cfg.residency}->{data_device}, {len(train_df)}+{len(val_df)} subj)"):
        if d.anatomy_pool:
            # TRAIN on synthetic anatomy (Rodero SSM label maps) — zero real data; val/test stay REAL.
            import torch as _t
            from core.data.dynamic.anatomy import load_pool
            pool = load_pool(d.anatomy_pool)
            Ytr = _t.as_tensor(pool, dtype=_t.long, device=data_device)               # [N,H,W] labels
            Xtr = _t.zeros((Ytr.shape[0], 1, d.size, d.size), device=data_device)     # unused (flat bg, synth_p=1)
            cfg.generator.synth.synth_p = 1.0
            cfg.generator.synth.bg_mode = "flat"                                      # no real image to partition
            log.info("ANATOMY POOL: %d synth-anatomy slices (train); real val/test held out", Ytr.shape[0])
        else:
            Xtr, Ytr = load_to_gpu(splits.paths(train_df), d.size, data_device)
        Xva, Yva = load_to_gpu(splits.paths(val_df), d.size, data_device)
    # the data engine: yields collapsed batches (real / synth / mixed by cfg.generator.synth)
    gen = Generator(cfg.generator, Xtr, Ytr, cfg.model.out_channels, device)
    nb = max(1, Xtr.shape[0] // cfg.batch)
    log.info("patients: %d train / %d val / %d test | slices: %d train / %d val (resident on %s, compute %s)",
             len(train_df), len(val_df), len(test_df), Xtr.shape[0], Xva.shape[0], data_device, device)
    model = build_unet(cfg.model).to(device)
    # Soft-label training (honest probabilistic boundary targets) takes the SoftDiceCE path; σ=0
    # (default) keeps the hard-label Dice+CE recipe bit-for-bit.
    soft_sigma = cfg.generator.aug.soft_label_sigma
    if soft_sigma > 0:
        from .losses import SoftDiceCE
        loss_fn = SoftDiceCE()
    else:
        loss_fn = build_loss(cfg.loss)
    opt = torch.optim.Adam(model.parameters(), cfg.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=pin)   # mixed precision

    # cfg.epochs is a ceiling — early stopping keeps the best-val checkpoint and bails on plateau.
    best_dice, best_state, bad = -1.0, None, 0   # nb (batches/epoch) computed above from the resident set
    trk = track_run("cardioseg", out.name, run_dir=out,
                    params={**cfg.model_dump(), "n_train": len(train_df), "n_val": len(val_df)},
                    tags={"split": "+".join(d.test_vendors) or "legacy", "seed": cfg.seed})
    fit_t0 = time.perf_counter()                            # real training wall-clock (run-duration is unreliable)
    for ep in range(cfg.epochs):
        t0 = time.perf_counter()
        model.train()
        if hasattr(loss_fn, "epoch"):
            loss_fn.epoch = ep                             # drives the HD-warmup ramp (dice_ce_hd)
        tot = 0.0
        perm = torch.randperm(Xtr.shape[0], device=Xtr.device)   # shuffle on the data's device
        for bi in progress(range(nb), f"epoch {ep}", total=nb):
            idx = perm[bi * cfg.batch:(bi + 1) * cfg.batch]
            x, yt = gen.batch(idx, pin)                         # collapsed batch (real/synth/mixed)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=pin):
                loss = loss_fn(model(x), yt)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item()
        vd = _val_dice(model, Xva, Yva, cfg.batch, device)        # fast batched slice-Dice (no TTA)
        improved = vd > best_dice + 1e-4
        if improved:
            best_dice, bad = vd, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        dt = time.perf_counter() - t0
        log.info("epoch %2d  train_loss %.4f  val_dice %.4f%s  (%.1fs, %.1f batch/s)",
                 ep, tot / nb, vd, " *" if improved else "", dt, nb / dt)
        trk.metric("train_loss", tot / nb, step=ep); trk.metric("val_dice", vd, step=ep)
        if bad >= cfg.patience:
            log.info("early stop @ epoch %d (no val gain for %d); best val_dice %.4f", ep, cfg.patience, best_dice)
            break
    trk.metric("train_minutes", (time.perf_counter() - fit_t0) / 60)   # trustworthy compute time
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

    torch.save(model.state_dict(), out / "model.pth")  # out already created by to_json(config) above
    meta = {"config": cfg.model_dump(), "n_train": len(train_df), "n_val": len(val_df),
            "val_patients": val_df.get_column("subject_id").to_list(), "results": results}
    (out / "metrics.json").write_text(json.dumps(meta, indent=2))
    log.info("saved model + config + metrics -> %s/", out)

    trk.summary(results)                                    # final per-axis dice/EF (metrics in the run)
    if quick:                                               # experiment sweep: skip the artifact tail
        log.info("quick mode: skipping model card / attribution / ONNX / registry")
        trk.end()
        return model, results
    # build the rest of the artifacts into staging (card + onnx) BEFORE registering the complete set.
    try:
        from ..evaluation.modelcard import generate
        generate(out)
        log.info("wrote %s/MODEL_CARD.md", out)
    except Exception as e:
        log.warning("model card skipped: %s", e)
    try:                                                   # attribution diagnostic (confusion + saliency)
        from core.data.analysis.attribution import run_attribution
        s = run_attribution(model, Xva, Yva, cfg.model.out_channels, device, out)
        log.info("attribution: recall=%s saliency=%s -> %s/attribution.png", s["recall"], s["saliency"], out.name)
    except Exception as e:
        log.warning("attribution skipped: %s", e)
    try:
        from core.export_onnx import export
        export(out, splits.paths(val_df)[0])               # ONNX + INT8, parity-gated
    except ImportError:
        log.info("ONNX export skipped (pip install .[export] for onnxruntime)")
    except Exception as e:
        log.warning("ONNX export skipped: %s", e)

    # register the COMPLETE artifact set (model.pth + config + metrics + onnx + card) to the mlflow
    # registry — the sole model store. alias='production' makes this the flagship.
    try:
        import mlflow
        from core.registry import save_model, MODEL_NAME
        rid = mlflow.active_run().info.run_id if mlflow.active_run() else None
        split = "+".join(d.test_vendors) or "legacy"
        kind = "flagship" if alias == "production" else "candidate"
        save_model(out, run_name=out.name, run_id=rid, alias=alias,
                   description=f"{out.name} · split={split} · seed={cfg.seed}",
                   tags={"kind": kind, "split": split, "seed": cfg.seed})
        log.info("registered to mlflow registry '%s'%s", MODEL_NAME, f" (alias={alias})" if alias else "")
    except Exception as e:
        log.warning("registry save skipped: %s", e)
    trk.end()
    return model, results


if __name__ == "__main__":
    from core.hparams import apply_overrides

    # Defaults = the generalization split (hold out ACDC + Canon). Change the split via the criteria
    # on DataCfg with --set, e.g. legacy train M&M-2 -> test ACDC:
    #   --set generator.data.sources=('mnm2','acdc') generator.data.test_datasets=('acdc',) generator.data.test_vendors=()
    ap = argparse.ArgumentParser(description="train a 2D U-Net from a TrainCfg (split = DataCfg criteria)")
    ap.add_argument("--epochs", type=int); ap.add_argument("--batch", type=int)
    ap.add_argument("--patience", type=int); ap.add_argument("--workers", type=int)
    ap.add_argument("--seed", type=int); ap.add_argument("--n-patients", type=int, dest="n_patients")
    ap.add_argument("--n4", action="store_true"); ap.add_argument("--out", default=None)
    ap.add_argument("--alias", default=None,
                    help="registry alias to set (e.g. 'production' to make this run the flagship)")
    ap.add_argument("--set", nargs="*", default=[], dest="overrides",
                    help="deep cfg overrides, e.g. generator.data.test_vendors=('GE',) generator.aug.gamma_p=0.5")
    ap.add_argument("--quick", action="store_true",
                    help="experiment sweep: train + eval only, skip ONNX/INT8/attribution/registry (~2x faster)")
    a = ap.parse_args()

    cfg = TrainCfg()
    for attr in ("epochs", "batch", "patience", "workers", "seed", "n_patients"):
        if getattr(a, attr) is not None:
            setattr(cfg, attr, getattr(a, attr))
    if a.n4:
        cfg.generator.data.n4 = True
    if a.out:
        cfg.out_dir = a.out
    apply_overrides(cfg, a.overrides)
    train_seg(cfg, alias=a.alias, quick=a.quick)
