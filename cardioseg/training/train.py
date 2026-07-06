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
    log.info("device=%s torch=%s seed=%s | split=%s | criteria datasets=%s vendors=%s", device,
             torch.__version__, cfg.seed, d.split or "(legacy criteria)",
             list(d.test_datasets), list(d.test_vendors))

    # split = criteria over the consolidated store (builds processed/<ds>/ if missing). A coded split
    # family may declare its own `sources` (e.g. static_all adds SCD) — load those, not just d.sources.
    srcs = list(d.sources)
    if d.split:
        from core.data.ingest.splits import load_split, parse_ref
        srcs = list(getattr(load_split(parse_ref(d.split)[0]), "sources", None) or d.sources)
    with timed(log, "store.load + split"):
        meta = store.load(srcs, inplane=d.inplane, n4=d.n4, n4_params=d.n4_params,
                          workers=cfg.workers, nyul=d.nyul, norm=d.norm)
        if d.split:                                 # NEW-STYLE: a coded-filter family owns the partition
            from core.data.ingest.splits import resolve_cfg
            r = resolve_cfg(d, meta)
            train_src, val_src = r.train, r.val      # Sources (static OR dynamic) -> the train_gen seam
            test_df = r.test.frame                   # test + val are always StaticSource (frozen real)
            val_df = r.val.frame                     # val is real -> its frame drives scoring/export/params
            if r.train.kind == "static":
                train_df = r.train.frame             # dynamic train has no frame (counts via tensors)
            log.info("split=%s@%s test_hash=%s | train=%s val=%s test n=%d",
                     d.split.split("@")[0], r.version, r.test_hash[:19], r.train.kind, r.val.kind, len(test_df))
        else:
            train_src = val_src = None               # legacy: DataCfg criteria + inline anatomy block
            train_df, val_df, test_df = splits.split_from_cfg(d, meta, cfg.seed)
    if cfg.n_patients:                          # debug cap (old-style frames only; test always capped)
        test_df = test_df.head(cfg.n_patients)
        if train_src is None:
            train_df, val_df = train_df.head(cfg.n_patients), val_df.head(max(1, cfg.n_patients // 4))

    # Preload ALL slices into device memory (VRAM): after this, the epoch loop is pure GPU — index a
    # permutation, augment, train; zero per-epoch CPU/disk/host↔device copy. The slice set fits the
    # card (~3 GB at 256px). No DataLoader/workers (which on Windows pickle the whole RAM dataset per
    # worker and starve the GPU). Prefer fast all-GPU epochs over disk-streamed ones.
    pin = device == "cuda"
    # residency = where the preloaded tensors live (gpu=VRAM-resident / cpu=RAM, copied per batch).
    # gpu only makes sense with a cuda device; fall back to cpu residency otherwise.
    data_device = device if (cfg.residency == "gpu" and device == "cuda") else "cpu"
    with timed(log, f"preload slices (residency={cfg.residency}->{data_device})"):
        import torch as _t
        if train_src is not None:
            # NEW: each Source OWNS its batch engine (train_gen). No static/dynamic if, no force_synth in
            # the interface, no bg_mode poke — StaticSource = real + DR-aug, DynamicSource = synth painter.
            gen = train_src.train_gen(d.size, data_device, cfg.generator, cfg.model.out_channels)
            Xva, Yva = val_src.resident(d.size, data_device)
        else:
            # LEGACY criteria path: build resident tensors (+ inline anatomy) and the shared Generator.
            force_synth = None
            if d.anatomy_pool:
                # TRAIN masks from synthetic anatomy (Rodero SSM label maps); val/test stay REAL held-out.
                from core.data.dynamic.anatomy import load_pool
                Ys = _t.as_tensor(load_pool(d.anatomy_pool), dtype=_t.long, device=data_device)  # [N,H,W]
                if d.anatomy_mode == "mix":
                    # AUGMENTATION (bd pwih): REAL train + synth-anatomy UNION; synth rows force-painted.
                    Xr, Yr = load_to_gpu(splits.paths(train_df), d.size, data_device)
                    Xsy = _t.zeros((Ys.shape[0], 1, d.size, d.size), device=data_device)
                    Xtr = _t.cat([Xr, Xsy]); Ytr = _t.cat([Yr, Ys])
                    force_synth = _t.cat([_t.zeros(Xr.shape[0], dtype=_t.bool, device=data_device),
                                          _t.ones(Ys.shape[0], dtype=_t.bool, device=data_device)])
                    log.info("ANATOMY MIX: %d real + %d synth-anatomy (bg=%s)", Xr.shape[0], Ys.shape[0],
                             cfg.generator.synth.bg_mode)
                else:                                                        # "replace": synth anatomy ONLY
                    Ytr = Ys
                    cfg.generator.synth.synth_p = 1.0
                    if cfg.generator.synth.bg_mode in ("flat", "procedural", "mrxcat"):
                        Xtr = _t.zeros((Ytr.shape[0], 1, d.size, d.size), device=data_device)  # ZERO-REAL
                        log.info("ANATOMY POOL: %d slices, ZERO-REAL bg=%s", Ytr.shape[0],
                                 cfg.generator.synth.bg_mode)
                    else:                                                    # Rodero heart on real bg (excised)
                        from core.data.dynamic.synth import excise_heart
                        Xr, Yr = load_to_gpu(splits.paths(train_df), d.size, data_device)
                        Xr = excise_heart(Xr, Yr)
                        Xtr = Xr[_t.randint(Xr.shape[0], (Ytr.shape[0],), device=Xr.device)]
                        log.info("ANATOMY POOL: %d Rodero on real bg (excised, %s)", Ytr.shape[0],
                                 cfg.generator.synth.bg_mode)
            else:
                Xtr, Ytr = load_to_gpu(splits.paths(train_df), d.size, data_device)
            Xva, Yva = load_to_gpu(splits.paths(val_df), d.size, data_device)
            gen = Generator(cfg.generator, Xtr, Ytr, cfg.model.out_channels, device, force_synth=force_synth)
    nb = max(1, gen.X.shape[0] // cfg.batch)
    log.info("engine train=%s | slices: %d train / %d val / %d test-subj (resident %s, compute %s)",
             (train_src.kind if train_src else "legacy"), gen.X.shape[0], Xva.shape[0], len(test_df),
             data_device, device)
    model = build_unet(cfg.model).to(device)
    # Loss selection. PARTIAL-LABEL (a train source carries a per-slice valid mask, e.g. SCD LV-only)
    # overrides -> PartialLabelDiceCE (handles soft yt + the [B,C] mask). Else soft-label -> SoftDiceCE;
    # else the configured (MONAI) loss. Full-label runs are untouched (partial=False).
    soft_sigma = cfg.generator.aug.soft_label_sigma
    partial = getattr(gen, "valid", None) is not None
    if partial:
        from .losses import PartialLabelDiceCE
        loss_fn = PartialLabelDiceCE()
        log.info("PARTIAL-LABEL loss: %d/%d slices class-masked",
                 int((~gen.valid.all(1)).sum()), gen.valid.shape[0])
    elif soft_sigma > 0:
        from .losses import SoftDiceCE
        loss_fn = SoftDiceCE()
    else:
        loss_fn = build_loss(cfg.loss)
    opt = torch.optim.Adam(model.parameters(), cfg.lr)
    # Kendall learned seg-vs-EF balance (ef_learn): two log-variances trained alongside the model.
    log_sig = torch.zeros(2, device=device, requires_grad=True) if (cfg.ef_lambda > 0 and cfg.ef_learn) else None
    if log_sig is not None:
        opt.add_param_group({"params": [log_sig]})
    scaler = torch.amp.GradScaler("cuda", enabled=pin)   # mixed precision

    # cfg.epochs is a ceiling — early stopping keeps the best-val checkpoint and bails on plateau.
    best_dice, best_state, bad = -1.0, None, 0   # nb (batches/epoch) computed above from the resident set
    # n_train in slices when the train source is dynamic (no patient frame); else patient rows.
    n_train = gen.X.shape[0] if (train_src is not None and train_src.kind == "dynamic") else len(train_df)
    split_tag = d.split or ("+".join(d.test_vendors) or "legacy")
    trk = track_run("cardioseg", out.name, run_dir=out,
                    params={**cfg.model_dump(), "n_train": n_train, "n_val": len(val_df)},
                    tags={"split": split_tag, "seed": cfg.seed})
    fit_t0 = time.perf_counter()                            # real training wall-clock (run-duration is unreliable)
    for ep in range(cfg.epochs):
        t0 = time.perf_counter()
        model.train()
        if hasattr(loss_fn, "epoch"):
            loss_fn.epoch = ep                             # drives the HD-warmup ramp (dice_ce_hd)
        tot = 0.0
        perm = torch.randperm(gen.X.shape[0], device=gen.X.device)   # shuffle on the data's device
        for bi in progress(range(nb), f"epoch {ep}", total=nb):
            idx = perm[bi * cfg.batch:(bi + 1) * cfg.batch]
            x, yt, valid = gen.batch(idx, pin)                  # collapsed batch (+ partial-label mask)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=pin):
                loss = loss_fn(model(x), yt, valid) if partial else loss_fn(model(x), yt)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item()
        if cfg.ef_lambda > 0 and ep >= cfg.ef_warmup and train_src is not None and train_src.kind == "static":
            # EF/volume-consistency NUDGE: add ef_lambda·vol_loss INTO one seg gradient step (not a
            # separate vote) — seg's dense signal keeps the direction, EF only bends the cavity volume.
            from .ef_lane import volume_loss_batch
            paths = splits.paths(train_df)
            sub = [paths[i] for i in np.random.permutation(len(paths))[:cfg.ef_subjects]]
            vl = volume_loss_batch(model, sub, d.size, device, amp=pin)
            if vl is not None:
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", enabled=pin):
                    seg = loss_fn(model(x), yt, valid) if partial else loss_fn(model(x), yt)
                if log_sig is not None:                                # learned Kendall balance
                    from .losses import uncertainty_weighted
                    loss_j = uncertainty_weighted([seg, vl], [log_sig[0], log_sig[1]])
                else:                                                  # fixed nudge (seg dominant)
                    loss_j = seg + cfg.ef_lambda * vl
                scaler.scale(loss_j).backward()
                scaler.step(opt)
                scaler.update()
                if log_sig is not None:
                    log.info("ef balance (learned): seg_w=%.3f vol_w=%.3f",
                             float(torch.exp(-log_sig[0])), float(torch.exp(-log_sig[1])))
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
    meta = {"config": cfg.model_dump(), "n_train": n_train, "n_val": len(val_df),
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
    ap.add_argument("--ef-lambda", type=float, dest="ef_lambda",
                    help="weight of the EF/volume-consistency auxiliary lane (0 = off)")
    ap.add_argument("--ef-learn", action="store_true", dest="ef_learn",
                    help="LEARN the seg-vs-EF balance (Kendall) instead of the fixed ef_lambda")
    ap.add_argument("--split", default=None,
                    help="coded-filter split family 'name[@ver]' (core.data.ingest.splits, e.g. "
                         "static_main / synth_main); sets generator.data.split before --set overrides")
    ap.add_argument("--n4", action="store_true"); ap.add_argument("--out", default=None)
    ap.add_argument("--alias", default=None,
                    help="registry alias to set (e.g. 'production' to make this run the flagship)")
    ap.add_argument("--set", nargs="*", default=[], dest="overrides",
                    help="deep cfg overrides, e.g. generator.data.test_vendors=('GE',) generator.aug.gamma_p=0.5")
    ap.add_argument("--quick", action="store_true",
                    help="experiment sweep: train + eval only, skip ONNX/INT8/attribution/registry (~2x faster)")
    a = ap.parse_args()

    cfg = TrainCfg()
    if a.split:                                      # coded-filter family owns the partition (core.data.ingest.splits)
        from core.data.ingest.splits import list_splits
        name = a.split.split("@", 1)[0]
        if name not in list_splits():
            raise SystemExit(f"unknown split {name!r}; known: {list_splits()}")
        cfg.generator.data.split = a.split
    for attr in ("epochs", "batch", "patience", "workers", "seed", "n_patients", "ef_lambda"):
        if getattr(a, attr) is not None:
            setattr(cfg, attr, getattr(a, attr))
    if a.n4:
        cfg.generator.data.n4 = True
    if a.ef_learn:
        cfg.ef_learn = True
    if a.out:
        cfg.out_dir = a.out
    apply_overrides(cfg, a.overrides)
    train_seg(cfg, alias=a.alias, quick=a.quick)
