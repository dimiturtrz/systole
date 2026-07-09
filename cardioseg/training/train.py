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
from contextlib import contextmanager
from pathlib import Path

import mlflow
import numpy as np
import torch
from mlflow.exceptions import MlflowException

from core.data.analysis.attribution import Attribution
from core.data.dynamic.anatomy import load_pool
from core.data.dynamic.dataset import load_to_gpu
from core.data.dynamic.generator import Generator
from core.data.dynamic.synth import excise_heart
from core.data.ingest.splits import list_splits, load_split, parse_ref, resolve_cfg
from core.data.static import splits
from core.data.static.labels import FOREGROUND
from core.data.static.store import build as store
from core.export_onnx import export
from core.hparams import TrainCfg, apply_overrides, to_json
from core.losses import PartialLabelDiceCE, SoftDiceCE, uncertainty_weighted
from core.model import build_unet, resolve_device
from core.obs import progress, setup, timed
from core.registry import MODEL_NAME, save_model

from ..evaluation.modelcard import ModelCard
from ..evaluation.validate import EvalCfg, Evaluator, summarize
from ..tracking import Tracker
from .ef_lane import build_aux


def parse_seeds(seeds_arg: str | None) -> list[int] | None:
    """CLI '--seeds 0,1,2' -> [0,1,2]; None/'' -> None (falls back to cfg.seed). Pure string parse."""
    return [int(s) for s in seeds_arg.split(",")] if seeds_arg else None


def resolve_seeds(cfg_seed: int, seeds) -> list[int]:
    """The effective seed list for a run: explicit `seeds` if given, else the single cfg seed. Mirrors
    train_seg's `seeds = list(seeds) if seeds else [cfg.seed]`, pulled out so it's testable off-GPU."""
    return list(seeds) if seeds else [cfg_seed]


def check_multiseed_split(seeds: list, split) -> None:
    """Multi-seed A/B needs a coded --split (legacy criteria splits key their partition on the seed, so
    they can't share one loaded dataset). Raises ValueError on the invalid combination; else no-op."""
    if len(seeds) > 1 and not split:
        raise ValueError("multi-seed needs a coded --split: legacy criteria splits key their partition "
                         "on the seed, so seeds can't share one loaded dataset")


def seed_out_dir(base_out: Path, seed: int, *, single: bool) -> Path:
    """Per-seed artifact dir: the shared base for a single seed, else `<base>_s<seed>` as a sibling."""
    return base_out if single else base_out.parent / f"{base_out.name}_s{seed}"


def split_tag_of(d) -> str:
    """The tracker's split tag from a DataCfg: coded split name, else joined test vendors, else 'legacy'."""
    return d.split or ("+".join(d.test_vendors) or "legacy")


def n_train_of(train_src, gen, train_df) -> int:
    """n_train is the resident slice count (gen.X) when the train source is dynamic (no patient frame);
    else the patient-row count of the train frame."""
    return gen.n if (train_src is not None and getattr(train_src, "kind", None) in ("dynamic", "composite")) \
        else len(train_df)


def apply_cli_args(cfg: TrainCfg, args) -> TrainCfg:
    """Map the argparse Namespace onto a TrainCfg IN PLACE (then returned): scalar attrs copied when
    set, the store-flags folded in, then the deep `--set` overrides applied last (so --set wins). The
    coded --split is applied by the caller (it needs list_splits validation) before this. Pure mapping —
    no IO, so the whole CLI contract is testable without training."""
    a = args if isinstance(args, dict) else vars(args)         # Namespace -> dict; also accept a plain dict
    for attr in ("epochs", "batch", "patience", "workers", "seed", "n_patients", "ef_lambda"):
        if a.get(attr) is not None:
            setattr(cfg, attr, a[attr])
    if a.get("n4"):
        cfg.generator.data.n4 = True
    if a.get("ef_learn"):
        cfg.ef_learn = True
    if a.get("ef_kaggle"):
        cfg.ef_kaggle = True
    if a.get("out"):
        cfg.out_dir = a["out"]
    apply_overrides(cfg, a.get("overrides") or [])
    return cfg


def _val_dice(model, Ximg, Ymsk, batch: int, device) -> float:
    """Fast batched mean foreground Dice (pooled over val slices, no TTA) — the early-stop signal.
    Ximg/Ymsk are the resident val tensors; .to(device) is a no-op when they're already on the GPU."""
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


class SeedTrainer:
    """Trains + evaluates + registers ONE seed on the shared resident data `sh` (built once by
    train_seg). Holds the seed's OWN state (model/optimizer/loss/tracker/output dir); the read-only
    shared data (resident tensors, val, aux EF lanes) comes via `sh`. Construct per seed, call .run().
    N seeds cost 1xdata + Nxmodel. (bd 01fh: former _train_loop/_finalize threaded ~10 args -> methods.)"""

    def __init__(self, cfg: TrainCfg, seed: int, sh: dict, alias: str | None, *, quick: bool):  # pragma: no cover  (builds a GPU model/optimizer/tracker on the resident data bundle)
        self.cfg, self.seed, self.sh, self.alias, self.quick = cfg, seed, sh, alias, quick
        self.d = cfg.generator.data
        self.device, self.pin, self.gen, self.aux = sh["device"], sh["pin"], sh["gen"], sh["aux"]
        cfg.seed = seed
        self.out = seed_out_dir(sh["base_out"], seed, single=sh["single"])
        self.out.mkdir(parents=True, exist_ok=True)
        self.log = setup(self.out / "train.log")
        to_json(cfg, self.out / "config.json")
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.model = build_unet(cfg.model).to(self.device)
        self.partial, self.loss_fn = self._build_loss()
        self.opt = torch.optim.Adam(self.model.parameters(), cfg.lr)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.pin)
        split_tag = split_tag_of(self.d)
        self.trk = Tracker.track_run("cardioseg", self.out.name, run_dir=self.out,
                             params={**cfg.model_dump(), "n_train": sh["n_train"], "n_val": len(sh["val_df"])},
                             tags={"split": split_tag, "seed": seed})
        self.log_sig = None
        if cfg.ef_learn and self.aux:
            self.log_sig = torch.zeros(1 + len(self.aux), device=self.device, requires_grad=True)
            self.opt.add_param_group({"params": [self.log_sig]})

    def _build_loss(self):  # pragma: no cover  (branches on the resident GPU Generator's partial/soft flags)
        """PARTIAL-LABEL mask -> PartialLabelDiceCE; soft-label -> SoftDiceCE; else the configured loss."""
        gen = self.gen
        partial = gen.valid is not None
        if partial:
            self.log.info("PARTIAL-LABEL loss: %d/%d slices class-masked",
                          int((~gen.valid.all(1)).sum()), gen.valid.shape[0])
            return partial, PartialLabelDiceCE()
        if self.cfg.generator.aug.soft_label_sigma > 0:
            return partial, SoftDiceCE()
        return partial, self.cfg.loss.build()

    def run(self):  # pragma: no cover  (drives the full GPU train->eval->save->finalize sequence)
        """train loop -> eval (val + held-out test) -> save -> finalize (artifacts+registry unless quick)."""
        best_state = self._train_loop()
        if best_state is not None:
            self.model.load_state_dict(best_state)
        results = self._evaluate()
        self._save(results)
        if self.quick:
            self.log.info("quick mode: skipping model card / attribution / ONNX / registry")
        else:
            self._finalize()
        self.trk.end()
        return self.model, results

    def _train_loop(self):  # pragma: no cover  (the GPU forward/backward epoch loop — needs a real model + resident batches)
        """The epoch loop for one seed — a long but LINEAR procedure (the training step, by nature): each
        epoch forward/loss/backward over the resident batches (+ the EF aux-lane nudge folded into one seg
        step), then a fast batched val-Dice for early stopping. Returns the best-val `state_dict` (or None)."""
        cfg, model, opt, scaler, loss_fn = self.cfg, self.model, self.opt, self.scaler, self.loss_fn
        partial, log_sig, log, trk = self.partial, self.log_sig, self.log, self.trk
        gen, aux, Xva, Yva = self.gen, self.aux, self.sh["Xva"], self.sh["Yva"]
        nb, pin, device = self.sh["nb"], self.pin, self.device
        best_dice, best_state, bad = -1.0, None, 0
        fit_t0 = time.perf_counter()                            # real training wall-clock (run-duration is unreliable)
        for ep in range(cfg.epochs):                            # cfg.epochs is a ceiling — early stopping bails sooner
            t0 = time.perf_counter()
            model.train()
            loss_fn.epoch = ep                                 # drives the HD-warmup ramp (dice_ce_hd); no-op for others
            tot = 0.0
            perm = torch.randperm(gen.n, device=gen.device)   # shuffle on the data's device
            for bi in progress(range(nb), f"epoch {ep}", total=nb):
                idx = perm[bi * cfg.batch:(bi + 1) * cfg.batch]
                x, yt, valid = gen.batch(idx, pin=pin)              # collapsed batch (+ partial-label mask)
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", enabled=pin):
                    loss = loss_fn(model(x), yt, valid) if partial else loss_fn(model(x), yt)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                tot += loss.item()
            if aux and ep >= cfg.ef_warmup:
                # EF/volume-consistency NUDGE: fold the aux lanes INTO one seg gradient step (not a separate
                # vote) — seg's dense signal keeps the direction, the lanes only bend the cavity volume.
                # Fixed λ nudge (seg dominant) or learned Kendall balance (log_sig); seg stays dominant.
                auxs = [l for l in (lane.loss(model, amp=pin) for lane in aux) if l is not None]
                if auxs:
                    opt.zero_grad(set_to_none=True)
                    with torch.autocast("cuda", enabled=pin):
                        seg = loss_fn(model(x), yt, valid) if partial else loss_fn(model(x), yt)
                    loss_j = (uncertainty_weighted([seg, *auxs], list(log_sig)) if log_sig is not None
                              else seg + cfg.ef_lambda * sum(auxs))     # learned Kendall | fixed nudge
                    scaler.scale(loss_j).backward()
                    scaler.step(opt)
                    scaler.update()
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
        return best_state

    def _evaluate(self) -> dict:  # pragma: no cover  (Evaluator runs GPU inference over the val/test frames)
        """VALIDATION + HELD-OUT TEST: one Evaluator, score val + the criteria test split, summarize."""
        model, device, d, log = self.model, self.device, self.d, self.log
        test_df = self.sh["test_df"]
        log.info("===== VALIDATION =====")
        ev = Evaluator(model, device, EvalCfg(size=d.size, boundary=not self.quick))  # skip HD95/ASSD on quick sweeps
        dice_per_class, ef_rows, surf = ev.validate(splits.paths(self.sh["val_df"]))
        results = {"val": summarize(dice_per_class, ef_rows, surf)}
        if len(test_df):                                            # held-out test = the criteria split
            log.info("===== HELD-OUT TEST: datasets=%s vendors=%s (n=%d) =====",
                     list(d.test_datasets), list(d.test_vendors), len(test_df))
            tdice, tef, tsurf = ev.validate(splits.paths(test_df))
            results["test"] = summarize(tdice, tef, tsurf)
        return results

    def _save(self, results: dict):  # pragma: no cover  (torch.save model.pth + metrics.json write + tracker summary)
        """Persist model.pth + metrics.json, then log the final per-axis summary to the tracker."""
        model, out, cfg, val_df = self.model, self.out, self.cfg, self.sh["val_df"]
        torch.save(model.state_dict(), out / "model.pth")  # out already created by to_json(config) above
        meta = {"config": cfg.model_dump(), "n_train": self.sh["n_train"], "n_val": len(val_df),
                "val_patients": val_df.get_column("subject_id").to_list(), "results": results}
        (out / "metrics.json").write_text(json.dumps(meta, indent=2))
        self.log.info("saved model + config + metrics -> %s/", out)
        self.trk.summary(results)                               # final per-axis dice/EF (metrics in the run)

    def _finalize(self):  # pragma: no cover  (model-card + attribution + ONNX export + mlflow registry save)
        """The non-quick artifact tail: build model card + attribution + ONNX (each best-effort, logged on
        failure), then register the COMPLETE set (model.pth + config + metrics + onnx + card) to the mlflow
        registry — the sole model store. alias='production' makes this the flagship."""
        model, out, log = self.model, self.out, self.log
        Xva, Yva, val_df, device = self.sh["Xva"], self.sh["Yva"], self.sh["val_df"], self.device
        cfg, d, alias, seed = self.cfg, self.d, self.alias, self.seed
        with self._artifact_step("model card"):
            ModelCard.generate(out)
            log.info("wrote %s/MODEL_CARD.md", out)
        with self._artifact_step("attribution"):               # attribution diagnostic (confusion + saliency)
            s = Attribution(model, device, cfg.model.out_channels).run(Xva, Yva, out)
            log.info("attribution: recall=%s saliency=%s -> %s/attribution.png", s["recall"], s["saliency"], out.name)
        with self._artifact_step("ONNX export"):
            export(out, splits.paths(val_df)[0])               # ONNX + INT8, parity-gated
        with self._artifact_step("registry save"):
            rid = mlflow.active_run().info.run_id if mlflow.active_run() else None
            split = "+".join(d.test_vendors) or "legacy"
            kind = "flagship" if alias == "production" else "candidate"
            save_model(out, run_name=out.name, run_id=rid, alias=alias,
                       description=f"{out.name} · split={split} · seed={seed}",
                       tags={"kind": kind, "split": split, "seed": seed})
            log.info("registered to mlflow registry '%s'%s", MODEL_NAME, f" (alias={alias})" if alias else "")

    # the realistic failure modes of the 4 artifact steps: file I/O (card write), torch/onnx + matplotlib
    # (attribution, ONNX export -> RuntimeError/ValueError), mlflow registry (MlflowException). NOT just
    # MlflowException — only the registry step is mlflow. A truly unexpected error propagates (surfaces the
    # bug); the trained model.pth is already on disk by now, so nothing is lost either way.
    _ARTIFACT_FAILURES = (OSError, RuntimeError, ValueError, MlflowException)

    @contextmanager
    def _artifact_step(self, name):  # pragma: no cover  (best-effort wrapper around the finalize artifact steps)
        """Run one post-training artifact step best-effort. By the time _finalize runs the model.pth is
        ALREADY saved, so a card / attribution / ONNX / registry hiccup (a missing optional dep, one bad
        val slice, a locked mlflow db) is logged and swallowed — the trained run is never lost."""
        try:
            yield
        except self._ARTIFACT_FAILURES as e:
            self.log.warning("%s skipped: %s", name, e)


def _legacy_resident(cfg: TrainCfg, train_df, val_df, data_device: str, device: str, log):  # noqa: PLR0913  # pragma: no cover  (loads slice tensors to VRAM + builds the Generator — needs the real store)
    """LEGACY criteria path (no coded --split): build the resident TRAIN tensors + shared Generator +
    val tensors. Optional synth-anatomy (Rodero SSM label maps; val/test stay REAL held-out): 'mix' =
    real + synth-anatomy UNION with synth rows force-painted (bd pwih); 'replace' = synth anatomy only,
    painted on a ZERO/procedural bg or on excised-real bg. Returns (gen, Xva, Yva)."""
    d = cfg.generator.data
    force_synth = None
    if d.anatomy_pool:
        Ys = torch.as_tensor(load_pool(d.anatomy_pool), dtype=torch.long, device=data_device)  # [N,H,W]
        if d.anatomy_mode == "mix":
            Xr, Yr = load_to_gpu(splits.paths(train_df), d.size, data_device)
            Xsy = torch.zeros((Ys.shape[0], 1, d.size, d.size), device=data_device)
            Xtr = torch.cat([Xr, Xsy]); Ytr = torch.cat([Yr, Ys])
            force_synth = torch.cat([torch.zeros(Xr.shape[0], dtype=torch.bool, device=data_device),
                                     torch.ones(Ys.shape[0], dtype=torch.bool, device=data_device)])
            log.info("ANATOMY MIX: %d real + %d synth-anatomy (bg=%s)", Xr.shape[0], Ys.shape[0],
                     cfg.generator.synth.bg.mode)
        else:                                                        # "replace": synth anatomy ONLY
            Ytr = Ys
            cfg.generator.synth.synth_p = 1.0
            if cfg.generator.synth.bg.mode in ("flat", "procedural", "mrxcat"):
                Xtr = torch.zeros((Ytr.shape[0], 1, d.size, d.size), device=data_device)  # ZERO-REAL
                log.info("ANATOMY POOL: %d slices, ZERO-REAL bg=%s", Ytr.shape[0], cfg.generator.synth.bg.mode)
            else:                                                    # Rodero heart on real bg (excised)
                Xr, Yr = load_to_gpu(splits.paths(train_df), d.size, data_device)
                Xr = excise_heart(Xr, Yr)
                Xtr = Xr[torch.randint(Xr.shape[0], (Ytr.shape[0],), device=Xr.device)]
                log.info("ANATOMY POOL: %d Rodero on real bg (excised, %s)", Ytr.shape[0], cfg.generator.synth.bg.mode)
    else:
        Xtr, Ytr = load_to_gpu(splits.paths(train_df), d.size, data_device)
    Xva, Yva = load_to_gpu(splits.paths(val_df), d.size, data_device)
    gen = Generator(cfg.generator, Xtr, Ytr, cfg.model.out_channels, device, force_synth=force_synth)
    return gen, Xva, Yva


def _run_seeds(cfg: TrainCfg, seeds: list, sh: dict, alias: str | None, *, quick: bool):  # pragma: no cover  (per-seed GPU train loop)
    """Train each seed on the shared bundle. Between seeds (multi-seed A/B only) move the finished seed's
    weights to host + reclaim its CUDA pool, so a later seed's memory-heavy TTA test doesn't OOM (bd fpav)."""
    res = []
    for s in seeds:
        out = SeedTrainer(cfg, s, sh, alias, quick=quick).run()
        if len(seeds) > 1 and sh["device"] == "cuda":
            out[0].cpu()
            torch.cuda.empty_cache()
        res.append(out)
    return res[0] if len(seeds) == 1 else res


def train_seg(cfg: TrainCfg, alias: str | None = None, *, quick: bool = False, seeds=None):  # pragma: no cover  (composition root: store.load + split + VRAM preload + per-seed GPU training)
    """Train from one TrainCfg over one or more seeds. Returns (model, results) for a single seed, or a
    list of them for many. The resident data (store, split, preloaded tensors, aux EF lanes) is built
    ONCE and shared across seeds (`SeedTrainer` does the per-seed work) — N seeds cost 1×data + N×model,
    no per-seed disk reload or VRAM duplication. Artifacts register to the mlflow model registry (the
    sole store); `alias='production'` makes a run the flagship. Multi-seed needs a coded `--split`
    (legacy criteria splits key their partition on the seed, so they can't share one dataset)."""
    d = cfg.generator.data
    # staging dir (gitignored) — per-seed artifacts build under here, then register to mlflow (the
    # store). base_out itself only holds the shared load log; each seed gets base_out(_s<seed>).
    base_out = Path(cfg.out_dir or ".staging/run")
    base_out.mkdir(parents=True, exist_ok=True)
    log = setup(base_out / "load.log")          # shared data-loading phase (per-seed logs are separate)
    seeds = resolve_seeds(cfg.seed, seeds)
    check_multiseed_split(seeds, d.split)
    device = resolve_device(cfg.device)
    torch.backends.cudnn.benchmark = True       # fixed input size -> autotune fastest convs
    log.info("device=%s torch=%s cudnn.benchmark=%s seeds=%s | split=%s | criteria datasets=%s vendors=%s",
             device, torch.__version__, torch.backends.cudnn.benchmark, seeds,
             d.split or "(legacy criteria)", list(d.test_datasets), list(d.test_vendors))

    # split = criteria over the consolidated store (builds processed/<ds>/ if missing). A coded split
    # family may declare its own `sources` (e.g. static_all adds SCD) — load those, not just d.sources.
    srcs = list(d.sources)
    if d.split:
        srcs = list(load_split(parse_ref(d.split)[0]).sources or d.sources)
    with timed(log, "store.load + split"):
        meta = store.load(srcs, inplane=d.inplane, n4=d.n4, n4_params=d.n4_params,
                          workers=cfg.workers, nyul=d.nyul, norm=d.norm)
        if d.split:                                 # NEW-STYLE: a coded-filter family owns the partition
            r = resolve_cfg(d, meta)
            train_src, val_src = r.train, r.val      # Sources (static OR dynamic) -> the train_gen seam
            test_df = r.test.frame                   # test + val are always StaticSource (frozen real)
            val_df = r.val.frame                     # val is real -> its frame drives scoring/export/params
            train_df = r.train.frame if r.train.kind == "static" else None   # dynamic train has no frame (counts via tensors)
            log.info("split=%s@%s test_hash=%s | train=%s val=%s test n=%d",
                     d.split.split("@")[0], r.version, r.test_hash[:19], r.train.kind, r.val.kind, len(test_df))
        else:
            train_src = val_src = None               # legacy: DataCfg criteria + inline anatomy block
            train_df, val_df, test_df = splits.split_from_cfg(d, meta, seeds[0])   # single-seed only
    if cfg.n_patients:                          # debug cap — bound test + val (+ legacy train frame)
        n = cfg.n_patients
        test_df = test_df.head(n)
        if val_df is not None:                  # coded split OR legacy: cap the val frame the same way
            val_df = val_df.head(max(1, n))
        if train_src is None:                   # legacy criteria path also has a train frame to cap
            train_df = train_df.head(n)

    # Preload ALL slices into device memory (VRAM): after this, the epoch loop is pure GPU — index a
    # permutation, augment, train; zero per-epoch CPU/disk/host↔device copy. The slice set fits the
    # card (~3 GB at 256px). No DataLoader/workers (which on Windows pickle the whole RAM dataset per
    # worker and starve the GPU). Prefer fast all-GPU epochs over disk-streamed ones.
    pin = device == "cuda"
    # residency = where the preloaded tensors live (gpu=VRAM-resident / cpu=RAM, copied per batch).
    # gpu only makes sense with a cuda device; fall back to cpu residency otherwise.
    data_device = device if (cfg.residency == "gpu" and device == "cuda") else "cpu"
    with timed(log, f"preload slices (residency={cfg.residency}->{data_device})"):
        if train_src is not None:
            # NEW: each Source OWNS its batch engine (train_gen). No static/dynamic if, no force_synth in
            # the interface, no bg_mode poke — StaticSource = real + DR-aug, DynamicSource = synth painter.
            # residency governs the STATIC real pool (VRAM vs host RAM). A dynamic painter has no big real
            # pool; it must paint ON the compute device or its float batch mismatches the autocast Half model
            # (bd uoiu) — so residency=cpu is a no-op for dynamic (paint always on `device`).
            gen_device = data_device if train_src.kind == "static" else device
            gen = train_src.train_gen(d.size, gen_device, cfg.generator, cfg.model.out_channels)
            Xva, Yva = val_src.resident(d.size, data_device)
        else:
            gen, Xva, Yva = _legacy_resident(cfg, train_df, val_df, data_device, device, log)
    nb = max(1, gen.n // cfg.batch)
    log.info("engine train=%s | slices: %d train / %d val / %d test-subj (resident %s, compute %s)",
             (train_src.kind if train_src else "legacy"), gen.n, Xva.shape[0], len(test_df),
             data_device, device)
    # n_train in slices when the train source is dynamic (no patient frame); else patient rows.
    n_train = n_train_of(train_src, gen, train_df)
    # Auxiliary EF lanes (GPU-resident, built ONCE and shared across seeds) — a list the epoch loop
    # iterates, so it never branches on cfg.ef_*. Empty when the lane is off / train source isn't static.
    aux = build_aux(cfg, splits, train_df, device,
                    is_static=train_src is not None and train_src.kind == "static")
    for lane in aux:
        log.info("aux lane: %s (%d items cached)", type(lane).__name__, lane.n)

    # Everything above is seed-invariant (coded split + resident tensors + aux). Hand the shared bundle
    # to a SeedTrainer per seed — each builds its own model/optimizer/artifacts on the SAME data.
    sh = {"device": device, "pin": pin, "data_device": data_device, "gen": gen, "Xva": Xva, "Yva": Yva,
          "train_df": train_df, "val_df": val_df, "test_df": test_df, "train_src": train_src,
          "aux": aux, "nb": nb, "n_train": n_train, "base_out": base_out, "single": len(seeds) == 1}
    log.info("shared data ready — training %d seed(s): %s", len(seeds), seeds)
    return _run_seeds(cfg, seeds, sh, alias, quick=quick)


if __name__ == "__main__":
    # Defaults = the generalization split (hold out ACDC + Canon). Change the split via the criteria
    # on DataCfg with --set, e.g. legacy train M&M-2 -> test ACDC:
    #   --set generator.data.sources=('mnm2','acdc') generator.data.test_datasets=('acdc',) generator.data.test_vendors=()
    ap = argparse.ArgumentParser(description="train a 2D U-Net from a TrainCfg (split = DataCfg criteria)")
    ap.add_argument("--epochs", type=int); ap.add_argument("--batch", type=int)
    ap.add_argument("--patience", type=int); ap.add_argument("--workers", type=int)
    ap.add_argument("--seed", type=int); ap.add_argument("--n-patients", type=int, dest="n_patients")
    ap.add_argument("--seeds", default=None,
                    help="comma-sep seeds to train on ONE shared dataset (e.g. '0,1,2') — multi-seed A/B "
                         "in one process, no per-seed reload/VRAM dup. Needs a coded --split.")
    ap.add_argument("--ef-lambda", type=float, dest="ef_lambda",
                    help="weight of the EF/volume-consistency auxiliary lane (0 = off)")
    ap.add_argument("--ef-learn", action="store_true", dest="ef_learn",
                    help="LEARN the seg-vs-EF balance (Kendall) instead of the fixed ef_lambda")
    ap.add_argument("--ef-kaggle", action="store_true", dest="ef_kaggle",
                    help="add the Kaggle EF-only cases to the vol lane (EF-ratio weak supervision)")
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
    args = ap.parse_args()

    cfg = TrainCfg()
    if args.split:                                      # coded-filter family owns the partition (core.data.ingest.splits)
        name = args.split.split("@", 1)[0]
        if name not in list_splits():
            raise SystemExit(f"unknown split {name!r}; known: {list_splits()}")
        cfg.generator.data.split = args.split
    apply_cli_args(cfg, args)
    train_seg(cfg, alias=args.alias, quick=args.quick, seeds=parse_seeds(args.seeds))
