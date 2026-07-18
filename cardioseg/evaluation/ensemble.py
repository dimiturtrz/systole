"""Deep-ensemble uncertainty decomposition — the *strong* epistemic estimate.

The single-model BALD in uncertainty.py uses the 4 TTA flips as a cheap ensemble (input-perturbation,
same weights) -> it UNDER-counts epistemic. A deep ensemble (K models, different seeds = different
weights) disagrees properly, so its mutual-information term is the honest reducible-uncertainty number.

Each member is itself a model's TTA-mean softmax, so this is "an ensemble of TTA models". We report
aleatoric / epistemic / epistemic_fraction on an eval axis, and the single-model (TTA) fraction beside
it — the gap is how much reducible headroom the weak estimate was hiding.

    python -m cardioseg.evaluation.ensemble --runs runs/gen runs/seed1 runs/seed2 runs/seed3 --eval canon
"""
import argparse
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import polars as pl
import torch
from jaxtyping import Float, Integer

from core.data.static import splits
from core.data.static.labels import FOREGROUND
from core.data.static.mri.base import Phase
from core.data.static.store.build import Build as store
from core.data.static.store.query import Recipe
from core.hparams import TrainCfg
from core.inference import Inference
from core.measure import Measure
from core.model import Model
from core.postprocess import Postprocess
from core.preprocessing.preprocess import SIZE, Preprocess
from core.registry import Registry
from core.run import Run
from core.types import shapecheck

from ..tracking import Tracker
from .uncertainty import Uncertainty

log = logging.getLogger("cardioseg.ensemble")


class Ensemble:
    """Deep-ensemble uncertainty decomposition + canonical scoring (the free helpers folded in as
    staticmethods): TTA-member BALD decompose, per-class Dice folds, EF-MAE finalize, reducible-fraction,
    and the eval-frame / headroom orchestration that the CLI drives."""

    @staticmethod
    @shapecheck
    def decompose(models: Sequence[torch.nn.Module], vol_img: Float[np.ndarray, "d h w"], size: int, device: str):
        """Members = each model's TTA-mean softmax. Returns (pred, total, aleatoric, epistemic) maps in
        [0,1] (normalized by log C). epistemic = mutual information across the weight-diverse members."""
        mems = [Inference(m, size, device).predict_volume_probs(vol_img)[1] for m in models]   # each [D,C,H,W]
        members = torch.stack(mems)                                                  # [K,D,C,H,W]
        mean = members.mean(0)
        logc = np.log(mean.shape[1])
        total = -(mean * (mean + 1e-12).log()).sum(1) / logc
        aleat = (-(members * (members + 1e-12).log()).sum(2)).mean(0) / logc
        epi = (total - aleat).clamp(min=0)
        pred = mean.argmax(1).to(torch.uint8)
        return pred.cpu().numpy(), total.cpu().numpy(), aleat.cpu().numpy(), epi.cpu().numpy()

    @staticmethod
    @shapecheck
    def _dice_fold(pred: Integer[np.ndarray, "*grid"], gt: Integer[np.ndarray, "*grid"], inter: dict[Any, float], den: dict[Any, float]) -> None:
        """Fold one (pred, gt) label-map pair into the running per-class Dice inter/den accumulators."""
        for cl in FOREGROUND:
            p, g = pred == cl, gt == cl
            inter[cl] += 2.0 * np.logical_and(p, g).sum(); den[cl] += p.sum() + g.sum()

    @staticmethod
    def _score_summary(inter: dict[Any, float], den: dict[Any, float], diffs: list[float]) -> dict[str, float | int]:
        """Finalize the ensemble accumulators: mean per-class Dice + EF MAE over the collected EF diffs."""
        dice = {cl: (inter[cl] / den[cl] if den[cl] else float("nan")) for cl in FOREGROUND}
        return {"dice_mean": round(float(np.nanmean(list(dice.values()))), 3),
                "ef_mae": round(float(np.mean(np.abs(diffs))), 1) if diffs else float("nan")}

    @staticmethod
    def score(models: Sequence[torch.nn.Module], df: pl.DataFrame, size: int, device: str) -> dict[str, float | int]:
        """Canonical Dice (pooled ED+ES, per class) + EF MAE for the ensemble prediction (largest-CC,
        like the single-model pipeline). K=1 model -> the single-model score, so the same fn compares both."""
        inter = dict.fromkeys(FOREGROUND, 0.0); den = dict.fromkeys(FOREGROUND, 0.0)
        diffs: list[float] = []
        for r in df.iter_rows(named=True):
            case = store.load_arrays(r["path"]); sp = tuple(float(s) for s in case["spacing"])
            preds: dict[str, Integer[np.ndarray, "d h w"]] = {}; gts: dict[str, Integer[np.ndarray, "d h w"]] = {}
            for tag in (p.lower() for p in Phase):
                if f"{tag}_img" not in case:
                    continue
                pred = Postprocess.largest_cc_per_class(Ensemble.decompose(models, case[f"{tag}_img"], size, device)[0])
                gt = Preprocess.stack_slices(case[f"{tag}_gt"], size, dtype=np.uint8)
                preds[tag], gts[tag] = pred, gt
                Ensemble._dice_fold(pred, gt, inter, den)
            if Phase.ED.lower() in preds and Phase.ES.lower() in preds:
                efp = Measure.ejection_fraction(preds[Phase.ED.lower()], preds[Phase.ES.lower()], sp)[0]
                efg = Measure.ejection_fraction(gts[Phase.ED.lower()], gts[Phase.ES.lower()], sp)[0]
                if not (np.isnan(efp) or np.isnan(efg)):
                    diffs.append(efp - efg)
        return Ensemble._score_summary(inter, den, diffs)

    @staticmethod
    def _eval_df(cfg: TrainCfg | None, which: str) -> pl.DataFrame:  # pragma: no cover  store.load + split resolution (disk/metadata I/O)
        d = cfg.generator.data
        meta = store.load(list(d.sources), Recipe(inplane=d.inplane, n4=d.n4)).filter(pl.col("labelled"))
        ms = splits.ModelSplit(d, meta)
        if which.lower() == "val":                          # the held-out val split (split-derived, not a literal)
            return ms.val
        test = ms.test                                             # a vendor axis carves the frozen test by vendor
        return test.filter(pl.col("vendor").str.to_lowercase() == which.lower())

    @staticmethod
    def reducible_frac(aleatoric: list[Float[np.ndarray, "..."]], epistemic: list[Float[np.ndarray, "..."]]) -> float:
        """epistemic / (aleatoric + epistemic) over pooled foreground samples (lists of arrays) —
        the reducible (model) fraction of total uncertainty. Guards the all-zero denominator."""
        a = float(np.concatenate(aleatoric).mean()); e = float(np.concatenate(epistemic).mean())
        return e / max(a + e, 1e-9)

    @staticmethod
    def _headroom(models: Sequence[torch.nn.Module], df: pl.DataFrame, size: int, device: str) -> tuple[float, float]:
        """Foreground aleatoric/epistemic for the ensemble + the single-model (TTA) lower bound."""
        ea: list[Float[np.ndarray, "..."]] = []; ee: list[Float[np.ndarray, "..."]] = []; ta: list[Float[np.ndarray, "..."]] = []; te: list[Float[np.ndarray, "..."]] = []
        for r in df.iter_rows(named=True):
            case = store.load_arrays(r["path"])
            for tag in (p.lower() for p in Phase):
                if f"{tag}_img" not in case:
                    continue
                pred, _, ale, epi = Ensemble.decompose(models, case[f"{tag}_img"], size, device)
                gt = Preprocess.stack_slices(case[f"{tag}_gt"], size, dtype=np.uint8)
                fg = (pred > 0) | (gt > 0)
                ea.append(ale[fg]); ee.append(epi[fg])
                _, _, _, sa, se_ = Uncertainty.tta_uncertainty(models[0], case[f"{tag}_img"], size, device)
                ta.append(sa[fg]); te.append(se_[fg])
        return Ensemble.reducible_frac(ea, ee), Ensemble.reducible_frac(ta, te)   # ensemble, single(TTA)

    @staticmethod
    def add_args(ap: argparse.ArgumentParser) -> None:
        ap.add_argument("--runs", nargs="+", required=True, help="K run dirs (different seeds)")
        ap.add_argument("--eval", nargs="+", default=["canon", "ge"], help="axes: canon ge val")

    @staticmethod
    def run(args: argparse.Namespace) -> None:  # pragma: no cover  CLI entrypoint: mlflow model loading (network) + GPU + tracking
        device = Model.resolve_device()
        loaded = [Run.load_run(Registry.resolve(r), device) for r in args.runs]
        models = [m for m, _, _ in loaded]
        cfg = loaded[0][1]
        tracker = Tracker("cardioseg", f"ensemble-{len(models)}seed",
                    {"members": len(models), "runs": ",".join(Path(r).name for r in args.runs)})
        trk = tracker.start()
        for ax in args.eval:
            df = Ensemble._eval_df(cfg, ax)
            if not len(df):
                continue
            ens = Ensemble.score(models, df, SIZE, device)          # K models
            sgl = Ensemble.score(models[:1], df, SIZE, device)      # single (gen)
            ef_red, tf_red = Ensemble._headroom(models, df, SIZE, device)
            dd = ens["dice_mean"] - sgl["dice_mean"]
            log.info(f"axis {ax} (n={len(df)}, K={len(models)}): "
                  f"Dice ensemble {ens['dice_mean']} vs single {sgl['dice_mean']} ({dd:+.3f}) | "
                  f"EF MAE {ens['ef_mae']} vs {sgl['ef_mae']} | reducible {ef_red:.0%} (ensemble) / {tf_red:.0%} (TTA)")
            trk.metric(f"{ax}_dice_ensemble", ens["dice_mean"]); trk.metric(f"{ax}_dice_single", sgl["dice_mean"])
            trk.metric(f"{ax}_ef_ensemble", ens["ef_mae"]); trk.metric(f"{ax}_ef_single", sgl["ef_mae"])
            trk.metric(f"{ax}_reducible_ensemble", round(ef_red, 3))
            trk.metric(f"{ax}_reducible_tta", round(tf_red, 3))
        trk.end()
