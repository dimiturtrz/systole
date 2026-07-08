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

import numpy as np
import polars as pl
import torch

from core.data.static import splits, store
from core.data.static.labels import FOREGROUND
from core.inference import predict_volume_probs
from core.measure import ejection_fraction
from core.model import resolve_device
from core.obs import setup
from core.postprocess import largest_cc_per_class
from core.preprocessing.preprocess import SIZE, stack_slices
from core.registry import resolve
from core.run import load_run

from ..tracking import start
from .uncertainty import tta_uncertainty

log = logging.getLogger("cardioseg.ensemble")


def ensemble_decompose(models, vol_img, size, device):
    """Members = each model's TTA-mean softmax. Returns (pred, total, aleatoric, epistemic) maps in
    [0,1] (normalized by log C). epistemic = mutual information across the weight-diverse members."""
    mems = [predict_volume_probs(m, vol_img, size, device)[1] for m in models]   # each [D,C,H,W]
    members = torch.stack(mems)                                                  # [K,D,C,H,W]
    mean = members.mean(0)
    logc = np.log(mean.shape[1])
    total = -(mean * (mean + 1e-12).log()).sum(1) / logc
    aleat = (-(members * (members + 1e-12).log()).sum(2)).mean(0) / logc
    epi = (total - aleat).clamp(min=0)
    pred = mean.argmax(1).to(torch.uint8)
    return pred.cpu().numpy(), total.cpu().numpy(), aleat.cpu().numpy(), epi.cpu().numpy()


def _dice_fold(pred, gt, inter, den):
    """Fold one (pred, gt) label-map pair into the running per-class Dice inter/den accumulators."""
    for cl in FOREGROUND:
        p, g = pred == cl, gt == cl
        inter[cl] += 2.0 * np.logical_and(p, g).sum(); den[cl] += p.sum() + g.sum()


def _score_summary(inter, den, diffs):
    """Finalize the ensemble accumulators: mean per-class Dice + EF MAE over the collected EF diffs."""
    dice = {cl: (inter[cl] / den[cl] if den[cl] else float("nan")) for cl in FOREGROUND}
    return {"dice_mean": round(float(np.nanmean(list(dice.values()))), 3),
            "ef_mae": round(float(np.mean(np.abs(diffs))), 1) if diffs else float("nan")}


def ensemble_score(models, df, size, device):
    """Canonical Dice (pooled ED+ES, per class) + EF MAE for the ensemble prediction (largest-CC,
    like the single-model pipeline). K=1 model -> the single-model score, so the same fn compares both."""
    inter = {c: 0.0 for c in FOREGROUND}; den = {c: 0.0 for c in FOREGROUND}
    diffs = []
    for r in df.iter_rows(named=True):
        case = store.load_arrays(r["path"]); sp = tuple(float(s) for s in case["spacing"])
        preds, gts = {}, {}
        for tag in ("ed", "es"):
            if f"{tag}_img" not in case:
                continue
            pred = largest_cc_per_class(ensemble_decompose(models, case[f"{tag}_img"], size, device)[0])
            gt = stack_slices(case[f"{tag}_gt"], size, dtype=np.uint8)
            preds[tag], gts[tag] = pred, gt
            _dice_fold(pred, gt, inter, den)
        if "ed" in preds and "es" in preds:
            efp = ejection_fraction(preds["ed"], preds["es"], sp)[0]
            efg = ejection_fraction(gts["ed"], gts["es"], sp)[0]
            if not (np.isnan(efp) or np.isnan(efg)):
                diffs.append(efp - efg)
    return _score_summary(inter, den, diffs)


def _eval_df(cfg, which):  # pragma: no cover  store.load + split resolution (disk/metadata I/O)
    d = cfg.generator.data
    meta = store.load(list(d.sources), inplane=d.inplane, n4=d.n4).filter(pl.col("labelled"))
    if which.lower() == "val":                          # the held-out val split (split-derived, not a literal)
        return splits.model_val(d, meta)
    test = splits.model_test(d, meta)                   # a vendor axis carves the frozen test by vendor
    return test.filter(pl.col("vendor").str.to_lowercase() == which.lower())


def reducible_frac(aleatoric, epistemic):
    """epistemic / (aleatoric + epistemic) over pooled foreground samples (lists of arrays) —
    the reducible (model) fraction of total uncertainty. Guards the all-zero denominator."""
    a = float(np.concatenate(aleatoric).mean()); e = float(np.concatenate(epistemic).mean())
    return e / max(a + e, 1e-9)


def _headroom(models, df, size, device):
    """Foreground aleatoric/epistemic for the ensemble + the single-model (TTA) lower bound."""
    ea, ee, ta, te = [], [], [], []
    for r in df.iter_rows(named=True):
        case = store.load_arrays(r["path"])
        for tag in ("ed", "es"):
            if f"{tag}_img" not in case:
                continue
            pred, _, ale, epi = ensemble_decompose(models, case[f"{tag}_img"], size, device)
            gt = stack_slices(case[f"{tag}_gt"], size, dtype=np.uint8)
            fg = (pred > 0) | (gt > 0)
            ea.append(ale[fg]); ee.append(epi[fg])
            _, _, _, sa, se_ = tta_uncertainty(models[0], case[f"{tag}_img"], size, device)
            ta.append(sa[fg]); te.append(se_[fg])
    return reducible_frac(ea, ee), reducible_frac(ta, te)   # ensemble reducible-frac, single(TTA) reducible-frac


def main():  # pragma: no cover  CLI entrypoint: mlflow model loading (network) + GPU + tracking
    setup()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", required=True, help="K run dirs (different seeds)")
    ap.add_argument("--eval", nargs="+", default=["canon", "ge"], help="axes: canon ge val")
    args = ap.parse_args()
    device = resolve_device()
    loaded = [load_run(resolve(r), device) for r in args.runs]
    models = [m for m, _, _ in loaded]
    cfg = loaded[0][1]
    trk = start("cardioseg", f"ensemble-{len(models)}seed",
                {"members": len(models), "runs": ",".join(Path(r).name for r in args.runs)})
    for ax in args.eval:
        df = _eval_df(cfg, ax)
        if not len(df):
            continue
        ens = ensemble_score(models, df, SIZE, device)          # K models
        sgl = ensemble_score(models[:1], df, SIZE, device)      # single (gen)
        ef_red, tf_red = _headroom(models, df, SIZE, device)
        dd = ens["dice_mean"] - sgl["dice_mean"]
        log.info(f"axis {ax} (n={len(df)}, K={len(models)}): "
              f"Dice ensemble {ens['dice_mean']} vs single {sgl['dice_mean']} ({dd:+.3f}) | "
              f"EF MAE {ens['ef_mae']} vs {sgl['ef_mae']} | reducible {ef_red:.0%} (ensemble) / {tf_red:.0%} (TTA)")
        trk.metric(f"{ax}_dice_ensemble", ens["dice_mean"]); trk.metric(f"{ax}_dice_single", sgl["dice_mean"])
        trk.metric(f"{ax}_ef_ensemble", ens["ef_mae"]); trk.metric(f"{ax}_ef_single", sgl["ef_mae"])
        trk.metric(f"{ax}_reducible_ensemble", round(ef_red, 3)); trk.metric(f"{ax}_reducible_tta", round(tf_red, 3))
    trk.end()


if __name__ == "__main__":
    main()
