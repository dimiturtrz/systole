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
from pathlib import Path

import numpy as np


def ensemble_decompose(models, vol_img, size, device):
    """Members = each model's TTA-mean softmax. Returns (pred, total, aleatoric, epistemic) maps in
    [0,1] (normalized by log C). epistemic = mutual information across the weight-diverse members."""
    import torch
    from core.inference import predict_volume_probs

    mems = [predict_volume_probs(m, vol_img, size, device)[1] for m in models]   # each [D,C,H,W]
    members = torch.stack(mems)                                                  # [K,D,C,H,W]
    mean = members.mean(0)
    logc = np.log(mean.shape[1])
    total = -(mean * (mean + 1e-12).log()).sum(1) / logc
    aleat = (-(members * (members + 1e-12).log()).sum(2)).mean(0) / logc
    epi = (total - aleat).clamp(min=0)
    pred = mean.argmax(1).to(torch.uint8)
    return pred.cpu().numpy(), total.cpu().numpy(), aleat.cpu().numpy(), epi.cpu().numpy()


def ensemble_score(models, df, size, device):
    """Canonical Dice (pooled ED+ES, per class) + EF MAE for the ensemble prediction (largest-CC,
    like the single-model pipeline). K=1 model -> the single-model score, so the same fn compares both."""
    import numpy as np
    from core.data.static import store
    from core.preprocessing.preprocess import stack_slices
    from core.data.static.labels import FOREGROUND, LV_CAV
    from core.postprocess import largest_cc_per_class
    from core.measure import ejection_fraction

    inter = {c: 0.0 for c in FOREGROUND}; den = {c: 0.0 for c in FOREGROUND}
    diffs = []
    for r in df.iter_rows(named=True):
        c = store.load_arrays(r["path"]); sp = tuple(float(s) for s in c["spacing"])
        preds, gts = {}, {}
        for tag in ("ed", "es"):
            if f"{tag}_img" not in c:
                continue
            pred = largest_cc_per_class(ensemble_decompose(models, c[f"{tag}_img"], size, device)[0])
            gt = stack_slices(c[f"{tag}_gt"], size, dtype=np.uint8)
            preds[tag], gts[tag] = pred, gt
            for cl in FOREGROUND:
                p, g = pred == cl, gt == cl
                inter[cl] += 2.0 * np.logical_and(p, g).sum(); den[cl] += p.sum() + g.sum()
        if "ed" in preds and "es" in preds:
            efp = ejection_fraction(preds["ed"], preds["es"], sp)[0]
            efg = ejection_fraction(gts["ed"], gts["es"], sp)[0]
            if not (np.isnan(efp) or np.isnan(efg)):
                diffs.append(efp - efg)
    dice = {cl: (inter[cl] / den[cl] if den[cl] else float("nan")) for cl in FOREGROUND}
    return {"dice_mean": round(float(np.nanmean(list(dice.values()))), 3),
            "ef_mae": round(float(np.mean(np.abs(diffs))), 1) if diffs else float("nan")}


def _eval_df(cfg, which):
    import polars as pl
    from core.data.static import store, splits
    d = cfg.generator.data
    meta = store.load(list(d.sources), inplane=d.inplane, n4=d.n4).filter(pl.col("labelled"))
    _, val, test = splits.make_split(meta, d.test_datasets, d.test_vendors, d.val_frac, 0,
                                     val_datasets=d.val_datasets, val_vendors=d.val_vendors)
    if which == "acdc":
        return val
    return test.filter(pl.col("vendor").str.to_lowercase() == which.lower())


def _headroom(models, df, size, device):
    """Foreground aleatoric/epistemic for the ensemble + the single-model (TTA) lower bound."""
    from core.data.static import store
    from core.preprocessing.preprocess import stack_slices
    from .uncertainty import tta_uncertainty
    ea, ee, ta, te = [], [], [], []
    for r in df.iter_rows(named=True):
        c = store.load_arrays(r["path"])
        for tag in ("ed", "es"):
            if f"{tag}_img" not in c:
                continue
            pred, _, ale, epi = ensemble_decompose(models, c[f"{tag}_img"], size, device)
            gt = stack_slices(c[f"{tag}_gt"], size, dtype=np.uint8)
            fg = (pred > 0) | (gt > 0)
            ea.append(ale[fg]); ee.append(epi[fg])
            _, _, _, sa, se_ = tta_uncertainty(models[0], c[f"{tag}_img"], size, device)
            ta.append(sa[fg]); te.append(se_[fg])
    f = lambda al, ep: (float(np.concatenate(ep).mean()) /
                        max(float(np.concatenate(al).mean()) + float(np.concatenate(ep).mean()), 1e-9))
    return f(ea, ee), f(ta, te)             # ensemble reducible-frac, single(TTA) reducible-frac


def main():
    from core.preprocessing.preprocess import SIZE
    from core.model import load_run, resolve_device
    from core.registry import resolve
    from ..tracking import start

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", required=True, help="K run dirs (different seeds)")
    ap.add_argument("--eval", nargs="+", default=["canon", "ge"], help="axes: canon ge acdc")
    a = ap.parse_args()
    device = resolve_device()
    loaded = [load_run(resolve(r), device) for r in a.runs]
    models = [m for m, _, _ in loaded]
    cfg = loaded[0][1]
    trk = start("cardioseg", f"ensemble-{len(models)}seed",
                {"members": len(models), "runs": ",".join(Path(r).name for r in a.runs)})
    for ax in a.eval:
        df = _eval_df(cfg, ax)
        if not len(df):
            continue
        ens = ensemble_score(models, df, SIZE, device)          # K models
        sgl = ensemble_score(models[:1], df, SIZE, device)      # single (gen)
        ef_red, tf_red = _headroom(models, df, SIZE, device)
        dd = ens["dice_mean"] - sgl["dice_mean"]
        print(f"axis {ax} (n={len(df)}, K={len(models)}): "
              f"Dice ensemble {ens['dice_mean']} vs single {sgl['dice_mean']} ({dd:+.3f}) | "
              f"EF MAE {ens['ef_mae']} vs {sgl['ef_mae']} | reducible {ef_red:.0%} (ensemble) / {tf_red:.0%} (TTA)")
        trk.metric(f"{ax}_dice_ensemble", ens["dice_mean"]); trk.metric(f"{ax}_dice_single", sgl["dice_mean"])
        trk.metric(f"{ax}_ef_ensemble", ens["ef_mae"]); trk.metric(f"{ax}_ef_single", sgl["ef_mae"])
        trk.metric(f"{ax}_reducible_ensemble", round(ef_red, 3)); trk.metric(f"{ax}_reducible_tta", round(tf_red, 3))
    trk.end()


if __name__ == "__main__":
    main()
