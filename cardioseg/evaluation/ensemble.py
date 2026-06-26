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
    from .validate import predict_volume_probs

    mems = [predict_volume_probs(m, vol_img, size, device)[1] for m in models]   # each [D,C,H,W]
    members = torch.stack(mems)                                                  # [K,D,C,H,W]
    mean = members.mean(0)
    logc = np.log(mean.shape[1])
    total = -(mean * (mean + 1e-12).log()).sum(1) / logc
    aleat = (-(members * (members + 1e-12).log()).sum(2)).mean(0) / logc
    epi = (total - aleat).clamp(min=0)
    pred = mean.argmax(1).to(torch.uint8)
    return pred.cpu().numpy(), total.cpu().numpy(), aleat.cpu().numpy(), epi.cpu().numpy()


def _eval_df(cfg, which):
    import polars as pl
    from ..data import store, splits
    d = cfg.data
    meta = store.load(list(d.sources), inplane=d.inplane, n4=d.n4).filter(pl.col("labelled"))
    _, val, test = splits.make_split(meta, d.test_datasets, d.test_vendors, d.val_frac, 0,
                                     val_datasets=d.val_datasets, val_vendors=d.val_vendors)
    if which == "acdc":
        return val
    return test.filter(pl.col("vendor").str.to_lowercase() == which.lower())


def main():
    from ..data import store
    from ..training.dataset import fit_square, SIZE
    from ..training.model import load_run, resolve_device
    from .uncertainty import tta_uncertainty

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", required=True, help="K run dirs (different seeds)")
    ap.add_argument("--eval", default="canon", help="axis: canon | ge | acdc")
    a = ap.parse_args()
    device = resolve_device()
    loaded = [load_run(r, device) for r in a.runs]
    models = [m for m, _, _ in loaded]
    cfg = loaded[0][1]                                   # split criteria from the first run's config
    df = _eval_df(cfg, a.eval)

    ens_ale, ens_epi = [], []                            # deep-ensemble foreground voxels
    tta_ale, tta_epi = [], []                            # single-model (gen) TTA, same voxels
    for r in df.iter_rows(named=True):
        c = store.load_arrays(r["path"])
        for tag in ("ed", "es"):
            if f"{tag}_img" not in c:
                continue
            pred, _, ale, epi = ensemble_decompose(models, c[f"{tag}_img"], SIZE, device)
            gt = np.stack([fit_square(s, SIZE, 0) for s in c[f"{tag}_gt"]]).astype(np.uint8)
            fg = (pred > 0) | (gt > 0)
            ens_ale.append(ale[fg]); ens_epi.append(epi[fg])
            _, _, _, ta, te = tta_uncertainty(models[0], c[f"{tag}_img"], SIZE, device)
            tta_ale.append(ta[fg]); tta_epi.append(te[fg])

    def frac(ale, epi):
        a_, e_ = float(np.concatenate(ale).mean()), float(np.concatenate(epi).mean())
        return a_, e_, e_ / max(a_ + e_, 1e-9)
    ea, ee, ef = frac(ens_ale, ens_epi)
    ta, te, tf = frac(tta_ale, tta_epi)
    print(f"axis {a.eval} (n={len(df)}), K={len(models)} models")
    print(f"  deep-ensemble : aleatoric {ea:.3f} / epistemic {ee:.3f}  -> {ef:.0%} reducible")
    print(f"  single (TTA)  : aleatoric {ta:.3f} / epistemic {te:.3f}  -> {tf:.0%} reducible")
    print(f"  => weak TTA hid {ef - tf:+.0%} of reducible headroom" if ef > tf else
          f"  => TTA estimate already near-ensemble ({ef:.0%} vs {tf:.0%})")


if __name__ == "__main__":
    main()
