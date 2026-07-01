"""Temperature scaling (Guo 2017) — post-hoc calibration of the softmax confidence.

Fit a single scalar T (logits -> logits/T) on a held-out set by minimizing NLL, with the model
frozen; T>1 softens overconfidence. It does NOT change argmax, so Dice/accuracy are untouched — only
the confidence numbers (ECE) move.

We fit T on VAL (the held-out ACDC centre) and report ECE before/after on val AND each test axis
(Canon, GE). The point of interest is the *transfer*: T calibrated in-domain typically fixes val but
not the unseen-vendor test — i.e. post-hoc calibration is itself domain-shift-limited.

    python -m cardioseg.evaluation.calibrate --run runs/gen
"""
import argparse
import json
from pathlib import Path

import numpy as np


def _gather(model, paths, size, device, per_vol=4000, seed=0):
    """Foreground (logits[N,C], labels[N]) over the given subjects (single forward, no TTA),
    subsampled to ~per_vol voxels/volume — plenty for a 1-param fit + ECE, bounded memory."""
    import torch
    from core.data.static import store
    from core.preprocessing.preprocess import fit_square

    rng = np.random.RandomState(seed)
    L, Y = [], []
    model.eval()
    for p in paths:
        c = store.load_arrays(p)
        for tag in ("ed", "es"):
            if f"{tag}_img" not in c:
                continue
            xs = np.stack([fit_square(s.astype(np.float32), size, 0.0) for s in c[f"{tag}_img"]])
            gt = np.stack([fit_square(s, size, 0) for s in c[f"{tag}_gt"]]).astype(np.int64)
            with torch.no_grad():
                logits = model(torch.from_numpy(xs)[:, None].to(device))   # [D,C,H,W]
            logits = logits.permute(0, 2, 3, 1).reshape(-1, logits.shape[1]).cpu().numpy()  # [Npix,C]
            y = gt.reshape(-1)
            pred = logits.argmax(1)
            fg = (y > 0) | (pred > 0)
            idx = np.where(fg)[0]
            if idx.size > per_vol:
                idx = rng.choice(idx, per_vol, replace=False)
            L.append(logits[idx]); Y.append(y[idx])
    return np.concatenate(L), np.concatenate(Y)


def fit_temperature(logits: np.ndarray, labels: np.ndarray, device: str = "cpu") -> float:
    """T minimizing NLL of softmax(logits/T) vs labels (LBFGS). Model frozen; one scalar."""
    import torch
    z = torch.tensor(logits, dtype=torch.float32, device=device)
    y = torch.tensor(labels, dtype=torch.long, device=device)
    logT = torch.zeros(1, requires_grad=True, device=device)   # optimize log T -> T>0
    opt = torch.optim.LBFGS([logT], lr=0.05, max_iter=80)
    nll = torch.nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        loss = nll(z / logT.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(logT.exp().detach())


def _ece_at(logits: np.ndarray, labels: np.ndarray, T: float) -> float:
    """ECE of softmax(logits/T) vs labels (reuses uncertainty.ece on max-prob conf / correctness)."""
    from .uncertainty import ece
    z = logits / T
    z = z - z.max(1, keepdims=True)
    p = np.exp(z); p /= p.sum(1, keepdims=True)
    conf, pred = p.max(1), p.argmax(1)
    return ece(conf, (pred == labels).astype(float))[0]


def main():
    import torch
    import polars as pl
    from core.data.static import store, splits
    from core.hparams import from_json
    from core.model import load_run
    from core.registry import resolve
    from core.config import FLAGSHIP_REF

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default=FLAGSHIP_REF)
    a = ap.parse_args()
    run = resolve(a.run)
    model, cfg, device = load_run(run)
    d = cfg.generator.data
    meta = store.load(list(d.sources), inplane=d.inplane, n4=d.n4).filter(pl.col("labelled"))
    _, val, test = splits.make_split(meta, d.test_datasets, d.test_vendors, d.val_frac, 0,
                                     val_datasets=d.val_datasets, val_vendors=d.val_vendors)
    size = d.size

    val_logits, val_labels = _gather(model, splits.paths(val), size, device)
    T = fit_temperature(val_logits, val_labels, device)

    axes = {"val": val}
    for v in d.test_vendors:                         # report each test vendor separately
        axes[v] = test.filter(pl.col("vendor") == v)
    report = {"T": round(T, 3), "axes": {}}
    print(f"fitted T = {T:.3f} on val (n={len(val)})")
    for name, df in axes.items():
        if not len(df):
            continue
        lg, lb = (val_logits, val_labels) if name == "val" else _gather(model, splits.paths(df), size, device)
        e0, e1 = _ece_at(lg, lb, 1.0), _ece_at(lg, lb, T)
        report["axes"][name] = {"n": len(df), "ece_uncal": round(e0, 4), "ece_temp": round(e1, 4)}
        print(f"  {name:8} (n={len(df):3}) ECE {e0:.3f} -> {e1:.3f}  ({e1-e0:+.3f})")
    (run / "plots").mkdir(parents=True, exist_ok=True)
    (run / "plots" / "calibration.json").write_text(json.dumps(report, indent=2))
    print(f"-> {run}/plots/calibration.json")

    from ..tracking import track_run
    trk = track_run("cardioseg", run.name, run_dir=run)      # resume the train run
    trk.metric("temp_T", T)
    for name, ax in report["axes"].items():
        trk.metric(f"{name}_ece_uncal", ax["ece_uncal"]); trk.metric(f"{name}_ece_temp", ax["ece_temp"])
    trk.artifact(run / "plots" / "calibration.json")
    trk.end()


if __name__ == "__main__":
    main()
