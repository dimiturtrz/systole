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
import logging

import numpy as np
import polars as pl
import torch

from core.config import FLAGSHIP_REF
from core.data.ingest.splits import resolve_cfg
from core.data.static import splits, store
from core.obs import setup
from core.registry import resolve
from core.run import load_run

from ..tracking import track_run
from .uncertainty import ece
from .validate import EvalCfg, Evaluator

log = logging.getLogger("cardioseg.calibrate")


def fit_temperature(logits: np.ndarray, labels: np.ndarray, device: str = "cpu") -> float:
    """T minimizing NLL of softmax(logits/T) vs labels (LBFGS). Model frozen; one scalar."""
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
    z = logits / T
    z = z - z.max(1, keepdims=True)
    p = np.exp(z); p /= p.sum(1, keepdims=True)
    conf, pred = p.max(1), p.argmax(1)
    return ece(conf, (pred == labels).astype(float))[0]


def main():
    setup()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default=FLAGSHIP_REF)
    a = ap.parse_args()
    run = resolve(a.run)
    model, cfg, device = load_run(run)
    d = cfg.generator.data
    meta = store.load_cfg(d).filter(pl.col("labelled"))   # all preprocessing params (nyul/norm too)
    if getattr(d, "split", ""):                           # coded split -> its resolved val/test
        r = resolve_cfg(d, meta)
        val, test = r.val.frame, r.test.frame
    else:
        _, val, test = splits.make_split(meta, d.test_datasets, d.test_vendors, d.val_frac, 0,
                                         val_datasets=d.val_datasets, val_vendors=d.val_vendors)
    ev = Evaluator(model, device, EvalCfg(size=d.size))   # state (model/device/size) once; call many
    val_logits, val_labels = ev.gather(splits.paths(val))
    T = fit_temperature(val_logits, val_labels, device)

    axes = {"val": val}
    for v in d.test_vendors:                         # report each test vendor separately
        axes[v] = test.filter(pl.col("vendor") == v)
    report = {"T": round(T, 3), "axes": {}}
    log.info(f"fitted T = {T:.3f} on val (n={len(val)})")
    for name, df in axes.items():
        if not len(df):
            continue
        lg, lb = (val_logits, val_labels) if name == "val" else ev.gather(splits.paths(df))
        e0, e1 = _ece_at(lg, lb, 1.0), _ece_at(lg, lb, T)
        report["axes"][name] = {"n": len(df), "ece_uncal": round(e0, 4), "ece_temp": round(e1, 4)}
        log.info(f"  {name:8} (n={len(df):3}) ECE {e0:.3f} -> {e1:.3f}  ({e1-e0:+.3f})")
    (run / "plots").mkdir(parents=True, exist_ok=True)
    (run / "plots" / "calibration.json").write_text(json.dumps(report, indent=2))
    log.info(f"-> {run}/plots/calibration.json")

    trk = track_run("cardioseg", run.name, run_dir=run)      # resume the train run
    trk.metric("temp_T", T)
    for name, ax in report["axes"].items():
        trk.metric(f"{name}_ece_uncal", ax["ece_uncal"]); trk.metric(f"{name}_ece_temp", ax["ece_temp"])
    trk.artifact(run / "plots" / "calibration.json")
    trk.end()


if __name__ == "__main__":
    main()
