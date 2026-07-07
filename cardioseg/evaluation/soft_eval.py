"""Soft-label evaluation (bd duw): for a run, compare EF read two ways + report calibration.

  - HARD EF  : argmax -> largest-CC -> voxel-count volume (the current readout)
  - SOFT EF  : expected volume = Σ blood-prob within the CC gate (collapse-never, late)
  - ECE      : calibration of the model's own softmax (the provable soft-label win)

Run on the hard baseline and the soft run and compare by hand:
    python -m cardioseg.evaluation.soft_eval --run runs/gen
    python -m cardioseg.evaluation.soft_eval --run runs/soft
EF-vs-GT is GT-bound (GT is hard, drawn on 10mm slices) — read EF as directional, ECE as the
clean number.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from cardioseg.evaluation.uncertainty import ece
from core.data.static import splits, store
from core.data.static.labels import LV_CAV
from core.hparams import from_json
from core.inference import predict_volume_probs
from core.measure import ef_statistics, expected_volume_ml, label_volume_ml
from core.model import load_run, resolve_device
from core.postprocess import largest_cc_per_class
from core.preprocessing.preprocess import SIZE, stack_slices
from core.registry import resolve


def _val(run: Path):
    d = from_json(run / "config.json").generator.data
    meta = store.load(list(d.sources), inplane=d.inplane).filter(pl.col("labelled"))
    _, val, _ = splits.make_split(meta, d.test_datasets, d.test_vendors, d.val_frac, 0,
                                  val_datasets=d.val_datasets, val_vendors=d.val_vendors)
    return val


def _ef(edv: float, esv: float) -> float:
    return (edv - esv) / edv * 100.0 if edv > 0 else float("nan")


def evaluate(run: Path):
    dev = resolve_device()
    model, _, _ = load_run(run, dev)
    rows, conf_all, corr_all = [], [], []
    for r in _val(run).iter_rows(named=True):
        c = store.load_arrays(r["path"])
        if "ed_img" not in c or "es_img" not in c:
            continue
        sp = tuple(float(s) for s in c["spacing"])
        vols = {}
        for tag in ("ed", "es"):
            _, mean = predict_volume_probs(model, c[f"{tag}_img"], SIZE, dev)   # [D,C,H,W] softmax
            p = mean.float().cpu().numpy()
            blood = p[:, LV_CAV]                                                 # [D,H,W] blood prob
            hard = largest_cc_per_class(p.argmax(1).astype(np.uint8))            # argmax + CC
            gate = hard == LV_CAV
            gt = stack_slices(c[f"{tag}_gt"], SIZE, dtype=np.uint8)
            vols[tag] = {"hard": label_volume_ml(hard, LV_CAV, sp),
                         "soft": expected_volume_ml(blood * gate, sp),
                         "gt": label_volume_ml(gt, LV_CAV, sp)}
            # calibration over foreground voxels (pred or gt non-bg)
            fg = (gt > 0) | (hard > 0)
            conf_all.append(p.max(1)[fg]); corr_all.append((p.argmax(1)[fg] == gt[fg]).astype(float))
        rows.append((_ef(vols["ed"]["gt"], vols["es"]["gt"]),
                     _ef(vols["ed"]["hard"], vols["es"]["hard"]),
                     _ef(vols["ed"]["soft"], vols["es"]["soft"])))
    a = np.array(rows)                                  # [n, 3] = gt, hard, soft
    conf = np.concatenate(conf_all); corr = np.concatenate(corr_all)
    return a, ece(conf, corr)[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    arr, e = evaluate(resolve(a.run))
    gt, hard, soft = arr[:, 0], arr[:, 1], arr[:, 2]
    print(f"\n=== {a.run}  (n={len(arr)}) ===")
    print(f"ECE: {e:.4f}")
    for name, pred in (("HARD (argmax+CC count)", hard), ("SOFT (expected vol, late)", soft)):
        s = ef_statistics(gt, pred)
        print(f"{name:28} EF MAE {s['mae']:5.1f}%  bias {s['bias']:+5.1f}%")


if __name__ == "__main__":
    main()
