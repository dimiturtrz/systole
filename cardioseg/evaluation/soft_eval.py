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

import logging
from pathlib import Path

import numpy as np
import polars as pl

from cardioseg.evaluation.uncertainty import Uncertainty
from core.data.static import splits
from core.data.static.labels import LV_CAV
from core.data.static.store.build import Build as store
from core.data.static.store.query import Recipe
from core.hparams import Hparams
from core.inference import Inference
from core.measure import Measure
from core.model import Model
from core.postprocess import Postprocess
from core.preprocessing.preprocess import SIZE, Preprocess
from core.registry import Registry
from core.run import Run

log = logging.getLogger("cardioseg.soft_eval")


class SoftEval:
    """Soft-label EF readout + calibration for a run: hard EF (argmax+CC voxel count) vs soft EF (expected
    blood volume, collapse-never) plus the softmax ECE. Free helpers folded in as staticmethods."""

    @staticmethod
    def _val(run: Path):  # pragma: no cover  (store.load + split need the real data tree on disk)
        d = Hparams.from_json(run / "config.json").generator.data
        meta = store.load(list(d.sources), Recipe(inplane=d.inplane)).filter(pl.col("labelled"))
        _, val, _ = splits.Splits.make_split(meta, d.test_datasets, d.test_vendors, d.val_frac, 0,
                                      val_datasets=d.val_datasets, val_vendors=d.val_vendors)
        return val

    @staticmethod
    def _ef(edv: float, esv: float) -> float:
        return (edv - esv) / edv * 100.0 if edv > 0 else float("nan")

    @staticmethod
    def evaluate(run: Path):  # pragma: no cover  (loads the model + runs GPU inference over real val cases)
        dev = Model.resolve_device()
        model, _, _ = Run.load_run(run, dev)
        rows, conf_all, corr_all = [], [], []
        for r in SoftEval._val(run).iter_rows(named=True):
            case = store.load_arrays(r["path"])
            if "ed_img" not in case or "es_img" not in case:
                continue
            sp = tuple(float(s) for s in case["spacing"])
            vols = {}
            for tag in ("ed", "es"):
                _, mean = Inference(model, SIZE, dev).predict_volume_probs(case[f"{tag}_img"])   # [D,C,H,W] softmax
                p = mean.float().cpu().numpy()
                blood = p[:, LV_CAV]                                                 # [D,H,W] blood prob
                hard = Postprocess.largest_cc_per_class(p.argmax(1).astype(np.uint8))            # argmax + CC
                gate = hard == LV_CAV
                gt = Preprocess.stack_slices(case[f"{tag}_gt"], SIZE, dtype=np.uint8)
                vols[tag] = {"hard": Measure.label_volume_ml(hard, LV_CAV, sp),
                             "soft": Measure.expected_volume_ml(blood * gate, sp),
                             "gt": Measure.label_volume_ml(gt, LV_CAV, sp)}
                # calibration over foreground voxels (pred or gt non-bg)
                fg = (gt > 0) | (hard > 0)
                conf_all.append(p.max(1)[fg]); corr_all.append((p.argmax(1)[fg] == gt[fg]).astype(float))
            rows.append((SoftEval._ef(vols["ed"]["gt"], vols["es"]["gt"]),
                         SoftEval._ef(vols["ed"]["hard"], vols["es"]["hard"]),
                         SoftEval._ef(vols["ed"]["soft"], vols["es"]["soft"])))
        a = np.array(rows)                                  # [n, 3] = gt, hard, soft
        conf = np.concatenate(conf_all); corr = np.concatenate(corr_all)
        return a, Uncertainty.ece(conf, corr)[0]

    @staticmethod
    def add_args(ap):
        ap.add_argument("--run", required=True)

    @staticmethod
    def run(args):  # pragma: no cover  (CLI: resolve registry ref + GPU eval + log)
        arr, e = SoftEval.evaluate(Registry.resolve(args.run))
        gt, hard, soft = arr[:, 0], arr[:, 1], arr[:, 2]
        log.info(f"\n=== {args.run}  (n={len(arr)}) ===")
        log.info(f"ECE: {e:.4f}")
        for name, pred in (("HARD (argmax+CC count)", hard), ("SOFT (expected vol, late)", soft)):
            s = Measure.ef_statistics(gt, pred)
            log.info(f"{name:28} EF MAE {s['mae']:5.1f}%  bias {s['bias']:+5.1f}%")
