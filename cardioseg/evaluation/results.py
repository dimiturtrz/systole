"""Emit cardioseg/RESULTS.json — the ONE canonical source of published numbers (flagship ACDC + Canon,
the nnU-Net baseline, efficiency). `cardioseg/evaluation/sync_numbers.py` renders the doc tables/stats from this,
so a number lives in a single place and the docs can't drift. Regenerate after any eval:

    python -m cardioseg.evaluation results --run runs/gen

All flagship numbers pool ED+ES (the honest read). nnU-Net numbers are read from
baselines/nnunet/results.json (emitted by baselines/nnunet/score.py --out, ED+ES) — single source,
no hand-copy; refresh by re-running that baseline's score.py.
"""
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import mlflow
import numpy as np
import polars as pl
from mlflow.exceptions import MlflowException

from cardioseg.evaluation.distribution import Distribution
from core.config import FLAGSHIP_REF
from core.data.static import splits
from core.data.static.store.build import Build as store
from core.data.static.store.query import Recipe
from core.evaluate import CLASSES, Evaluate, SurfaceMetrics
from core.hparams import Hparams
from core.measure import AgreementStats, EfCalibration, Measure
from core.model import Model
from core.registry import _DB_URI, Registry

log = logging.getLogger("cardioseg.results")

ROOT = Path(__file__).resolve().parents[2]  # repo root (…/cardioseg/evaluation/ -> repo)

# nnU-Net v2 (2d, fold0, 50ep) on the SAME split, scored by baselines/nnunet/score.py (ED+ES).
_bj = json.loads((ROOT / "baselines/nnunet/results.json").read_text())
# nnU-Net axes = the held-out vendors it was tested on (Canon, GE) — same split + macro eval as ours.
NNUNET = {v: {"dice": _bj[v]["dice"], "hd95": _bj[v]["hd95"],
              "ef_mae": _bj[v]["ef_mae"], "ef_bias": _bj[v]["ef_bias"]}
          for v in ("canon", "ge") if v in _bj}
# Architectural constants (fvcore single-forward), not a baseline measurement — stay static.
EFFICIENCY = {"ours": {"params": "1.6 M", "flops": "0.8 G"},
              "nnunet": {"params": "92 M", "flops": "19 G"}}
_NAMES = [CLASSES[c][0] for c in CLASSES]  # RV, LV-myo, LV-cav


@dataclass(frozen=True)
class _Collected:
    """One axis' GPU-collected pieces, held raw so the VAL calibration can be fit before assembly."""
    n: int
    dists: dict
    dice_acc: dict
    ef_gt: np.ndarray
    ef_pred: np.ndarray
    strata: dict | None


class Results:
    """Emits the canonical RESULTS.json axes: pure axis-record assembly (`axis_dict`) + the GPU-shell
    `_axis`/`build` that feed it. Grouped so the published-JSON shape lives with the eval that fills it."""

    @staticmethod
    def axis_dict(n_rows: int, dists: dict, dice_acc: dict, ef_stats: AgreementStats,
                  cal_stats: AgreementStats | None = None) -> dict:
        """The pure axis-record assembler: pooled per-class boundary dists + per-class dice lists + the EF
        stats dict -> the published axis dict (per-class Dice/HD95/ASSD at their fixed rounding + mean Dice +
        EF MAE/bias/LoA). No model, no store — extracted from `_collect` so the exact JSON shape + rounding
        (the thing the docs read) is testable off synthetic pooled arrays; `build` supplies these from the
        GPU `collect` (the shell) and appends `strata` after. `cal_stats` (EF stats after the VAL-fit linear
        correction) adds the disclosed `ef_*_cal` companions alongside the raw EF — never replaces them."""
        dice, hd95, assd = {}, {}, {}
        for cl, (name, _) in CLASSES.items():
            pooled = (
                np.concatenate([d for d in dists[cl] if d.size]) if any(d.size for d in dists[cl]) else np.array([])
            )
            nan = float("nan")
            surf = Evaluate.surface_metrics(pooled) if pooled.size else SurfaceMetrics(nan, nan, nan)
            dice[name] = round(float(np.mean(dice_acc[cl])), 3)
            hd95[name] = round(float(surf.hd95), 1)
            assd[name] = round(float(surf.assd), 2)
        out = {"n": n_rows, "dice": {**dice, "mean": round(float(np.mean(list(dice.values()))), 3)},
               "hd95": hd95, "assd": assd, "ef_mae": round(ef_stats.mae, 1),
               "ef_bias": round(ef_stats.bias, 1),
               "ef_loa": [round(ef_stats.loa[0], 1), round(ef_stats.loa[1], 1)]}
        if cal_stats is not None:
            out |= {"ef_mae_cal": round(cal_stats.mae, 1), "ef_bias_cal": round(cal_stats.bias, 1),
                    "ef_loa_cal": [round(cal_stats.loa[0], 1), round(cal_stats.loa[1], 1)]}
        return out

    @staticmethod
    def _collect(run: Path, device: str, df, *, with_strata: bool) -> "_Collected":  # pragma: no cover
        """GPU shell for one axis: pooled boundary dists + per-class dice + EF pairs (+ pathology strata).
        Returns raw pieces so `build` can fit the calibration on VAL before assembling any axis."""
        rows = Distribution.collect(run, device, df.iter_rows(named=True))
        dists, dice_acc, ef_gt, ef_pred = Distribution.pooled(rows)
        strata = Distribution.strata_table(rows, "pathology") if with_strata else None
        return _Collected(len(rows), dists, dice_acc, np.asarray(ef_gt), np.asarray(ef_pred), strata)

    @staticmethod
    def _axis(c: "_Collected", cal: EfCalibration) -> dict:
        """Assemble one axis dict from collected pieces + the VAL-fit calibration (raw EF stats plus the
        disclosed calibrated companions). Pure over `_Collected` — the GPU cost lives in `_collect`."""
        ef = Measure.ef_statistics(c.ef_gt, c.ef_pred)
        cal_stats = Measure.ef_statistics(c.ef_gt, cal.apply(c.ef_pred))
        out = Results.axis_dict(c.n, c.dists, c.dice_acc, ef, cal_stats)
        if c.strata is not None:
            out["strata"] = c.strata
        return out

    @staticmethod
    def build(run: Path) -> dict:  # pragma: no cover  (store.load + make_split need the real data tree on disk)
        """Axes derived from the run's own split: VAL = ACDC (held-out centre, with pathology strata);
        TEST = each held-out vendor (Canon, GE) separately. So the published numbers always match what
        the run actually held out. EF calibration is fit on VAL only (leak rule) and applied to every axis;
        each axis reports raw EF plus a disclosed `ef_*_cal` companion — the reported EF stays uncalibrated,
        the correction is surfaced, not silently applied."""
        device = Model.resolve_device()
        d = Hparams.from_json(run / "config.json").generator.data
        meta = store.load(list(d.sources), Recipe(inplane=d.inplane, n4=d.n4)).filter(pl.col("labelled"))
        _, val, test = splits.Splits.make_split(meta, d.test_datasets, d.test_vendors, d.val_frac, 0,
                                         val_datasets=d.val_datasets, val_vendors=d.val_vendors)
        collected = {"acdc": Results._collect(run, device, val, with_strata=True)}   # val = ACDC, with strata
        for v in d.test_vendors:                                         # test = unseen vendors, each its own axis
            collected[v.lower()] = Results._collect(run, device, test.filter(pl.col("vendor") == v), with_strata=False)
        vc = collected["acdc"]
        cal = Measure.fit_ef_calibration(vc.ef_gt, vc.ef_pred)           # fit on VAL only (leak-safe)
        flagship = {k: Results._axis(c, cal) for k, c in collected.items()}
        return {
            "_note": "Canonical published numbers — generated by cardioseg.evaluation.results; do not hand-edit.",
            "run": run.name,
            "split": {"val": "acdc (held-out centre)", "test": "+".join(d.test_vendors) + " (unseen vendors)"},
            "ef_calibration": {
                "slope": round(float(cal.slope), 4), "intercept": round(float(cal.intercept), 4),
                "note": "ef_*_cal = post-hoc linear EF correction ef_corr=slope*ef_pred+intercept, fit on VAL "
                        "(acdc) only, applied to all axes. Reported EF stays the raw ef_*; calibration is "
                        "disclosed alongside, not substituted. See interpretations/ef/2026-07-15_ef_defensibility.md.",
            },
            "flagship": flagship,
            "nnunet": NNUNET,        # NOTE: still on the OLD split (ACDC-as-test) until re-run — provisional
            "efficiency": EFFICIENCY,
        }

    @staticmethod
    def add_args(ap):
        ap.add_argument("--run", default=FLAGSHIP_REF)
        ap.add_argument("--out", default="cardioseg/RESULTS.json")

    @staticmethod
    def run(args):  # pragma: no cover  (CLI: resolve registry ref + GPU build + mlflow metric logging + file write)
        res = Results.build(Registry.resolve(args.run))
        Path(args.out).write_text(json.dumps(res, indent=2))
        f = res["flagship"]
        log.info(f"wrote {args.out}: " + " · ".join(
            f"{k.upper()} mean {v['dice']['mean']}/EF {v['ef_mae']}%" for k, v in f.items()))

        # log the CANONICAL per-axis numbers into the model's registry run (resolve ref -> run-id)
        try:
            mlflow.set_tracking_uri(_DB_URI)
            with mlflow.start_run(run_id=Registry.run_id_for(args.run)):
                for ax, v in f.items():
                    mlflow.log_metric(f"{ax}_dice_mean", v["dice"]["mean"])
                    mlflow.log_metric(f"{ax}_ef_mae", v["ef_mae"])
                    mlflow.log_metric(f"{ax}_ef_bias", v["ef_bias"])
                mlflow.log_artifact(args.out)
        except MlflowException as e:
            log.warning("mlflow metric logging skipped: %s", e)
