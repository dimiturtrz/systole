"""Post-hoc EF bias calibration (bd cardiac-seg-tb58): fit a linear EF correction on VAL, apply once
to TEST — the cheapest lever on the derived EF number (masks good, ratio biased). No retrain.

EF is a ratio of predicted volumes, so a systematic under/over-fill of the ES cavity shows up as a
constant EF bias with the masks still scoring well on Dice. A held-out linear fit ef_corr = a*ef_pred
+ b (fit on VAL, the held-out ACDC centre) removes the bias without touching the model. The point of
interest is the TRANSFER: bias fit in-domain may or may not carry to the unseen-vendor test — post-hoc
calibration is itself domain-shift-limited (cf. temperature scaling in calibrate.py). Selection lives
on VAL only; TEST is scored once, uncalibrated vs calibrated, for the report (leak rule).

    python -m cardioseg.evaluation ef_calibrate --run runs/gen
"""
import json
import logging

import numpy as np
import polars as pl

from core.config import FLAGSHIP_REF
from core.data.ingest.splits import Splits
from core.data.static import splits
from core.data.static.store.build import Build as store
from core.measure import EfCalibration, Measure
from core.registry import Registry
from core.run import Run

from ..tracking import Tracker
from .validate import EvalCfg, Evaluator

log = logging.getLogger("cardioseg.ef_calibrate")


class EfCalibrate:
    """Linear EF-bias calibration: fit slope+intercept on VAL EF pairs, report Bland–Altman agreement
    (MAE/bias/LoA) uncalibrated vs calibrated on val + each test axis. Fit lives on VAL only."""

    @staticmethod
    def _ef_pairs(ef_rows) -> tuple[np.ndarray, np.ndarray]:
        """(ef_gt, ef_pred) arrays from an Evaluator.validate ef_rows list."""
        gt = np.array([r["ef_gt"] for r in ef_rows], dtype=float)
        pred = np.array([r["ef_pred"] for r in ef_rows], dtype=float)
        return gt, pred

    @staticmethod
    def _round2(pair) -> list[float]:
        """Round a [lo, hi] CI bracket to 1 dp for the report."""
        return [round(pair[0], 1), round(pair[1], 1)]

    @staticmethod
    def axis_report(cal: EfCalibration, ef_rows) -> dict:
        """Agreement stats before/after applying `cal` to one axis' EF pairs, each with a bootstrap 95% CI
        on MAE + bias (the defensible error bar a single held-out split otherwise lacks). Pure — the
        testable core. Lists are [uncalibrated, calibrated]; `*_ci` entries are [[lo, hi], [lo, hi]]."""
        gt, pred = EfCalibrate._ef_pairs(ef_rows)
        uncal = Measure.ef_statistics(gt, pred)
        recal = Measure.ef_statistics(gt, cal.apply(pred))
        ci_u = Measure.bootstrap_ef_ci(gt, pred)
        ci_c = Measure.bootstrap_ef_ci(gt, cal.apply(pred))
        return {
            "n": uncal.n,
            "mae": [round(uncal.mae, 1), round(recal.mae, 1)],
            "mae_ci": [EfCalibrate._round2(ci_u.mae_ci), EfCalibrate._round2(ci_c.mae_ci)],
            "bias": [round(uncal.bias, 1), round(recal.bias, 1)],
            "bias_ci": [EfCalibrate._round2(ci_u.bias_ci), EfCalibrate._round2(ci_c.bias_ci)],
            "loa": [[round(uncal.loa[0], 1), round(uncal.loa[1], 1)],
                    [round(recal.loa[0], 1), round(recal.loa[1], 1)]],
        }

    @staticmethod
    def add_args(ap):
        ap.add_argument("--run", default=FLAGSHIP_REF)

    @staticmethod
    def run(args):  # pragma: no cover  CLI entrypoint: mlflow model loading (network) + GPU + tracking + file writes
        run = Registry.resolve(args.run)
        model, cfg, device = Run.load_run(run)
        data_cfg = cfg.generator.data
        meta = store.load_cfg(data_cfg).filter(pl.col("labelled"))
        if data_cfg.split:
            resolved = Splits.resolve_cfg(data_cfg, meta)
            val, test = resolved.val.frame, resolved.test.frame
        else:
            _, val, test = splits.Splits.make_split(meta, data_cfg.test_datasets, data_cfg.test_vendors, data_cfg.val_frac, 0,
                                             val_datasets=data_cfg.val_datasets, val_vendors=data_cfg.val_vendors)
        evaluator = Evaluator(model, device, EvalCfg(size=data_cfg.size, boundary=False))   # EF-only -> skip EDT

        _, val_ef_rows, _ = evaluator.validate(splits.Splits.paths(val))
        val_gt, val_pred = EfCalibrate._ef_pairs(val_ef_rows)
        cal = Measure.fit_ef_calibration(val_gt, val_pred)
        log.info(f"fitted EF calibration on val (n={len(val_ef_rows)}): ef_corr = {cal.slope:.3f}*ef_pred + {cal.intercept:.2f}")

        axes = {"val": val_ef_rows}
        for vendor in data_cfg.test_vendors:
            vf = test.filter(pl.col("vendor") == vendor)
            if len(vf):
                _, axes[vendor], _ = evaluator.validate(splits.Splits.paths(vf))
        report = {"slope": round(cal.slope, 4), "intercept": round(cal.intercept, 4), "axes": {}}
        for name, ef_rows in axes.items():
            if not ef_rows:
                continue
            ax = EfCalibrate.axis_report(cal, ef_rows)
            report["axes"][name] = ax
            log.info(f"  {name:8} (n={ax['n']:3})  MAE {ax['mae'][0]:4.1f}->{ax['mae'][1]:4.1f} "
                     f"95%CI {ax['mae_ci'][1]}  bias {ax['bias'][0]:+5.1f}->{ax['bias'][1]:+5.1f} "
                     f"95%CI {ax['bias_ci'][1]}  LoA {ax['loa'][0]}->{ax['loa'][1]}")
        (run / "plots").mkdir(parents=True, exist_ok=True)
        (run / "plots" / "ef_calibration.json").write_text(json.dumps(report, indent=2))
        log.info(f"-> {run}/plots/ef_calibration.json")

        tracker = Tracker("cardioseg", run.name)
        tracked_run = tracker.track_run(run_dir=run)
        tracked_run.metric("ef_cal_slope", cal.slope); tracked_run.metric("ef_cal_intercept", cal.intercept)
        for name, ax in report["axes"].items():
            tracked_run.metric(f"{name}_ef_mae_uncal", ax["mae"][0]); tracked_run.metric(f"{name}_ef_mae_cal", ax["mae"][1])
        tracked_run.artifact(run / "plots" / "ef_calibration.json")
        tracked_run.end()
