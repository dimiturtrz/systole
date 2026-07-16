"""Temperature scaling (Guo 2017) — post-hoc calibration of the softmax confidence.

Fit a single scalar T (logits -> logits/T) on a held-out set by minimizing NLL, with the model
frozen; T>1 softens overconfidence. It does NOT change argmax, so Dice/accuracy are untouched — only
the confidence numbers (ECE) move.

We fit T on VAL (the held-out ACDC centre) and report ECE before/after on val AND each test axis
(Canon, GE). The point of interest is the *transfer*: T calibrated in-domain typically fixes val but
not the unseen-vendor test — i.e. post-hoc calibration is itself domain-shift-limited.

    python -m cardioseg.evaluation.calibrate --run runs/gen
"""
import json
import logging

import numpy as np
import polars as pl
import torch
from jaxtyping import Float, Integer

from core.config import FLAGSHIP_REF
from core.data.ingest.splits import Splits
from core.data.static import splits
from core.data.static.store.build import Build as store
from core.registry import Registry
from core.run import Run
from core.types import shapecheck

from ..tracking import Tracker
from .uncertainty import Uncertainty
from .validate import EvalCfg, Evaluator

log = logging.getLogger("cardioseg.calibrate")


class Calibrate:
    """Temperature scaling (Guo 2017): fit one scalar T on val (LBFGS/NLL), report ECE before/after per
    axis. The free helpers folded in as staticmethods — the fit and the ECE-at-T evaluation."""

    @staticmethod
    @shapecheck
    def fit_temperature(
        logits: Float[np.ndarray, "*n c"], labels: Integer[np.ndarray, "*n"], device: str = "cpu"
    ) -> float:
        """T minimizing NLL of softmax(logits/T) vs labels (LBFGS). Model frozen; one scalar."""
        logits_tensor = torch.tensor(logits, dtype=torch.float32, device=device)
        labels_tensor = torch.tensor(labels, dtype=torch.long, device=device)
        log_temperature = torch.zeros(1, requires_grad=True, device=device)   # optimize log T -> T>0
        optimizer = torch.optim.LBFGS([log_temperature], lr=0.05, max_iter=80)
        cross_entropy = torch.nn.CrossEntropyLoss()

        def closure():
            optimizer.zero_grad()
            loss = cross_entropy(logits_tensor / log_temperature.exp(), labels_tensor)
            loss.backward()
            return loss

        optimizer.step(closure)
        return float(log_temperature.exp().detach())

    @staticmethod
    @shapecheck
    def _ece_at(logits: Float[np.ndarray, "*n c"], labels: Integer[np.ndarray, "*n"], temperature: float) -> float:
        """ECE of softmax(logits/T) vs labels (reuses uncertainty.ece on max-prob conf / correctness)."""
        scaled_logits = logits / temperature
        scaled_logits = scaled_logits - scaled_logits.max(1, keepdims=True)
        probs = np.exp(scaled_logits); probs /= probs.sum(1, keepdims=True)
        confidence, pred = probs.max(1), probs.argmax(1)
        return Uncertainty.ece(confidence, (pred == labels).astype(float))[0]

    @staticmethod
    def add_args(ap):
        ap.add_argument("--run", default=FLAGSHIP_REF)

    @staticmethod
    def run(args):  # pragma: no cover  CLI entrypoint: mlflow model loading (network) + GPU + tracking + file writes
        run = Registry.resolve(args.run)
        model, cfg, device = Run.load_run(run)
        data_cfg = cfg.generator.data
        meta = store.load_cfg(data_cfg).filter(pl.col("labelled"))   # all preprocessing params (nyul/norm too)
        if data_cfg.split:                                    # coded split -> its resolved val/test
            resolved = Splits.resolve_cfg(data_cfg, meta)
            val, test = resolved.val.frame, resolved.test.frame
        else:
            _, val, test = splits.Splits.make_split(
                meta, data_cfg.test_datasets, data_cfg.test_vendors, data_cfg.val_frac, 0,
                val_datasets=data_cfg.val_datasets, val_vendors=data_cfg.val_vendors)
        evaluator = Evaluator(model, device, EvalCfg(size=data_cfg.size))   # state (model/device/size) once; call many
        val_logits, val_labels = evaluator.gather(splits.Splits.paths(val))
        temperature = Calibrate.fit_temperature(val_logits, val_labels, device)

        axes = {"val": val}
        for vendor in data_cfg.test_vendors:             # report each test vendor separately
            axes[vendor] = test.filter(pl.col("vendor") == vendor)
        report = {"T": round(temperature, 3), "axes": {}}
        log.info(f"fitted T = {temperature:.3f} on val (n={len(val)})")
        for name, df in axes.items():
            if not len(df):
                continue
            logits, labels = (val_logits, val_labels) if name == "val" else evaluator.gather(splits.Splits.paths(df))
            ece_uncal, ece_temp = Calibrate._ece_at(logits, labels, 1.0), Calibrate._ece_at(logits, labels, temperature)
            report["axes"][name] = {"n": len(df), "ece_uncal": round(ece_uncal, 4), "ece_temp": round(ece_temp, 4)}
            log.info(f"  {name:8} (n={len(df):3}) ECE {ece_uncal:.3f} -> {ece_temp:.3f}  ({ece_temp-ece_uncal:+.3f})")
        (run / "plots").mkdir(parents=True, exist_ok=True)
        (run / "plots" / "calibration.json").write_text(json.dumps(report, indent=2))
        log.info(f"-> {run}/plots/calibration.json")

        tracker = Tracker("cardioseg", run.name)
        tracked_run = tracker.track_run(run_dir=run)      # resume the train run
        tracked_run.metric("temp_T", temperature)
        for name, axis_report in report["axes"].items():
            tracked_run.metric(f"{name}_ece_uncal", axis_report["ece_uncal"])
            tracked_run.metric(f"{name}_ece_temp", axis_report["ece_temp"])
        tracked_run.artifact(run / "plots" / "calibration.json")
        tracked_run.end()
