"""Optional MLflow experiment tracking — Phase 1: local file backend, runs/ stays canonical.

A thin, GUARDED layer: if mlflow is absent or CARDIOSEG_NO_MLFLOW is set, every call is a no-op, so
training/scoring never depend on it. It's the cross-run comparison UI (`mlflow ui`), not the source of
truth — runs/<name>/{config,metrics}.json + RESULTS.json remain authoritative.

    trk = start("cardioseg", run_name, params=cfg.model_dump())
    trk.metric("val_dice", vd, step=ep)
    trk.summary({"test": {...}}); trk.artifact("runs/x/metrics.json"); trk.end()
"""
from __future__ import annotations

import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_MLRUNS = _ROOT / "mlruns"


def _mlflow():
    """The mlflow module if tracking is enabled, else None (absent dep or opt-out env)."""
    if os.environ.get("CARDIOSEG_NO_MLFLOW"):
        return None
    try:
        import mlflow
    except ImportError:
        return None
    return mlflow


def _flat(d: dict, prefix: str = "") -> dict:
    """Flatten nested config dicts to dotted keys (mlflow params are scalar)."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flat(v, key + "."))
        else:
            out[key] = v
    return out


class _Noop:
    def metric(self, *a, **k): pass
    def summary(self, *a, **k): pass
    def artifact(self, *a, **k): pass
    def end(self): pass


class _Live:
    def __init__(self, mlflow):
        self._m = mlflow

    def metric(self, key: str, value, step: int | None = None):
        try: self._m.log_metric(key, float(value), step=step)
        except Exception: pass

    def summary(self, results: dict):
        """Log final per-axis scalars (e.g. {'test': {'dice_mean':..,'ef_mae':..}}) as <axis>_<k>."""
        for axis, d in (results or {}).items():
            if isinstance(d, dict):
                for k in ("dice_mean", "ef_mae"):
                    if isinstance(d.get(k), (int, float)):
                        self.metric(f"{axis}_{k}", d[k])

    def artifact(self, path):
        try:
            if Path(path).exists():
                self._m.log_artifact(str(path))
        except Exception: pass

    def end(self):
        try: self._m.end_run()
        except Exception: pass


def start(experiment: str, run_name: str, params: dict | None = None):
    """Begin a tracked run (local mlruns/). Returns a handle; a no-op handle if tracking is off."""
    mlflow = _mlflow()
    if mlflow is None:
        return _Noop()
    try:
        _MLRUNS.mkdir(exist_ok=True)
        mlflow.set_tracking_uri(_MLRUNS.as_uri())
        mlflow.set_experiment(experiment)
        mlflow.start_run(run_name=run_name)
        if params:
            mlflow.log_params(_flat(params))
        return _Live(mlflow)
    except Exception:
        return _Noop()      # tracking must never break a run
