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
    """Begin a fresh tracked run (local mlruns/). Returns a handle; no-op if tracking is off."""
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


def track_run(experiment: str, run_name: str, run_dir=None, params: dict | None = None):
    """Resume the run tied to `run_dir` (via runs/<name>/.mlflow_run_id) if it exists, else start a
    fresh one and persist its id there. Lets the post-hoc eval (results/uncertainty/calibrate) log the
    CANONICAL numbers into the SAME run train.py created — so the UI compares real numbers, not just
    training curves."""
    mlflow = _mlflow()
    if mlflow is None:
        return _Noop()
    try:
        _MLRUNS.mkdir(exist_ok=True)
        mlflow.set_tracking_uri(_MLRUNS.as_uri())
        mlflow.set_experiment(experiment)
        idf = Path(run_dir) / ".mlflow_run_id" if run_dir else None
        rid = idf.read_text().strip() if idf and idf.exists() else None
        if rid:
            mlflow.start_run(run_id=rid)                 # resume — don't re-log params
        else:
            mlflow.start_run(run_name=run_name)
            if params:
                mlflow.log_params(_flat(params))
            if idf:
                idf.write_text(mlflow.active_run().info.run_id)
        return _Live(mlflow)
    except Exception:
        return _Noop()


def backfill(experiment: str = "cardioseg"):
    """One-shot: log existing runs/<name>/{config,metrics}.json as runs, so the UI has history.
    Skips runs already tracked (have .mlflow_run_id). `python -m cardioseg.tracking`."""
    import json
    runs_dir = _ROOT / "runs"
    n = 0
    for mj in sorted(runs_dir.glob("*/metrics.json")):
        rd = mj.parent
        if (rd / ".mlflow_run_id").exists():
            continue
        m = json.loads(mj.read_text())
        trk = track_run(experiment, rd.name, run_dir=rd, params=m.get("config", {}))
        trk.summary(m.get("results", {}))               # val/test train-time numbers
        for f in ("config.json", "metrics.json"):
            trk.artifact(rd / f)
        trk.end()
        print(f"backfilled {rd.name}")
        n += 1
    print(f"backfilled {n} run(s)")


if __name__ == "__main__":
    backfill()
