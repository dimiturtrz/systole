"""Optional MLflow experiment tracking — Phase 1: local file backend, runs/ stays canonical.

mlflow is a required dep; the only opt-out is CARDIOSEG_NO_MLFLOW, which makes every call a no-op so
training/scoring never depend on it. It's the cross-run comparison UI (`mlflow ui`), not the source of
truth — runs/<name>/{config,metrics}.json + RESULTS.json remain authoritative.

    trk = Tracker("cardioseg", run_name, params=cfg.model_dump()).start()
    trk.metric("val_dice", vd, step=ep)
    trk.summary({"test": {...}}); trk.artifact("runs/x/metrics.json"); trk.end()
"""
from __future__ import annotations

import os
from pathlib import Path

import mlflow
import mlflow.pytorch

_ROOT = Path(__file__).resolve().parents[1]
_MLRUNS = _ROOT / "mlruns"                          # artifact store (default root)
_DB_URI = f"sqlite:///{(_ROOT / 'mlflow.db').as_posix()}"   # metadata + registry (file store deprecated)


class _Noop:
    def metric(self, *a, **k): pass
    def summary(self, *a, **k): pass
    def artifact(self, *a, **k): pass
    def tag(self, *a, **k): pass
    def log_model(self, *a, **k): pass
    def end(self): pass


class _Live:
    def __init__(self, mlflow):
        self._m = mlflow

    def metric(self, key: str, value, step: int | None = None):
        try: self._m.log_metric(key, float(value), step=step)
        except Exception: pass

    def summary(self, results: dict):
        """Log the TRAIN-TIME per-axis scalars as fit_<axis>_<k> (e.g. fit_val_dice_mean). The 'fit_'
        prefix marks these as in-loop validate() numbers — distinct from the authoritative canonical
        metrics (acdc/canon/ge_*) that results.py logs. Don't let them masquerade as the test number."""
        for axis, d in (results or {}).items():
            if isinstance(d, dict):
                for k in ("dice_mean", "ef_mae"):
                    if isinstance(d.get(k), (int, float)):
                        self.metric(f"fit_{axis}_{k}", d[k])

    def artifact(self, path):
        try:
            if Path(path).exists():
                self._m.log_artifact(str(path))
        except Exception: pass

    def tag(self, key, value):
        try: self._m.set_tag(key, str(value))
        except Exception: pass

    def log_model(self, model, registered_name, alias=None, description=None, version_tags=None):
        """Log the torch model + register a version (catalog). `alias` (e.g. 'production') points at it;
        `description`/`version_tags` make the auto-numbered version readable. Guarded."""
        try:
            mlflow.pytorch.log_model(model, name="model", registered_model_name=registered_name)
            c = self._m.tracking.MlflowClient()
            v = str(max(int(mv.version) for mv in c.search_model_versions(f"name='{registered_name}'")))
            if description:
                c.update_model_version(registered_name, v, description=description)
            for k, val in (version_tags or {}).items():
                c.set_model_version_tag(registered_name, v, k, str(val))
            if alias:
                c.set_registered_model_alias(registered_name, alias, v)
        except Exception: pass

    def end(self):
        try: self._m.end_run()
        except Exception: pass


class Tracker:
    """A tracking RUN is a session: experiment + run_name + the params/tags logged once at open.
    Construct with that session config, then open the run — fresh via `start`, or resume-or-create
    tied to a `run_dir` via `track_run`. Both return a live handle (or a no-op if tracking's off).
    `_mlflow`/`_flat` stay static — pure helpers that thread no session state."""

    MODEL_NAME = "cardioseg-2dunet"        # the one deployable model line (registry lives in core.registry)

    def __init__(self, experiment: str, run_name: str, params: dict | None = None,
                 tags: dict | None = None):
        self.experiment, self.run_name = experiment, run_name
        self.params, self.tags = params, tags

    @staticmethod
    def _mlflow():
        """The mlflow module if tracking is enabled, else None (CARDIOSEG_NO_MLFLOW opt-out)."""
        if os.environ.get("CARDIOSEG_NO_MLFLOW"):
            return None
        return mlflow

    @staticmethod
    def _flat(d: dict, prefix: str = "") -> dict:
        """Flatten nested config dicts to dotted keys (mlflow params are scalar)."""
        out = {}
        for k, v in d.items():
            key = f"{prefix}{k}"
            if isinstance(v, dict):
                out.update(Tracker._flat(v, key + "."))
            else:
                out[key] = v
        return out

    def _configure(self, mlflow):
        """Shared open: point mlflow at the local store + this session's experiment + system metrics.
        Raises propagate to the caller's outer guard (a fatal setup error -> no-op handle)."""
        _MLRUNS.mkdir(exist_ok=True)
        mlflow.set_tracking_uri(_DB_URI)
        mlflow.set_experiment(self.experiment)
        try: mlflow.enable_system_metrics_logging()          # GPU/CPU/mem if psutil+pynvml present
        except Exception: pass

    def _apply_tags(self, mlflow):
        for k, v in (self.tags or {}).items():
            mlflow.set_tag(k, str(v))

    def start(self):
        """Begin a fresh tracked run (local mlruns/). Returns a handle; no-op if tracking is off."""
        mlflow = Tracker._mlflow()
        if mlflow is None:
            return _Noop()
        try:
            self._configure(mlflow)
            mlflow.start_run(run_name=self.run_name)
            if self.params:
                mlflow.log_params(Tracker._flat(self.params))
            self._apply_tags(mlflow)
            return _Live(mlflow)
        except Exception:
            return _Noop()      # tracking must never break a run

    def track_run(self, run_dir=None):
        """Resume the run tied to `run_dir` (via runs/<name>/.mlflow_run_id) if it exists, else start a
        fresh one and persist its id there. Lets the post-hoc eval (results/uncertainty/calibrate) log the
        CANONICAL numbers into the SAME run train.py created — so the UI compares real numbers, not just
        training curves."""
        mlflow = Tracker._mlflow()
        if mlflow is None:
            return _Noop()
        try:
            self._configure(mlflow)
            idf = Path(run_dir) / ".mlflow_run_id" if run_dir else None
            rid = idf.read_text().strip() if idf and idf.exists() else None
            if rid:
                mlflow.start_run(run_id=rid)                 # resume — don't re-log params
            else:
                mlflow.start_run(run_name=self.run_name)
                if self.params:
                    mlflow.log_params(Tracker._flat(self.params))
                if idf:
                    idf.write_text(mlflow.active_run().info.run_id)
            self._apply_tags(mlflow)
            return _Live(mlflow)
        except Exception:
            return _Noop()


# Model registration + resolution live in core.registry (the model store). train.py registers via
# core.registry.save_model; this module is now metrics/params/system-metrics tracking only. The old
# runs/-based backfill + recurate utilities were removed with runs/ (model store = the mlflow registry).
