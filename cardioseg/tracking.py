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
_MLRUNS = _ROOT / "mlruns"                          # artifact store (default root)
_DB_URI = f"sqlite:///{(_ROOT / 'mlflow.db').as_posix()}"   # metadata + registry (file store deprecated)


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

    def tag(self, key, value):
        try: self._m.set_tag(key, str(value))
        except Exception: pass

    def log_model(self, model, registered_name, alias=None, description=None, version_tags=None):
        """Log the torch model + register a version (catalog). `alias` (e.g. 'production') points at it;
        `description`/`version_tags` make the auto-numbered version readable. Guarded."""
        try:
            import mlflow.pytorch
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


def start(experiment: str, run_name: str, params: dict | None = None, tags: dict | None = None):
    """Begin a fresh tracked run (local mlruns/). Returns a handle; no-op if tracking is off."""
    mlflow = _mlflow()
    if mlflow is None:
        return _Noop()
    try:
        _MLRUNS.mkdir(exist_ok=True)
        mlflow.set_tracking_uri(_DB_URI)
        mlflow.set_experiment(experiment)
        try: mlflow.enable_system_metrics_logging()      # GPU/CPU/mem if psutil+pynvml present
        except Exception: pass
        mlflow.start_run(run_name=run_name)
        if params:
            mlflow.log_params(_flat(params))
        for k, v in (tags or {}).items():
            mlflow.set_tag(k, str(v))
        return _Live(mlflow)
    except Exception:
        return _Noop()      # tracking must never break a run


def track_run(experiment: str, run_name: str, run_dir=None, params: dict | None = None,
              tags: dict | None = None):
    """Resume the run tied to `run_dir` (via runs/<name>/.mlflow_run_id) if it exists, else start a
    fresh one and persist its id there. Lets the post-hoc eval (results/uncertainty/calibrate) log the
    CANONICAL numbers into the SAME run train.py created — so the UI compares real numbers, not just
    training curves."""
    mlflow = _mlflow()
    if mlflow is None:
        return _Noop()
    try:
        _MLRUNS.mkdir(exist_ok=True)
        mlflow.set_tracking_uri(_DB_URI)
        mlflow.set_experiment(experiment)
        try: mlflow.enable_system_metrics_logging()      # GPU/CPU/mem if psutil+pynvml present
        except Exception: pass
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
        for k, v in (tags or {}).items():
            mlflow.set_tag(k, str(v))
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


MODEL_NAME = "cardioseg-2dunet"        # the one deployable model line


def run_kind(name: str, is_flagship: bool, same_split: bool) -> str:
    """Classify a run for tagging/grouping."""
    if is_flagship: return "flagship"
    if name.startswith("seed"): return "seed"
    if "aug" in name or "bias" in name: return "ablation"
    return "candidate" if same_split else "xdataset"


def recurate(experiment: str = "cardioseg", old: str = "cardioseg-unet"):
    """Curate the registry: drop the lumped `old` model; tag every run (kind/split); register ONLY
    deployable candidates (runs whose split matches the flagship's) as versions of MODEL_NAME with
    readable descriptions + version tags; flagship gets the 'production' alias. Old experiments stay
    tracked runs, not registry versions. runs/ + FLAGSHIP_RUN remain the resolver."""
    from .training.model import load_run
    from .config import FLAGSHIP_RUN
    from .hparams import from_json
    mlflow = _mlflow()
    if mlflow is None:
        print("mlflow off"); return
    mlflow.set_tracking_uri(_DB_URI)
    c = mlflow.tracking.MlflowClient()
    try: c.delete_registered_model(old)                 # drop the polluted catalog
    except Exception: pass

    flagname = Path(FLAGSHIP_RUN).name
    fcfg = from_json(Path(FLAGSHIP_RUN) / "config.json").data
    fsplit = (tuple(fcfg.test_vendors), tuple(fcfg.val_datasets))
    for pth in sorted((_ROOT / "runs").glob("*/model.pth")):
        rd = pth.parent
        cp = rd / "config.json"
        cfg = from_json(cp).data if cp.exists() else None
        same = bool(cfg) and (tuple(cfg.test_vendors), tuple(cfg.val_datasets)) == fsplit
        split = "+".join(cfg.test_vendors) if cfg else "legacy"
        kind = run_kind(rd.name, rd.name == flagname, same)
        trk = track_run(experiment, rd.name, run_dir=rd, tags={"kind": kind, "split": split})
        if same:                                        # deployable candidate -> register a version
            model, _, _ = load_run(rd, "cpu")
            trk.log_model(model, MODEL_NAME, alias=("production" if rd.name == flagname else None),
                          description=f"{rd.name} · split={split} · {kind}",
                          version_tags={"run": rd.name, "split": split, "kind": kind})
        trk.end()
        print(f"{rd.name:14} kind={kind:10} {'REGISTERED' if same else 'run-only'}")
    print(f"-> registry '{MODEL_NAME}': candidates only; production -> {flagname}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="MLflow utilities (Phase 1).")
    ap.add_argument("--recurate", action="store_true", help="curate registry + tag runs (kind/split)")
    a = ap.parse_args()
    recurate() if a.recurate else backfill()
