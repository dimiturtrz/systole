"""MLflow model registry as the SINGLE source of truth for trained models (replaces the runs/ dir).

A model = an mlflow run's artifacts — `model.pth` (state_dict), `config.json`, `model.onnx`/`int8`,
`MODEL_CARD.md` — registered as a version of MODEL_NAME with aliases (`production` = flagship). We
store the state_dict + config as artifacts (NOT a pickled mlflow.pytorch model) so loading stays
robust across torch/monai versions: rebuild the arch from config, load_state_dict.

`resolve(ref)` downloads a version's artifacts to a local cache dir and returns that dir — so every
existing dir-consumer (`core.run.Run.load_run`, `results.build`, modelcard, export) works UNCHANGED on
the cached dir. `ref` = alias ('production') | version number | run-id.

    from core.registry import Registry
    run_dir = Registry.resolve("production")  # cached dir with model.pth + config.json + ...
    model, cfg, device = Run.load_run(run_dir)
"""
from __future__ import annotations

from pathlib import Path

import mlflow
from mlflow.exceptions import MlflowException

_ROOT = Path(__file__).resolve().parents[1]
_DB_URI = f"sqlite:///{(_ROOT / 'mlflow.db').as_posix()}"
_CACHE = _ROOT / ".mlflow_cache"                       # resolved-artifact cache (gitignored)
MODEL_NAME = "cardioseg-2dunet"
PRODUCTION = "production"                               # the flagship alias


class Registry:
    """MLflow model registry access (free functions folded in as staticmethods; public names kept):
    save/resolve/load a trained model as a version of MODEL_NAME with aliases (`production` = flagship)."""

    @staticmethod
    def _mlflow():
        mlflow.set_tracking_uri(_DB_URI)
        return mlflow

    @staticmethod
    def _client():
        return Registry._mlflow().tracking.MlflowClient()

    @staticmethod
    def _run_id_for(ref: str) -> str:
        """Resolve a ref (alias | version number | run-id) to an mlflow run-id."""
        c = Registry._client()
        # alias (e.g. 'production')
        try:
            return c.get_model_version_by_alias(MODEL_NAME, ref).run_id
        except MlflowException:   # no such alias (ref is a version number or run-id) -> fall through
            pass
        # explicit version number
        if str(ref).isdigit():
            return c.get_model_version(MODEL_NAME, str(ref)).run_id
        # assume it's already a run-id
        return str(ref)

    @staticmethod
    def resolve(ref: str = PRODUCTION) -> Path:
        """Download the model version's artifacts to a cache dir and return it (a dir with model.pth +
        config.json + …, ready for core.run.Run.load_run). ref = alias | version | run-id — OR an existing
        dir (returned as-is, so callers can pass a path too)."""
        p = Path(ref)
        if (p / "model.pth").exists():                     # already a dir (back-compat / explicit path)
            return p
        Registry._mlflow()
        rid = Registry._run_id_for(ref)
        dst = _CACHE / rid
        if not (dst / "model.pth").exists():               # cache: download once per run-id
            dst.mkdir(parents=True, exist_ok=True)
            mlflow.artifacts.download_artifacts(run_id=rid, artifact_path="model", dst_path=str(dst))
        # artifacts land under <dst>/model/* ; flatten to the dir load_run expects
        inner = dst / "model"
        return inner if (inner / "model.pth").exists() else dst

    @staticmethod
    def save_model(staging_dir, *, run_name: str, params: dict | None = None,  # noqa: PLR0913  already keyword-only (*); legit registry inputs
                   alias: str | None = None, description: str | None = None,
                   tags: dict | None = None, run_id: str | None = None) -> str:
        """Log a trained model's artifacts (everything in `staging_dir`: model.pth/config.json/onnx/card)
        to mlflow under artifact_path 'model', register a version of MODEL_NAME, optionally set `alias`.
        Reuses `run_id` if given (the train run), else starts one. Returns the run-id."""
        mlflow = Registry._mlflow()
        staging = Path(staging_dir)
        own = run_id is None
        if own:
            mlflow.start_run(run_name=run_name)
            if params:
                mlflow.log_params(Registry._flat(params))
        rid = run_id or mlflow.active_run().info.run_id
        c = Registry._client()
        # log each artifact under 'model/'
        for f in staging.iterdir():
            if f.is_file():
                (mlflow.log_artifact(str(f), artifact_path="model") if own
                 else c.log_artifact(rid, str(f), artifact_path="model"))
        src = f"runs:/{rid}/model"
        mv = c.create_model_version(MODEL_NAME, source=src, run_id=rid)
        if description:
            c.update_model_version(MODEL_NAME, mv.version, description=description)
        for k, v in (tags or {}).items():
            c.set_model_version_tag(MODEL_NAME, mv.version, k, str(v))
        if alias:
            c.set_registered_model_alias(MODEL_NAME, alias, mv.version)
        if own:
            mlflow.end_run()
        return rid

    @staticmethod
    def _flat(d: dict, prefix: str = "") -> dict:
        out = {}
        for k, v in d.items():
            key = f"{prefix}{k}"
            if isinstance(v, dict):
                out.update(Registry._flat(v, key + "."))
            else:
                out[key] = v
        return out
