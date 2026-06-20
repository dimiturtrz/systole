"""Central path config. Reads paths.yaml (OmegaConf); env vars override.

paths.yaml (gitignored, machine-specific) holds the absolute data roots — copy
paths.example.yaml -> paths.yaml and edit. This replaces the scattered
CARDIAC_DATA_ROOT / CARDIAC_PROCESSED_ROOT env lookups with one place; the env
vars still override (handy for tests / CI on another machine).

Resolution order for a root: env var -> paths.yaml -> repo-relative fallback.
"""
import os
from pathlib import Path

from omegaconf import OmegaConf

_REPO = Path(__file__).resolve().parents[1]
_PATHS_FILE = Path(os.environ.get("CARDIAC_PATHS", _REPO / "paths.yaml"))
_cfg = OmegaConf.load(_PATHS_FILE) if _PATHS_FILE.exists() else OmegaConf.create({})

_ENV = {"raw": "CARDIAC_DATA_ROOT", "processed": "CARDIAC_PROCESSED_ROOT"}
# repo-relative fallbacks (gitignored). Per-domain convention: everything under
# data/volumetric/mri/ — inputs in acdc/, preprocess cache in processed/.
_FALLBACK = {"raw": "data/volumetric/mri/acdc", "processed": "data/volumetric/mri/processed"}


def data_root(kind: str = "raw") -> str:
    """Absolute root for 'raw' (inputs) or 'processed' (cache) data."""
    env = os.environ.get(_ENV[kind])
    if env:
        return env
    val = OmegaConf.select(_cfg, f"data.{kind}")
    if val:
        return str(val)
    return _FALLBACK[kind]
