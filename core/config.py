"""Central path config. ONE data root; raw/processed derived from it.

paths.yaml (gitignored, machine-specific) holds a single absolute `data` root:

    data: /abs/path/to/cardiac-data

Convention under it: `<data>/raw/<dataset>/` holds your downloads (create raw/ and drop the
datasets in); `<data>/processed/` is the preprocess cache (auto-created). Copy
paths.example.yaml -> paths.yaml and set the one path. Env CARDIAC_DATA overrides the file.
"""
import os
from pathlib import Path

from omegaconf import OmegaConf
from pydantic import ConfigDict

from core.paths import resolve_data_root

# Shared pydantic config used by every dispersed cfg (ModelCfg/AugCfg/…): setattr (used by the
# `--set` overrides) re-validates the field. Lives here — a leaf module both core.hparams and the
# per-class config homes import without cycling.
_VALIDATE = ConfigDict(validate_assignment=True)

_REPO = Path(__file__).resolve().parents[1]
_PATHS_FILE = Path(os.environ.get("CARDIAC_PATHS", _REPO / "paths.yaml"))
_cfg = OmegaConf.load(_PATHS_FILE) if _PATHS_FILE.exists() else OmegaConf.create({})

_FALLBACK_ROOT = "data"   # repo-relative (gitignored) if nothing configured


def _root() -> str:
    """The one data root: env CARDIAC_DATA -> paths.yaml `data` -> repo-relative fallback, then
    adapted to the current OS by core.paths.resolve_data_root — translates a Windows drive path to/
    from its WSL mount, and raises (never silently relative) for an untranslatable foreign path."""
    raw = str(os.environ.get("CARDIAC_DATA") or OmegaConf.select(_cfg, "data") or _FALLBACK_ROOT)
    return resolve_data_root(raw)


def data_root(kind: str = "raw") -> str:
    """Absolute root for a data kind ('raw' inputs, 'processed' cache, 'reference', 'meshes', …).
    Default `<data>/<kind>`, but a top-level `paths.yaml` key named `kind` overrides it — so e.g.
    `meshes: /big/scratch/meshes` in paths.yaml redirects exports off the data root (still OS-adapted,
    still never silently relative)."""
    override = OmegaConf.select(_cfg, kind)
    if override:
        return resolve_data_root(str(override))
    return str(Path(_root()) / kind)


# The flagship is the `production`-aliased version in the mlflow model registry (core.registry) —
# the SINGLE source for "which model is flagship". Re-point by moving the alias, not editing a path.
# All eval/export/viewer defaults reference this ref. See ROADMAP.
FLAGSHIP_REF = "production"


# --- pipeline constants (single source; config-default + module-constant readers reference these) ---
DEFAULT_SIZE = 256        # square in-plane grid the 2D model runs on (DataCfg.size + dataset.SIZE)
DEFAULT_INPLANE = 1.5     # in-plane resample target (mm); ACDC/M&M in-plane ~1.2-1.6 (DataCfg.inplane)
KNOWN_DATASETS = ("acdc", "mnm2", "mnms1", "cmrxmotion")  # the wired MRI datasets (DataCfg.sources, persist)
KNOWN_VENDORS = ("Siemens", "Philips", "GE", "Canon")  # canonical scanner-vendor names
