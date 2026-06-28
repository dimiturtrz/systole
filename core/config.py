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

_REPO = Path(__file__).resolve().parents[1]
_PATHS_FILE = Path(os.environ.get("CARDIAC_PATHS", _REPO / "paths.yaml"))
_cfg = OmegaConf.load(_PATHS_FILE) if _PATHS_FILE.exists() else OmegaConf.create({})

_FALLBACK_ROOT = "data"   # repo-relative (gitignored) if nothing configured


def _root() -> str:
    """The one data root: env CARDIAC_DATA -> paths.yaml `data` -> repo-relative fallback."""
    return os.environ.get("CARDIAC_DATA") or OmegaConf.select(_cfg, "data") or _FALLBACK_ROOT


def data_root(kind: str = "raw") -> str:
    """Absolute root for 'raw' (inputs) or 'processed' (cache): `<data>/<kind>`."""
    return str(Path(_root()) / kind)


# The shipped flagship run dir (repo-relative): trained on the pooled multi-vendor cloud
# (M&M-2 + M&Ms-1), held out ACDC + Canon. SINGLE SOURCE for "which run is flagship" —
# bump here only; all eval/export/viewer defaults reference it. See ROADMAP.
FLAGSHIP_RUN = "runs/gen"


def flagship_model() -> str:
    """Path to the flagship trained weights: `<FLAGSHIP_RUN>/model.pth`."""
    return f"{FLAGSHIP_RUN}/model.pth"


# --- pipeline constants (single source; config-default + module-constant readers reference these) ---
DEFAULT_SIZE = 256        # square in-plane grid the 2D model runs on (DataCfg.size + dataset.SIZE)
DEFAULT_INPLANE = 1.5     # in-plane resample target (mm); ACDC/M&M in-plane ~1.2-1.6 (DataCfg.inplane)
KNOWN_DATASETS = ("acdc", "mnm2", "mnms1", "cmrxmotion")  # the wired MRI datasets (DataCfg.sources, persist)
KNOWN_VENDORS = ("Siemens", "Philips", "GE", "Canon")  # canonical scanner-vendor names
