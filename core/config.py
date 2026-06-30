"""Central path config. ONE data root; raw/processed derived from it.

paths.yaml (gitignored, machine-specific) holds a single absolute `data` root:

    data: /abs/path/to/cardiac-data

Convention under it: `<data>/raw/<dataset>/` holds your downloads (create raw/ and drop the
datasets in); `<data>/processed/` is the preprocess cache (auto-created). Copy
paths.example.yaml -> paths.yaml and set the one path. Env CARDIAC_DATA overrides the file.
"""
import os
import re
from pathlib import Path

from omegaconf import OmegaConf

_REPO = Path(__file__).resolve().parents[1]
_PATHS_FILE = Path(os.environ.get("CARDIAC_PATHS", _REPO / "paths.yaml"))
_cfg = OmegaConf.load(_PATHS_FILE) if _PATHS_FILE.exists() else OmegaConf.create({})

_FALLBACK_ROOT = "data"   # repo-relative (gitignored) if nothing configured


def _root() -> str:
    """The one data root: env CARDIAC_DATA -> paths.yaml `data` -> repo-relative fallback.

    GUARD: a Windows drive path ('D:/…') on a POSIX OS is treated as RELATIVE, so any mkdir under it
    silently creates `repo/D:/…` — a data-leak landmine (this happened, then got committed). Fail loud
    instead: on POSIX, set CARDIAC_DATA to the mounted path (e.g. /mnt/d/…)."""
    r = str(os.environ.get("CARDIAC_DATA") or OmegaConf.select(_cfg, "data") or _FALLBACK_ROOT)
    if os.name != "nt" and re.match(r"^[A-Za-z]:[\\/]", r):
        raise RuntimeError(
            f"data root {r!r} is a Windows path but this is a POSIX OS — it would be treated as "
            f"relative and create a repo-local 'D:/' dir (data leak). Set CARDIAC_DATA to the mounted "
            f"path, e.g. CARDIAC_DATA=/mnt/d/{r[3:].replace(chr(92), '/')}")
    return r


def data_root(kind: str = "raw") -> str:
    """Absolute root for 'raw' (inputs) or 'processed' (cache): `<data>/<kind>`."""
    return str(Path(_root()) / kind)


# The flagship is the `production`-aliased version in the mlflow model registry (core.registry) —
# the SINGLE source for "which model is flagship". Re-point by moving the alias, not editing a path.
# All eval/export/viewer defaults reference this ref. See ROADMAP.
FLAGSHIP_REF = "production"


def flagship_dir() -> str:
    """Local dir of the flagship's resolved artifacts (model.pth + config.json + …), downloaded from
    the registry on demand. Use this where a run dir was expected. Lazy import — keep config light."""
    from core.registry import resolve
    return str(resolve(FLAGSHIP_REF))


def flagship_model() -> str:
    """Path to the flagship trained weights (resolved from the registry)."""
    from pathlib import Path
    return str(Path(flagship_dir()) / "model.pth")


# --- pipeline constants (single source; config-default + module-constant readers reference these) ---
DEFAULT_SIZE = 256        # square in-plane grid the 2D model runs on (DataCfg.size + dataset.SIZE)
DEFAULT_INPLANE = 1.5     # in-plane resample target (mm); ACDC/M&M in-plane ~1.2-1.6 (DataCfg.inplane)
KNOWN_DATASETS = ("acdc", "mnm2", "mnms1", "cmrxmotion")  # the wired MRI datasets (DataCfg.sources, persist)
KNOWN_VENDORS = ("Siemens", "Philips", "GE", "Canon")  # canonical scanner-vendor names
