"""Shared cardioview IO + constants: data paths, model loading, chamber colors.

Pulls the data root from core's central config (paths.yaml; CARDIAC_DATA_ROOT
overrides), so cardioview has no hardcoded machine paths.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

from core.config import FLAGSHIP_REF, data_root
from core.inference import predict_volume
from core.postprocess import Postprocess
from core.preprocessing.preprocess import fit_square
from core.registry import resolve
from core.run import load_run

# Label convention (verified on real masks): 1=RV, 2=LV-myo, 3=LV-cavity.
CHAMBERS = {
    3: ("LV cavity", "#ef5350"),
    2: ("LV myocardium", "#ffca5b"),
    1: ("RV cavity", "#5b8def"),
}
SIZE = 256  # square grid the 2D model runs on
# Models are mlflow registry refs (alias|version|run-id) — the flagship is the `production` alias
# (the pooled multi-vendor soft-label model). Add more refs here to compare in the viewer.
MODELS = {"gen": FLAGSHIP_REF}
DEFAULT_MODEL = "gen"  # single source for the viewer/export default


def log_setup(level: int = logging.INFO) -> None:  # pragma: no cover  (stdout logging handler wiring — IO shell)
    """Configure the `cardioview` logger -> stdout (mirrors core.obs.setup for this namespace).
    Handlers on the named logger with propagate=False survive third-party basicConfig(force=True)."""
    log = logging.getLogger("cardioview")
    log.setLevel(level)
    log.propagate = False
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s | %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(sh)


def model_dir(ref: str):  # pragma: no cover  (mlflow registry resolve + artifact download — registry shell)
    """Resolve a registry ref to its local artifact dir (model.pth + config.json + …)."""
    return resolve(ref)


def patient_dir(patient: str, root: str | None = None) -> Path:
    """Resolve a patient folder. Accepts a full path, or a bare ACDC ID resolved under
    <data>/raw/acdc/{training,testing} — so the canned-heart list can mix IDs and full paths."""
    p = Path(patient)
    if p.is_dir():
        return p
    base = Path(root or data_root("raw")) / "acdc"
    for split in ("training", "testing"):
        d = base / split / patient
        if d.is_dir():
            return d
    raise FileNotFoundError(f"{patient} not found (not a dir, nor under {base}/training|testing)")


def load_model(ref: str, device):  # pragma: no cover  (load_run reads model.pth weights + GPU — registry/model shell)
    """Load the trained U-Net for a registry ref (arch from its config.json, so it can't mismatch a
    default). `ref` = an mlflow registry ref (alias|version|run-id) resolved to its artifact dir."""
    model, _, _ = load_run(model_dir(ref), device)
    return model


def square_stack(vol_zyx, dtype=None):
    """Center pad/crop each slice to the SIZE square grid the model expects."""
    out = np.stack([fit_square(s.astype(np.float32), SIZE, 0.0) for s in vol_zyx])
    return out.astype(dtype) if dtype else out


def masks(case: dict, source: str, model=None, device=None) -> dict:
    """{ED, ES} chamber-label volumes on the square grid — ground truth or model prediction.

    Predictions get largest-CC post-processing (matches the cardioseg pipeline), so the
    displayed EF/volumes line up with the reported numbers.
    """
    out = {}
    for tag in ("ED", "ES"):
        k = tag.lower()
        if f"{k}_img" not in case:
            continue
        out[tag] = (
            square_stack(case[f"{k}_gt"], np.uint8)
            if source == "gt"
            else Postprocess.largest_cc_per_class(predict_volume(model, case[f"{k}_img"], SIZE, device))
        )
    return out
