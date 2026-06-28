"""Shared cardioview IO + constants: data paths, model loading, chamber colors.

Pulls the data root from cardioseg's central config (paths.yaml; CARDIAC_DATA_ROOT
overrides), so cardioview has no hardcoded machine paths.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from core.config import data_root, flagship_model
from core.preprocessing.preprocess import fit_square

# Label convention (verified on real masks): 1=RV, 2=LV-myo, 3=LV-cavity.
CHAMBERS = {
    3: ("LV cavity", "#ef5350"),
    2: ("LV myocardium", "#ffca5b"),
    1: ("RV cavity", "#5b8def"),
}
SIZE = 256  # square grid the 2D model runs on
# gen = the shipped flagship (core.config.FLAGSHIP_RUN): pooled multi-vendor cloud
# (M&M-2 + M&Ms-1), held out ACDC + Canon — its numbers are the ones the docs report.
# mnm2/acdc/acdc_aug = older single-/cross-dataset runs, kept for comparison.
MODELS = {
    "gen": flagship_model(),
    "mnm2": "runs/mnm2_to_acdc/model.pth",
    "acdc_aug": "runs/acdc_aug/model.pth",
    "acdc": "runs/acdc/model.pth",
}
DEFAULT_MODEL = "gen"  # single source for the viewer/export default


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


def load_model(weights: str, device):
    """Load the trained U-Net for a run (architecture from the run's config.json, so it can't
    mismatch a default). `weights` is the run's model.pth; its parent dir is the run."""
    from cardioseg.training.model import load_run

    model, _, _ = load_run(Path(weights).parent, device)
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
    from cardioseg.evaluation.validate import predict_volume
    from cardioseg.evaluation.postprocess import largest_cc_per_class

    out = {}
    for tag in ("ED", "ES"):
        k = tag.lower()
        if f"{k}_img" not in case:
            continue
        out[tag] = (
            square_stack(case[f"{k}_gt"], np.uint8)
            if source == "gt"
            else largest_cc_per_class(predict_volume(model, case[f"{k}_img"], SIZE, device))
        )
    return out
