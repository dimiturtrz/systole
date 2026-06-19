"""Shared cardioview IO + constants: data paths, model loading, chamber colors.

Pulls the data root from cardioseg's central config (paths.yaml; CARDIAC_DATA_ROOT
overrides), so cardioview has no hardcoded machine paths.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from omegaconf import OmegaConf

from cardioseg.config import data_root, _cfg as _PATHS  # reuse the one loaded paths.yaml
from cardioseg.training.dataset import fit_square

# Label convention (verified on real masks): 1=RV, 2=LV-myo, 3=LV-cavity.
CHAMBERS = {
    3: ("LV cavity", "#ef5350"),
    2: ("LV myocardium", "#ffca5b"),
    1: ("RV cavity", "#5b8def"),
}
SIZE = 256  # square grid the 2D model runs on
# mnm2 = flagship: trained on multi-vendor M&M-2, generalizes to ACDC (the domain-
# generalization setting). acdc/acdc_aug = single-centre baselines.
MODELS = {
    "mnm2": "runs/mnm2_to_acdc/model.pth",
    "acdc_aug": "runs/acdc_aug/model.pth",
    "acdc": "runs/acdc/model.pth",
}


def cardioview_default(key: str, fallback):
    """A cardioview default from paths.yaml (`cardioview.<key>`), else the fallback."""
    v = OmegaConf.select(_PATHS, f"cardioview.{key}")
    return list(v) if v is not None else fallback


def patient_dir(patient: str, root: str | None = None) -> Path:
    """Resolve a patient folder. Accepts a full path to the folder, or an ID resolved
    under data.raw/{training,testing} — so paths.yaml hearts can mix IDs and absolute paths."""
    p = Path(patient)
    if p.is_dir():
        return p
    base = Path(root or data_root("raw"))
    for split in ("training", "testing"):
        d = base / split / patient
        if d.is_dir():
            return d
    raise FileNotFoundError(f"{patient} not found (not a dir, nor under {base}/training|testing)")


def load_model(weights: str, device):
    """Build the 2D U-Net and load trained weights (matches training: spatial_dims=2, 4 classes)."""
    import torch
    from cardioseg.training.model import build_unet

    model = build_unet(spatial_dims=2, out_channels=4).to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()
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
