"""Shared cardioview IO + constants: data paths, model loading, chamber colors.

Pulls the data root from cardioseg's central config (paths.yaml; CARDIAC_DATA_ROOT
overrides), so cardioview has no hardcoded machine paths.
"""
from __future__ import annotations

from pathlib import Path

from cardioseg.config import data_root

# Label convention (verified on real masks): 1=RV, 2=LV-myo, 3=LV-cavity.
CHAMBERS = {
    3: ("LV cavity", "#ef5350"),
    2: ("LV myocardium", "#ffca5b"),
    1: ("RV cavity", "#5b8def"),
}
SIZE = 256  # square grid the 2D model runs on
MODELS = {"acdc": "runs/acdc/model.pth", "acdc_aug": "runs/acdc_aug/model.pth"}


def patient_dir(patient: str, root: str | None = None) -> Path:
    """Locate a patient folder under training/ or testing/ (root from paths.yaml)."""
    base = Path(root or data_root("raw"))
    for split in ("training", "testing"):
        p = base / split / patient
        if p.is_dir():
            return p
    raise FileNotFoundError(f"{patient} not found under {base}/training or /testing")


def load_model(weights: str, device):
    """Build the 2D U-Net and load trained weights (matches training: spatial_dims=2, 4 classes)."""
    import torch
    from cardioseg.training.model import build_unet

    model = build_unet(spatial_dims=2, out_channels=4).to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()
    return model
