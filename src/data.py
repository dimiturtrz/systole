"""ACDC loader (NIfTI). Register at Creatis / humanheart-project; drop under data/acdc/.

ACDC layout: patientXXX/ with cine frames (patientXXX_frameNN.nii.gz) and
ground-truth masks (patientXXX_frameNN_gt.nii.gz); the Info.cfg names the ED/ES
frame indices.
"""
import os
from pathlib import Path

# Data lives outside the repo (licensing + size). Point CARDIAC_DATA_ROOT at your
# local ACDC dir, e.g. D:/data/volumetric/mri/acdc. Falls back to ./data/acdc.
DATA_ROOT = os.environ.get("CARDIAC_DATA_ROOT", "data/acdc")


def load_nifti(path):
    """Return (array [D,H,W], spacing (z,y,x) mm)."""
    import nibabel as nib
    import numpy as np
    img = nib.load(str(path))
    arr = np.asanyarray(img.dataobj)          # NIfTI is x,y,z
    arr = np.transpose(arr, (2, 1, 0))        # -> z,y,x (D,H,W)
    zx, zy, zz = img.header.get_zooms()[:3]
    return arr, (zz, zy, zx)


def acdc_cases(root=None):
    """Yield patient dirs (ACDC: patientXXX/). Defaults to CARDIAC_DATA_ROOT."""
    return sorted(p for p in Path(root or DATA_ROOT).glob("patient*") if p.is_dir())
