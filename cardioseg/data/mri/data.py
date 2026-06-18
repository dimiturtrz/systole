"""ACDC loader (NIfTI). Register at Creatis / humanheart-project; drop under
data/raw/mri/acdc/.

ACDC layout: training/patientXXX/ with cine frames (patientXXX_frameNN.nii.gz)
and ground-truth masks (patientXXX_frameNN_gt.nii.gz); Info.cfg names the ED/ES
frame indices and the pathology Group.

Label convention (VERIFIED on real masks via myo-enclosure test, see
data/mri/eda.py): 0 background, 1 RV cavity, 2 LV myocardium, 3 LV cavity.
NB: LV cavity is label 3, NOT 1 — the synthetic fixture historically had this
flipped. Always disambiguate geometrically, never trust a remembered integer.
"""
import os
from pathlib import Path

# Data lives outside the repo (licensing + size). Point CARDIAC_DATA_ROOT at your
# local ACDC root (the dir holding training/), e.g. D:/data/raw/mri/acdc.
DATA_ROOT = os.environ.get("CARDIAC_DATA_ROOT", "data/raw/mri/acdc")

# ACDC ground-truth integer labels (verified on real data).
LV_CAVITY, LV_MYO, RV_CAVITY = 3, 2, 1


def load_nifti(path):
    """Return (array [D,H,W], spacing (z,y,x) mm)."""
    import nibabel as nib
    import numpy as np
    img = nib.load(str(path))
    arr = np.asanyarray(img.dataobj)          # NIfTI is x,y,z
    arr = np.transpose(arr, (2, 1, 0))        # -> z,y,x (D,H,W)
    zx, zy, zz = img.header.get_zooms()[:3]
    return arr, (zz, zy, zx)


def _training_dir(root=None):
    """Resolve the dir holding patient*/ — accepts the root or .../training."""
    base = Path(root or DATA_ROOT)
    for cand in (base, base / "training", base / "database" / "training"):
        if any(cand.glob("patient*")):
            return cand
    return base


def acdc_cases(root=None):
    """Yield patient dirs (ACDC: patientXXX/). Defaults to CARDIAC_DATA_ROOT."""
    return sorted(p for p in _training_dir(root).glob("patient*") if p.is_dir())


def parse_info_cfg(patient_dir):
    """ACDC Info.cfg -> dict (ED, ES frame numbers; Group = pathology; etc.)."""
    cfg = {}
    p = Path(patient_dir) / "Info.cfg"
    if p.exists():
        for line in p.read_text().splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                cfg[k.strip()] = v.strip()
    return cfg


def frame_paths(patient_dir, frame_no):
    """(image, gt) paths for one frame. ACDC: patientXXX_frameNN(.nii.gz)/_gt."""
    patient_dir = Path(patient_dir)
    stem = f"{patient_dir.name}_frame{int(frame_no):02d}"
    return patient_dir / f"{stem}.nii.gz", patient_dir / f"{stem}_gt.nii.gz"


def load_ed_es(patient_dir):
    """Load the ED and ES frames + masks for one patient.

    Returns dict: {group, spacing, ED:{img,gt}, ES:{img,gt}} — img/gt are [D,H,W],
    spacing is (z,y,x) mm. Frame indices come from Info.cfg.
    """
    patient_dir = Path(patient_dir)
    cfg = parse_info_cfg(patient_dir)
    out = {"group": cfg.get("Group"), "spacing": None}
    for tag in ("ED", "ES"):
        fno = cfg.get(tag)
        if fno is None:
            continue
        img_p, gt_p = frame_paths(patient_dir, fno)
        img, sp = load_nifti(img_p)
        gt, _ = load_nifti(gt_p)
        out["spacing"] = sp
        out[tag] = {"img": img, "gt": gt}
    return out


def identify_lv_cavity(mask, myo_label=LV_MYO):
    """Geometrically identify the LV-cavity label: the non-myo foreground label
    most enclosed by the myocardium ring. Trusts geometry, not a remembered int.

    Returns (lv_label, scores) where score = fraction of a label's 1-voxel shell
    that touches myocardium. LV (inside the ring) scores high; RV scores low.
    """
    import numpy as np
    from scipy import ndimage

    labels = [int(l) for l in np.unique(mask) if l != 0 and l != myo_label]
    myo = mask == myo_label
    scores = {}
    for lab in labels:
        cav = mask == lab
        shell = ndimage.binary_dilation(cav) & ~cav
        scores[lab] = float((shell & myo).sum()) / float(shell.sum()) if shell.sum() else 0.0
    lv = max(scores, key=scores.get) if scores else None
    return lv, scores
