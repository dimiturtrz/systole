"""ACDC loader (NIfTI). Register at Creatis / humanheart-project; drop under
data/raw/mri/acdc/.

ACDC layout: training/patientXXX/ with cine frames (patientXXX_frameNN.nii.gz)
and ground-truth masks (patientXXX_frameNN_gt.nii.gz); Info.cfg names the ED/ES
frame indices and the pathology Group.

Label convention (VERIFIED on real masks via myo-enclosure test, see
data/mri/eda.py): 0 background, 1 RV cavity, 2 LV myocardium, 3 LV cavity.
NB: LV cavity is label 3, NOT 1 — the synthetic fixture historically had this
flipped. Always disambiguate geometrically, never trust a remembered integer.

Shapes: volumes are [D, H, W] (D slices, H x W in-plane); spacing is (z, y, x) mm.
"""
from pathlib import Path
from typing import TypedDict

from cardioseg.config import data_root
from cardioseg.types import Image, Mask, Spacing, Volume

# Data lives outside the repo (licensing + size). Configured in paths.yaml
# (data.raw), overridable via CARDIAC_DATA_ROOT. See cardioseg/config.py.
DATA_ROOT = data_root("raw")

# ACDC ground-truth integer labels (verified on real data).
LV_CAVITY, LV_MYO, RV_CAVITY = 3, 2, 1


class Frame(TypedDict):
    """One cardiac-phase frame: image + its label mask, both [D, H, W]."""
    img: Image
    gt: Mask


class PatientData(TypedDict, total=False):
    """One patient's ED/ES frames + metadata (returned by load_ed_es)."""
    group: str | None          # pathology: NOR/DCM/HCM/MINF/ARV
    spacing: Spacing | None    # (z, y, x) mm
    ED: Frame                  # end-diastole (fullest)
    ES: Frame                  # end-systole (emptiest)


def load_nifti(path: str | Path) -> tuple[Volume, Spacing]:
    """Load a NIfTI volume. Returns (array [D, H, W], spacing (z, y, x) mm)."""
    import nibabel as nib
    import numpy as np
    img = nib.load(str(path))
    arr = np.asanyarray(img.dataobj)          # NIfTI is x,y,z
    arr = np.transpose(arr, (2, 1, 0))        # -> z,y,x (D,H,W)
    zx, zy, zz = img.header.get_zooms()[:3]
    return arr, (zz, zy, zx)


def _training_dir(root: str | Path | None = None) -> Path:
    """Resolve the dir holding patient*/ — accepts the root or .../training."""
    base = Path(root or DATA_ROOT)
    for cand in (base, base / "training", base / "database" / "training"):
        if any(cand.glob("patient*")):
            return cand
    return base


def acdc_cases(root: str | Path | None = None) -> list[Path]:
    """List patient dirs (ACDC: patientXXX/). Defaults to CARDIAC_DATA_ROOT."""
    return sorted(p for p in _training_dir(root).glob("patient*") if p.is_dir())


def parse_info_cfg(patient_dir: str | Path) -> dict[str, str]:
    """ACDC Info.cfg -> dict (ED, ES frame numbers; Group = pathology; etc.)."""
    cfg: dict[str, str] = {}
    p = Path(patient_dir) / "Info.cfg"
    if p.exists():
        for line in p.read_text().splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                cfg[k.strip()] = v.strip()
    return cfg


def frame_paths(patient_dir: str | Path, frame_no: int | str) -> tuple[Path, Path]:
    """(image, gt) paths for one frame. ACDC: patientXXX_frameNN(.nii.gz)/_gt."""
    patient_dir = Path(patient_dir)
    stem = f"{patient_dir.name}_frame{int(frame_no):02d}"
    return patient_dir / f"{stem}.nii.gz", patient_dir / f"{stem}_gt.nii.gz"


def load_ed_es(patient_dir: str | Path) -> PatientData:
    """Load the ED and ES frames + masks for one patient.

    img/gt are [D, H, W], spacing is (z, y, x) mm. Frame indices come from Info.cfg.
    """
    patient_dir = Path(patient_dir)
    cfg = parse_info_cfg(patient_dir)
    out: PatientData = {"group": cfg.get("Group"), "spacing": None}
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


def identify_lv_cavity(
    mask: Mask, myo_label: int = LV_MYO
) -> tuple[int | None, dict[int, float]]:
    """Geometrically identify the LV-cavity label: the non-myo foreground label
    most enclosed by the myocardium ring. Trusts geometry, not a remembered int.

    mask: [D, H, W] or [H, W] label map. Returns (lv_label, scores) where score =
    fraction of a label's 1-voxel shell that touches myocardium. LV (inside the
    ring) scores high; RV scores low.
    """
    import numpy as np
    from scipy import ndimage

    labels = [int(l) for l in np.unique(mask) if l != 0 and l != myo_label]
    myo = mask == myo_label
    scores: dict[int, float] = {}
    for lab in labels:
        cav = mask == lab
        shell = ndimage.binary_dilation(cav) & ~cav
        scores[lab] = float((shell & myo).sum()) / float(shell.sum()) if shell.sum() else 0.0
    lv = max(scores, key=scores.get) if scores else None
    return lv, scores
