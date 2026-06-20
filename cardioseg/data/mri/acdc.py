"""ACDC adapter (NIfTI). Register at Creatis / humanheart-project; drop under data/.../acdc/.

ACDC layout: training/patientXXX/ with cine frames (patientXXX_frameNN.nii.gz) + gt masks
(_gt.nii.gz); Info.cfg names the ED/ES frame indices, pathology Group, Height, Weight.

Labels are already the canonical convention (0 bg, 1 RV, 2 LV-myo, 3 LV-cav) — verified
geometrically (see base.identify_lv_cavity). Scanner: Siemens Aera 1.5T / Trio Tim 3T (Dijon).
"""
from pathlib import Path

from cardioseg.config import data_root
from cardioseg.data.mri.base import (
    Frame, PatientData, load_nifti, identify_lv_cavity,
    LV_CAVITY, LV_MYO, RV_CAVITY,
)

# Data lives outside the repo (paths.yaml data.raw; CARDIAC_DATA_ROOT overrides).
DATA_ROOT = data_root("raw")
LABEL_MAP = {0: 0, 1: 1, 2: 2, 3: 3}   # ACDC is the canonical convention (identity)


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
    """ACDC Info.cfg -> dict (ED, ES frame numbers; Group = pathology; Height; Weight; etc.)."""
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
    """Load ED + ES frames + masks for one patient (labels already canonical).

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


class AcdcAdapter:
    """ACDC: single-centre Siemens (Dijon), the canonical-label held-out test set."""
    name = "acdc"
    cache_ns = ""              # canonical -> default cache namespace
    label_map = LABEL_MAP

    def cases(self) -> list[Path]:
        return acdc_cases()

    def load_ed_es(self, case: Path) -> PatientData:
        return load_ed_es(case)   # already canonical; identity map

    def meta(self, case: Path) -> dict:
        """Acquisition + demographics (AUTO from Info.cfg; vendor/field cited constants)."""
        cfg = parse_info_cfg(case)
        return {
            "group": cfg.get("Group"),
            "height": _f(cfg.get("Height")), "weight": _f(cfg.get("Weight")),
            "age": None, "sex": None,
            "vendor": "Siemens", "field_T": [1.5, 3.0],   # Bernard 2018 (Aera 1.5T / Trio 3T)
            "centre": "Dijon", "_source": {"vendor": "paper", "field_T": "paper", "rest": "Info.cfg"},
        }


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
