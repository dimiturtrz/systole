"""ACDC adapter (NIfTI). Register at Creatis / humanheart-project; drop under data/.../acdc/.

ACDC layout: training/patientXXX/ with cine frames (patientXXX_frameNN.nii.gz) + gt masks
(_gt.nii.gz); Info.cfg names the ED/ES frame indices, pathology Group, Height, Weight.

Labels are already the canonical convention (0 bg, 1 RV, 2 LV-myo, 3 LV-cav) — verified
geometrically (see base.identify_lv_cavity). Scanner: Siemens Aera 1.5T / Trio Tim 3T (Dijon).
"""
from pathlib import Path

from core.config import data_root
from core.data.mri.base import (
    DatasetAdapter, Frame, PatientData, load_nifti, load_frames, identify_lv_cavity, to_float,
    LV_CAVITY, LV_MYO, RV_CAVITY,
)

# Data lives outside the repo at <data>/raw/acdc/ (paths.yaml `data`; CARDIAC_DATA_ROOT overrides).
DATA_ROOT = str(Path(data_root("raw")) / "acdc")
LABEL_MAP = {0: 0, 1: 1, 2: 2, 3: 3}   # ACDC is the canonical convention (identity)


def acdc_cases(root: str | Path | None = None) -> list[Path]:
    """ALL labelled ACDC patients — both training/ (100) and testing/ (50, also has GT). We pull
    everything and make our own train/val/test splits downstream (don't inherit the challenge split)."""
    base = Path(root or DATA_ROOT)
    out: list[Path] = []
    for split in ("training", "testing"):
        d = base / split
        if d.is_dir():
            out += [p for p in d.glob("patient*") if p.is_dir()]
    if not out:                                    # tolerate a flat or database/training layout
        for cand in (base, base / "database" / "training"):
            out += [p for p in cand.glob("patient*") if p.is_dir()]
    return sorted(out, key=lambda p: p.name)


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

    def resolve(tag):
        fno = cfg.get(tag)
        return (*frame_paths(patient_dir, fno), None) if fno is not None else None

    return load_frames(cfg.get("Group"), resolve, LABEL_MAP)   # identity map -> masks unchanged


class AcdcAdapter(DatasetAdapter):
    """ACDC: single-centre Siemens (Dijon), the canonical-label held-out test set."""
    name = "acdc"
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
            "height": to_float(cfg.get("Height")), "weight": to_float(cfg.get("Weight")),
            "age": None, "sex": None,
            "vendor": "Siemens", "field_T": [1.5, 3.0],   # Bernard 2018 (Aera 1.5T / Trio 3T)
            "scanner": "Siemens Aera/Trio",                # two units, not recorded per-subject
            "centre": "Dijon", "country": "France",        # CHU Dijon (single centre)
            "_source": {"vendor": "paper", "field_T": "paper", "country": "paper", "rest": "Info.cfg"},
        }
