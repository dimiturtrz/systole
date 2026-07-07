"""CMRxMotion adapter (NIfTI) — single-vendor (Siemens 3T), healthy volunteers, but a NEW
axis our other sources lack: respiratory-motion corruption. 20 volunteers × 4 breathing
conditions (breath-hold → free → intense) = 80 short-axis acquisitions, each ED + ES.

Layout: <root>/cmrxmotion/data/P###-{1..4}/P###-{n}-{ED,ES}.nii.gz (image, 4D w/ trailing
singleton) + P###-{n}-{ED,ES}-label.nii.gz (seg). A per-frame motion-quality grade (1 mild,
2 intermediate, 3 severe) lives in <root>/cmrxmotion/IQA.csv. Grade-3 (severe) frames ship
NO seg label — too corrupted to annotate — so they fall out via the store's `labelled` flag
(both ED+ES need usable GT). Scorable: ~69 cases, grades 1+2; grade 3 = predict-only.

Labels: same flip as M&Ms (raw 1=LV-cav, 2=myo, 3=RV) -> canonical via label_map (verified
geometrically on P001/P010). `motion_grade` is surfaced in meta() — a new stratification axis
the unified store grew a column for (null on the other datasets).
"""
import os
from pathlib import Path

from core.config import data_root
from core.data.static.mri.base import (
    MNM_LABEL_MAP,
    DatasetAdapter,
    PatientData,
    load_csv_info,
    load_frames,
)

LABEL_MAP = MNM_LABEL_MAP   # raw -> canonical (LV-cav 1->3, RV 3->1); shared M&Ms flip

# Single scanner across the challenge: Siemens MAGNETOM Vida 3T, at Fudan University
# (Zhangjiang International Brain Imaging Center, Shanghai); healthy volunteers (no pathology).
VENDOR, FIELD_T, SCANNER = "Siemens", 3.0, "MAGNETOM Vida"
CENTRE, COUNTRY = "Fudan (Shanghai)", "China"


def _root(root: str | Path | None = None) -> Path:
    """Resolve the dir holding the P###-n/ folders. Override with CARDIAC_CMRX_ROOT."""
    env = os.environ.get("CARDIAC_CMRX_ROOT")
    bases = [Path(env)] if env else []
    if root is not None:
        bases.append(Path(root))
    raw = Path(data_root("raw"))
    bases += [raw / "cmrxmotion", raw, raw.parent]
    subs = ("data", ".", "cmrxmotion/data")
    for base in bases:
        for sub in subs:
            cand = base if sub == "." else base / sub
            if cand.is_dir() and any(cand.glob("P[0-9][0-9][0-9]-[0-9]")):
                return cand
    return raw / "cmrxmotion" / "data"


def _iqa(root: str | Path | None = None) -> dict[str, dict[str, str]]:
    """{image-id (e.g. 'P001-1-ED') -> {Image, Label}} from IQA.csv (motion-quality grade)."""
    return load_csv_info(_root(root).parent / "IQA.csv", "Image")


def cmrx_cases(root: str | Path | None = None) -> list[Path]:
    """List acquisition dirs (P###-n: volunteer × breathing condition)."""
    d = _root(root)
    return sorted((p for p in d.glob("P[0-9][0-9][0-9]-[0-9]") if p.is_dir()), key=lambda p: p.name)


def _grade(case: Path) -> str | None:
    """Worst motion grade across the case's ED/ES frames (conservative case-level severity)."""
    iqa = _iqa(case.parent.parent)
    gs = [iqa.get(f"{case.name}-{t}", {}).get("Label") for t in ("ED", "ES")]
    gs = [g for g in gs if g]
    return max(gs) if gs else None


def load_ed_es(case: str | Path) -> PatientData:
    """Load ED + ES short-axis frames + canonical-remapped masks for one acquisition.
    Image is 4D with a trailing singleton -> frame=0 squeezes it; label is 3D. A frame whose
    label file is absent (grade-3 severe) is skipped -> the store marks the case unlabelled."""
    case = Path(case)
    cid = case.name

    def resolve(tag):
        gt = case / f"{cid}-{tag}-label.nii.gz"
        if not gt.exists():
            return None                                   # severe motion: no GT -> skip frame
        return case / f"{cid}-{tag}.nii.gz", gt, 0        # frame=0: drop trailing singleton

    return load_frames(None, resolve, LABEL_MAP)          # healthy volunteers -> no pathology group


class CmrxMotionAdapter(DatasetAdapter):
    """CMRxMotion: single-vendor motion-robustness axis; labels remapped to canonical."""
    name = "cmrxmotion"
    label_map = LABEL_MAP

    def cases(self) -> list[Path]:
        return cmrx_cases()

    def load_ed_es(self, case: Path) -> PatientData:
        return load_ed_es(case)

    def meta(self, case: Path) -> dict:
        """Acquisition + the motion-grade axis — AUTO from IQA.csv (single fixed scanner)."""
        return {
            "group": None, "vendor": VENDOR, "scanner": SCANNER, "field_T": FIELD_T,
            "centre": CENTRE, "country": COUNTRY,
            "age": None, "sex": None, "height": None, "weight": None,
            "motion_grade": _grade(case),
            "_source": {"vendor": "challenge", "scanner": "challenge", "country": "challenge",
                        "motion_grade": "IQA.csv"},
        }
