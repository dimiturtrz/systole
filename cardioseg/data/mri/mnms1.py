"""M&Ms-1 adapter (NIfTI) — the broadest multi-site set: 375 subjects, 6 centres
(Spain×4, Germany, Canada), 4 vendors (Siemens/Philips/GE/Canon).

Layout: <root>/MnM/{Training/{Labeled,Unlabeled}, Validation, Testing}/<code>/ with a 4D
short-axis cine <code>_sa.nii(.gz) + 4D gt <code>_sa_gt.nii(.gz) where only the ED/ES
frames are labelled. ED/ES frame indices + vendor/centre/age/sex/h+w in the CSV.

Labels: same flip as M&M-2 (raw 1=LV-cav) -> canonical via label_map.
"""
import os
from pathlib import Path

from cardioseg.data.mri.base import (
    DatasetAdapter, MNM_LABEL_MAP, PatientData, load_csv_info, load_frames, to_float,
)

LABEL_MAP = MNM_LABEL_MAP   # same M&Ms flip as M&M-2

# M&Ms-1 centre code -> (readable site name, country). From the challenge paper (Campello 2021).
# Our labelled split carries centres 1-5 (Spain + Germany); 6 (McGill, Canada) is test-only.
CENTRES = {
    "1": ("Vall d'Hebron (Barcelona)", "Spain"),
    "2": ("Sagrada Familia (Barcelona)", "Spain"),
    "3": ("UKE Hamburg", "Germany"),
    "4": ("Dexeus (Barcelona)", "Spain"),
    "5": ("Creu Blanca (Barcelona)", "Spain"),
    "6": ("McGill (Montreal)", "Canada"),
}

_SPLITS = ("Training/Labeled", "Validation", "Testing")


def _is_mnms1(base: Path) -> bool:
    """M&Ms-1 marker: a nested Training/Labeled (ACDC has only lowercase training/, no Labeled),
    or a CSV with the External-code column. Avoids case-insensitive false-matches (acdc/testing)."""
    if (base / "Training" / "Labeled").is_dir():
        return True
    for c in list(base.glob("*.csv")) + list(base.glob("*/*.csv")):
        try:
            if "External code" in c.read_text()[:200]:
                return True
        except OSError:
            pass
    return False


def _root(root: str | Path | None = None) -> Path:
    """Resolve the MnM/ root (holds Training/Validation/Testing). Override CARDIAC_MNMS1_ROOT.
    Prefers an explicit MnM dir + validates it's actually M&Ms-1 (not a case-insensitive match)."""
    env = os.environ.get("CARDIAC_MNMS1_ROOT")
    raw = Path(data_root_raw())
    bases = ([Path(env)] if env else []) + ([Path(root)] if root else [])
    bases += [raw.parent / "MnM", raw / "MnM", raw.parent, raw]
    for base in bases:
        if base.is_dir() and _is_mnms1(base):
            return base
    return raw.parent / "MnM"


def data_root_raw() -> str:
    from core.config import data_root
    return data_root("raw")


def mnms1_info(root: str | Path | None = None) -> dict[str, dict[str, str]]:
    """{External code -> CSV row (Vendor, Centre, ED, ES, Age, Sex, Height, Weight, Pathology)}."""
    base = _root(root)
    csvs = list(base.glob("*.csv")) + list(base.glob("*/*.csv"))
    if not csvs:
        return {}
    return load_csv_info(csvs[0], "External code", alt_key_col="External_code")


def _sa(case: Path) -> tuple[Path, Path]:
    code = case.name
    for ext in (".nii.gz", ".nii"):
        img = case / f"{code}_sa{ext}"
        if img.exists():
            return img, case / f"{code}_sa_gt{ext}"
    return case / f"{code}_sa.nii.gz", case / f"{code}_sa_gt.nii.gz"


def mnms1_cases(root: str | Path | None = None) -> list[Path]:
    """All subject dirs with a SA cine, across Labeled/Validation/Testing. The store flags which
    have usable (non-empty) masks — M&Ms-1 ships withheld/zero-filled GT for much of Testing, so the
    `labelled` column (computed from mask content) is what train/eval filter on; vendor (incl. the
    Canon unseen-vendor slice) is a query over that meta, not a separate case list."""
    base = _root(root)
    out: list[Path] = []
    for split in _SPLITS:
        d = base / split
        if d.is_dir():
            out += [p for p in sorted(d.iterdir()) if p.is_dir() and _sa(p)[1].exists()]
    return out


def load_ed_es(case: str | Path, root: str | Path | None = None) -> PatientData:
    """Load ED + ES frames (from the 4D cine at CSV indices) + canonical-remapped masks."""
    case = Path(case)
    row = mnms1_info(root).get(case.name, {})
    img_p, gt_p = _sa(case)

    def resolve(tag):
        idx = row.get(tag)
        if idx is None or idx == "":
            return None
        return img_p, gt_p, int(float(idx))   # 4D cine -> frame index from the CSV

    return load_frames(row.get("Pathology"), resolve, LABEL_MAP)


class Mnms1Adapter(DatasetAdapter):
    """M&Ms-1: 6-centre / 4-vendor (incl. Canon); richest demographics (age/sex/BSA)."""
    name = "mnms1"
    label_map = LABEL_MAP

    def cases(self) -> list[Path]:
        return mnms1_cases()

    def load_ed_es(self, case: Path) -> PatientData:
        return load_ed_es(case)

    def meta(self, case: Path) -> dict:
        """Acquisition + demographics — AUTO from the CSV (richest of the three)."""
        r = mnms1_info().get(case.name, {})
        name, country = CENTRES.get(str(r.get("Centre")).strip(), (r.get("Centre"), None))
        return {
            "group": r.get("Pathology"),
            "vendor": r.get("VendorName") or r.get("Vendor"),
            "centre": name, "country": country,   # code -> readable site + country (paper map)
            "age": to_float(r.get("Age")), "sex": r.get("Sex"),
            "height": to_float(r.get("Height")), "weight": to_float(r.get("Weight")),
            "scanner": None, "field_T": None,   # not in CSV (paper tables)
            "_source": {"all": "csv", "centre+country": "paper centre map", "field_T": "paper(unfilled)"},
        }
