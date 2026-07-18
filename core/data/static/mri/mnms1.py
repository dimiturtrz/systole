"""M&Ms-1 adapter (NIfTI) — the broadest multi-site set: 375 subjects, 6 centres
(Spain×4, Germany, Canada), 4 vendors (Siemens/Philips/GE/Canon).

Layout: <root>/MnM/{Training/{Labeled,Unlabeled}, Validation, Testing}/<code>/ with a 4D
short-axis cine <code>_sa.nii(.gz) + 4D gt <code>_sa_gt.nii(.gz) where only the ED/ES
frames are labelled. ED/ES frame indices + vendor/centre/age/sex/h+w in the CSV.

Labels: same flip as M&M-2 (raw 1=LV-cav) -> canonical via label_map.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, override

from core.config import Config
from core.data.static.mri.base import (
    MNM_LABEL_MAP,
    AdapterBase,
    Base,
    Dataset,
    DatasetAdapter,
    PatientData,
    PatientMeta,
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


class Mnms1Adapter(AdapterBase, DatasetAdapter):
    """M&Ms-1: 6-centre / 4-vendor (incl. Canon); richest demographics (age/sex/BSA). Owns its M&Ms-1
    marker detection, root resolution, CSV keying, SA path resolution, frame-index parsing, and meta
    assembly (staticmethods); root override via AdapterBase."""
    name = Dataset.MNMS1
    label_map = LABEL_MAP

    @staticmethod
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

    @staticmethod
    def _root(root: str | Path | None = None) -> Path:
        """Resolve the MnM/ root (holds Training/Validation/Testing). Override CARDIAC_MNMS1_ROOT.
        Prefers an explicit MnM dir + validates it's actually M&Ms-1 (not a case-insensitive match)."""
        raw = Path(Mnms1Adapter.data_root_raw())
        cands = Base.candidate_dirs("CARDIAC_MNMS1_ROOT", root,
                                    [raw.parent / "MnM", raw / "MnM", raw.parent, raw])
        return Base.first_dir(cands, lambda b: b.is_dir() and Mnms1Adapter._is_mnms1(b), raw.parent / "MnM")

    @staticmethod
    def data_root_raw() -> str:
        return Config.data_root("raw")

    @staticmethod
    def mnms1_info(root: str | Path | None = None) -> dict[str, dict[str, str]]:
        """{External code -> CSV row (Vendor, Centre, ED, ES, Age, Sex, Height, Weight, Pathology)}."""
        base = Mnms1Adapter._root(root)
        csvs = list(base.glob("*.csv")) + list(base.glob("*/*.csv"))
        if not csvs:
            return {}
        return Base.load_csv_info(csvs[0], "External code", alt_key_col="External_code")

    @staticmethod
    def _sa(case: Path) -> tuple[Path, Path]:
        code = case.name
        for ext in (".nii.gz", ".nii"):
            img = case / f"{code}_sa{ext}"
            if img.exists():
                return img, case / f"{code}_sa_gt{ext}"
        return case / f"{code}_sa.nii.gz", case / f"{code}_sa_gt.nii.gz"

    @staticmethod
    def mnms1_cases(root: str | Path | None = None) -> list[Path]:
        """All subject dirs with a SA cine, across Labeled/Validation/Testing. The store flags which
        have usable (non-empty) masks — M&Ms-1 ships withheld/zero-filled GT for much of Testing, so the
        `labelled` column (computed from mask content) is what train/eval filter on; vendor (incl. the
        Canon unseen-vendor slice) is a query over that meta, not a separate case list."""
        base = Mnms1Adapter._root(root)
        out: list[Path] = []
        for split in _SPLITS:
            d = base / split
            if d.is_dir():
                out += [p for p in sorted(d.iterdir()) if p.is_dir() and Mnms1Adapter._sa(p)[1].exists()]
        return out

    @staticmethod
    def _frame_idx(idx: Any) -> int | None:
        """CSV ED/ES frame cell -> int index (float-string tolerant), or None when blank/missing."""
        if idx is None or idx == "":
            return None
        return int(float(idx))

    @staticmethod
    def meta_from_row(row: dict[str, str]) -> PatientMeta:
        """PURE M&Ms-1 meta from one CSV row: centre code -> (readable site, country) via the paper map,
        vendor from either spelling, demographics float-parsed. Unknown centre -> (raw code, None country)."""
        name, country = CENTRES.get(str(row.get("Centre")).strip(), (row.get("Centre"), None))
        return {
            "group": row.get("Pathology"),
            "vendor": row.get("VendorName") or row.get("Vendor"),
            "centre": name, "country": country,   # code -> readable site + country (paper map)
            "age": Base.to_float(row.get("Age")), "sex": row.get("Sex"),
            "height": Base.to_float(row.get("Height")), "weight": Base.to_float(row.get("Weight")),
            "scanner": None, "field_T": None,   # not in CSV (paper tables)
            "_source": {"all": "csv", "centre+country": "paper centre map", "field_T": "paper(unfilled)"},
        }

    @override
    def cases(self) -> list[Path]:
        return Mnms1Adapter.mnms1_cases(self.root)

    @override
    def load_ed_es(self, case: str | Path) -> PatientData:
        """Load ED + ES frames (from the 4D cine at CSV indices) + canonical-remapped masks."""
        case = Path(case)
        row = Mnms1Adapter.mnms1_info().get(case.name, {})
        img_p, gt_p = Mnms1Adapter._sa(case)

        def resolve(tag: str) -> tuple[Path, Path, int] | None:
            idx = Mnms1Adapter._frame_idx(row.get(tag))
            return None if idx is None else (img_p, gt_p, idx)   # 4D cine -> frame index from the CSV

        return Base.load_frames(row.get("Pathology"), resolve, LABEL_MAP)

    @override
    def meta(self, case: Path) -> dict[str, Any]:
        """Acquisition + demographics — AUTO from the CSV (richest of the three)."""
        return dict(Mnms1Adapter.meta_from_row(Mnms1Adapter.mnms1_info().get(case.name, {})))
