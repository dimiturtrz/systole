"""M&M-2 adapter (NIfTI) — multi-vendor, multi-disease. 360 subjects, 3 vendors
(Siemens/Philips/GE), 1.5T + 3T. The diverse training source (ACDC held out).

Layout: <root>/mnm2/MnM2/dataset/NNN/ with SA (short-axis) + LA, each CINE(4D) + ED/ES
volumes + _gt: NNN_SA_ED.nii.gz, NNN_SA_ED_gt.nii.gz, NNN_SA_ES(.nii.gz/_gt). We use SA.
Per-subject DISEASE/VENDOR/SCANNER/FIELD in dataset_information.csv.

Labels: raw 1=LV-cav, 2=myo, 3=RV — opposite of ACDC. `label_map` remaps to canonical
(verified geometrically: myo=2 -> LV=1 in raw). The flip is interface data, not buried logic.
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
    to_float,
)

LABEL_MAP = MNM_LABEL_MAP   # raw -> canonical (LV-cav 1->3, RV 3->1); shared M&Ms flip


def _dataset_dir(root: str | Path | None = None) -> Path:
    """Resolve the dir holding the NNN/ subject folders, tolerating nesting.
    Searches the raw root + parent + common nestings. Override with CARDIAC_MNM2_ROOT."""
    env = os.environ.get("CARDIAC_MNM2_ROOT")
    bases = [Path(env)] if env else []
    if root is not None:
        bases.append(Path(root))
    raw = Path(data_root("raw"))
    bases += [raw, raw.parent]
    subs = (".", "mnm2/MnM2/dataset", "MnM2/dataset", "mnm2/dataset", "dataset",
            "mri/mnm2/MnM2/dataset")
    for base in bases:
        for sub in subs:
            cand = base if sub == "." else base / sub
            if cand.is_dir() and any(cand.glob("[0-9][0-9][0-9]")):
                return cand
    return raw


def mnm2_cases(root: str | Path | None = None) -> list[Path]:
    """List subject dirs (M&M-2: NNN/)."""
    d = _dataset_dir(root)
    return sorted((p for p in d.glob("[0-9][0-9][0-9]") if p.is_dir()), key=lambda p: p.name)


def mnm2_info(root: str | Path | None = None) -> dict[str, dict[str, str]]:
    """{subject_code (zero-padded NNN) -> {DISEASE, VENDOR, SCANNER, FIELD}}."""
    d = _dataset_dir(root)
    return load_csv_info(d.parent / "dataset_information.csv", "SUBJECT_CODE",
                         key_transform=lambda c: c.zfill(3))


def load_ed_es(patient_dir: str | Path, view: str = "SA") -> PatientData:
    """Load ED + ES short-axis frames + canonical-remapped masks for one M&M-2 subject."""
    patient_dir = Path(patient_dir)
    pid = patient_dir.name
    grp = mnm2_info(patient_dir.parent.parent).get(pid, {}).get("DISEASE")

    def resolve(tag):
        return (patient_dir / f"{pid}_{view}_{tag}.nii.gz",
                patient_dir / f"{pid}_{view}_{tag}_gt.nii.gz", None)

    return load_frames(grp, resolve, LABEL_MAP)


class Mnm2Adapter(DatasetAdapter):
    """M&M-2: multi-vendor training source; labels remapped to canonical via label_map."""
    name = "mnm2"
    label_map = LABEL_MAP

    def cases(self) -> list[Path]:
        return mnm2_cases()

    def load_ed_es(self, case: Path) -> PatientData:
        return load_ed_es(case)

    def meta(self, case: Path) -> dict:
        """Acquisition + disease — AUTO from dataset_information.csv."""
        info = mnm2_info(case.parent.parent).get(case.name, {})
        return {
            "group": info.get("DISEASE"),
            "vendor": info.get("VENDOR"), "scanner": info.get("SCANNER"),
            "field_T": to_float(info.get("FIELD")),
            # 3 Spanish hospitals; per-subject centre not published, but country is uniform.
            "centre": None, "country": "Spain",
            "age": None, "sex": None, "height": None, "weight": None,
            "_source": {"all": "dataset_information.csv", "country": "paper (3 Spanish hospitals)"},
        }
