"""M&M-2 adapter (NIfTI) — multi-vendor, multi-disease. 360 subjects, 3 vendors
(Siemens/Philips/GE), 1.5T + 3T. The diverse training source (ACDC held out).

Layout: <root>/mnm2/MnM2/dataset/NNN/ with SA (short-axis) + LA, each CINE(4D) + ED/ES
volumes + _gt: NNN_SA_ED.nii.gz, NNN_SA_ED_gt.nii.gz, NNN_SA_ES(.nii.gz/_gt). We use SA.
Per-subject DISEASE/VENDOR/SCANNER/FIELD in dataset_information.csv.

Labels: raw 1=LV-cav, 2=myo, 3=RV — opposite of ACDC. `label_map` remaps to canonical
(verified geometrically: myo=2 -> LV=1 in raw). The flip is interface data, not buried logic.
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

LABEL_MAP = MNM_LABEL_MAP   # raw -> canonical (LV-cav 1->3, RV 3->1); shared M&Ms flip


class Mnm2Adapter(AdapterBase, DatasetAdapter):
    """M&M-2: multi-vendor training source; owns its dataset-dir search, subject-code CSV keying, and
    meta assembly (staticmethods); root override via AdapterBase. Labels remapped to canonical via label_map."""
    name = Dataset.MNM2
    label_map = LABEL_MAP

    @staticmethod
    def _dataset_dir(root: str | Path | None = None) -> Path:
        """Resolve the dir holding the NNN/ subject folders, tolerating nesting.
        Searches the raw root + parent + common nestings. Override with CARDIAC_MNM2_ROOT."""
        raw = Path(Config.data_root("raw"))
        cands = Base.candidate_dirs("CARDIAC_MNM2_ROOT", root, [raw, raw.parent],
                                    subs=(".", "mnm2/MnM2/dataset", "MnM2/dataset", "mnm2/dataset",
                                          "dataset", "mri/mnm2/MnM2/dataset"))
        return Base.first_dir(cands, lambda c: c.is_dir() and any(c.glob("[0-9][0-9][0-9]")), raw)

    @staticmethod
    def _info(root: str | Path | None = None) -> dict[str, dict[str, str]]:
        """{subject_code (zero-padded NNN) -> {DISEASE, VENDOR, SCANNER, FIELD}}."""
        d = Mnm2Adapter._dataset_dir(root)
        return Base.load_csv_info(d.parent / "dataset_information.csv", "SUBJECT_CODE",
                             key_transform=lambda c: c.zfill(3))

    @staticmethod
    def _meta_from_info(info: dict[str, str]) -> PatientMeta:
        """PURE M&M-2 meta from one CSV row: disease/vendor/scanner pass through, FIELD float-parsed;
        centre null (not published) + country fixed Spain (3 Spanish hospitals, paper)."""
        return {
            "group": info.get("DISEASE"),
            "vendor": info.get("VENDOR"), "scanner": info.get("SCANNER"),
            "field_T": Base.to_float(info.get("FIELD")),
            # 3 Spanish hospitals; per-subject centre not published, but country is uniform.
            "centre": None, "country": "Spain",
            "age": None, "sex": None, "height": None, "weight": None,
            "_source": {"all": "dataset_information.csv", "country": "paper (3 Spanish hospitals)"},
        }

    @override
    def cases(self) -> list[Path]:
        """List subject dirs (M&M-2: NNN/)."""
        d = self._dataset_dir(self.root)
        return sorted((p for p in d.glob("[0-9][0-9][0-9]") if p.is_dir()), key=lambda p: p.name)

    @override
    def load_ed_es(self, case: Path, view: str = "SA") -> PatientData:
        """Load ED + ES short-axis frames + canonical-remapped masks for one M&M-2 subject."""
        patient_dir = Path(case)
        pid = patient_dir.name
        grp = self._info(patient_dir.parent.parent).get(pid, {}).get("DISEASE")

        def resolve(tag: str) -> tuple[Path, Path, None]:
            return (patient_dir / f"{pid}_{view}_{tag}.nii.gz",
                    patient_dir / f"{pid}_{view}_{tag}_gt.nii.gz", None)

        return Base.load_frames(grp, resolve, LABEL_MAP)

    @override
    def meta(self, case: Path) -> dict[str, Any]:
        """Acquisition + disease — AUTO from dataset_information.csv."""
        return self._meta_from_info(self._info(case.parent.parent).get(case.name, {}))
