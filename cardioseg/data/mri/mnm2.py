"""M&M-2 loader (NIfTI) — the multi-vendor, multi-disease external set.

M&M-2 (Multi-Disease, Multi-View & Multi-Centre, 2021) has 360 subjects across
3 vendors (Siemens / Philips / GE), 8 pathologies, 1.5T + 3T. ACDC is single-centre
Siemens, so M&M-2 is the out-of-distribution test (and, trained on, the diverse
*source* with ACDC held out — the stronger generalization story).

Layout: <root>/mnm2/MnM2/dataset/NNN/ with short-axis (SA) + long-axis (LA),
each as CINE (4D) + ED/ES volumes + _gt masks: NNN_SA_ED.nii.gz, NNN_SA_ED_gt.nii.gz,
NNN_SA_ES(.nii.gz/_gt), and the LA equivalents. We use SHORT-AXIS for the 2D model;
LA is ignored. Per-subject DISEASE/VENDOR/SCANNER/FIELD live in dataset_information.csv.

Label convention: M&M-2 ground truth is 1=LV-cavity, 2=LV-myocardium, 3=RV — the
*opposite* of ACDC (1=RV, 3=LV-cavity). VERIFIED geometrically on real masks
(identify_lv_cavity, myo=2 -> LV=1, score 0.5-0.65 vs RV 0.05-0.13). We remap to the
ACDC convention on load so one model's labels mean the same thing across datasets.

Shapes: volumes are [D, H, W]; spacing is (z, y, x) mm. See cardioseg/types.py.
"""
import csv as _csv
import os
from pathlib import Path

from cardioseg.config import data_root
from cardioseg.data.mri.data import PatientData, load_nifti
from cardioseg.types import Mask

# M&M-2 raw label -> ACDC convention (1 RV, 2 myo, 3 LV-cav). Verified on real masks.
_REMAP = {0: 0, 1: 3, 2: 2, 3: 1}

_INFO_CACHE: dict[str, dict[str, dict[str, str]]] = {}


def _dataset_dir(root: str | Path | None = None) -> Path:
    """Resolve the dir holding the NNN/ subject folders, tolerating nesting.

    M&M-2 sits beside ACDC (e.g. data/raw/mri/mnm2 while raw root is .../mri/acdc),
    so we search the raw root, its parent, and common sub-nestings. Override with
    CARDIAC_MNM2_ROOT.
    """
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
    """List subject dirs (M&M-2: NNN/). Defaults to CARDIAC_DATA_ROOT/mnm2/..."""
    d = _dataset_dir(root)
    return sorted((p for p in d.glob("[0-9][0-9][0-9]") if p.is_dir()), key=lambda p: p.name)


def mnm2_info(root: str | Path | None = None) -> dict[str, dict[str, str]]:
    """{subject_code (zero-padded NNN) -> {DISEASE, VENDOR, SCANNER, FIELD}}."""
    d = _dataset_dir(root)
    key = str(d)
    if key in _INFO_CACHE:
        return _INFO_CACHE[key]
    csvp = d.parent / "dataset_information.csv"
    info: dict[str, dict[str, str]] = {}
    if csvp.exists():
        with csvp.open(newline="") as f:
            for row in _csv.DictReader(f):
                code = (row.get("SUBJECT_CODE") or "").strip()
                if code:
                    info[code.zfill(3)] = {k: (v or "").strip() for k, v in row.items()}
    _INFO_CACHE[key] = info
    return info


def _remap(gt: Mask) -> Mask:
    """M&M-2 labels -> ACDC convention (no-op on background)."""
    import numpy as np
    out = np.zeros_like(gt)
    for src, dst in _REMAP.items():
        if dst:
            out[gt == src] = dst
    return out


def load_ed_es(patient_dir: str | Path, view: str = "SA") -> PatientData:
    """Load ED + ES short-axis frames + remapped masks for one M&M-2 subject.

    Mirrors data.load_ed_es so the rest of the pipeline is dataset-agnostic.
    `group` is the DISEASE code; img/gt are [D, H, W], spacing (z, y, x) mm.
    """
    patient_dir = Path(patient_dir)
    pid = patient_dir.name
    info = mnm2_info(patient_dir.parent.parent)
    grp = info.get(pid, {}).get("DISEASE")
    out: PatientData = {"group": grp, "spacing": None}
    for tag in ("ED", "ES"):
        img_p = patient_dir / f"{pid}_{view}_{tag}.nii.gz"
        gt_p = patient_dir / f"{pid}_{view}_{tag}_gt.nii.gz"
        if not img_p.exists():
            continue
        img, sp = load_nifti(img_p)
        gt, _ = load_nifti(gt_p)
        out["spacing"] = sp
        out[tag] = {"img": img, "gt": _remap(gt)}
    return out
