"""M&Ms-1 adapter (NIfTI) — the broadest multi-site set: 375 subjects, 6 centres
(Spain×4, Germany, Canada), 4 vendors (Siemens/Philips/GE/Canon).

Layout: <root>/MnM/{Training/{Labeled,Unlabeled}, Validation, Testing}/<code>/ with a 4D
short-axis cine <code>_sa.nii(.gz) + 4D gt <code>_sa_gt.nii(.gz) where only the ED/ES
frames are labelled. ED/ES frame indices + vendor/centre/age/sex/h+w in the CSV.

Labels: same flip as M&M-2 (raw 1=LV-cav) -> canonical via label_map.
"""
import csv as _csv
import os
from pathlib import Path

from cardioseg.data.mri.base import PatientData, apply_label_map

LABEL_MAP = {0: 0, 1: 3, 2: 2, 3: 1}

_INFO_CACHE: dict[str, dict[str, dict[str, str]]] = {}
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
    from cardioseg.config import data_root
    return data_root("raw")


def _load_4d_frame(path: Path, frame: int):
    """Load one time-frame of a 4D NIfTI -> ([D,H,W], spacing (z,y,x) mm)."""
    import nibabel as nib
    import numpy as np
    img = nib.load(str(path))
    arr = np.asanyarray(img.dataobj)              # (x, y, z, t)
    vol = arr[..., frame] if arr.ndim == 4 else arr
    vol = np.transpose(vol, (2, 1, 0))            # -> (z, y, x) = [D, H, W]
    zx, zy, zz = img.header.get_zooms()[:3]
    return vol, (zz, zy, zx)


def mnms1_info(root: str | Path | None = None) -> dict[str, dict[str, str]]:
    """{External code -> CSV row (Vendor, Centre, ED, ES, Age, Sex, Height, Weight, Pathology)}."""
    base = _root(root)
    key = str(base)
    if key in _INFO_CACHE:
        return _INFO_CACHE[key]
    info: dict[str, dict[str, str]] = {}
    csvs = list(base.glob("*.csv")) + list(base.glob("*/*.csv"))
    if csvs:
        with csvs[0].open(newline="") as f:
            for row in _csv.DictReader(f):
                code = (row.get("External code") or row.get("External_code") or "").strip()
                if code:
                    info[code] = {k: (v or "").strip() for k, v in row.items()}
    _INFO_CACHE[key] = info
    return info


def _sa(case: Path) -> tuple[Path, Path]:
    code = case.name
    for ext in (".nii.gz", ".nii"):
        img = case / f"{code}_sa{ext}"
        if img.exists():
            return img, case / f"{code}_sa_gt{ext}"
    return case / f"{code}_sa.nii.gz", case / f"{code}_sa_gt.nii.gz"


_LABELLED_CACHE: dict[str, bool] = {}


def _has_labels(case: Path) -> bool:
    """True iff BOTH ED and ES GT frames are non-empty. M&Ms-1 ships withheld (zero-filled) GT for
    much of Testing (and a few Validation/Training cases) — the gt *file* exists but is all-background,
    so a file-existence check is not enough. Cached per case (loading the 4D GT is the cost)."""
    key = str(case)
    if key not in _LABELLED_CACHE:
        d = load_ed_es(case)
        import numpy as np
        ok = all(tag in d and bool((d[tag]["gt"] > 0).any()) for tag in ("ED", "ES"))
        _LABELLED_CACHE[key] = ok
    return _LABELLED_CACHE[key]


def mnms1_cases(root: str | Path | None = None, vendor: str | None = None,
                labelled_only: bool = False) -> list[Path]:
    """Subject dirs with a SA cine, across Labeled/Validation/Testing.

    `vendor` (e.g. "Canon") filters to one vendor via the CSV — carves the unseen-vendor held-out
    test out of M&Ms-1. `labelled_only` drops cases whose GT is withheld (zero-filled) — REQUIRED for
    train/eval (320 cases on disk, only 213 have usable masks). Leave False for metadata inventory."""
    base = _root(root)
    out: list[Path] = []
    for split in _SPLITS:
        d = base / split
        if d.is_dir():
            out += [p for p in sorted(d.iterdir()) if p.is_dir() and _sa(p)[1].exists()]
    if vendor is not None:
        info = mnms1_info(root)
        out = [p for p in out
               if (info.get(p.name, {}).get("VendorName") or info.get(p.name, {}).get("Vendor")) == vendor]
    if labelled_only:
        out = [p for p in out if _has_labels(p)]
    return out


def load_ed_es(case: str | Path, root: str | Path | None = None) -> PatientData:
    """Load ED + ES frames (from the 4D cine at CSV indices) + canonical-remapped masks."""
    case = Path(case)
    row = mnms1_info(root).get(case.name, {})
    img_p, gt_p = _sa(case)
    out: PatientData = {"group": row.get("Pathology"), "spacing": None}
    for tag in ("ED", "ES"):
        idx = row.get(tag)
        if idx is None or idx == "" or not img_p.exists():
            continue
        f = int(float(idx))
        img, sp = _load_4d_frame(img_p, f)
        gt, _ = _load_4d_frame(gt_p, f)
        out["spacing"] = sp
        out[tag] = {"img": img, "gt": apply_label_map(gt, LABEL_MAP)}
    return out


class Mnms1Adapter:
    """M&Ms-1: 6-centre / 4-vendor (incl. Canon); richest demographics (age/sex/BSA)."""
    name = "mnms1"
    cache_ns = "mnms1"
    label_map = LABEL_MAP

    def cases(self) -> list[Path]:
        return mnms1_cases()

    def load_ed_es(self, case: Path) -> PatientData:
        return load_ed_es(case)

    def meta(self, case: Path) -> dict:
        """Acquisition + demographics — AUTO from the CSV (richest of the three)."""
        r = mnms1_info().get(case.name, {})
        return {
            "group": r.get("Pathology"),
            "vendor": r.get("VendorName") or r.get("Vendor"), "centre": r.get("Centre"),
            "age": _f(r.get("Age")), "sex": r.get("Sex"),
            "height": _f(r.get("Height")), "weight": _f(r.get("Weight")),
            "scanner": None, "field_T": None,   # not in CSV (paper tables)
            "_source": {"all": "csv", "field_T": "paper(unfilled)"},
        }


class CanonAdapter(Mnms1Adapter):
    """Canon-only slice of M&Ms-1 — the clean unseen-vendor held-out test (overlap-free with
    M&M-2, so usable WITHOUT dedup). Shares the mnms1 preprocess cache (these ARE mnms1 subjects)."""
    name = "canon"
    cache_ns = "mnms1"   # same subjects -> reuse mnms1's cache, no recompute

    def cases(self) -> list[Path]:
        return mnms1_cases(vendor="Canon", labelled_only=True)   # 9 labelled (Testing GT withheld)


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
