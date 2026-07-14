"""Shared MRI dataset primitives + the DatasetAdapter interface.

Canonical label convention (verified geometrically on real ACDC masks):
0 background, 1 RV cavity, 2 LV myocardium, 3 LV cavity. Every adapter remaps its raw
labels to this via `label_map`, so one model's labels mean the same thing across datasets.

Shapes: volumes are [D, H, W] (D slices, H×W in-plane); spacing is (z, y, x) mm.
"""
import csv
from enum import StrEnum
from pathlib import Path
from typing import Protocol, TypedDict, runtime_checkable

import nibabel as nib
import numpy as np
from scipy import ndimage

from core.types import Image, Mask, Spacing, Volume

LV_MYO = 2   # LV-myocardium; the identify_lv_cavity ring label (canonical home: core.data.static.labels)

# M&Ms raw labels (1=LV-cav, 2=myo, 3=RV) -> canonical; shared by the M&M-2 + M&Ms-1 adapters.
MNM_LABEL_MAP = {0: 0, 1: 3, 2: 2, 3: 1}

_CSV_CACHE: dict[str, dict[str, dict[str, str]]] = {}


class Vendor(StrEnum):
    """Scanner vendor, canonical casing as emitted in acquisition metadata. One source of truth
    for the four names that were scattered as literals across adapters/splits/reference."""
    SIEMENS = "Siemens"
    PHILIPS = "Philips"
    GE = "GE"
    CANON = "Canon"

    @classmethod
    def _missing_(cls, value: object) -> "Vendor | None":
        """Case-insensitive lookup (a CSV/DICOM field may ship 'SIEMENS'/'ge')."""
        if isinstance(value, str):
            low = value.strip().lower()
            return next((m for m in cls if m.value.lower() == low), None)
        return None


class Phase(StrEnum):
    """Cardiac-cycle phase tag. Members are the canonical uppercase forms; `_missing_` folds the
    ED/ed/ES/es case drift so Phase('ed') is Phase.ED. StrEnum == its str value, so it stays a
    drop-in key for the PatientData/Frame dicts keyed by 'ED'/'ES'."""
    ED = "ED"
    ES = "ES"

    @classmethod
    def _missing_(cls, value: object) -> "Phase | None":
        if isinstance(value, str):
            up = value.strip().upper()
            return next((m for m in cls if m.value == up), None)
        return None


class Frame(TypedDict):
    """One cardiac-phase frame: image + its label mask, both [D, H, W]."""
    img: Image
    gt: Mask


class PatientData(TypedDict, total=False):
    """One subject's ED/ES frames + metadata (returned by an adapter's load_ed_es)."""
    group: str | None          # pathology code
    spacing: Spacing | None    # (z, y, x) mm
    ED: Frame                  # end-diastole (fullest)
    ES: Frame                  # end-systole (emptiest)


class Base:
    """The shared MRI-adapter free helpers, folded in as staticmethods (CSV parse, NIfTI load, the ED/ES
    loader skeleton, label remap, geometric LV-cavity id). Every adapter + the store call these."""

    @staticmethod
    def to_float(v):
        """Parse a CSV/cfg field to float; None on missing/garbage. Shared by all adapters."""
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def load_csv_info(csv_path, key_col: str, *, alt_key_col: str | None = None,
                      key_transform=None) -> dict[str, dict[str, str]]:
        """Parse a metadata CSV -> {key -> {column: stripped value}}, cached by path.

        key comes from `key_col` (or `alt_key_col` if that column is absent — datasets ship both
        'External code'/'External_code' spellings); `key_transform` post-processes it (e.g. zero-pad)."""
        cache_key = str(csv_path)
        if cache_key in _CSV_CACHE:
            return _CSV_CACHE[cache_key]
        info: dict[str, dict[str, str]] = {}
        p = Path(csv_path)
        if p.exists():
            with p.open(newline="", encoding="utf-8-sig") as f:      # utf-8-sig strips a BOM (SCD csv has one)
                for row in csv.DictReader(f):
                    k = (row.get(key_col) or (row.get(alt_key_col) if alt_key_col else None) or "").strip()
                    if k:
                        info[key_transform(k) if key_transform else k] = {kk: (vv or "").strip() for kk, vv in row.items()}
        _CSV_CACHE[cache_key] = info
        return info

    @staticmethod
    def load_nifti(path: str | Path, frame: int | None = None) -> tuple[Volume, Spacing]:
        """Load a NIfTI volume -> (array [D, H, W], spacing (z, y, x) mm). For a 4D cine, pass
        `frame` to extract one time-index ([x,y,z,t] -> [D,H,W]); 3D inputs ignore `frame`."""
        img = nib.load(str(path))
        arr = np.asanyarray(img.dataobj)          # NIfTI is x,y,z (,t)
        if frame is not None and arr.ndim == 4:  # noqa: PLR2004 (4D cine: x,y,z,t)
            arr = arr[..., frame]
        arr = np.transpose(arr, (2, 1, 0))        # -> z,y,x (D,H,W)
        zx, zy, zz = img.header.get_zooms()[:3]
        return arr, (zz, zy, zx)

    @staticmethod
    def load_frames(group, resolve, label_map: dict[int, int]) -> "PatientData":
        """Shared ED/ES loader skeleton for the adapters. `resolve(tag)` returns (img_path, gt_path,
        frame|None) for that cardiac phase, or None to skip it; the adapter-specific bit is just that
        closure. Loads each frame (4D-aware via `frame`), remaps the mask to canonical, carries spacing."""
        out: PatientData = {"group": group, "spacing": None}
        for tag in Phase:
            r = resolve(tag)
            if r is None:
                continue
            img_p, gt_p, frame = r
            if not Path(img_p).exists():
                continue
            img, sp = Base.load_nifti(img_p, frame=frame)
            gt, _ = Base.load_nifti(gt_p, frame=frame)
            out["spacing"] = sp
            out[tag] = {"img": img, "gt": Base.apply_label_map(gt, label_map)}
        return out

    @staticmethod
    def apply_label_map(gt: Mask, label_map: dict[int, int]) -> Mask:
        """Remap raw integer labels to the canonical convention. Identity map -> unchanged."""
        if not label_map or all(k == v for k, v in label_map.items()):
            return gt
        out = np.zeros_like(gt)
        for src, dst in label_map.items():
            if dst:
                out[gt == src] = dst
        return out

    @staticmethod
    def identify_lv_cavity(mask: Mask, myo_label: int = LV_MYO) -> tuple[int | None, dict[int, float]]:
        """Geometrically identify the LV-cavity label: the non-myo foreground label most enclosed
        by the myocardium ring. Trusts geometry, not a remembered int. mask [D,H,W] or [H,W].
        Returns (lv_label, scores) where score = fraction of a label's shell touching myocardium."""
        labels = [int(label) for label in np.unique(mask) if label not in (0, myo_label)]
        myo = mask == myo_label
        scores: dict[int, float] = {}
        for lab in labels:
            cav = mask == lab
            shell = ndimage.binary_dilation(cav) & ~cav
            scores[lab] = float((shell & myo).sum()) / float(shell.sum()) if shell.sum() else 0.0
        lv = max(scores, key=scores.get) if scores else None
        return lv, scores


@runtime_checkable
class DatasetAdapter(Protocol):
    """What every dataset adapter provides — the dataset-agnostic interface the pipeline rides on.

    `label_map` makes the per-dataset label convention *interface data*, not hardcoded logic.
    `meta()` is the normalization hook: acquisition/demographics parsed from the dataset's own
    shipped files (the AUTO tier), feeding stratified eval + the reference store.
    """
    name: str                   # also the processed/<name>/ folder in the store
    label_map: dict[int, int]   # raw int -> canonical (0 bg, 1 RV, 2 myo, 3 LV-cav)

    def cases(self) -> list[Path]: ...
    def load_ed_es(self, case: Path) -> PatientData: ...   # labels already canonical
    def meta(self, case: Path) -> dict: ...                # parsed acquisition/demographics
