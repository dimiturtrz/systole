"""Shared MRI dataset primitives + the DatasetAdapter interface.

Canonical label convention (verified geometrically on real ACDC masks):
0 background, 1 RV cavity, 2 LV myocardium, 3 LV cavity. Every adapter remaps its raw
labels to this via `label_map`, so one model's labels mean the same thing across datasets.

Shapes: volumes are [D, H, W] (D slices, H×W in-plane); spacing is (z, y, x) mm.
"""
from pathlib import Path
from typing import Protocol, TypedDict, runtime_checkable

from cardioseg.types import Image, Mask, Spacing, Volume

LV_CAVITY, LV_MYO, RV_CAVITY = 3, 2, 1
CANONICAL_LABELS = {0: "background", 1: "RV", 2: "LV-myo", 3: "LV-cav"}

# M&Ms raw labels (1=LV-cav, 2=myo, 3=RV) -> canonical; shared by the M&M-2 + M&Ms-1 adapters.
MNM_LABEL_MAP = {0: 0, 1: 3, 2: 2, 3: 1}

_CSV_CACHE: dict[str, dict[str, dict[str, str]]] = {}


def to_float(v):
    """Parse a CSV/cfg field to float; None on missing/garbage. Shared by all adapters."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load_csv_info(csv_path, key_col: str, *, alt_key_col: str | None = None,
                  key_transform=None) -> dict[str, dict[str, str]]:
    """Parse a metadata CSV -> {key -> {column: stripped value}}, cached by path.

    key comes from `key_col` (or `alt_key_col` if that column is absent — datasets ship both
    'External code'/'External_code' spellings); `key_transform` post-processes it (e.g. zero-pad)."""
    cache_key = str(csv_path)
    if cache_key in _CSV_CACHE:
        return _CSV_CACHE[cache_key]
    import csv
    info: dict[str, dict[str, str]] = {}
    p = Path(csv_path)
    if p.exists():
        with p.open(newline="") as f:
            for row in csv.DictReader(f):
                k = (row.get(key_col) or (row.get(alt_key_col) if alt_key_col else None) or "").strip()
                if k:
                    info[key_transform(k) if key_transform else k] = {kk: (vv or "").strip() for kk, vv in row.items()}
    _CSV_CACHE[cache_key] = info
    return info


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


def load_nifti(path: str | Path, frame: int | None = None) -> tuple[Volume, Spacing]:
    """Load a NIfTI volume -> (array [D, H, W], spacing (z, y, x) mm). For a 4D cine, pass
    `frame` to extract one time-index ([x,y,z,t] -> [D,H,W]); 3D inputs ignore `frame`."""
    import nibabel as nib
    import numpy as np
    img = nib.load(str(path))
    arr = np.asanyarray(img.dataobj)          # NIfTI is x,y,z (,t)
    if frame is not None and arr.ndim == 4:
        arr = arr[..., frame]
    arr = np.transpose(arr, (2, 1, 0))        # -> z,y,x (D,H,W)
    zx, zy, zz = img.header.get_zooms()[:3]
    return arr, (zz, zy, zx)


def load_frames(group, resolve, label_map: dict[int, int]) -> "PatientData":
    """Shared ED/ES loader skeleton for the adapters. `resolve(tag)` returns (img_path, gt_path,
    frame|None) for that cardiac phase, or None to skip it; the adapter-specific bit is just that
    closure. Loads each frame (4D-aware via `frame`), remaps the mask to canonical, carries spacing."""
    out: PatientData = {"group": group, "spacing": None}
    for tag in ("ED", "ES"):
        r = resolve(tag)
        if r is None:
            continue
        img_p, gt_p, frame = r
        if not Path(img_p).exists():
            continue
        img, sp = load_nifti(img_p, frame=frame)
        gt, _ = load_nifti(gt_p, frame=frame)
        out["spacing"] = sp
        out[tag] = {"img": img, "gt": apply_label_map(gt, label_map)}
    return out


def apply_label_map(gt: Mask, label_map: dict[int, int]) -> Mask:
    """Remap raw integer labels to the canonical convention. Identity map -> unchanged."""
    import numpy as np
    if not label_map or all(k == v for k, v in label_map.items()):
        return gt
    out = np.zeros_like(gt)
    for src, dst in label_map.items():
        if dst:
            out[gt == src] = dst
    return out


def identify_lv_cavity(mask: Mask, myo_label: int = LV_MYO) -> tuple[int | None, dict[int, float]]:
    """Geometrically identify the LV-cavity label: the non-myo foreground label most enclosed
    by the myocardium ring. Trusts geometry, not a remembered int. mask [D,H,W] or [H,W].
    Returns (lv_label, scores) where score = fraction of a label's shell touching myocardium."""
    import numpy as np
    from scipy import ndimage

    labels = [int(l) for l in np.unique(mask) if l != 0 and l != myo_label]
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
