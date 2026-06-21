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


def load_nifti(path: str | Path) -> tuple[Volume, Spacing]:
    """Load a 3D NIfTI volume. Returns (array [D, H, W], spacing (z, y, x) mm)."""
    import nibabel as nib
    import numpy as np
    img = nib.load(str(path))
    arr = np.asanyarray(img.dataobj)          # NIfTI is x,y,z
    arr = np.transpose(arr, (2, 1, 0))        # -> z,y,x (D,H,W)
    zx, zy, zz = img.header.get_zooms()[:3]
    return arr, (zz, zy, zx)


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
