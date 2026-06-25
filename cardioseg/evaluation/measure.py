"""Spacing-aware chamber volumes + ejection fraction from a segmentation.

The measurement is the clinical output, so units matter: voxel counts become mL
via the physical voxel volume (mm^3). EF is derived from end-diastolic vs
end-systolic LV blood-pool volume.

Shapes: masks are [D, H, W] integer label maps; spacing is (z, y, x) mm. See
cardioseg/types.py for the convention.
"""
import numpy as np

from cardioseg.types import Mask, Spacing
from cardioseg.labels import LV_CAV

LOA_Z = 1.96  # z-multiplier for 95% limits of agreement (Bland–Altman)


def voxel_volume_ml(spacing: Spacing) -> float:
    """mm^3 per voxel -> mL (1 mL = 1000 mm^3). spacing = (z, y, x) mm."""
    return float(np.prod(spacing)) / 1000.0


def label_volume_ml(mask: Mask, label: int, spacing: Spacing) -> float:
    """Volume (mL) of one label = voxel count x voxel volume (the Riemann sum)."""
    return int(np.sum(mask == label)) * voxel_volume_ml(spacing)


def ejection_fraction(
    ed_mask: Mask, es_mask: Mask, spacing: Spacing, lv_label: int = LV_CAV
) -> tuple[float, float, float]:
    """EF = (EDV - ESV) / EDV in percent, from LV blood-pool volumes (mL).

    ed_mask / es_mask: [D, H, W] label maps at end-diastole / end-systole.
    Returns (ef_percent, edv_ml, esv_ml). EF is a ratio, so spacing cancels.
    """
    edv = label_volume_ml(ed_mask, lv_label, spacing)
    esv = label_volume_ml(es_mask, lv_label, spacing)
    if edv <= 0:
        return float("nan"), edv, esv
    return (edv - esv) / edv * 100.0, edv, esv
