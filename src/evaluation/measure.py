"""Spacing-aware chamber volumes + ejection fraction from a segmentation.

The measurement is the clinical output, so units matter: voxel counts become mL
via the physical voxel volume (mm^3). EF is derived from end-diastolic vs
end-systolic LV blood-pool volume.
"""
import numpy as np


def voxel_volume_ml(spacing):
    """mm^3 per voxel -> mL (1 mL = 1000 mm^3)."""
    return float(np.prod(spacing)) / 1000.0


def label_volume_ml(mask, label, spacing):
    return int(np.sum(mask == label)) * voxel_volume_ml(spacing)


def ejection_fraction(ed_mask, es_mask, spacing, lv_label=1):
    """EF = (EDV - ESV) / EDV in percent, from LV blood-pool volumes (mL).

    Returns (ef_percent, edv_ml, esv_ml).
    """
    edv = label_volume_ml(ed_mask, lv_label, spacing)
    esv = label_volume_ml(es_mask, lv_label, spacing)
    if edv <= 0:
        return float("nan"), edv, esv
    return (edv - esv) / edv * 100.0, edv, esv
