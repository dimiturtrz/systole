"""Segmentation metrics: Dice, Hausdorff, and failure ranking.

The failure ranking is the point — the worst cases are where a clinical measure
would be wrong, so they decide whether the model can be trusted, not the mean Dice.
"""
import numpy as np

try:
    from scipy.ndimage import distance_transform_edt
except ImportError:
    distance_transform_edt = None


def dice(pred, gt, label):
    p, g = (pred == label), (gt == label)
    denom = p.sum() + g.sum()
    if denom == 0:
        return 1.0  # both empty -> perfect by convention
    return 2.0 * np.logical_and(p, g).sum() / denom


def dice_all(pred, gt, labels=(1, 2, 3)):
    return {int(l): dice(pred, gt, l) for l in labels}


def hausdorff(pred, gt, label, spacing=None):
    """Symmetric Hausdorff for one label; mm if spacing (z,y,x) given, else voxels."""
    if distance_transform_edt is None:
        raise ImportError("scipy required for Hausdorff")
    p, g = (pred == label), (gt == label)
    if not p.any() or not g.any():
        return float("nan")
    dt_g = distance_transform_edt(~g, sampling=spacing)
    dt_p = distance_transform_edt(~p, sampling=spacing)
    return float(max(dt_g[p].max(), dt_p[g].max()))


def rank_failures(case_scores):
    """case_scores: dict name -> mean dice. Worst-first (where the model fails)."""
    return sorted(case_scores.items(), key=lambda kv: kv[1])
