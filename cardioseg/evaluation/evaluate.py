"""Segmentation metrics: Dice, Hausdorff, and failure ranking.

The failure ranking is the point — the worst cases are where a clinical measure
would be wrong, so they decide whether the model can be trusted, not the mean Dice.

Shapes: pred/gt are same-shape label maps ([H, W] or [D, H, W]); the ops are
shape-agnostic (they reduce over the whole array). spacing is (z, y, x) mm.
"""
import numpy as np

from cardioseg.types import Mask, Spacing
from cardioseg.labels import CLASSES, FOREGROUND  # re-exported here for back-compat callers

try:
    from scipy.ndimage import distance_transform_edt, binary_erosion
except ImportError:
    distance_transform_edt = binary_erosion = None


def dice(pred: Mask, gt: Mask, label: int) -> float:
    """Dice overlap for one label: 2|P∩G| / (|P|+|G|), in [0, 1]."""
    p, g = (pred == label), (gt == label)
    denom = p.sum() + g.sum()
    if denom == 0:
        return 1.0  # both empty -> perfect by convention
    return 2.0 * np.logical_and(p, g).sum() / denom


def dice_all(pred: Mask, gt: Mask, labels: tuple[int, ...] = FOREGROUND) -> dict[int, float]:
    """Per-label Dice -> {label: dice}."""
    return {int(l): dice(pred, gt, l) for l in labels}


def _surface(m: Mask) -> Mask:
    """Boundary voxels of a binary mask (region minus its erosion)."""
    return m & ~binary_erosion(m)


def surface_distances(pred: Mask, gt: Mask, label: int, spacing: Spacing | None = None) -> np.ndarray:
    """Symmetric boundary-distance array (mm if spacing given): distance from each surface
    voxel of pred to gt, and gt to pred, pooled. This IS the error distribution — HD/HD95/ASSD
    are summaries of it, and the KDE plots it directly. Empty array if either label is absent."""
    if distance_transform_edt is None:
        raise ImportError("scipy required for surface distances")
    p, g = (pred == label), (gt == label)
    if not p.any() or not g.any():
        return np.array([])
    dt_g = distance_transform_edt(~g, sampling=spacing)
    dt_p = distance_transform_edt(~p, sampling=spacing)
    return np.concatenate([dt_g[_surface(p)], dt_p[_surface(g)]]).astype(float)


def surface_metrics(sd: np.ndarray) -> dict[str, float]:
    """HD / HD95 / ASSD from a precomputed surface-distance array (one pass, no recompute)."""
    if sd.size == 0:
        return {"hd": float("nan"), "hd95": float("nan"), "assd": float("nan")}
    return {"hd": float(sd.max()), "hd95": float(np.percentile(sd, 95)), "assd": float(sd.mean())}


def hausdorff(pred: Mask, gt: Mask, label: int, spacing: Spacing | None = None) -> float:
    """Symmetric Hausdorff (max boundary distance); mm if spacing given. Fragile to one
    outlier — prefer hd95 for reporting (G4)."""
    return surface_metrics(surface_distances(pred, gt, label, spacing))["hd"]


def hd95(pred: Mask, gt: Mask, label: int, spacing: Spacing | None = None) -> float:
    """95th-percentile boundary distance — robust Hausdorff (drops the top 5% outliers)."""
    return surface_metrics(surface_distances(pred, gt, label, spacing))["hd95"]


def assd(pred: Mask, gt: Mask, label: int, spacing: Spacing | None = None) -> float:
    """Average symmetric surface distance — the mean of the boundary-distance distribution."""
    return surface_metrics(surface_distances(pred, gt, label, spacing))["assd"]


def rank_failures(case_scores: dict[str, float]) -> list[tuple[str, float]]:
    """case_scores: {name -> mean dice}. Worst-first (where the model fails)."""
    return sorted(case_scores.items(), key=lambda kv: kv[1])
