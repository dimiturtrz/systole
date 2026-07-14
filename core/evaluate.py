"""Segmentation metrics: Dice, Hausdorff, and failure ranking.

The failure ranking is the point — the worst cases are where a clinical measure
would be wrong, so they decide whether the model can be trusted, not the mean Dice.

Shapes: pred/gt are same-shape label maps ([H, W] or [D, H, W]); the ops are
shape-agnostic (they reduce over the whole array). spacing is (z, y, x) mm.
"""
from dataclasses import dataclass
from numbers import Real

import numpy as np
from jaxtyping import Bool, Float, Integer

from core.data.static.labels import (  # noqa: F401  (CLASSES re-exported for back-compat callers: validate/distribution/results)
    CLASSES,
    FOREGROUND,
)
from core.types import shapecheck

try:
    from scipy.ndimage import binary_erosion, distance_transform_edt
except ImportError:
    distance_transform_edt = binary_erosion = None


HD_PERCENTILE = 95  # robust-Hausdorff percentile (drops the top 5% boundary-distance outliers)


@dataclass(frozen=True)
class SurfaceMetrics:
    """Boundary-distance summary: max Hausdorff, robust HD95, average symmetric surface distance (mm)."""
    hd: float
    hd95: float
    assd: float


class Evaluate:
    """Segmentation metrics: Dice + boundary-distance summaries (HD/HD95/ASSD). The failure
    ranking is the point — the worst cases are where a clinical measure would be wrong."""

    @staticmethod
    @shapecheck
    def dice(pred: Integer[np.ndarray, "*grid"], gt: Integer[np.ndarray, "*grid"], label: int) -> float:
        """Dice overlap for one label: 2|P∩G| / (|P|+|G|), in [0, 1]."""
        p, g = (pred == label), (gt == label)
        denom = p.sum() + g.sum()
        if denom == 0:
            return 1.0  # both empty -> perfect by convention
        return 2.0 * np.logical_and(p, g).sum() / denom

    @staticmethod
    @shapecheck
    def _surface(m: Bool[np.ndarray, "*grid"]) -> Bool[np.ndarray, "*grid"]:
        """Boundary voxels of a binary mask (region minus its erosion)."""
        return m & ~binary_erosion(m)

    @staticmethod
    @shapecheck
    def surface_distances(pred: Integer[np.ndarray, "*grid"], gt: Integer[np.ndarray, "*grid"],
                          label: int, spacing: tuple[Real, ...] | None = None) -> Float[np.ndarray, "..."]:
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
        return np.concatenate([dt_g[Evaluate._surface(p)], dt_p[Evaluate._surface(g)]]).astype(float)

    @staticmethod
    @shapecheck
    def surface_metrics(sd: Float[np.ndarray, "..."]) -> SurfaceMetrics:
        """HD / HD95 / ASSD from a precomputed surface-distance array (one pass, no recompute)."""
        if sd.size == 0:
            return SurfaceMetrics(float("nan"), float("nan"), float("nan"))
        return SurfaceMetrics(float(sd.max()), float(np.percentile(sd, HD_PERCENTILE)), float(sd.mean()))

    @staticmethod
    @shapecheck
    def hd95(pred: Integer[np.ndarray, "*grid"], gt: Integer[np.ndarray, "*grid"],
             label: int, spacing: tuple[Real, ...] | None = None) -> float:
        """95th-percentile boundary distance — robust Hausdorff (drops the top 5% outliers)."""
        return Evaluate.surface_metrics(Evaluate.surface_distances(pred, gt, label, spacing)).hd95

    @staticmethod
    @shapecheck
    def assd(pred: Integer[np.ndarray, "*grid"], gt: Integer[np.ndarray, "*grid"],
             label: int, spacing: tuple[Real, ...] | None = None) -> float:
        """Average symmetric surface distance — the mean of the boundary-distance distribution."""
        return Evaluate.surface_metrics(Evaluate.surface_distances(pred, gt, label, spacing)).assd
