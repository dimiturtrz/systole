"""Spacing-aware chamber volumes + ejection fraction from a segmentation.

The measurement is the clinical output, so units matter: voxel counts become mL
via the physical voxel volume (mm^3). EF is derived from end-diastolic vs
end-systolic LV blood-pool volume.

Shapes: masks are [D, H, W] integer label maps; spacing is (z, y, x) mm. See
cardioseg/types.py for the convention.
"""
import numpy as np

from core.data.static.labels import LV_CAV
from core.types import Mask, Spacing

LOA_Z = 1.96  # z-multiplier for 95% limits of agreement (Bland–Altman)


class Measure:
    """Spacing-aware chamber volumes + ejection fraction (free helpers folded in as staticmethods)."""

    @staticmethod
    def voxel_volume_ml(spacing: Spacing) -> float:
        """mm^3 per voxel -> mL (1 mL = 1000 mm^3). spacing = (z, y, x) mm."""
        return float(np.prod(spacing)) / 1000.0

    @staticmethod
    def label_volume_ml(mask: Mask, label: int, spacing: Spacing) -> float:
        """Volume (mL) of one label = voxel count x voxel volume (the Riemann sum)."""
        return int(np.sum(mask == label)) * Measure.voxel_volume_ml(spacing)

    @staticmethod
    def expected_volume_ml(prob: np.ndarray, spacing: Spacing) -> float:
        """EXPECTED volume (mL) from a per-voxel probability map [D,H,W] of one class = Σ prob × voxel
        volume — the 'collapse-never' soft readout. A boundary voxel at p=0.6 contributes 0.6 of a voxel
        instead of 0 or 1, so sub-voxel boundary mass survives into the volume (only meaningful for a
        model that outputs a real gradient, i.e. soft-label-trained + calibrated)."""
        return float(np.asarray(prob).sum()) * Measure.voxel_volume_ml(spacing)

    @staticmethod
    def ef_statistics(ef_gt, ef_pred) -> dict:
        """Bland–Altman EF agreement over paired EF arrays (percent). Single source for the
        bias / SD / MAE / 95% limits-of-agreement quadruple that eval + results otherwise recompute
        (with drifting ddof / NaN handling). NaN pairs (EDV<=0 -> undefined EF) are dropped first.
        SD uses ddof=1 (sample). Returns {n, bias, sd, mae, loa=[lo,hi], mean_gt}; all-NaN -> NaNs."""
        g = np.asarray(ef_gt, dtype=float)
        p = np.asarray(ef_pred, dtype=float)
        diff = p - g
        ok = ~np.isnan(diff)
        diff, g = diff[ok], g[ok]
        n = int(diff.size)
        if n == 0:
            nan = float("nan")
            return {"n": 0, "bias": nan, "sd": nan, "mae": nan, "loa": [nan, nan], "mean_gt": nan}
        bias = float(diff.mean())
        sd = float(diff.std(ddof=1)) if n > 1 else 0.0
        return {"n": n, "bias": bias, "sd": sd, "mae": float(np.abs(diff).mean()),
                "loa": [bias - LOA_Z * sd, bias + LOA_Z * sd], "mean_gt": float(g.mean())}

    @staticmethod
    def ejection_fraction(
        ed_mask: Mask, es_mask: Mask, spacing: Spacing, lv_label: int = LV_CAV
    ) -> tuple[float, float, float]:
        """EF = (EDV - ESV) / EDV in percent, from LV blood-pool volumes (mL).

        ed_mask / es_mask: [D, H, W] label maps at end-diastole / end-systole.
        Returns (ef_percent, edv_ml, esv_ml). EF is a ratio, so spacing cancels.
        """
        edv = Measure.label_volume_ml(ed_mask, lv_label, spacing)
        esv = Measure.label_volume_ml(es_mask, lv_label, spacing)
        if edv <= 0:
            return float("nan"), edv, esv
        return (edv - esv) / edv * 100.0, edv, esv
