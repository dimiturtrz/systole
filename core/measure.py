"""Spacing-aware chamber volumes + ejection fraction from a segmentation.

The measurement is the clinical output, so units matter: voxel counts become mL
via the physical voxel volume (mm^3). EF is derived from end-diastolic vs
end-systolic LV blood-pool volume.

Shapes: masks are [D, H, W] integer label maps; spacing is (z, y, x) mm. See
cardioseg/types.py for the convention.
"""
from dataclasses import dataclass
from typing import Any

import numpy as np
from jaxtyping import Float, Integer

from core.data.static.labels import LV_CAV
from core.types import Spacing, shapecheck

_MIN_FIT_PAIRS = 2  # a line needs >=2 distinct points; below this a calibration fit is undefined
LOA_Z = 1.96  # z-multiplier for 95% limits of agreement (Bland–Altman)
_N_BOOT = 2000  # bootstrap resamples for the EF-agreement CI (percentile method)
_CI = 0.95      # coverage of the reported bootstrap interval


@dataclass(frozen=True)
class AgreementStats:
    """Bland–Altman EF agreement summary: n pairs, bias, sample SD, MAE, 95% limits-of-agreement
    [lo, hi], and mean ground-truth EF. One declared schema for the quadruple eval + results share."""
    n: int
    bias: float
    sd: float
    mae: float
    loa: list[float]
    mean_gt: float


@dataclass(frozen=True)
class EfCalibration:
    """Post-hoc linear EF correction ef_corr = slope * ef_pred + intercept, fit on a held-out set.
    Removes a systematic bias in the derived EF number (masks good, ratio biased) without touching the
    model. Identity (slope=1, intercept=0) is the no-op — degenerate fits fall back to it."""
    slope: float
    intercept: float

    def apply(self, ef_pred: Float[np.ndarray, "*n"]) -> Float[np.ndarray, "*n"]:
        """Correct predicted EF(s); NaN (undefined EF) passes through unchanged."""
        return self.slope * np.asarray(ef_pred, dtype=float) + self.intercept


@dataclass(frozen=True)
class EfInterval:
    """Bootstrap EF-agreement point estimate + 95% CI: MAE and bias each with a [lo, hi] over resampled
    pairs. The CI is the defensible error bar a single held-out split (esp. thin ones) otherwise lacks."""
    n: int
    mae: float
    mae_ci: list[float]
    bias: float
    bias_ci: list[float]


class Measure:
    """Spacing-aware chamber volumes + ejection fraction (free helpers folded in as staticmethods)."""

    @staticmethod
    def voxel_volume_ml(spacing: Spacing) -> float:
        """mm^3 per voxel -> mL (1 mL = 1000 mm^3). spacing = (z, y, x) mm."""
        return float(np.prod(spacing)) / 1000.0

    @staticmethod
    @shapecheck
    def label_volume_ml(mask: Integer[np.ndarray, "..."], label: int, spacing: Spacing) -> float:
        """Volume (mL) of one label = voxel count x voxel volume (the Riemann sum). Shape-agnostic (a
        count works on 2D/3D/batched alike); only the dtype (integer label map) is the contract."""
        return int(np.sum(mask == label)) * Measure.voxel_volume_ml(spacing)

    @staticmethod
    @shapecheck
    def expected_volume_ml(prob: Float[np.ndarray, "..."], spacing: Spacing) -> float:
        """EXPECTED volume (mL) from a per-voxel probability map [D,H,W] of one class = Σ prob × voxel
        volume — the 'collapse-never' soft readout. A boundary voxel at p=0.6 contributes 0.6 of a voxel
        instead of 0 or 1, so sub-voxel boundary mass survives into the volume (only meaningful for a
        model that outputs a real gradient, i.e. soft-label-trained + calibrated)."""
        return float(np.asarray(prob).sum()) * Measure.voxel_volume_ml(spacing)

    @staticmethod
    def ef_statistics(ef_gt: Any, ef_pred: Any) -> AgreementStats:
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
            return AgreementStats(0, nan, nan, nan, [nan, nan], nan)
        bias = float(diff.mean())
        sd = float(diff.std(ddof=1)) if n > 1 else 0.0
        return AgreementStats(n, bias, sd, float(np.abs(diff).mean()),
                              [bias - LOA_Z * sd, bias + LOA_Z * sd], float(g.mean()))

    @staticmethod
    def bootstrap_ef_ci(ef_gt: Any, ef_pred: Any, *, n_boot: int = _N_BOOT, seed: int = 0,
                        ci: float = _CI) -> "EfInterval":
        """Percentile bootstrap 95% CI for EF-agreement MAE + bias over paired EF arrays (percent).
        Resamples the (gt, pred) PAIRS with replacement n_boot times; each draw's MAE/bias forms the
        sampling distribution, and the [ci] central percentiles are the interval. This is the honest
        error bar on a single held-out split (esp. thin ones, e.g. Canon n=9) WITHOUT retraining — a
        point estimate with no CI is not defensible. NaN pairs (undefined EF) dropped; n<2 -> point==CI."""
        g = np.asarray(ef_gt, dtype=float)
        p = np.asarray(ef_pred, dtype=float)
        ok = ~(np.isnan(g) | np.isnan(p))
        g, p = g[ok], p[ok]
        n = int(g.size)
        pt = Measure.ef_statistics(g, p)
        if n < _MIN_FIT_PAIRS:
            return EfInterval(n, pt.mae, [pt.mae, pt.mae], pt.bias, [pt.bias, pt.bias])
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, n, size=(n_boot, n))
        diff = p[idx] - g[idx]
        maes = np.abs(diff).mean(axis=1)
        biases = diff.mean(axis=1)
        lo, hi = 100 * (1 - ci) / 2, 100 * (1 + ci) / 2
        return EfInterval(n, pt.mae, [float(np.percentile(maes, lo)), float(np.percentile(maes, hi))],
                          pt.bias, [float(np.percentile(biases, lo)), float(np.percentile(biases, hi))])

    @staticmethod
    def fit_ef_calibration(ef_gt: Any, ef_pred: Any) -> EfCalibration:
        """Least-squares linear fit ef_gt ~ slope * ef_pred + intercept over paired EF arrays (percent).
        NaN pairs (undefined EF) dropped. Selection axis: fit on VAL, apply once to TEST (leak rule).
        Degenerate (<2 pairs or no spread in ef_pred) -> identity, so a bad axis never distorts EF."""
        g = np.asarray(ef_gt, dtype=float)
        p = np.asarray(ef_pred, dtype=float)
        ok = ~(np.isnan(g) | np.isnan(p))
        g, p = g[ok], p[ok]
        if p.size < _MIN_FIT_PAIRS or np.ptp(p) == 0:
            return EfCalibration(1.0, 0.0)
        slope, intercept = np.polyfit(p, g, 1)
        return EfCalibration(float(slope), float(intercept))

    @staticmethod
    @shapecheck
    def ejection_fraction(
        ed_mask: Integer[np.ndarray, "*grid"], es_mask: Integer[np.ndarray, "*grid"],
        spacing: Spacing, lv_label: int = LV_CAV,
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
