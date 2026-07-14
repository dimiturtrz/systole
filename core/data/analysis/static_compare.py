"""Multi-metric static comparison: synth vs real, beyond voxel-voxel intensity (bd uy4d/analysis).

synth_fidelity.py covers the COLOR axis (per-class intensity W1). shape_coverage.py covers SHAPE
coverage. This adds the geometric/clinical metric PANEL — distributions of interpretable quantities
(chamber VOLUME, myo WALL THICKNESS, LV SPHERICITY, chamber ratios, compactness) compared real-vs-synth
by Wasserstein-1. Mask-derived (no painting needed), so it's a pure SHAPE/GEOMETRY comparison.

NOT here (need ED+ES pairs — synth is single-frame ED-only from Rodero ED meshes): EF, ESV/EDV,
ED->ES DEFORMATION. Those require a contraction model to generate ES from ED (flagged for later).
"""
from __future__ import annotations

import json
import logging

import numpy as np
import torch
from jaxtyping import Integer
from scipy.ndimage import binary_erosion, distance_transform_edt

from core.data.dynamic.anatomy import Anatomy

from .shape_coverage import ShapeCoverage
from .synth_fidelity import SynthFidelity

log = logging.getLogger("cardioseg.static_compare")

_RV, _MYO, _LVC = 1, 2, 3

_MIN_FG_PX = 40      # below this many foreground px a 2D label map is ~empty -> skip
_MIN_LVC_PX = 20     # below this many LV-cavity px sphericity is too noisy to report


class StaticCompare:
    """Static geometry/biomarker panel: mask-derived interpretable quantities compared real-vs-synth by
    W1 (the free helpers folded in as staticmethods): per-mask geometry metrics, per-metric distribution
    assembly, and the W1 + median comparison panel."""

    @staticmethod
    def geom_metrics(mask: Integer[np.ndarray, "*grid"]) -> dict | None:
        """Interpretable geometry/biomarkers for one 2D label map (px units), or None if ~empty."""
        foreground = mask > 0
        if int(foreground.sum()) < _MIN_FG_PX:
            return None
        rv, myo, lvc = mask == _RV, mask == _MYO, mask == _LVC
        rv_area, myo_area, lvc_area = int(rv.sum()), int(myo.sum()), int(lvc.sum())
        metrics = {"rv_area": rv_area, "myo_area": myo_area, "lvc_area": lvc_area, "fg_area": int(foreground.sum())}
        # myo WALL THICKNESS: mean over the myo of the distance-to-non-myo (×2 ≈ local thickness)
        distance_transform = distance_transform_edt(myo)
        metrics["myo_thickness"] = float(distance_transform[myo].mean() * 2.0) if myo_area else 0.0   # over myo pixels only (px)
        # LV-cavity SPHERICITY (2D roundness): 4πA / P²  (1 = perfect circle). P ≈ boundary-pixel count.
        if lvc_area >= _MIN_LVC_PX:
            perimeter = int((lvc & ~binary_erosion(lvc)).sum())
            metrics["lvc_sphericity"] = float(4 * np.pi * lvc_area / (perimeter * perimeter)) if perimeter else 0.0
        else:
            metrics["lvc_sphericity"] = 0.0
        metrics["rv_lv_ratio"] = rv_area / (lvc_area + 1.0)                 # RV vs LV cavity balance
        metrics["myo_lv_ratio"] = myo_area / (lvc_area + 1.0)              # myo mass per cavity (hypertrophy proxy)
        return metrics

    @staticmethod
    def _dist(masks) -> dict:
        rows = [metrics for metrics in (StaticCompare.geom_metrics(mask) for mask in masks) if metrics is not None]
        keys = rows[0].keys()
        return {metric: np.array([row[metric] for row in rows], dtype=np.float64) for metric in keys}

    @staticmethod
    def compare(real_masks, synth_masks) -> dict:
        """Per-metric W1(real, synth) + real/synth medians — the geometry/biomarker panel."""
        real_distributions, synth_distributions = StaticCompare._dist(real_masks), StaticCompare._dist(synth_masks)
        comparison = {}
        for metric in real_distributions:
            real_values, synth_values = torch.tensor(real_distributions[metric]), torch.tensor(synth_distributions[metric])
            comparison[metric] = {"w1": round(SynthFidelity.wasserstein1d(real_values, synth_values), 3),
                      "real_median": round(float(np.median(real_distributions[metric])), 2),
                      "synth_median": round(float(np.median(synth_distributions[metric])), 2)}
        return comparison


    @staticmethod
    def add_args(ap):
        ap.add_argument("--real", required=True, help="processed ACDC data dir (patient*.npz)")
        ap.add_argument("--pool", required=True, help="synth anatomy pool .npz")

    @staticmethod
    def run(args):  # pragma: no cover
        results = StaticCompare.compare(ShapeCoverage.real_masks(args.real), Anatomy.load_pool(args.pool))
        log.info(json.dumps(results, indent=2))
        worst = max(results, key=lambda metric: results[metric]["w1"])
        log.info(f"# worst-matched geometry metric: {worst} (W1={results[worst]['w1']})")
