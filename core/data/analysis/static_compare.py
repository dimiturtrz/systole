"""Multi-metric static comparison: synth vs real, beyond voxel-voxel intensity (bd uy4d/analysis).

synth_fidelity.py covers the COLOR axis (per-class intensity W1). shape_coverage.py covers SHAPE
coverage. This adds the geometric/clinical metric PANEL — distributions of interpretable quantities
(chamber VOLUME, myo WALL THICKNESS, LV SPHERICITY, chamber ratios, compactness) compared real-vs-synth
by Wasserstein-1. Mask-derived (no painting needed), so it's a pure SHAPE/GEOMETRY comparison.

NOT here (need ED+ES pairs — synth is single-frame ED-only from Rodero ED meshes): EF, ESV/EDV,
ED->ES DEFORMATION. Those require a contraction model to generate ES from ED (flagged for later).
"""
from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt

from core.data.dynamic.anatomy import Anatomy
from core.obs import setup

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
    def geom_metrics(mask: np.ndarray) -> dict | None:
        """Interpretable geometry/biomarkers for one 2D label map (px units), or None if ~empty."""
        fg = mask > 0
        if int(fg.sum()) < _MIN_FG_PX:
            return None
        rv, myo, lvc = mask == _RV, mask == _MYO, mask == _LVC
        a_rv, a_myo, a_lvc = int(rv.sum()), int(myo.sum()), int(lvc.sum())
        m = {"rv_area": a_rv, "myo_area": a_myo, "lvc_area": a_lvc, "fg_area": int(fg.sum())}
        # myo WALL THICKNESS: mean over the myo of the distance-to-non-myo (×2 ≈ local thickness)
        dt = distance_transform_edt(myo)
        m["myo_thickness"] = float(dt[myo].mean() * 2.0) if a_myo else 0.0   # over myo pixels only (px)
        # LV-cavity SPHERICITY (2D roundness): 4πA / P²  (1 = perfect circle). P ≈ boundary-pixel count.
        if a_lvc >= _MIN_LVC_PX:
            per = int((lvc & ~binary_erosion(lvc)).sum())
            m["lvc_sphericity"] = float(4 * np.pi * a_lvc / (per * per)) if per else 0.0
        else:
            m["lvc_sphericity"] = 0.0
        m["rv_lv_ratio"] = a_rv / (a_lvc + 1.0)                 # RV vs LV cavity balance
        m["myo_lv_ratio"] = a_myo / (a_lvc + 1.0)              # myo mass per cavity (hypertrophy proxy)
        return m

    @staticmethod
    def _dist(masks) -> dict:
        rows = [g for g in (StaticCompare.geom_metrics(m) for m in masks) if g is not None]
        keys = rows[0].keys()
        return {k: np.array([r[k] for r in rows], dtype=np.float64) for k in keys}

    @staticmethod
    def compare(real_masks, synth_masks) -> dict:
        """Per-metric W1(real, synth) + real/synth medians — the geometry/biomarker panel."""
        R, S = StaticCompare._dist(real_masks), StaticCompare._dist(synth_masks)
        out = {}
        for k in R:
            r, s = torch.tensor(R[k]), torch.tensor(S[k])
            out[k] = {"w1": round(SynthFidelity.wasserstein1d(r, s), 3),
                      "real_median": round(float(np.median(R[k])), 2),
                      "synth_median": round(float(np.median(S[k])), 2)}
        return out


def main():
    ap = argparse.ArgumentParser(description="Static geometry/biomarker panel: synth vs real (uy4d).")
    ap.add_argument("--real", required=True, help="processed ACDC data dir (patient*.npz)")
    ap.add_argument("--pool", required=True, help="synth anatomy pool .npz")
    args = ap.parse_args()
    setup()
    res = StaticCompare.compare(ShapeCoverage._real_masks(args.real), Anatomy.load_pool(args.pool))
    log.info(json.dumps(res, indent=2))
    worst = max(res, key=lambda k: res[k]["w1"])
    log.info(f"# worst-matched geometry metric: {worst} (W1={res[worst]['w1']})")


if __name__ == "__main__":
    main()
