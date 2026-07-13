"""Shape-coverage analysis (bd uy4d): does the GENERATED anatomy encompass the REAL anatomy?

Per-class intensity W1 (synth_fidelity) is anatomy-agnostic — it measures the COLOR axis and can't see
the generation gap, which lives in SHAPE (repaint 0.68 vs generate 0.56). This embeds the LABEL MAPS
(masks) into an interpretable shape-descriptor space and asks whether the synth (Rodero) shape cloud
COVERS the real (ACDC/M&M) shape cloud — the honest generation metric.

Descriptors are shape-only (area fractions, ratios, framing) so the two clouds are comparable even
though the hearts don't match one-to-one. Coverage = for each real mask, is there a synth mask nearby
in shape space; extrapolation = does synth also go BEYOND real (wanted, for domain randomization).
numpy-only (SVD for PCA); no sklearn/umap dependency.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt

from core.data.dynamic.anatomy import Anatomy

log = logging.getLogger("cardioseg.shape_coverage")

# canonical labels: 1 RV-cav, 2 LV-myo, 3 LV-cav (0 bg)
_RV, _MYO, _LVC = 1, 2, 3

_MIN_FG_PX = 40   # below this many foreground px a 2D label map is ~empty -> skip (shape stats unstable)


class ShapeCoverage:
    """Shape-coverage analysis (bd uy4d): does the GENERATED anatomy encompass the REAL anatomy? The free
    analysis helpers folded in as staticmethods — shape descriptors, real-mask loading, and the coverage
    metric over the shape-descriptor cloud."""

    @staticmethod
    def shape_features(mask: np.ndarray) -> np.ndarray | None:
        """Interpretable SHAPE-only descriptors for one 2D label map, or None if ~empty. Scale/position-
        invariant where sensible so it compares generated vs real anatomy, not framing artifacts."""
        foreground = mask > 0
        n = int(foreground.sum())
        if n < _MIN_FG_PX:
            return None
        rv, myo, lvc = (mask == _RV).sum(), (mask == _MYO).sum(), (mask == _LVC).sum()
        row_indices, col_indices = np.where(foreground)
        h = row_indices.max() - row_indices.min() + 1
        w = col_indices.max() - col_indices.min() + 1
        eps = 1.0
        return np.array([
            rv / n, myo / n, lvc / n,                       # class area fractions (composition)
            myo / (lvc + eps),                              # myo thickness proxy (myo per unit cavity)
            rv / (lvc + eps),                               # RV/LV cavity balance
            min(h, w) / max(h, w),                          # fg bbox aspect (roundness)
            n / (h * w + eps),                              # fill fraction of bbox (compactness)
        ], dtype=np.float64)

    @staticmethod
    def _feats_from_masks(masks) -> np.ndarray:
        rows = [features for features in (ShapeCoverage.shape_features(mask) for mask in masks) if features is not None]
        return np.stack(rows) if rows else np.zeros((0, 7))

    @staticmethod
    def _real_masks(acdc_dir: str) -> list[np.ndarray]:
        out = []
        for npz_path in sorted(Path(acdc_dir).glob("*.npz")):
            data = np.load(npz_path)
            for gt in (data["ed_gt"], data["es_gt"]):
                out.extend(gt[k] for k in range(gt.shape[0]))
        return out

    @staticmethod
    def coverage(real: np.ndarray, synth: np.ndarray) -> dict:
        """Standardize by REAL stats; for each real point the nearest synth point (does synth cover real),
        and the reverse (does synth extrapolate beyond real). Distances in std-units of the real cloud."""
        real_mean, real_std = real.mean(0), real.std(0) + 1e-9
        real_standardized, synth_standardized = (real - real_mean) / real_std, (synth - real_mean) / real_std
        def nn(a, b):                                        # nearest-neighbour distance a->b, per row of a
            distances = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))
            return distances.min(1)
        real_to_synth = nn(real_standardized, synth_standardized)   # real -> nearest synth (coverage)
        synth_to_real = nn(synth_standardized, real_standardized)   # synth -> nearest real (extrapolation if large)
        synth_to_synth = nn(synth_standardized, synth_standardized) if len(synth_standardized) > 1 else np.array([1.0]) # synth internal scale (for a radius)
        radius = float(np.median(synth_to_synth[synth_to_synth > 0])) if (synth_to_synth > 0).any() else 1.0
        return {
            "n_real": len(real), "n_synth": len(synth),
            "real_to_synth_nn_median": round(float(np.median(real_to_synth)), 3),
            "real_to_synth_nn_p95": round(float(np.percentile(real_to_synth, 95)), 3),
            "coverage_at_synth_radius": round(float((real_to_synth <= radius).mean()), 3),   # frac real with a synth neighbour
            "synth_to_real_nn_median": round(float(np.median(synth_to_real)), 3),         # >real-internal => extrapolates
            "synth_beyond_real_frac": round(float((synth_to_real > np.percentile(real_to_synth, 95)).mean()), 3),
        }


    @staticmethod
    def add_args(ap):
        ap.add_argument("--real", required=True, help="processed ACDC data dir (patient*.npz)")
        ap.add_argument("--pool", required=True, help="synth anatomy pool .npz (build_pool)")
        ap.add_argument("--out", default=None, help="PCA scatter PNG (real vs synth)")

    @staticmethod
    def run(args):  # pragma: no cover
        real = ShapeCoverage._feats_from_masks(ShapeCoverage._real_masks(args.real))
        synth = ShapeCoverage._feats_from_masks(Anatomy.load_pool(args.pool))
        log.info(json.dumps(ShapeCoverage.coverage(real, synth), indent=2))
        # 2D PCA (SVD) fit on the union, standardized by real, for the scatter
        real_mean, real_std = real.mean(0), real.std(0) + 1e-9
        real_standardized, synth_standardized = (real - real_mean) / real_std, (synth - real_mean) / real_std
        union = np.concatenate([real_standardized, synth_standardized])
        _, _, right_singular_vectors = np.linalg.svd(union - union.mean(0), full_matrices=False)
        projected_real = (real_standardized - union.mean(0)) @ right_singular_vectors[:2].T
        projected_synth = (synth_standardized - union.mean(0)) @ right_singular_vectors[:2].T
        plt.figure(figsize=(6, 6))
        plt.scatter(projected_synth[:, 0], projected_synth[:, 1], s=6, alpha=0.3, label=f"synth (n={len(projected_synth)})", color="#e35")
        plt.scatter(projected_real[:, 0], projected_real[:, 1], s=6, alpha=0.3, label=f"real (n={len(projected_real)})", color="#38e")
        plt.legend(); plt.title("shape-descriptor PCA: does synth (red) cover real (blue)?")
        output_path = args.out or (str(Path(args.pool).with_suffix("")) + "_shapecov.png")
        plt.savefig(output_path, dpi=110, bbox_inches="tight")
        log.info(f"wrote {output_path}")
