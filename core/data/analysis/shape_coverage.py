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

from pathlib import Path

import numpy as np

# canonical labels: 1 RV-cav, 2 LV-myo, 3 LV-cav (0 bg)
_RV, _MYO, _LVC = 1, 2, 3


def shape_features(mask: np.ndarray) -> np.ndarray | None:
    """Interpretable SHAPE-only descriptors for one 2D label map, or None if ~empty. Scale/position-
    invariant where sensible so it compares generated vs real anatomy, not framing artifacts."""
    fg = mask > 0
    n = int(fg.sum())
    if n < 40:
        return None
    rv, myo, lvc = (mask == _RV).sum(), (mask == _MYO).sum(), (mask == _LVC).sum()
    ys, xs = np.where(fg)
    h = ys.max() - ys.min() + 1
    w = xs.max() - xs.min() + 1
    eps = 1.0
    return np.array([
        rv / n, myo / n, lvc / n,                       # class area fractions (composition)
        myo / (lvc + eps),                              # myo thickness proxy (myo per unit cavity)
        rv / (lvc + eps),                               # RV/LV cavity balance
        min(h, w) / max(h, w),                          # fg bbox aspect (roundness)
        n / (h * w + eps),                              # fill fraction of bbox (compactness)
    ], dtype=np.float64)


def _feats_from_masks(masks) -> np.ndarray:
    rows = [f for f in (shape_features(m) for m in masks) if f is not None]
    return np.stack(rows) if rows else np.zeros((0, 7))


def _real_masks(acdc_dir: str) -> list[np.ndarray]:
    out = []
    for f in sorted(Path(acdc_dir).glob("*.npz")):
        d = np.load(f)
        for gt in (d["ed_gt"], d["es_gt"]):
            out.extend(gt[k] for k in range(gt.shape[0]))
    return out


def coverage(real: np.ndarray, synth: np.ndarray) -> dict:
    """Standardize by REAL stats; for each real point the nearest synth point (does synth cover real),
    and the reverse (does synth extrapolate beyond real). Distances in std-units of the real cloud."""
    mu, sd = real.mean(0), real.std(0) + 1e-9
    r, s = (real - mu) / sd, (synth - mu) / sd
    def nn(a, b):                                        # nearest-neighbour distance a->b, per row of a
        d = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))
        return d.min(1)
    r2s = nn(r, s)                                       # real -> nearest synth (coverage)
    s2r = nn(s, r)                                       # synth -> nearest real (extrapolation if large)
    s_self = nn(s, s) if len(s) > 1 else np.array([1.0]) # synth internal scale (for a radius)
    rad = float(np.median(s_self[s_self > 0])) if (s_self > 0).any() else 1.0
    return {
        "n_real": int(len(real)), "n_synth": int(len(synth)),
        "real_to_synth_nn_median": round(float(np.median(r2s)), 3),
        "real_to_synth_nn_p95": round(float(np.percentile(r2s, 95)), 3),
        "coverage_at_synth_radius": round(float((r2s <= rad).mean()), 3),   # frac real with a synth neighbour
        "synth_to_real_nn_median": round(float(np.median(s2r)), 3),         # >real-internal => extrapolates
        "synth_beyond_real_frac": round(float((s2r > np.percentile(r2s, 95)).mean()), 3),
    }


def _main():
    import argparse
    from core.data.dynamic.anatomy import load_pool
    ap = argparse.ArgumentParser(description="Shape coverage: does generated anatomy encompass real? (uy4d)")
    ap.add_argument("--real", required=True, help="processed ACDC data dir (patient*.npz)")
    ap.add_argument("--pool", required=True, help="synth anatomy pool .npz (build_pool)")
    ap.add_argument("--out", default=None, help="PCA scatter PNG (real vs synth)")
    a = ap.parse_args()
    real = _feats_from_masks(_real_masks(a.real))
    synth = _feats_from_masks(load_pool(a.pool))
    import json
    print(json.dumps(coverage(real, synth), indent=2))
    # 2D PCA (SVD) fit on the union, standardized by real, for the scatter
    mu, sd = real.mean(0), real.std(0) + 1e-9
    R, S = (real - mu) / sd, (synth - mu) / sd
    U = np.concatenate([R, S])
    _, _, vt = np.linalg.svd(U - U.mean(0), full_matrices=False)
    pr, ps = (R - U.mean(0)) @ vt[:2].T, (S - U.mean(0)) @ vt[:2].T
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 6))
        plt.scatter(ps[:, 0], ps[:, 1], s=6, alpha=0.3, label=f"synth (n={len(ps)})", color="#e35")
        plt.scatter(pr[:, 0], pr[:, 1], s=6, alpha=0.3, label=f"real (n={len(pr)})", color="#38e")
        plt.legend(); plt.title("shape-descriptor PCA: does synth (red) cover real (blue)?")
        out = a.out or (str(Path(a.pool).with_suffix("")) + "_shapecov.png")
        plt.savefig(out, dpi=110, bbox_inches="tight")
        print(f"wrote {out}")
    except ImportError:
        print("(matplotlib not available — skipped scatter)")


if __name__ == "__main__":
    _main()
