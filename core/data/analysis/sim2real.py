"""sim2real fitting probe — can parametrized bSSFP REPRODUCE each vendor's real per-class contrast?

For each vendor, take the real per-class mean intensities (z-scored real images) and grid-fit the
acquisition (field, TR, flip) to match the STANDARDIZED per-class heart contrast. Reports best params +
residual per vendor. The point (vs the pooled synth_fidelity distance): a per-sample INVERSE test that
separates "wrong params" (calibratable) from "missing physics" (irreducible residual).

Findings (2026-07-01, deep_dives/2026-06-30_synth-fidelity-investigation): residuals tiny (GE 0.000,
Siemens 0.08) -> physics is expressive enough for heart contrast; the residual concentrates in the
RV≠cav flow artifact (vendor-varying); best-fit params ~degenerate across vendors -> per-vendor
acquisition calibration is NOT the fidelity lever (the gap is composition/z-norm, not acquisition).
"""
from __future__ import annotations

import argparse
import logging
import math

import polars as pl
import torch

from core.data.dynamic.dataset import ACDCSliceDataset
from core.data.dynamic.mri_physics import MriPhysics
from core.data.static import splits
from core.data.static.labels import CLASSES
from core.data.static.store import build as store
from core.hparams import TrainCfg
from core.obs import Obs

log = logging.getLogger("cardioseg.sim2real")


class Sim2Real:
    """sim2real acquisition-fit CLI (free helpers folded in as staticmethods): per-class standardization +
    the (field, TR, flip) grid-fit that asks whether parametrized bSSFP can reproduce each vendor's real
    STANDARDIZED heart contrast."""

    @staticmethod
    def _standardize(v: torch.Tensor) -> torch.Tensor:
        return (v - v.mean()) / v.std().clamp_min(1e-6)

    @staticmethod
    def fit_acquisition(real_means: torch.Tensor, n_classes: int,
                        tr_grid=(2.5, 6.0, 36), fl_grid=(20.0, 80.0, 31), fields=(1.5, 3.0)) -> dict:
        """Grid-fit (field, TR, flip) so the bSSFP STANDARDIZED heart contrast matches `real_means` (the
        real per-class heart-class means, [n_classes-1]). Returns {field, tr, flip, residual, synth_z}.
        Standardized so it fits the CONTRAST SHAPE (bSSFP scale is arbitrary)."""
        real_z = Sim2Real._standardize(real_means)
        trs = torch.linspace(*tr_grid[:2], int(tr_grid[2]))
        fls = torch.linspace(*fl_grid[:2], int(fl_grid[2]))
        best = {"residual": float("inf")}
        for field in fields:
            t1, t2, pd = MriPhysics.tissue_params(n_classes, 0, field, "cpu")
            for tr in trs:
                for fl in fls:
                    sig = MriPhysics.bssfp_signal(t1, t2, pd, tr, fl * math.pi / 180)
                    syn_z = Sim2Real._standardize(sig[1:n_classes])
                    res = float(((syn_z - real_z) ** 2).mean())
                    if res < best["residual"]:
                        best = {"field": field, "tr": float(tr), "flip": float(fl),
                                "residual": round(res, 4), "synth_z": [round(v, 3) for v in syn_z.tolist()]}
        return best


def main():
    ap = argparse.ArgumentParser(description="Per-vendor sim2real acquisition fit.")
    ap.add_argument("--n", type=int, default=20, help="subjects per vendor")
    args = ap.parse_args()
    Obs.setup()
    d = TrainCfg().generator.data
    n_classes = len(CLASSES) + 1
    meta = store.load_cfg(d)                          # ALL preprocessing params (nyul/norm too)
    log.info(f"{'vendor':10} {'n':>4}  field  TR   flip  | residual | real z(heart) vs synth")
    for vendor in ("Siemens", "Philips", "GE", "Canon"):
        df = meta.filter(pl.col("labelled") & (pl.col("vendor") == vendor))
        if df.height == 0:
            continue
        X, Y = ACDCSliceDataset.load_to_gpu(splits.Splits.paths(df.head(args.n)), d.size, "cpu")
        real = torch.tensor([X[:, 0][Y == c].mean() for c in range(1, n_classes)])
        b = Sim2Real.fit_acquisition(real, n_classes)
        real_z = [round(v, 2) for v in Sim2Real._standardize(real).tolist()]
        log.info(f"{vendor:10} {X.shape[0]:>4}  {b['field']}  {b['tr']:.1f}  {b['flip']:.0f}  | "
              f"{b['residual']:.4f} | real {real_z} synth {b['synth_z']}")


if __name__ == "__main__":
    main()
