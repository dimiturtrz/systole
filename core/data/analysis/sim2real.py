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

import logging
import math

import polars as pl
import torch
from jaxtyping import Float

from core.data.dynamic.dataset import ACDCSliceDataset
from core.data.dynamic.mri_physics import MriPhysics
from core.data.static import splits
from core.data.static.labels import CLASSES
from core.data.static.mri.base import Vendor
from core.data.static.store.build import Build as store
from core.hparams import TrainCfg

log = logging.getLogger("cardioseg.sim2real")


class Sim2Real:
    """sim2real acquisition-fit CLI (free helpers folded in as staticmethods): per-class standardization +
    the (field, TR, flip) grid-fit that asks whether parametrized bSSFP can reproduce each vendor's real
    STANDARDIZED heart contrast."""

    @staticmethod
    def _standardize(v: torch.Tensor) -> torch.Tensor:
        return (v - v.mean()) / v.std().clamp_min(1e-6)

    @staticmethod
    def fit_acquisition(real_means: Float[torch.Tensor, "*n"], n_classes: int,
                        tr_grid=(2.5, 6.0, 36), fl_grid=(20.0, 80.0, 31), fields=(1.5, 3.0)) -> dict:
        """Grid-fit (field, TR, flip) so the bSSFP STANDARDIZED heart contrast matches `real_means` (the
        real per-class heart-class means, [n_classes-1]). Returns {field, tr, flip, residual, synth_z}.
        Standardized so it fits the CONTRAST SHAPE (bSSFP scale is arbitrary)."""
        real_standardized = Sim2Real._standardize(real_means)
        tr_values = torch.linspace(*tr_grid[:2], int(tr_grid[2]))
        flip_values = torch.linspace(*fl_grid[:2], int(fl_grid[2]))
        best = {"residual": float("inf")}
        for field in fields:
            t1, t2, pd = MriPhysics.tissue_params(n_classes, 0, field, "cpu")
            for tr in tr_values:
                for flip in flip_values:
                    signal = MriPhysics.bssfp_signal(t1, t2, pd, tr, flip * math.pi / 180)
                    synth_standardized = Sim2Real._standardize(signal[1:n_classes])
                    residual = float(((synth_standardized - real_standardized) ** 2).mean())
                    if residual < best["residual"]:
                        best = {"field": field, "tr": float(tr), "flip": float(flip),
                                "residual": round(residual, 4),
                                "synth_z": [round(value, 3) for value in synth_standardized.tolist()]}
        return best


    @staticmethod
    def add_args(ap):
        ap.add_argument("--n", type=int, default=20, help="subjects per vendor")

    @staticmethod
    def run(args):  # pragma: no cover
        data_cfg = TrainCfg().generator.data
        n_classes = len(CLASSES) + 1
        meta = store.load_cfg(data_cfg)                          # ALL preprocessing params (nyul/norm too)
        log.info(f"{'vendor':10} {'n':>4}  field  TR   flip  | residual | real z(heart) vs synth")
        for vendor in Vendor:
            df = meta.filter(pl.col("labelled") & (pl.col("vendor") == vendor))
            if df.height == 0:
                continue
            X, Y = ACDCSliceDataset.load_to_gpu(splits.Splits.paths(df.head(args.n)), data_cfg.size, "cpu")
            real = torch.tensor([X[:, 0][c == Y].mean() for c in range(1, n_classes)])
            best_fit = Sim2Real.fit_acquisition(real, n_classes)
            real_z = [round(value, 2) for value in Sim2Real._standardize(real).tolist()]
            log.info(f"{vendor:10} {X.shape[0]:>4}  {best_fit['field']}  {best_fit['tr']:.1f}  "
                  f"{best_fit['flip']:.0f}  | {best_fit['residual']:.4f} | real {real_z} synth {best_fit['synth_z']}")
