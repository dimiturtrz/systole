"""sim2real acquisition fit (core.data.analysis.sim2real). The testable cores are _standardize (zero-
mean/unit-std) and fit_acquisition — the (field, TR, flip) grid-fit that asks whether parametrized
bSSFP can reproduce a real STANDARDIZED per-class heart contrast. The CLI (main) loads real data and is
coverage-omitted; here we drive the pure fit with synthetic contrast targets."""
import math

import torch

from core.data.analysis.sim2real import Sim2Real
from core.data.dynamic.mri_physics import MriPhysics


def test_standardize_zero_mean_unit_std():
    v = torch.tensor([1.0, 2.0, 3.0, 4.0])
    z = Sim2Real._standardize(v)
    assert abs(float(z.mean())) < 1e-6
    assert abs(float(z.std()) - 1.0) < 1e-6


def test_standardize_constant_input_no_nan():
    """Constant vector -> std clamped, no div-by-zero (result finite, ~zero)."""
    z = Sim2Real._standardize(torch.ones(5))
    assert torch.isfinite(z).all()


def test_fit_acquisition_reproduces_own_contrast():
    """bSSFP can reproduce a contrast it itself generated -> ~0 residual (expressiveness sanity; the real
    residual on real data is the physics gap, e.g. RV/cav flow). Moved from test_synth_fidelity."""
    n = 4
    t1, t2, pd = MriPhysics.tissue_params(n, 0, 1.5, "cpu")
    target = MriPhysics.bssfp_signal(t1, t2, pd, torch.tensor(3.0), torch.tensor(50 * math.pi / 180))[1:n]
    assert Sim2Real.fit_acquisition(target, n)["residual"] < 0.01


def test_fit_acquisition_report_shape():
    """The fit report carries the selected params + a standardized synth contrast of the right length."""
    n = 4
    target = torch.tensor([2.0, -1.0, 1.5])            # arbitrary heart contrast [n-1]
    best = Sim2Real.fit_acquisition(target, n)
    assert set(best) == {"field", "tr", "flip", "residual", "synth_z"}
    assert best["field"] in (1.5, 3.0)
    assert len(best["synth_z"]) == n - 1


def test_fit_acquisition_scale_invariant_target():
    """Fit matches CONTRAST SHAPE (standardized) -> scaling/shifting the target picks the same params."""
    n = 4
    t1, t2, pd = MriPhysics.tissue_params(n, 0, 3.0, "cpu")
    base = MriPhysics.bssfp_signal(t1, t2, pd, torch.tensor(3.0), torch.tensor(45 * math.pi / 180))[1:n]
    a = Sim2Real.fit_acquisition(base, n)
    b = Sim2Real.fit_acquisition(base * 5.0 - 2.0, n)
    assert (a["field"], a["tr"], a["flip"]) == (b["field"], b["tr"], b["flip"])
