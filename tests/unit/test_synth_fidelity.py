"""Synth fidelity eval (core.data.analysis.synth_fidelity). The testable core is wasserstein1d —
the per-class synth-vs-real distance metric that localizes where the generator breaks."""
import torch
from core.data.analysis.synth_fidelity import wasserstein1d


def test_w1_zero_for_identical():
    a = torch.randn(1000)
    assert wasserstein1d(a, a) < 1e-5


def test_w1_recovers_shift():
    torch.manual_seed(0)
    a = torch.randn(5000); b = a + 2.0                 # pure translation by 2
    assert abs(wasserstein1d(a, b) - 2.0) < 0.05       # W1 of a shift == the shift


def test_w1_empty_is_nan():
    v = wasserstein1d(torch.tensor([]), torch.randn(10))
    assert v != v                                       # NaN (absent class)


def test_fit_acquisition_reproduces_own_contrast():
    """sim2real fit: bSSFP can reproduce a contrast it itself generated -> ~0 residual (expressiveness
    sanity; the real residual on real data is the physics gap, e.g. RV/cav flow)."""
    import math, torch
    from core.data.dynamic.mri_physics import bssfp_signal, tissue_params
    from core.data.analysis.sim2real import fit_acquisition
    n = 4
    t1, t2, pd = tissue_params(n, 0, 1.5, "cpu")
    target = bssfp_signal(t1, t2, pd, torch.tensor(3.0), torch.tensor(50 * math.pi / 180))[1:n]
    assert fit_acquisition(target, n)["residual"] < 0.01
