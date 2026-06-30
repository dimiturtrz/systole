"""Synth fidelity eval (cardioseg.evaluation.synth_fidelity). The testable core is wasserstein1d —
the per-class synth-vs-real distance metric that localizes where the generator breaks."""
import torch
from cardioseg.evaluation.synth_fidelity import wasserstein1d


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
