"""Synth fidelity eval (core.data.analysis.synth_fidelity). The testable cores are wasserstein1d —
the per-class synth-vs-real distance metric — and dprime, the separability (distinguishability) metric."""
import math

import torch

from core.data.analysis.synth_fidelity import SynthFidelity
from core.data.dynamic.synth import FlatBgCfg, SynthCfg


def test_w1_zero_for_identical():
    a = torch.randn(1000)
    assert SynthFidelity.wasserstein1d(a, a) < 1e-5


def test_w1_recovers_shift():
    torch.manual_seed(0)
    a = torch.randn(5000); b = a + 2.0                 # pure translation by 2
    assert abs(SynthFidelity.wasserstein1d(a, b) - 2.0) < 0.05       # W1 of a shift == the shift


def test_w1_empty_is_nan():
    v = SynthFidelity.wasserstein1d(torch.tensor([]), torch.randn(10))
    assert math.isnan(v)                                # NaN (absent class)


def test_dprime_separated_vs_overlapping():
    torch.manual_seed(0)
    far = SynthFidelity.dprime(torch.randn(2000), torch.randn(2000) + 4.0)     # 4 SD apart -> d' ~ 4
    same = SynthFidelity.dprime(torch.randn(2000), torch.randn(2000))          # same dist -> d' ~ 0
    assert 3.5 < far < 4.5 and same < 0.2


def test_dprime_affine_invariant():
    """d' is scale/shift invariant (ratio of mean-gap to pooled SD) -> z-scoring can't change it."""
    torch.manual_seed(1)
    a, b = torch.randn(3000), torch.randn(3000) + 2.0
    base = SynthFidelity.dprime(a, b)
    assert abs(SynthFidelity.dprime(a * 7.3 - 1.5, b * 7.3 - 1.5) - base) < 0.02


def test_dprime_tiny_sample_is_nan():
    v = SynthFidelity.dprime(torch.randn(10), torch.randn(3000))              # < 50 pts -> unstable -> NaN
    assert math.isnan(v)


def test_paint_ignores_deform_so_anatomy_stays_on_the_mask():
    """_paint must force deform OFF: a fidelity comparison measures synth intensity at the REAL mask, so
    the painted heart has to stay aligned to it. deform>0 warps the anatomy off the mask -> the thin myo
    ring samples adjacent blood+bg -> spurious per-class spread that inverts the d' verdict (bd f4hk)."""
    mask = torch.zeros(2, 32, 32, dtype=torch.long)
    mask[:, 10:22, 10:22] = 2; mask[:, 14:18, 14:18] = 3                      # a myo block around a cavity
    torch.manual_seed(1); warped = SynthFidelity._paint(mask, SynthCfg(synth_p=1.0, deform=0.5, bg=FlatBgCfg()), 4)
    torch.manual_seed(1); flat = SynthFidelity._paint(mask, SynthCfg(synth_p=1.0, deform=0.0, bg=FlatBgCfg()), 4)
    assert torch.allclose(warped, flat)                                       # deform had no effect -> forced off
