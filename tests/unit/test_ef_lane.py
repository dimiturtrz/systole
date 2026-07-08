"""EF-ratio weak-supervision math (ef_lane) — the spacing-cancelling (ED-ES)/ED core + its Huber loss,
extracted out of KaggleEF.loss (which needs a cine pool + GPU forward — that's the shell). Equivalence
classes over cavity-total inputs; synthetic tensors, no model. Mirrors test_volumes' vol_loss idiom.
"""
from types import SimpleNamespace

import numpy as np
import torch

from cardioseg.training.ef_lane import _zscore, build_aux, ef_ratio, ef_ratio_loss


def test_ef_ratio_matches_hand_formula():
    """Normal class: EF = (ED-ES)/ED*100 elementwise on cavity pixel totals (units cancel)."""
    ed = torch.tensor([100.0, 200.0])
    es = torch.tensor([40.0, 100.0])
    assert torch.allclose(ef_ratio(ed, es), torch.tensor([60.0, 50.0]))


def test_ef_ratio_scale_invariant():
    """Ratio class: scaling both ED and ES by any factor (spacing/px) leaves EF unchanged."""
    ed, es = torch.tensor([120.0]), torch.tensor([50.0])
    assert torch.allclose(ef_ratio(ed, es), ef_ratio(ed * 7.3, es * 7.3))


def test_ef_ratio_empty_ed_clamped_no_nan():
    """Collapse class: ED=0 (empty prediction) clamps at 1e-6 -> ~0% not NaN/inf."""
    r = ef_ratio(torch.tensor(0.0), torch.tensor(0.0))
    assert torch.isfinite(r) and abs(float(r)) < 1e-3


def test_ef_ratio_loss_zero_when_pred_equals_target():
    """Exact class: predicted EF-ratio == csv target -> zero Huber."""
    ed, es = torch.tensor([100.0]), torch.tensor([40.0])   # EF 60%
    assert float(ef_ratio_loss(ed, es, [60.0])) == 0.0


def test_ef_ratio_loss_positive_on_mismatch():
    """Mismatch class: wrong target -> positive loss."""
    ed, es = torch.tensor([100.0]), torch.tensor([40.0])   # EF 60%
    assert float(ef_ratio_loss(ed, es, [45.0])) > 0


def test_ef_ratio_loss_accepts_tensor_targets():
    """Target-type class: a tensor target behaves like a list target (same value)."""
    ed, es = torch.tensor([100.0]), torch.tensor([40.0])
    as_list = float(ef_ratio_loss(ed, es, [45.0]))
    as_tensor = float(ef_ratio_loss(ed, es, torch.tensor([45.0])))
    assert abs(as_list - as_tensor) < 1e-9


def test_ef_ratio_loss_is_differentiable():
    """Grad class: loss carries a gradient back to the (soft-cavity) inputs — usable as a loss."""
    ed = torch.tensor([100.0], requires_grad=True)
    es = torch.tensor([40.0], requires_grad=True)
    ef_ratio_loss(ed, es, [45.0]).backward()
    assert ed.grad is not None and float(ed.grad.abs().sum()) > 0


# --- _zscore: per-slice standardization the Kaggle lane feeds the net (pure numpy) ---
def test_zscore_zero_mean_unit_std():
    """Normal class: output has ~0 mean and ~1 std (the ε is negligible on a spread input)."""
    s = _zscore(np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert abs(float(s.mean())) < 1e-5 and abs(float(s.std()) - 1.0) < 1e-3


def test_zscore_constant_input_no_nan():
    """Degenerate class: a constant slice (std=0) -> finite zeros via the 1e-6 floor, never NaN."""
    s = _zscore(np.full((2, 2), 7.0))
    assert np.isfinite(s).all() and float(np.abs(s).max()) < 1e-3


# --- build_aux: the lane-assembly OFF switches (return [] without touching GPU/data) ---
def _cfg(ef_lambda=1.0, ef_kaggle=False):
    return SimpleNamespace(ef_lambda=ef_lambda, ef_kaggle=ef_kaggle, ef_subjects=4,
                           ef_kaggle_subjects=4, seed=0,
                           generator=SimpleNamespace(data=SimpleNamespace(size=64)))


def test_build_aux_off_when_lambda_nonpositive():
    """Off class: ef_lambda<=0 disables the lane -> empty list (no VolConsistency built)."""
    assert build_aux(_cfg(ef_lambda=0.0), None, None, "cpu", is_static=True) == []


def test_build_aux_off_when_not_static():
    """Source class: a non-static train source has no labeled EDV/ESV frame -> empty list."""
    assert build_aux(_cfg(ef_lambda=1.0), None, None, "cpu", is_static=False) == []
