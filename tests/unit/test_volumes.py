"""Differentiable soft EDV/ESV/EF — must equal the hard core.measure readout on a one-hot (GT) mask
(the sanity that lets us trust it as a loss), and must carry a gradient.
"""
import numpy as np
import torch
import torch.nn.functional as F

from cardioseg.training.volumes import soft_ef, soft_lv_volume
from core.data.static.labels import LV_CAV
from core.measure import ejection_fraction, label_volume_ml

SP = (10.0, 1.5, 1.5)                                  # (z,y,x) mm


def _onehot(mask: np.ndarray, C: int = 4) -> torch.Tensor:
    return F.one_hot(torch.as_tensor(mask).long(), C).permute(0, 3, 1, 2).float()  # [D,C,H,W]


def test_soft_volume_equals_hard_on_onehot():
    m = np.zeros((3, 12, 12), int); m[1, 2:7, 2:7] = LV_CAV      # a cavity blob on the mid slice
    soft = float(soft_lv_volume(_onehot(m), SP))
    assert abs(soft - label_volume_ml(m, LV_CAV, SP)) < 1e-4


def test_soft_ef_matches_measure_on_onehot():
    ed = np.zeros((3, 12, 12), int); ed[1, 2:8, 2:8] = LV_CAV    # larger (diastole)
    es = np.zeros((3, 12, 12), int); es[1, 3:6, 3:6] = LV_CAV    # smaller (systole)
    ef, edv, esv = soft_ef(_onehot(ed), _onehot(es), SP)
    ef2, edv2, esv2 = ejection_fraction(ed, es, SP)
    assert abs(float(ef) - ef2) < 1e-3 and abs(float(edv) - edv2) < 1e-4 and abs(float(esv) - esv2) < 1e-4


def test_soft_volume_is_differentiable():
    logits = torch.randn(3, 4, 8, 8, requires_grad=True)
    soft_lv_volume(logits.softmax(1), SP).backward()
    assert logits.grad is not None and float(logits.grad.abs().sum()) > 0


def test_vol_loss_zero_when_exact_and_positive_on_mismatch():
    from cardioseg.training.volumes import vol_loss
    assert float(vol_loss(torch.tensor(120.0), torch.tensor(50.0), 120.0, 50.0)) == 0.0
    assert float(vol_loss(torch.tensor(120.0), torch.tensor(50.0), 100.0, 50.0)) > 0


def test_vol_loss_batched_equals_per_subject_mean():
    # the vectorization invariant: one [K]-vector call == mean of K scalar calls (Huber over 2K elems).
    from cardioseg.training.volumes import vol_loss
    edp, esp = torch.tensor([120., 90., 200.]), torch.tensor([50., 40., 95.])
    edg, esg = torch.tensor([115., 95., 190.]), torch.tensor([55., 38., 90.])
    batched = float(vol_loss(edp, esp, edg, esg))
    per = torch.stack([vol_loss(edp[i], esp[i], edg[i], esg[i]) for i in range(3)]).mean()
    assert abs(batched - float(per)) < 1e-6
