"""Volume-consistency loss (vol_loss) — dimensionless EDV/ESV Huber: zero on an exact match, positive
on a mismatch, and batched == per-subject-mean (the vectorization invariant).
"""
import torch

from cardioseg.training.volumes import vol_loss


def test_vol_loss_zero_when_exact_and_positive_on_mismatch():
    """Exact vs mismatch: pred == gt volumes -> 0 loss; a wrong EDV_gt -> positive."""
    assert float(vol_loss(torch.tensor(120.0), torch.tensor(50.0), 120.0, 50.0)) == 0.0
    assert float(vol_loss(torch.tensor(120.0), torch.tensor(50.0), 100.0, 50.0)) > 0


def test_vol_loss_batched_equals_per_subject_mean():
    """Vectorization invariant: one [K]-vector call == mean of K scalar calls (Huber over 2K elems)."""
    edp, esp = torch.tensor([120., 90., 200.]), torch.tensor([50., 40., 95.])
    edg, esg = torch.tensor([115., 95., 190.]), torch.tensor([55., 38., 90.])
    batched = float(vol_loss(edp, esp, edg, esg))
    per = torch.stack([vol_loss(edp[i], esp[i], edg[i], esg[i]) for i in range(3)]).mean()
    assert abs(batched - float(per)) < 1e-6
