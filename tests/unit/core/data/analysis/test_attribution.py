"""Attribution diagnostic (core.data.analysis.attribution). The load-bearing, model-free piece is
the class-confusion matrix — it surfaces directional mistakes (e.g. foreground predicted as bg) that a
mean-Dice number hides. Saliency needs captum + a model (smoke-tested at train time, not here)."""
import torch

from core.data.analysis.attribution import Attribution


def test_confusion_row_normalized_per_gt_class():
    """M[g,p] = fraction of GT-g voxels predicted p; rows over present classes sum to 1."""
    gt = torch.tensor([0, 0, 1, 1, 1, 2])
    pred = torch.tensor([0, 0, 1, 1, 0, 2])            # class1: 2/3 ->1, 1/3 ->0 ; others perfect
    M = Attribution.class_confusion(pred, gt, 3)
    assert torch.allclose(M[0], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(M[1], torch.tensor([1 / 3, 2 / 3, 0.0]))
    assert torch.allclose(M[2], torch.tensor([0.0, 0.0, 1.0]))
    assert torch.allclose(M[:3].sum(1), torch.ones(3))   # each present row sums to 1


def test_confusion_absent_class_row_is_zero():
    """A GT class with no voxels leaves its row at zero (no div-by-zero)."""
    gt = torch.tensor([0, 0, 0])
    pred = torch.tensor([0, 0, 1])
    M = Attribution.class_confusion(pred, gt, 4)
    assert torch.allclose(M[1], torch.zeros(4)) and torch.allclose(M[2], torch.zeros(4))
    assert M[0, 0] == 2 / 3 and M[0, 1] == 1 / 3


def test_confusion_captures_undersegmentation():
    """The systematic mistake this exists to catch: foreground (1) mostly predicted as bg (0)."""
    gt = torch.tensor([1, 1, 1, 1])
    pred = torch.tensor([0, 0, 0, 1])                  # 75% of class-1 -> bg
    M = Attribution.class_confusion(pred, gt, 2)
    assert M[1, 0] == 0.75 and M[1, 1] == 0.25         # under-seg shows as off-diagonal leak to bg
