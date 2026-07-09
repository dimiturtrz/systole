"""FIT operator (core.data.dynamic.inverse, bd ncph/ixea): differentiable render + acquisition fit.
Contract: render_heart is differentiable wrt acquisition; fit_acquisition recovers a KNOWN flip from a
synthetic target (round-trip, where flip-only IS identifiable)."""
import torch

from core.data.dynamic.inverse import Inverse

N = 4  # 0 bg, 1 RV-cav, 2 myo, 3 LV-cav


def _seg(b=1, h=32, w=32):
    m = torch.zeros(b, h, w, dtype=torch.long)
    m[:, 8:24, 8:24] = 2          # myo block
    m[:, 12:20, 12:20] = 3        # LV-cav (blood) inside
    m[:, 8:14, 24:30] = 1         # RV-cav (blood) beside
    return m


def test_render_is_differentiable_wrt_acquisition():
    seg = _seg()
    tr = torch.tensor([[3.0]])
    flip = torch.tensor([[45.0]], requires_grad=True)
    img = Inverse.render_heart(seg, tr, flip, N, 1.5, "cpu")
    assert img.shape == (1, 1, 32, 32)
    img.sum().backward()
    assert flip.grad is not None and torch.isfinite(flip.grad).all() and flip.grad.abs() > 0


def test_acquisition_unidentifiable_from_two_tissue_heart():
    """KEY FINDING (bd ixea): the heart has only 2 tissue levels (blood, myo; RV-cav==LV-cav==blood), so
    after gain/bias normalization ANY flip matches ANY other (2 levels map onto 2 levels exactly). The
    fit therefore reaches ~0 loss at every init but DOESN'T converge to a unique flip -> acquisition is
    NOT identifiable from one frame's heart region. Motivates multi-acquisition / more tissues (bd 5ev5)."""
    seg = _seg()
    target = Inverse.render_heart(seg, torch.tensor([[3.0]]), torch.tensor([[62.0]]), N, 1.5, "cpu")
    lo = Inverse.fit_acquisition(target, seg, N, fit_params=("flip",), flip0=25.0, steps=400, lr=1.0)
    hi = Inverse.fit_acquisition(target, seg, N, fit_params=("flip",), flip0=75.0, steps=400, lr=1.0)
    assert lo["recon_loss"] < 1e-3 and hi["recon_loss"] < 1e-3    # both fit the 2-level pattern perfectly
    assert abs(lo["flip"] - hi["flip"]) > 20.0                    # ...at very different flips = non-unique


def test_fit_needs_foreground():
    seg = torch.zeros(1, 16, 16, dtype=torch.long)
    img = torch.randn(1, 1, 16, 16)
    try:
        Inverse.fit_acquisition(img, seg, N)
        raise AssertionError
    except ValueError:
        pass
