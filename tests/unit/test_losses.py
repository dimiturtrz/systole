"""PartialLabelDiceCE — the general per-sample valid-class-mask loss. Locks: valid=None reduces to
all-valid; masking a class drops its Dice penalty; CE ignores pixels whose GT class is invalid; soft
targets accepted.
"""
import torch

from cardioseg.training.losses import PartialLabelDiceCE

FULL = torch.tensor([[True, True, True, True]])       # all classes valid
SCD = torch.tensor([[False, False, True, True]])      # LV-only: bg + RV untrustworthy


def test_valid_none_equals_all_true():
    torch.manual_seed(0)
    logits, y = torch.randn(2, 4, 8, 8), torch.randint(0, 4, (2, 8, 8))
    L = PartialLabelDiceCE()
    assert torch.allclose(L(logits, y, valid=None), L(logits, y, valid=torch.ones(2, 4, dtype=torch.bool)))


def test_masking_class_drops_its_dice_penalty():
    # GT has only myo/cav (no RV); logits scream RV everywhere = false positives
    y = torch.zeros(1, 8, 8, dtype=torch.long); y[:, :, :4] = 2; y[:, :, 4:] = 3
    logits = torch.zeros(1, 4, 8, 8); logits[:, 1] = 5.0
    L = PartialLabelDiceCE()
    assert L(logits, y, valid=SCD) < L(logits, y, valid=FULL)   # RV FP not penalized when RV masked


def test_ce_ignores_pixels_with_invalid_gt_class():
    # half the pixels are bg (invalid for SCD); model mispredicts them
    y = torch.zeros(1, 8, 8, dtype=torch.long); y[:, :, :4] = 2      # half myo, half bg
    logits = torch.zeros(1, 4, 8, 8); logits[:, 1] = 5.0             # predict RV (wrong everywhere)
    L = PartialLabelDiceCE()
    assert L(logits, y, valid=SCD) < L(logits, y, valid=FULL)       # bg pixels ignored under SCD mask


def test_soft_target_accepted():
    logits = torch.randn(1, 4, 8, 8)
    soft = torch.softmax(torch.randn(1, 4, 8, 8), 1)                 # [B,C,H,W] soft
    out = PartialLabelDiceCE()(logits, soft, valid=None)
    assert out.dim() == 0 and torch.isfinite(out)


def test_valid_row_registry():
    from core.data.static.labels import valid_row
    assert valid_row("scd") == [False, False, True, True]      # LV-only: bg + RV untrusted
    assert valid_row("acdc") == [True, True, True, True]       # full-label


def test_per_sample_mask_mixes_full_and_partial():
    torch.manual_seed(3)
    logits = torch.randn(2, 4, 8, 8)
    y = torch.randint(0, 4, (2, 8, 8))
    valid = torch.cat([FULL, SCD])                                  # sample 0 full, sample 1 LV-only
    out = PartialLabelDiceCE()(logits, y, valid=valid)
    assert out.dim() == 0 and torch.isfinite(out)


def test_uncertainty_weighted_matches_kendall_formula():
    from cardioseg.training.losses import uncertainty_weighted
    L1, L2 = torch.tensor(2.0), torch.tensor(4.0)
    s1, s2 = torch.tensor(0.5), torch.tensor(-0.3)
    got = uncertainty_weighted([L1, L2], [s1, s2])
    want = torch.exp(-s1) * L1 + s1 + torch.exp(-s2) * L2 + s2
    assert torch.allclose(got, want)
