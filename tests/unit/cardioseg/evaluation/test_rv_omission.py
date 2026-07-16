"""nttu.6: RV-omission diagnostic pure cores — omission classification, recall/coverage split, RV bias."""
import numpy as np
import torch

from cardioseg.evaluation.rv_omission import RV, RvOmission


def _slice(gt_rv_px, argmax_rv_px, size=40):
    """A [size,size] GT + argmax pair with the requested RV pixel counts (disjoint corners)."""
    gt = np.zeros((size, size), dtype=np.int64)
    argmax = np.zeros((size, size), dtype=np.int64)
    gt.flat[:gt_rv_px] = RV
    argmax.flat[size * size - argmax_rv_px:] = RV
    return gt, argmax


def test_omission_row_none_when_gt_rv_small():
    gt, argmax = _slice(gt_rv_px=5, argmax_rv_px=0)          # GT-RV below OMIT_PX -> not an omission
    prob = np.zeros((40, 40), np.float32)
    assert RvOmission.omission_row(prob, argmax, gt, argmax) is None


def test_omission_row_none_when_argmax_fires():
    gt, argmax = _slice(gt_rv_px=100, argmax_rv_px=100)      # argmax fired plenty of RV -> not omitted
    prob = np.full((40, 40), 0.9, np.float32)
    assert RvOmission.omission_row(prob, argmax, gt, argmax) is None


def test_omission_row_records_recoverable_activation():
    gt, argmax = _slice(gt_rv_px=100, argmax_rv_px=0)        # GT-RV present, argmax omitted
    prob = np.zeros((40, 40), np.float32)
    gz = gt == RV
    prob[gz] = 0.4                                           # sub-dominant but present RV softmax
    pred = np.zeros_like(gt)                                 # bg wins the region
    row = RvOmission.omission_row(prob, argmax, gt, pred)
    assert row is not None
    assert row["gtpx"] == 100
    assert abs(row["maxp_in_gt"] - 0.4) < 1e-6
    assert row["win"] == "bg" and abs(row["win_frac"] - 1.0) < 1e-6


def test_split_verdict_partitions_on_floor():
    v = RvOmission.split_verdict([0.01, 0.04, 0.05, 0.4, 0.57])
    assert v["n"] == 5
    assert v["recoverable"] == 3                             # >= 0.05
    assert v["zero_activation"] == 2
    assert abs(v["max"] - 0.57) < 1e-6 and abs(v["min"] - 0.01) < 1e-6


def test_split_verdict_empty():
    v = RvOmission.split_verdict([])
    assert v["n"] == 0 and v["recoverable"] == 0
    assert np.isnan(v["med"])


def test_biased_pred_tilts_rv_over_bg():
    # one slice, 2 classes-of-interest: RV logit just under bg -> a large positive bias flips argmax to RV.
    d, c, h, w = 1, 4, 2, 2
    logits = torch.zeros((4, c, h, w))                      # [K*D=4, C, H, W] (4 FLIPS, d=1)
    logits[:, 0] = 1.0                                       # bg dominant
    logits[:, RV] = 0.5                                      # RV runner-up
    base = RvOmission.biased_pred(logits, d, 0.0)
    tilted = RvOmission.biased_pred(logits, d, 2.0)
    assert (base == 0).all()                                # bg wins with no bias
    assert (tilted == RV).all()                             # +2.0 RV bias flips it
