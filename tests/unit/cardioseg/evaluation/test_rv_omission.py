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


def test_dice_buckets_histograms_the_deficit():
    # egeh: half-open [lo,hi) bins; a value ON an edge lands in the upper bin, 1.0 caught by the 1.01 top.
    edges = (0.0, 0.05, 0.3, 0.6, 1.01)
    got = RvOmission.dice_buckets([0.0, 0.04, 0.05, 0.3, 0.7, 1.0], edges)
    assert [c for _, _, c in got] == [2, 1, 1, 2]             # {0,.04}|{.05}|{.3}|{.7,1.0}
    assert [(lo, hi) for lo, hi, _ in got] == [(0.0, 0.05), (0.05, 0.3), (0.3, 0.6), (0.6, 1.01)]
    assert RvOmission.dice_buckets([], edges) == [(0.0, 0.05, 0), (0.05, 0.3, 0), (0.3, 0.6, 0), (0.6, 1.01, 0)]


def test_select_bias_picks_best_mean_when_cav_held():
    # RV rises with b, cav flat -> pick the b with the best foreground mean (b=2.0 here).
    sweep = {0.0: {"RV": 0.40, "myo": 0.60, "cav": 0.70},
             1.0: {"RV": 0.46, "myo": 0.60, "cav": 0.70},
             2.0: {"RV": 0.50, "myo": 0.59, "cav": 0.70}}
    assert RvOmission.select_bias(sweep) == 2.0


def test_select_bias_guards_cav_regression():
    # b=2.0 has the best mean BUT cav drops 0.70->0.66 (> guard 0.01) -> declined; b=1.0 kept.
    sweep = {0.0: {"RV": 0.40, "myo": 0.60, "cav": 0.70},
             1.0: {"RV": 0.47, "myo": 0.60, "cav": 0.70},
             2.0: {"RV": 0.60, "myo": 0.60, "cav": 0.66}}
    assert RvOmission.select_bias(sweep) == 1.0


def test_select_bias_declines_when_lever_only_hurts():
    # every b < b=0 mean (or breaks the cav guard) -> fall back to no bias.
    sweep = {0.0: {"RV": 0.50, "myo": 0.60, "cav": 0.70},
             1.0: {"RV": 0.48, "myo": 0.59, "cav": 0.69},
             2.0: {"RV": 0.45, "myo": 0.58, "cav": 0.62}}
    assert RvOmission.select_bias(sweep) == 0.0


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


def test_gated_bias_preserves_healthy_rv_and_recovers_omission():
    # ru27: gate the RV bias on per-slice max RV softmax < tau. z0 healthy (RV wins center strongly ->
    # NOT gated), z1 under-fired (bg just over RV everywhere -> gated). Global bias over-segments z0's
    # border (cav->RV); gating leaves z0 alone yet still recovers z1. Flip-symmetric so TTA is a no-op.
    d, c, h, w = 2, 4, 4, 4
    pat = torch.zeros((d, c, h, w))
    pat[0, 3] = 1.5                                         # z0: cav border baseline (moderate)
    pat[0, 1, 1:3, 1:3] = 3.0; pat[0, 3, 1:3, 1:3] = 0.0   # z0: strong RV center (RV wins, maxP>tau)
    pat[1, 0] = 1.0; pat[1, RV] = 0.6                       # z1: bg just over RV (omission, maxP<tau)
    logits = pat.repeat(4, 1, 1, 1)                         # [K*D=8, C, H, W]: 4 identical flip blocks
    base = RvOmission.biased_pred(logits, d, 0.0)
    glob = RvOmission.biased_pred(logits, d, 2.0)
    gated = RvOmission.gated_biased_pred(logits, d, 2.0, 0.6)
    assert (base[1] == 0).all()                            # z1 omitted: bg wins
    assert base[0, 0, 0] == 3 and base[0, 1, 1] == RV      # z0 healthy: cav border, RV center
    assert (glob[0] != base[0]).any()                      # global over-segments z0 (border cav->RV)
    assert np.array_equal(gated[0], base[0])               # gate preserves healthy z0
    assert (gated[1] == RV).all()                          # gate still recovers omitted z1
