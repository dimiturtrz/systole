"""Module-pair / pipeline-chain integration (data-free, synthetic).

Unit tests check each stage in isolation. These check that stages *chain*: A's output
is a valid input to B, and the A->B chain produces what each unit promises. Pairs cover
the cardioseg pipeline end to end:

    preprocess (resample, zscore) -> dataset (fit_square) -> model (predict_volume)
        -> measure (EF, volumes) <-> evaluate (Dice, surface metrics)

No ACDC data, no real weights, no GPU — synthetic masks/volumes + a deterministic
threshold "model" — so these always run. Inputs are partitioned into equivalence
classes (upsample/downsample/identity; pad/crop; normal/zero/full/degenerate EF).
"""
import numpy as np
import pytest

from cardioseg.preprocessing.preprocess import resample_inplane, zscore
from cardioseg.training.dataset import fit_square
from cardioseg.evaluation.validate import predict_volume
from cardioseg.evaluation.measure import ejection_fraction, label_volume_ml, voxel_volume_ml
from cardioseg.evaluation.evaluate import (
    dice, assd, hd95, hausdorff, surface_distances, surface_metrics,
)


def _cube(d, hw, half, label=3, value=None):
    """[d, hw, hw] array with a centred (2*half)^3 cube. Mask if value is None (uses label).

    Ranges are explicitly clamped to [0, n] — a negative slice start would *wrap*, not clamp.
    """
    fill = label if value is None else value
    dtype = np.uint8 if value is None else np.float32
    a = np.zeros((d, hw, hw), dtype=dtype)

    def span(n):
        return max(0, n // 2 - half), min(n, n // 2 + half)

    z0, z1 = span(d)
    y0, y1 = span(hw)
    a[z0:z1, y0:y1, y0:y1] = fill
    return a


# === preprocess -> dataset ================================================
# resample_inplane (rescale H,W to target mm) feeds fit_square (centre pad/crop
# to the model's square grid). The chain must always land on [D, size, size].

@pytest.mark.parametrize("spacing_in,note", [
    ((10.0, 3.0, 3.0), "downsample: source coarser than 1.5mm -> shrinks"),
    ((10.0, 0.75, 0.75), "upsample: source finer than 1.5mm -> grows"),
    ((10.0, 1.5, 1.5), "identity: already at target"),
])
def test_resample_then_fit_square_lands_on_grid(spacing_in, note):
    vol = np.random.rand(3, 48, 56).astype(np.float32)
    res, sp = resample_inplane(vol, spacing_in, target_inplane=1.5)
    assert sp == (10.0, 1.5, 1.5)                       # spacing convention carried through
    size = 64
    stack = np.stack([fit_square(s, size) for s in res])
    assert stack.shape == (3, size, size), note         # both crop & pad paths reach the grid
    assert stack.dtype == res.dtype


def test_mask_chain_preserves_integer_labels():
    """A label volume through resample(is_mask) -> fit_square stays integer-labelled (no blur)."""
    mask = _cube(3, 48, 10, label=3)
    res, _ = resample_inplane(mask, (10.0, 3.0, 3.0), target_inplane=1.5, is_mask=True)
    stack = np.stack([fit_square(s, 128) for s in res])  # 128 > resampled H,W -> pad path
    assert set(np.unique(stack).tolist()).issubset({0, 3})
    assert stack.shape == (3, 128, 128)


def test_zscore_then_fit_square_pad_is_background():
    """zscore centres the volume at ~0; fit_square pads with 0 -> padding == the mean (no bias)."""
    vol = (np.random.rand(2, 30, 30) * 50 + 100).astype(np.float32)  # uncalibrated intensities
    z = zscore(vol)
    assert abs(float(z.mean())) < 1e-4
    padded = np.stack([fit_square(s, 80) for s in z])   # 80 > 30 -> all-pad border
    border = padded[:, 0, 0]                              # a corner that is pure padding
    assert np.allclose(border, 0.0)                      # pad value sits at the z-score mean


# === dataset -> evaluate ==================================================
# fit_square'd label stacks are valid Dice inputs (same shape both sides -> no error).

def test_squared_stacks_are_valid_dice_inputs():
    a = np.stack([fit_square(_cube(1, 40, 8)[0], 64) for _ in range(4)])
    b = np.stack([fit_square(_cube(1, 40, 5)[0], 64) for _ in range(4)])
    assert dice(a, a, 3) == 1.0                           # identity
    disjoint = np.zeros_like(a); disjoint[a == 0] = 3     # never overlaps a
    assert dice(a, disjoint, 3) == 0.0                    # disjoint
    assert 0.0 < dice(a, b, 3) < 1.0                      # partial (b inside a)


# === model -> measure / evaluate ==========================================
# A deterministic threshold "model": bright voxels -> LV-cav (label 3), else bg.
# Exercises the real predict_volume plumbing (slice loop, fit_square, argmax, stack)
# without trained weights, then feeds measure (EF) and evaluate (Dice) the result.

class _ThreshModel:
    """Stand-in for the U-Net: forward([1,1,H,W]) -> logits [1,4,H,W], argmax = label 3 where bright."""
    def eval(self):
        return self

    def __call__(self, x):
        import torch
        b, _, h, w = x.shape
        logits = torch.zeros(b, 4, h, w)
        bright = (x[:, 0] > 0.5).float()
        logits[:, 3] = bright * 10.0
        logits[:, 0] = (1.0 - bright) * 10.0
        return logits


def test_predict_then_ejection_fraction():
    """predict_volume output is a valid EF input; bigger ED pool than ES -> EF in (0,100)."""
    pytest.importorskip("torch")
    model = _ThreshModel()
    ed_vol = _cube(8, 64, 12, value=1.0)                  # large bright pool
    es_vol = _cube(8, 64, 7, value=1.0)                   # small bright pool
    pred_ed = predict_volume(model, ed_vol, size=64, device="cpu")
    pred_es = predict_volume(model, es_vol, size=64, device="cpu")
    assert pred_ed.shape == (8, 64, 64)
    assert set(np.unique(pred_ed).tolist()) == {0, 3}     # model emits bg + LV-cav only
    ef, edv, esv = ejection_fraction(pred_ed, pred_es, (10.0, 1.5, 1.5))
    assert edv > esv > 0 and 0 < ef < 100


def test_predict_matches_groundtruth_threshold():
    """predict_volume vs the equivalent hand-thresholded GT -> Dice 1.0 (chain is faithful)."""
    pytest.importorskip("torch")
    model = _ThreshModel()
    vol = _cube(6, 64, 10, value=1.0)
    pred = predict_volume(model, vol, size=64, device="cpu")
    gt = np.where(vol > 0.5, 3, 0).astype(np.uint8)        # what the threshold model "should" give
    assert dice(pred, gt, 3) == 1.0
    assert hd95(pred, gt, 3) == 0.0                        # identical surfaces -> zero distance


# === measure <-> evaluate (one mask pair feeds both) ======================

def test_one_mask_pair_feeds_ef_and_overlap():
    ed, es = _cube(8, 64, 10), _cube(8, 64, 6)
    spacing = (10.0, 1.5, 1.5)
    ef, edv, esv = ejection_fraction(ed, es, spacing)
    assert edv > esv > 0 and 0 < ef < 100
    assert np.isclose(edv, label_volume_ml(ed, 3, spacing))
    assert dice(ed, ed, 3) == 1.0
    assert dice(ed, es, 3) < 1.0
    assert assd(ed, es, 3) > 0


@pytest.mark.parametrize("ed_half,es_half,lo,hi", [
    (10, 10, -0.01, 0.01),   # ED == ES   -> EF ~ 0   (no contraction)
    (10, 0,  99.99, 100.01), # ES empty   -> EF ~ 100 (full ejection)
    (10, 6,  60.0, 80.0),    # normal contraction band
])
def test_ef_equivalence_classes(ed_half, es_half, lo, hi):
    ed = _cube(8, 64, ed_half)
    es = _cube(8, 64, es_half) if es_half else np.zeros((8, 64, 64), np.uint8)
    ef, _, _ = ejection_fraction(ed, es, (10.0, 1.5, 1.5))
    assert lo <= ef <= hi


def test_degenerate_ed_propagates_nan_not_crash():
    """Empty ED (EDV==0) -> EF NaN; evaluate stays defined on the same input (no crash)."""
    empty = np.zeros((8, 64, 64), dtype=np.uint8)
    es = _cube(8, 64, 6)
    ef, edv, _ = ejection_fraction(empty, es, (10.0, 1.5, 1.5))
    assert edv == 0 and np.isnan(ef)
    assert dice(empty, empty, 3) == 1.0                   # vacuous overlap defined
    sd = surface_distances(empty, es, 3)
    assert sd.size == 0 or np.all(np.isnan(sd))           # absent label -> empty/nan, not error


def test_ef_invariant_to_spacing_volumes_scale():
    """EF is a ratio -> spacing cancels; absolute volumes scale by the voxel-volume factor."""
    ed, es = _cube(8, 64, 10), _cube(8, 64, 6)
    ef1, edv1, _ = ejection_fraction(ed, es, (10.0, 1.5, 1.5))
    ef2, edv2, _ = ejection_fraction(ed, es, (5.0, 0.75, 0.75))  # every dim halved -> 1/8 volume
    assert np.isclose(ef1, ef2)
    assert np.isclose(edv2 / edv1, voxel_volume_ml((5.0, 0.75, 0.75)) / voxel_volume_ml((10.0, 1.5, 1.5)))


# === surface chain: distances -> metrics -> hd/hd95/assd ==================

def test_surface_metric_ordering_is_consistent():
    """surface_distances -> surface_metrics agrees with the hd/hd95/assd helpers, mean<=p95<=max."""
    a = _cube(8, 64, 12)
    b = _cube(8, 64, 8)                                    # concentric -> a gap all around
    sd = surface_distances(a, b, 3, spacing=(1.5, 1.5, 1.5))
    m = surface_metrics(sd)
    assert m["assd"] <= m["hd95"] <= m["hd"]               # mean <= 95th pct <= max
    assert np.isclose(m["hd"], hausdorff(a, b, 3, (1.5, 1.5, 1.5)))
    assert np.isclose(m["hd95"], hd95(a, b, 3, (1.5, 1.5, 1.5)))
    assert np.isclose(m["assd"], assd(a, b, 3, (1.5, 1.5, 1.5)))
