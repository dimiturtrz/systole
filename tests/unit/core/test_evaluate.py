"""Evaluate metrics: Dice + boundary-distance summaries (HD / HD95 / ASSD).

Tiny hand-built label arrays test the math, not a model. Real-data behaviour is covered
by tests/integration. The dice equivalence classes (perfect / disjoint / one-empty /
both-empty / partial) + the surface-distance summaries live together here (mirror of
core/evaluate.py).
"""
import math

import numpy as np

from core.evaluate import Evaluate

# --- dice ---


def _enclosed_lv():
    """A myo (2) block enclosing an LV core (3); an RV blob (1) off in a corner."""
    m = np.zeros((1, 7, 7), dtype=np.uint8)
    m[0, 1:6, 1:6] = 2               # 5x5 myocardium block
    m[0, 2:5, 2:5] = 3               # 3x3 cavity fully inside the myo block
    m[0, 0, 0] = 1                   # RV blob in the corner
    m[0, 0, 1] = 1
    m[0, 1, 0] = 1
    return m


def test_dice_perfect_and_disjoint():
    m = _enclosed_lv()
    assert Evaluate.dice(m, m, 3) == 1.0
    a = np.zeros((4, 8, 8), dtype=np.uint8); a[0, 0, 0] = 1
    b = np.zeros((4, 8, 8), dtype=np.uint8); b[0, 0, 1] = 1
    assert Evaluate.dice(a, b, 1) == 0.0


def test_dice_empty_and_partial_classes():
    z = np.zeros((4, 8, 8), dtype=np.uint8)
    assert Evaluate.dice(z, z, 3) == 1.0                  # both absent -> vacuously perfect (the convention)
    a = np.zeros((4, 8, 8), dtype=np.uint8); a[0, :4, :4] = 3   # 16 vox
    assert Evaluate.dice(a, z, 3) == 0.0                  # one empty -> no overlap
    b = np.zeros((4, 8, 8), dtype=np.uint8); b[0, :4, :2] = 3   # 8 vox, subset of a
    assert abs(Evaluate.dice(a, b, 3) - 2 * 8 / (16 + 8)) < 1e-9         # partial: 2|∩|/(|a|+|b|)


# --- surface distances (HD / HD95 / ASSD) ---


def _square(n=30, lo=10, hi=20, shift=0, label=3):
    mask = np.zeros((n, n), int)
    mask[lo:hi, lo + shift:hi + shift] = label
    return mask


def _hd(mask_a, mask_b, label, spacing=None):
    return Evaluate.surface_metrics(Evaluate.surface_distances(mask_a, mask_b, label, spacing)).hd


def test_identical_is_zero():
    mask = _square()
    surf_dist = Evaluate.surface_distances(mask, mask, 3)
    assert surf_dist.size > 0
    surf = Evaluate.surface_metrics(surf_dist)
    assert surf.hd == 0 and surf.hd95 == 0 and surf.assd == 0


def test_metrics_ordered_assd_le_hd95_le_hd():
    surf = Evaluate.surface_metrics(Evaluate.surface_distances(_square(), _square(shift=3), 3))
    assert surf.assd <= surf.hd95 <= surf.hd


def test_hd_matches_shift():
    # boundary shifted 3 voxels -> worst boundary gap ~3 (spacing 1)
    h = _hd(_square(), _square(shift=3), 3)
    assert 3 - 1e-6 <= h <= 3 + 1.0


def test_hd95_robust_to_one_outlier():
    a = _square()
    b = _square()
    b[0, 0] = 3  # a stray far speck
    full, robust = _hd(a, b, 3), Evaluate.hd95(a, b, 3)
    assert full > robust  # HD detonates on the outlier, HD95 shrugs it off


def test_spacing_scales_to_mm():
    a, b = _square(), _square(shift=2)
    vox = _hd(a, b, 3)
    mm = _hd(a, b, 3, spacing=(2.0, 2.0))
    assert math.isclose(mm, 2 * vox, rel_tol=1e-6)


def test_absent_label_is_nan():
    a = np.zeros((10, 10), int)
    b = _square(n=10, lo=2, hi=5)
    assert math.isnan(Evaluate.hd95(a, b, 3))
    assert math.isnan(Evaluate.assd(a, b, 3))
