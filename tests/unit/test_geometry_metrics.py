"""Unit tests for the spacing-aware geometry/measurement + metric functions.

Tiny hand-built label arrays (not a fake-anatomy generator) — these test the
math, not a model. Real-data behaviour is covered by tests/integration.
"""
import numpy as np

from cardioseg.data.mri.data import identify_lv_cavity, LV_CAVITY, LV_MYO, RV_CAVITY
from cardioseg.evaluation.measure import voxel_volume_ml, label_volume_ml, ejection_fraction
from cardioseg.evaluation.evaluate import dice, hausdorff


def _enclosed_lv():
    """A myo (2) block enclosing an LV core (3); an RV blob (1) off in a corner."""
    m = np.zeros((1, 7, 7), dtype=np.uint8)
    m[0, 1:6, 1:6] = LV_MYO          # 5x5 myocardium block
    m[0, 2:5, 2:5] = LV_CAVITY       # 3x3 cavity fully inside the myo block
    m[0, 0, 0] = RV_CAVITY           # RV blob in the corner (mostly touches bg)
    m[0, 0, 1] = RV_CAVITY
    m[0, 1, 0] = RV_CAVITY
    return m


def test_voxel_volume_mm3_to_ml():
    # 8 * 1.5 * 1.5 mm = 18 mm^3 = 0.018 mL
    assert abs(voxel_volume_ml((8.0, 1.5, 1.5)) - 0.018) < 1e-9


def test_label_volume_scales_with_spacing():
    m = _enclosed_lv()
    v1 = label_volume_ml(m, LV_CAVITY, (8.0, 1.5, 1.5))
    v2 = label_volume_ml(m, LV_CAVITY, (16.0, 1.5, 1.5))
    assert abs(v2 - 2 * v1) < 1e-6           # double slice thickness -> double volume


def test_identify_lv_cavity_is_label_3_geometrically():
    lv, scores = identify_lv_cavity(_enclosed_lv())
    assert lv == LV_CAVITY == 3
    assert scores[LV_CAVITY] > scores[RV_CAVITY]   # LV more myo-enclosed than RV


def test_ejection_fraction_default_label_is_lv():
    ed = np.zeros((1, 7, 7), dtype=np.uint8); ed[0, 2:5, 2:5] = LV_CAVITY   # 9 vox
    es = np.zeros((1, 7, 7), dtype=np.uint8); es[0, 3, 3] = LV_CAVITY        # 1 vox
    ef, edv, esv = ejection_fraction(ed, es, (8.0, 1.5, 1.5))               # default lv_label=3
    assert edv > esv > 0
    assert abs(ef - (9 - 1) / 9 * 100) < 1e-6


def test_ef_nan_when_edv_zero():
    empty = np.zeros((4, 8, 8), dtype=np.uint8)
    ef, edv, esv = ejection_fraction(empty, empty, (8.0, 1.5, 1.5))
    assert np.isnan(ef) and edv == 0


def test_dice_perfect_and_disjoint():
    m = _enclosed_lv()
    assert dice(m, m, LV_CAVITY) == 1.0
    a = np.zeros((4, 8, 8), dtype=np.uint8); a[0, 0, 0] = 1
    b = np.zeros((4, 8, 8), dtype=np.uint8); b[0, 0, 1] = 1
    assert dice(a, b, 1) == 0.0


def test_dice_empty_and_partial_classes():
    z = np.zeros((4, 8, 8), dtype=np.uint8)
    assert dice(z, z, 3) == 1.0                  # both absent -> vacuously perfect (the convention)
    a = np.zeros((4, 8, 8), dtype=np.uint8); a[0, :4, :4] = 3   # 16 vox
    assert dice(a, z, 3) == 0.0                  # one empty -> no overlap
    b = np.zeros((4, 8, 8), dtype=np.uint8); b[0, :4, :2] = 3   # 8 vox, subset of a
    assert abs(dice(a, b, 3) - 2 * 8 / (16 + 8)) < 1e-9         # partial: 2|∩|/(|a|+|b|)


def test_hausdorff_zero_on_identical():
    m = _enclosed_lv()
    assert hausdorff(m, m, LV_CAVITY, spacing=(8.0, 1.5, 1.5)) == 0.0
