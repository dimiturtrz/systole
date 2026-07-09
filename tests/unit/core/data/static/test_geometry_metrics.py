"""Unit tests for the spacing-aware geometry/measurement + metric functions.

Tiny hand-built label arrays (not a fake-anatomy generator) — these test the
math, not a model. Real-data behaviour is covered by tests/integration.
"""
import numpy as np

from core.data.static.labels import LV_CAV as LV_CAVITY
from core.data.static.labels import MYO as LV_MYO
from core.data.static.labels import RV as RV_CAVITY
from core.data.static.mri.base import Base
from core.measure import Measure


def test_expected_volume_equals_count_when_binary():
    """Expected volume on a 0/1 prob map == hard label volume (Σp = count); a fractional voxel
    contributes its fraction (the late-collapse soft readout)."""
    sp = (8.0, 1.5, 1.5)
    prob = np.zeros((1, 4, 4), np.float32); prob[0, :2, :2] = 1.0    # 4 whole voxels
    assert abs(Measure.expected_volume_ml(prob, sp) - 4 * Measure.voxel_volume_ml(sp)) < 1e-9
    prob[0, 0, 2] = 0.5                                              # one half voxel
    assert abs(Measure.expected_volume_ml(prob, sp) - 4.5 * Measure.voxel_volume_ml(sp)) < 1e-9


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
    assert abs(Measure.voxel_volume_ml((8.0, 1.5, 1.5)) - 0.018) < 1e-9


def test_label_volume_scales_with_spacing():
    m = _enclosed_lv()
    v1 = Measure.label_volume_ml(m, LV_CAVITY, (8.0, 1.5, 1.5))
    v2 = Measure.label_volume_ml(m, LV_CAVITY, (16.0, 1.5, 1.5))
    assert abs(v2 - 2 * v1) < 1e-6           # double slice thickness -> double volume


def test_identify_lv_cavity_is_label_3_geometrically():
    lv, scores = Base.identify_lv_cavity(_enclosed_lv())
    assert lv == LV_CAVITY == 3
    assert scores[LV_CAVITY] > scores[RV_CAVITY]   # LV more myo-enclosed than RV


def test_ejection_fraction_default_label_is_lv():
    ed = np.zeros((1, 7, 7), dtype=np.uint8); ed[0, 2:5, 2:5] = LV_CAVITY   # 9 vox
    es = np.zeros((1, 7, 7), dtype=np.uint8); es[0, 3, 3] = LV_CAVITY        # 1 vox
    ef, edv, esv = Measure.ejection_fraction(ed, es, (8.0, 1.5, 1.5))               # default lv_label=3
    assert edv > esv > 0
    assert abs(ef - (9 - 1) / 9 * 100) < 1e-6


def test_ef_nan_when_edv_zero():
    empty = np.zeros((4, 8, 8), dtype=np.uint8)
    ef, edv, esv = Measure.ejection_fraction(empty, empty, (8.0, 1.5, 1.5))
    assert np.isnan(ef) and edv == 0
