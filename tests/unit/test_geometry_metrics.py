"""Unit tests for the spacing-aware geometry/measurement + metric functions."""
import numpy as np

from cardioseg.data.mri.synth import make_volume, ed_es_pair, LABELS
from cardioseg.data.mri.data import identify_lv_cavity
from cardioseg.evaluation.measure import voxel_volume_ml, label_volume_ml, ejection_fraction
from cardioseg.evaluation.evaluate import dice, hausdorff


def test_voxel_volume_mm3_to_ml():
    # 8 * 1.5 * 1.5 mm = 18 mm^3 = 0.018 mL
    assert abs(voxel_volume_ml((8.0, 1.5, 1.5)) - 0.018) < 1e-9


def test_label_volume_scales_with_spacing():
    _, mask, sp = make_volume()
    v1 = label_volume_ml(mask, LABELS["lv"], sp)
    v2 = label_volume_ml(mask, LABELS["lv"], (sp[0] * 2, sp[1], sp[2]))
    assert abs(v2 - 2 * v1) < 1e-6           # double slice thickness -> double volume


def test_identify_lv_cavity_is_label_3():
    # synth uses the verified ACDC convention: LV cavity (enclosed by myo) = 3.
    _, mask, _ = make_volume()
    lv, scores = identify_lv_cavity(mask)
    assert lv == LABELS["lv"] == 3
    assert scores[3] > scores[1]              # LV more myo-enclosed than RV


def test_ejection_fraction_default_label_is_lv():
    (_, ed, sp), (_, es, _) = ed_es_pair()
    ef, edv, esv = ejection_fraction(ed, es, sp)      # default lv_label=3
    assert edv > esv > 0
    assert 0 < ef < 100


def test_ef_nan_when_edv_zero():
    empty = np.zeros((4, 8, 8), dtype=np.uint8)
    ef, edv, esv = ejection_fraction(empty, empty, (8.0, 1.5, 1.5))
    assert np.isnan(ef) and edv == 0


def test_dice_perfect_and_disjoint():
    _, mask, _ = make_volume()
    assert dice(mask, mask, LABELS["lv"]) == 1.0
    a = np.zeros((4, 8, 8), dtype=np.uint8); a[0, 0, 0] = 1
    b = np.zeros((4, 8, 8), dtype=np.uint8); b[0, 0, 1] = 1
    assert dice(a, b, 1) == 0.0


def test_hausdorff_zero_on_identical():
    _, mask, sp = make_volume()
    assert hausdorff(mask, mask, LABELS["lv"], spacing=sp) == 0.0
