"""End-to-end integration on REAL ACDC (no model/GPU): load -> identify LV ->
measure EF -> Dice. Skips if the ACDC data isn't present, so a fresh clone
without the (gated, out-of-repo) dataset still passes.
"""
import numpy as np
import pytest

from core.data.static.mri.acdc import (
    acdc_cases, load_ed_es, identify_lv_cavity, LV_CAVITY,
)
from core.measure import ejection_fraction
from core.evaluate import dice

_CASES = acdc_cases()
needs_data = pytest.mark.skipif(not _CASES, reason="ACDC data not present (set CARDIAC_DATA_ROOT)")


@needs_data
def test_real_patient_labels_and_lv_identification():
    d = load_ed_es(_CASES[0])
    gt = d["ED"]["gt"]
    assert set(np.unique(gt).tolist()).issubset({0, 1, 2, 3})
    lv, scores = identify_lv_cavity(gt)
    assert lv == LV_CAVITY == 3                      # LV cavity is label 3 on real masks
    assert scores[LV_CAVITY] == max(scores.values())


@needs_data
def test_real_ef_is_physiological():
    d = load_ed_es(_CASES[0])
    ef, edv, esv = ejection_fraction(d["ED"]["gt"], d["ES"]["gt"], d["spacing"])
    assert edv > esv > 0                              # diastole larger than systole
    assert 0 < ef < 100


@needs_data
def test_dice_perfect_on_real_mask_self():
    gt = load_ed_es(_CASES[0])["ED"]["gt"]
    assert dice(gt, gt, LV_CAVITY) == 1.0
