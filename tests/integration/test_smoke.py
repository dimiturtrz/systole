"""End-to-end smoke on the synthetic fixture — no real data, no GPU needed.

    python -m pytest tests/ -q
"""
import numpy as np

from src.modalities.mri.synth import make_volume, ed_es_pair
from src.core.measure import ejection_fraction
from src.core.evaluate import dice, dice_all


def test_synth_shapes():
    img, mask, spacing = make_volume()
    assert img.shape == mask.shape
    assert set(np.unique(mask)).issubset({0, 1, 2, 3})
    assert spacing.shape == (3,)


def test_ef_in_physiological_range():
    (_, ed_mask, sp), (_, es_mask, _) = ed_es_pair()
    ef, edv, esv = ejection_fraction(ed_mask, es_mask, sp)
    assert edv > esv > 0          # diastole larger than systole
    assert 0 < ef < 100           # EF is a sane percentage


def test_dice_perfect_on_self():
    _, mask, _ = make_volume()
    assert dice(mask, mask, 1) == 1.0
    assert all(v == 1.0 for v in dice_all(mask, mask).values())
