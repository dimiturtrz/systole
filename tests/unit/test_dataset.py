"""Unit tests for ACDC dataset helpers (pure; no real data, no GPU)."""
from types import SimpleNamespace

import numpy as np

from core.preprocessing.preprocess import fit_square
from cardioseg.training.dataset import split_patients


def test_fit_square_pads_small_centred():
    a = np.ones((4, 6), dtype=np.float32)
    out = fit_square(a, 8, pad_value=0.0)
    assert out.shape == (8, 8)
    assert out.sum() == 4 * 6                      # content preserved
    assert out[0, 0] == 0 and out[4, 4] == 1       # padded border, centred content


def test_fit_square_crops_large_centred():
    a = np.arange(100, dtype=np.float32).reshape(10, 10)
    out = fit_square(a, 4)
    assert out.shape == (4, 4)
    assert out[0, 0] == a[3, 3]                     # centred crop window


def test_fit_square_mask_keeps_integer_labels():
    m = np.full((5, 5), 3, dtype=np.uint8)
    out = fit_square(m, 9, pad_value=0)
    assert set(np.unique(out).tolist()) == {0, 3}   # only pad + label, no interpolation


def _cases(n):
    return [SimpleNamespace(name=f"patient{i:03d}") for i in range(n)]


def test_split_is_patient_level_and_disjoint():
    cases = _cases(20)
    train, val = split_patients(cases, val_frac=0.2, seed=0)
    tn = {c.name for c in train}
    vn = {c.name for c in val}
    assert tn.isdisjoint(vn)                         # no patient in both folds
    assert len(tn) + len(vn) == 20
    assert len(vn) == 4                              # 20% of 20


def test_split_is_deterministic():
    cases = _cases(20)
    a = [c.name for c in split_patients(cases, seed=0)[1]]
    b = [c.name for c in split_patients(cases, seed=0)[1]]
    assert a == b
