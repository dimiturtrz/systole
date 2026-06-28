"""Unit tests for largest-CC post-processing — equivalence classes of the input mask."""
import numpy as np

from core.postprocess import largest_cc_per_class


def _blob(mask, lab, z, y, x, r):
    mask[z, y - r:y + r, x - r:x + r] = lab


def test_no_islands_unchanged():
    """Single component per class -> identical output (identity class)."""
    m = np.zeros((4, 32, 32), np.uint8)
    _blob(m, 1, 2, 8, 8, 3)
    _blob(m, 3, 2, 20, 20, 4)
    out = largest_cc_per_class(m)
    assert np.array_equal(out, m)


def test_island_dropped_largest_kept():
    """A small FP island of a class is removed; the big component survives."""
    m = np.zeros((4, 32, 32), np.uint8)
    _blob(m, 3, 2, 16, 16, 5)   # main blood pool
    m[0, 0, 0] = 3              # stray speck, disconnected
    out = largest_cc_per_class(m)
    assert out[0, 0, 0] == 0           # island gone
    assert (out == 3).sum() == (m[2] == 3).sum()  # main kept intact


def test_each_class_independent():
    """Cleaning one class doesn't touch another; absent class stays absent."""
    m = np.zeros((4, 32, 32), np.uint8)
    _blob(m, 1, 2, 8, 8, 3)     # RV, single
    _blob(m, 2, 2, 20, 20, 4)   # myo, single
    m[3, 31, 31] = 1            # RV island
    out = largest_cc_per_class(m)
    assert (out == 2).sum() == (m == 2).sum()      # myo untouched
    assert out[3, 31, 31] == 0                      # RV island dropped
    assert (out == 3).sum() == 0                    # absent class stays absent


def test_empty_mask_returns_empty():
    m = np.zeros((3, 16, 16), np.uint8)
    assert largest_cc_per_class(m).sum() == 0
