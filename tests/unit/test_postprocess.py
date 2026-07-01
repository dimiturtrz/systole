"""Unit tests for largest-CC post-processing — equivalence classes of the input mask."""
import numpy as np
import pytest

from core.postprocess import largest_cc_per_class, _CUCIM_LABEL


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


@pytest.mark.skipif(_CUCIM_LABEL is None, reason="no cucim (GPU lane) — scipy path covered above")
def test_gpu_cucim_matches_cpu_parity():
    """Linux GPU lane: the cucim largest-CC must give bit-identical output to the scipy CPU path.
    Skipped where cucim is absent (Windows); the scipy path is covered by the tests above."""
    from scipy.ndimage import label as cpu_label
    from core.data.static.labels import FOREGROUND
    rng = np.random.default_rng(0)
    m = np.zeros((6, 64, 64), np.uint8)
    m[:, 16:48, 16:48] = 1; m[:, 24:40, 24:40] = 2; m[:, 28:36, 28:36] = 3
    for _ in range(20):                                  # scatter FP islands to drop
        z, y, x = rng.integers(0, 6), rng.integers(0, 60), rng.integers(0, 60)
        m[z, y:y + 3, x:x + 3] = rng.integers(1, 4)
    ref = np.zeros_like(m)
    for lab in FOREGROUND:
        b = m == lab
        if not b.any():
            continue
        cc, n = cpu_label(b)
        s = np.bincount(cc.ravel()); s[0] = 0
        ref[cc == int(s.argmax())] = lab
    assert np.array_equal(largest_cc_per_class(m), ref)   # cucim == scipy
