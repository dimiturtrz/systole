"""Nyúl histogram standardization (harmonization, bd qfz). Contract: fit a standard scale from a
cohort, then any image maps onto it -> different-scanner intensity distributions align (strip vendor
variance) while tissue ordering is preserved (piecewise-linear, monotonic)."""
import numpy as np
from cardioseg.preprocessing.normalization.nyul import (
    image_landmarks, fit_standard, transform, LANDMARKS)


def test_fit_standard_monotone_unit_range():
    rng = np.random.default_rng(0)
    rows = np.stack([image_landmarks(rng.normal(m, s, 3000))
                     for m, s in [(100, 15), (500, 80), (0, 1)]])
    std = fit_standard(rows)
    assert std.shape == (len(LANDMARKS),)
    assert np.all(np.diff(std) > 0)                      # strictly increasing
    assert 0.0 <= std[0] < std[-1] <= 1.0                # rescaled to [0,1]


def test_nyul_harmonizes_two_scanners():
    """Two very different intensity scales -> after transform to a shared standard, their percentile
    landmarks match (the point of harmonization)."""
    rng = np.random.default_rng(1)
    a = rng.normal(100, 15, 8000)                        # scanner A
    b = rng.normal(500, 90, 8000)                        # scanner B (different offset + scale)
    std = fit_standard(np.stack([image_landmarks(a), image_landmarks(b)]))
    ta, tb = transform(a, std), transform(b, std)
    pa, pb = np.percentile(ta, [10, 50, 90]), np.percentile(tb, [10, 50, 90])
    assert np.allclose(pa, pb, atol=0.03)               # distributions aligned


def test_transform_preserves_order():
    rng = np.random.default_rng(2)
    img = rng.normal(300, 50, 5000)
    std = fit_standard(np.stack([image_landmarks(img)]))
    t = transform(img, std)
    order = np.argsort(img)
    assert np.all(np.diff(t[order]) >= -1e-9)           # monotone map -> tissue ordering kept
