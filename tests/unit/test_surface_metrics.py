"""Surface-distance metrics (HD / HD95 / ASSD) from the boundary-distance array."""
import math

import numpy as np

from cardioseg.evaluation.evaluate import surface_distances, surface_metrics, hausdorff, hd95, assd


def _square(n=30, lo=10, hi=20, shift=0, label=3):
    a = np.zeros((n, n), int)
    a[lo:hi, lo + shift:hi + shift] = label
    return a


def test_identical_is_zero():
    a = _square()
    sd = surface_distances(a, a, 3)
    assert sd.size > 0
    m = surface_metrics(sd)
    assert m["hd"] == 0 and m["hd95"] == 0 and m["assd"] == 0


def test_metrics_ordered_assd_le_hd95_le_hd():
    m = surface_metrics(surface_distances(_square(), _square(shift=3), 3))
    assert m["assd"] <= m["hd95"] <= m["hd"]


def test_hd_matches_shift():
    # boundary shifted 3 voxels -> worst boundary gap ~3 (spacing 1)
    h = hausdorff(_square(), _square(shift=3), 3)
    assert 3 - 1e-6 <= h <= 3 + 1.0


def test_hd95_robust_to_one_outlier():
    a = _square()
    b = _square()
    b[0, 0] = 3  # a stray far speck
    full, robust = hausdorff(a, b, 3), hd95(a, b, 3)
    assert full > robust  # HD detonates on the outlier, HD95 shrugs it off


def test_spacing_scales_to_mm():
    a, b = _square(), _square(shift=2)
    vox = hausdorff(a, b, 3)
    mm = hausdorff(a, b, 3, spacing=(2.0, 2.0))
    assert math.isclose(mm, 2 * vox, rel_tol=1e-6)


def test_absent_label_is_nan():
    a = np.zeros((10, 10), int)
    b = _square(n=10, lo=2, hi=5)
    assert math.isnan(hausdorff(a, b, 3))
    assert math.isnan(assd(a, b, 3))
